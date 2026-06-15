"""
CQT Baseline for ALS Detection
Dataset: VOC-ALS (VOiCe signals in Amyotrophic Lateral Sclerosis)
"""

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import scipy.signal
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. AUDIO PREPROCESSING UTILITIES
# =============================================================================

class AudioPreprocessor:
    """
    Loads and preprocesses raw audio for LEAF.
    VOC-ALS recordings are typically sustained vowels (/a/) at various sample rates.
    """
    def __init__(self, target_sr=16000, duration=1.5):
        self.target_sr = target_sr
        self.target_len = int(target_sr * duration)  # 24,000 samples for 1.5s
    
    def load_audio(self, path):
        """Load audio using scipy (no torchaudio dependency)."""
        try:
            sr, wav = scipy.io.wavfile.read(path)
        except:
            # Fallback for different formats
            import soundfile as sf
            wav, sr = sf.read(path)
        
        # Convert to float32 in [-1, 1]
        if wav.dtype == np.int16:
            wav = wav.astype(np.float32) / 32768.0
        elif wav.dtype == np.int32:
            wav = wav.astype(np.float32) / 2147483648.0
        else:
            wav = wav.astype(np.float32)
        
        # Mono
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        
        # Resample if needed (simple linear interpolation for speed)
        if sr != self.target_sr:
            wav = self._resample(wav, sr, self.target_sr)
        
        return wav.astype(np.float32)
    
    def _resample(self, x, orig_sr, target_sr):
        """Fast scipy resampling using polyphase filtering."""
        gcd = math.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        return scipy.signal.resample_poly(x, up, down)
    
    def normalize_length(self, wav):
        """Pad with zeros or truncate to fixed length."""
        if len(wav) < self.target_len:
            pad = self.target_len - len(wav)
            wav = np.concatenate([wav, np.zeros(pad, dtype=np.float32)])
        else:
            wav = wav[:self.target_len]
        return wav
    
    def augment(self, wav, noise_level=0.005):
        # Random noise
        noise = np.random.randn(len(wav)).astype(np.float32) * noise_level
        wav = wav + noise
        
        # Random time shift (circular or zero-pad)
        shift = np.random.randint(-800, 800)  # ~50ms at 16kHz
        if shift != 0:
            wav = np.roll(wav, shift)
            if shift > 0:
                wav[:shift] = 0
            else:
                wav[shift:] = 0
        
        # Random amplitude scaling
        scale = np.random.uniform(0.9, 1.1)
        wav = wav * scale
        
        return wav.astype(np.float32)

    def extract_cqt(self, wav, n_bins=72, hop_length=256):
        # Compute CQT
        C = librosa.cqt(wav, sr=self.target_sr, hop_length=hop_length, fmin=librosa.note_to_hz('C1'), n_bins=n_bins, bins_per_octave=12)
        C = librosa.amplitude_to_db(np.abs(C), ref=np.max)
        return C.astype(np.float32)


# =============================================================================
# 3. CLASSIFIER (CNN ON LEAF FEATURES)
# =============================================================================

class ResNetBlock(nn.Module):
    """
    Standard residual block with 2D convolutions.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ALSClassifier(nn.Module):
    """
    ResNet-style classifier on top of LEAF features.
    Input: LEAF spectrogram [batch, n_filters, time]
    Output: logits [batch, num_classes]
    """
    def __init__(self, n_filters=40, num_classes=2, dropout=0.3):
        super().__init__()
        
        # Treat LEAF output as a 1-channel "image" [B, 1, n_filters, time]
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.layer1 = ResNetBlock(32, 32, stride=1)
        self.layer2 = ResNetBlock(32, 64, stride=2)   # downsamples spatial dims by 2
        self.layer3 = ResNetBlock(64, 128, stride=2)  # downsamples spatial dims by 2
        self.layer4 = ResNetBlock(128, 256, stride=2) # downsamples spatial dims by 2
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, leaf_features):
        # Add channel dimension: [B, 1, n_filters, time]
        x = leaf_features.unsqueeze(1)
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# =============================================================================
# 4. COMPLETE MODEL
# =============================================================================

class CQT_ALS_Model(nn.Module):
    """End-to-end: CQT -> CNN -> ALS/Healthy."""
    def __init__(self, n_filters=72, sr=16000, num_classes=2):
        super().__init__()
        self.classifier = ALSClassifier(n_filters=n_filters, num_classes=num_classes)
    
    def forward(self, features):
        # features: [B, n_filters, time]
        logits = self.classifier(features)  # [B, num_classes]
        return logits, features


# =============================================================================
# 5. DATASET FOR VOC-ALS
# =============================================================================

class VOCALSDataset(Dataset):
    """
    VOC-ALS dataset loader.
    Expected structure:
        data_dir/
            audio/
                ALS_001_a.wav
                H_001_a.wav
                ...
            metadata.csv (columns: filename, label, patient_id, vowel, ...)
    
    Labels:
        1 = ALS
        0 = Healthy
    """
    def __init__(self, data_dir, metadata_csv='metadata.csv', 
                 sr=16000, duration=1.5, augment=False, preprocessor=None):
        self.data_dir = data_dir
        self.audio_dir = os.path.join(data_dir, 'audio')
        self.augment = augment
        
        self.preprocessor = preprocessor if preprocessor else AudioPreprocessor(sr, duration)
        
        # Load metadata
        meta_path = os.path.join(data_dir, metadata_csv)
        if os.path.exists(meta_path):
            self.meta = pd.read_csv(meta_path)
        else:
            # Auto-generate from filenames if no CSV
            files = sorted([f for f in os.listdir(self.audio_dir) if f.endswith('.wav')])
            labels = [1 if f.startswith('ALS') else 0 for f in files]
            self.meta = pd.DataFrame({
                'filename': files,
                'label': labels
            })
        
        self.files = self.meta['filename'].values
        self.labels = self.meta['label'].values
        
        # Cache preprocessed audio for speed
        self.cache = {}
        print(f"Loaded {len(self)} samples from VOC-ALS")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if idx in self.cache:
            wav = self.cache[idx]
        else:
            fname = self.files[idx]
            fpath = os.path.join(self.audio_dir, fname)
            
            # Load and preprocess
            wav = self.preprocessor.load_audio(fpath)
            wav = self.preprocessor.normalize_length(wav)
            self.cache[idx] = wav
        
        # Apply augmentation if enabled
        if self.augment:
            wav = self.preprocessor.augment(wav)
            
        cqt_feat = self.preprocessor.extract_cqt(wav, n_bins=72, hop_length=256)
        
        # Convert to tensor
        waveform = torch.from_numpy(cqt_feat).float()  # [n_bins, time]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return waveform, label


# =============================================================================
# 6. TRAINING LOOP
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for waveforms, labels in loader:
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(waveforms)
        loss = criterion(logits, labels)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * waveforms.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for waveforms, labels in loader:
            waveforms = waveforms.to(device)
            logits, _ = model(waveforms)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # P(ALS)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    return acc, f1, auc, all_preds, all_labels, all_probs


# =============================================================================
# 7. MAIN EXECUTION
# =============================================================================

def main():
    # Configuration
    CONFIG = {
        'data_dir': '/Users/mansikhamar/Desktop/College/Research/leaf-als/data/VOC-ALS',          # <-- CHANGE THIS
        'sr': 16000,
        'duration': 1.5,
        'n_filters': 72,                  # LEAF filterbank size
        'batch_size': 16,
        'lr': 1e-3,
        'epochs': 30,                     # Increased epochs for convergence
        'device': 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    }
    
    print(f"Using device: {CONFIG['device']}")
    
    # Preprocessor
    preprocessor = AudioPreprocessor(CONFIG['sr'], CONFIG['duration'])
    
    # Load separate dataset instances for train vs val/test to prevent leakage of augment = True
    train_dataset = VOCALSDataset(
        CONFIG['data_dir'],
        sr=CONFIG['sr'],
        duration=CONFIG['duration'],
        augment=True,
        preprocessor=preprocessor
    )
    val_dataset = VOCALSDataset(
        CONFIG['data_dir'],
        sr=CONFIG['sr'],
        duration=CONFIG['duration'],
        augment=False,
        preprocessor=preprocessor
    )
    test_dataset = VOCALSDataset(
        CONFIG['data_dir'],
        sr=CONFIG['sr'],
        duration=CONFIG['duration'],
        augment=False,
        preprocessor=preprocessor
    )
    
    # Train/val/test split (stratified)
    indices = np.arange(len(train_dataset))
    labels = train_dataset.labels
    
    # First: train+val vs test (80/20)
    trainval_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Then: train vs val (75/25 of trainval)
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=0.25, 
        stratify=labels[trainval_idx], random_state=42
    )
    
    # Create subsets using the correct instances
    train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(val_dataset, val_idx)
    test_dataset = torch.utils.data.Subset(test_dataset, test_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], 
                             shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Model
    model = CQT_ALS_Model(
        n_filters=CONFIG['n_filters'],
        sr=CONFIG['sr'],
        num_classes=2
    ).to(CONFIG['device'])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    # Split optimizer learning rates: lower learning rate for the sensitive LEAF frontend
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Calculate class weights dynamically from training labels to handle class imbalance
    train_labels = labels[train_idx]
    class_counts = np.bincount(train_labels)
    class_weights = len(train_labels) / (len(class_counts) * class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(CONFIG['device'])
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Training
    best_val_f1 = 0.0
    best_state = None
    
    for epoch in range(CONFIG['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, CONFIG['device'])
        val_acc, val_f1, val_auc, _, _, _ = evaluate(model, val_loader, CONFIG['device'])
        
        scheduler.step(val_f1)
        
        print(f"Epoch {epoch+1:02d} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
            print(f"  -> New best model! (Val F1: {val_f1:.4f})")
    
    # Load best model and test
    model.load_state_dict(best_state)
    test_acc, test_f1, test_auc, preds, labels, probs = evaluate(model, test_loader, CONFIG['device'])
    
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"F1-Score:  {test_f1:.4f}")
    print(f"AUC-ROC:   {test_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=['Healthy', 'ALS']))
    
    
    
    # Save model
    torch.save({
        'model': model.state_dict(),
        'config': CONFIG,
        'test_metrics': {'acc': test_acc, 'f1': test_f1, 'auc': test_auc}
    }, 'cqt_als_best.pth')
    print("\nModel saved to cqt_als_best.pth")


if __name__ == '__main__':
    main()
