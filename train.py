"""
LEAF (Learnable Frontend) from scratch for ALS Detection
Dataset: VOC-ALS (VOiCe signals in Amyotrophic Lateral Sclerosis)
"""

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        """Simple augmentation: random noise + small time shift."""
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


# =============================================================================
# 2. LEAF: LEARNABLE FRONTEND
# =============================================================================

class GaborFilterbank(nn.Module):
    """
    Learnable Gabor filterbank.
    Each filter is a Gaussian-windowed complex sinusoid:
        h(t) = exp(-t^2 / (2*sigma^2)) * exp(i * 2*pi * f * t)
    
    Learnable parameters per filter:
        - center frequency (f)
        - bandwidth (sigma)
    
    Initialized on a mel-scale or log-scale to mimic auditory frequency spacing.
    """
    def __init__(self, n_filters=40, filter_len=401, sr=16000, 
                 min_freq=80.0, max_freq=4000.0):
        super().__init__()
        self.n_filters = n_filters
        self.filter_len = filter_len  # 401 samples @ 16kHz = 25ms
        self.sr = sr
        
        # Time vector for filter construction: [-L/2, ..., L/2]
        half_len = (filter_len - 1) // 2
        self.register_buffer('t', torch.arange(-half_len, half_len + 1).float() / sr)
        
        # Initialize center frequencies on mel-scale
        mel_min = 2595.0 * math.log10(1.0 + min_freq / 700.0)
        mel_max = 2595.0 * math.log10(1.0 + max_freq / 700.0)
        mels = torch.linspace(mel_min, mel_max, n_filters)
        f_init = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
        
        # Initialize bandwidths: constant-Q like (sigma proportional to 1/f)
        # In LEAF, bandwidth is learnable but initialized narrow
        sigma_init = 1.5 / f_init  # ~1.5 cycles at center freq
        
        # Parameterize in log-space to ensure positivity and stable gradient scaling
        self.log_f = nn.Parameter(torch.log(f_init))
        self.log_sigma = nn.Parameter(torch.log(sigma_init))
        
        # Optional: initialize gains (per-filter amplitude)
        self.gain = nn.Parameter(torch.ones(n_filters))
    
    def _build_filters(self):
        """
        Construct real and imaginary Gabor filters on every forward pass.
        This allows gradients to flow back to log_f and log_sigma.
        """
        # Shape: [n_filters, 1, filter_len]
        t = self.t.unsqueeze(0)           # [1, filter_len]
        f = torch.exp(self.log_f).unsqueeze(1)         # [n_filters, 1]
        sigma = torch.exp(self.log_sigma).unsqueeze(1) # [n_filters, 1]
        gain = self.gain.unsqueeze(1)   # [n_filters, 1]
        
        # Gaussian envelope: exp(-t^2 / (2*sigma^2))
        envelope = torch.exp(-0.5 * (t / sigma) ** 2)
        
        # Normalize envelope to unit energy
        envelope = envelope / (envelope.sum(dim=1, keepdim=True) + 1e-8)
        
        # Complex sinusoid
        sinusoid = 2.0 * math.pi * f * t  # [n_filters, filter_len]
        
        real_filter = gain * envelope * torch.cos(sinusoid)   # [n_filters, filter_len]
        imag_filter = gain * envelope * torch.sin(sinusoid)   # [n_filters, filter_len]
        
        return real_filter.unsqueeze(1), imag_filter.unsqueeze(1)  # [n, 1, L]
    
    def forward(self, x):
        """
        x: [batch, 1, time] raw waveform
        Returns: energy [batch, n_filters, time]
        """
        real_kernels, imag_kernels = self._build_filters()  # [n_filters, 1, L]
        
        # Complex convolution
        real_out = F.conv1d(x, real_kernels, padding='same', groups=1)   # [B, n_filt, T]
        imag_out = F.conv1d(x, imag_kernels, padding='same', groups=1)   # [B, n_filt, T]
        
        # Squared modulus (energy)
        energy = real_out ** 2 + imag_out ** 2  # [B, n_filters, T]
        
        # Stabilize
        energy = torch.clamp(energy, min=1e-10)
        
        return energy


class LearnablePooling(nn.Module):
    """
    Learnable Gaussian low-pass pooling (depthwise).
    Simulates the smoothing after envelope extraction.
    """
    def __init__(self, n_filters=40, kernel_size=128, stride=160):
        super().__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        
        half = (kernel_size - 1) // 2
        self.register_buffer('t', torch.arange(-half, half + 1).float())
        
        # Learnable bandwidth per filter
        self.sigma = nn.Parameter(torch.ones(n_filters) * 0.5)
    
    def _build_kernels(self):
        t = self.t.unsqueeze(0)              # [1, kernel_size]
        sigma = self.sigma.unsqueeze(1)      # [n_filters, 1]
        sigma = torch.clamp(sigma, min=0.05)
        
        gauss = torch.exp(-0.5 * (t / sigma) ** 2)
        gauss = gauss / (gauss.sum(dim=1, keepdim=True) + 1e-8)
        return gauss.unsqueeze(1)  # [n_filters, 1, kernel_size]
    
    def forward(self, x):
        """
        x: [batch, n_filters, time]
        Returns: [batch, n_filters, time / stride] (smoothed and downsampled)
        """
        kernels = self._build_kernels()  # [n_filters, 1, kernel_size]
        # Depthwise conv1d: groups=n_filters
        half = (self.kernel_size - 1) // 2
        out = F.conv1d(x, kernels, padding=half, stride=self.stride, groups=self.n_filters)
        return out


class PCENLayer(nn.Module):
    """
    Per-Channel Energy Normalization with learnable parameters.
    
    PCEN(x) = (x / (eps + M)^alpha + delta)^r - delta^r
    
    where M is a causal exponential moving average (EMA) of x:
        M[t] = s * x[t] + (1 - s) * M[t-1]
    
    Learnable per channel:
        - s (smoothing coefficient, 0 < s < 1)
        - alpha (exponent for denominator)
        - delta (offset)
        - r (power compression)
    """
    def __init__(self, n_filters=40, eps=1e-6, init_s=0.025):
        super().__init__()
        self.n_filters = n_filters
        self.eps = eps
        
        # Initialize parameters
        self.s = nn.Parameter(torch.ones(n_filters) * init_s)      # EMA coefficient
        self.alpha = nn.Parameter(torch.ones(n_filters) * 0.98)    # ~1.0
        self.delta = nn.Parameter(torch.ones(n_filters) * 2.0)     # offset
        self.r = nn.Parameter(torch.ones(n_filters) * 0.5)         # power compression (sqrt-like)
    
    def forward(self, x):
        """
        x: [batch, n_filters, time]
        Returns: [batch, n_filters, time]
        """
        B, C, T = x.shape
        
        # Clamp parameters to valid ranges for stability
        s = torch.clamp(torch.sigmoid(self.s), min=1e-3, max=1.0)      # (0, 1)
        alpha = torch.clamp(self.alpha, min=0.0, max=2.0)
        delta = torch.clamp(F.softplus(self.delta), min=1e-2)           # > 0
        r = torch.clamp(torch.sigmoid(self.r) * 2.0, min=0.01, max=2.0)  # (0, 2)
        
        # Expand to [1, C, 1]
        s = s.view(1, C, 1)
        alpha = alpha.view(1, C, 1)
        delta = delta.view(1, C, 1)
        r = r.view(1, C, 1)
        
        # Compute causal EMA along time dimension
        # M[t] = s * x[t] + (1-s) * M[t-1]
        # Implemented with a loop (T is small, ~100-500 frames)
        M = []
        m_prev = x[:, :, 0:1]  # Initialize with first frame
        
        for t in range(T):
            xt = x[:, :, t:t+1]
            if t == 0:
                mt = xt  # First frame
            else:
                mt = s * xt + (1.0 - s) * m_prev
            M.append(mt)
            m_prev = mt
        
        M = torch.cat(M, dim=2)  # [B, C, T]
        
        # PCEN formula
        smooth = M ** alpha
        pcen = (x / (self.eps + smooth) + delta) ** r - (delta ** r)
        
        return pcen


class LEAF(nn.Module):
    """
    Complete LEAF frontend.
    Input: raw waveform [batch, 1, time]
    Output: LEAF features [batch, n_filters, time]
    """
    def __init__(self, n_filters=40, filter_len=401, sr=16000, 
                 min_freq=80.0, max_freq=4000.0, pool_size=128):
        super().__init__()
        
        self.gabor = GaborFilterbank(
            n_filters=n_filters,
            filter_len=filter_len,
            sr=sr,
            min_freq=min_freq,
            max_freq=max_freq
        )
        
        self.pooling = LearnablePooling(n_filters=n_filters, kernel_size=pool_size, stride=160)
        self.pcen = PCENLayer(n_filters=n_filters)
    
    def forward(self, x):
        # 1. Gabor filterbank + squared modulus
        x = self.gabor(x)        # [B, n_filters, T]
        
        # 2. Learnable Gaussian pooling
        x = self.pooling(x)      # [B, n_filters, T]
        
        # 3. PCEN normalization
        x = self.pcen(x)         # [B, n_filters, T]
        
        return x


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

class LEAF_ALS_Model(nn.Module):
    """End-to-end: raw waveform -> LEAF -> CNN -> ALS/Healthy."""
    def __init__(self, n_filters=40, sr=16000, num_classes=2):
        super().__init__()
        self.leaf = LEAF(n_filters=n_filters, sr=sr)
        self.classifier = ALSClassifier(n_filters=n_filters, num_classes=num_classes)
    
    def forward(self, waveform):
        features = self.leaf(waveform)      # [B, n_filters, T]
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
        
        # Convert to tensor
        waveform = torch.from_numpy(wav).float().unsqueeze(0)  # [1, time]
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
        'n_filters': 40,                  # LEAF filterbank size
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
    model = LEAF_ALS_Model(
        n_filters=CONFIG['n_filters'],
        sr=CONFIG['sr'],
        num_classes=2
    ).to(CONFIG['device'])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    # Split optimizer learning rates: lower learning rate for the sensitive LEAF frontend
    optimizer = torch.optim.AdamW([
        {'params': model.leaf.parameters(), 'lr': CONFIG['lr'] * 0.1},
        {'params': model.classifier.parameters(), 'lr': CONFIG['lr']}
    ], weight_decay=1e-4)
    
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
    
    # Inspect learned LEAF parameters
    print("\n" + "="*60)
    print("LEARNED LEAF PARAMETERS (Sample)")
    print("="*60)
    print(f"Center frequencies (first 5): {torch.exp(model.leaf.gabor.log_f[:5]).detach().cpu().numpy()}")
    print(f"Bandwidths (first 5): {torch.exp(model.leaf.gabor.log_sigma[:5]).detach().cpu().numpy()}")
    print(f"PCEN s (first 5): {model.leaf.pcen.s[:5].detach().cpu().numpy()}")
    print(f"PCEN alpha (first 5): {model.leaf.pcen.alpha[:5].detach().cpu().numpy()}")
    
    # Save model
    torch.save({
        'model': model.state_dict(),
        'config': CONFIG,
        'test_metrics': {'acc': test_acc, 'f1': test_f1, 'auc': test_auc}
    }, 'leaf_als_best.pth')
    print("\nModel saved to leaf_als_best.pth")


if __name__ == '__main__':
    main()
