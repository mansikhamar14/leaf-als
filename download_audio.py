import os
import synapseclient

# 1. Ensure the audio directory exists
output_dir = "data/VOC-ALS/audio"
os.makedirs(output_dir, exist_ok=True)

# 2. Log in to Synapse
syn = synapseclient.Synapse()
# IMPORTANT: Replace YOUR_TOKEN_HERE with your actual Synapse authorization token
syn.login(authToken="eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIiwibW9kaWZ5Il0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc4MDQ5MjQ4MiwiaWF0IjoxNzgwNDkyNDgyLCJqdGkiOiIzODk5MiIsInN1YiI6IjM1OTEzMjUifQ.OhmZYxBCS61l4hK81Uc1tzctOzJFMzCBeyLGnGxtThuFYIVMtJr6_jzEWBDLSDYMkGX-k4qt7Yp9MemTPb8uuCk8UmmZONroBzVGRrXGK6IaL__bAdhPSM3QJoqRjmmogajRQxhYNpy7C-jgwX7m_cAmLKtrRFuKtkGyHBl-G_NwFfTsQcKy-bcd0r0Iq9NFA5NN7PgXxNPDQdj3DvDsndbqXVceDOZVIse_iMxZH0CvbF7HxbWgkUmGRAHMk6gKVx86DyMwN4ottG9qTlhvDQsL5fuSxMpOCoXIcy_ln4zzme-Fg6FFQr_AL81EcbMbp7OGQu5rXT3IrgujPrcdLg")

# 3. Retrieve your download list
print("Retrieving download list...")
dl_list_file_entities = syn.get_download_list()

print(f"Downloading files to {output_dir}/ ...")

# 4. Download each entity in the list to the designated folder
for entity in dl_list_file_entities:
    try:
        # syn.get() downloads the file entity to the designated location
        # Depending on Synapse version, 'entity' might be an object, a dict, or an ID string
        entity_id = entity['fileEntityId'] if isinstance(entity, dict) and 'fileEntityId' in entity else entity
        downloaded_file = syn.get(entity_id, downloadLocation=output_dir)
        print(f"Successfully downloaded: {downloaded_file.name}")
    except Exception as e:
        print(f"Failed to download entity {entity}. Error: {e}")

print("All done!")
