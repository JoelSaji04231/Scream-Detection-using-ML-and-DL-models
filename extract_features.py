import pandas as pd
import numpy as np
import librosa
from glob import glob
import os

print("\n" + "="*60)
print("FEATURE EXTRACTION WITH ESC-50 DATASET")
print("="*60)

# Load original files
scream_audio_files = glob('Converted_Separately/scream/*.wav')
non_scream_audio_files = glob('Converted_Separately/non_scream/*.wav')

print(f"\nOriginal dataset:")
print(f"  Scream files: {len(scream_audio_files)}")
print(f"  Non-scream files: {len(non_scream_audio_files)}")

# Load ESC-50 files
esc50_audio_files = glob('ESC-50-master/audio/*.wav')
print(f"  ESC-50 files: {len(esc50_audio_files)}")
print(f"  Total: {len(scream_audio_files) + len(non_scream_audio_files) + len(esc50_audio_files)}")

def clean_audio(y, sr):
    """
    Clean audio data before feature extraction
    - Remove silence
    - Normalize amplitude
    - Handle NaN and Inf values
    """
    # Remove leading and trailing silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    
    # Normalize audio to [-1, 1] range
    if np.max(np.abs(y_trimmed)) > 0:
        y_normalized = y_trimmed / np.max(np.abs(y_trimmed))
    else:
        y_normalized = y_trimmed
    
    # Handle NaN and Inf values
    y_cleaned = np.nan_to_num(y_normalized, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Ensure we have valid audio data
    if len(y_cleaned) == 0:
        # If trimming removed everything, use original normalized audio
        y_cleaned = np.nan_to_num(y / (np.max(np.abs(y)) + 1e-8), nan=0.0, posinf=1.0, neginf=-1.0)
    
    return y_cleaned

def extract_all_features(file_path, label, n_mfcc=13):
    """Extract comprehensive audio features for classification with data cleaning"""
    # Load audio
    y, sr = librosa.load(file_path, sr=None)
    
    # Clean audio data BEFORE feature extraction
    y_clean = clean_audio(y, sr)
    
    # MFCC features
    mfcc = librosa.feature.mfcc(y=y_clean, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    # RMS Energy
    rms = librosa.feature.rms(y=y_clean)[0]
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)
    
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y_clean)[0]
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)
    
    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y_clean, sr=sr)[0]
    sc_mean = np.mean(spectral_centroid)
    sc_std = np.std(spectral_centroid)
    
    # Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y_clean, sr=sr)[0]
    sr_mean = np.mean(spectral_rolloff)
    sr_std = np.std(spectral_rolloff)
    

    
    # Handle NaN/Inf in extracted features
    def clean_value(val):
        if np.isnan(val) or np.isinf(val):
            return 0.0
        return float(val)
    
    # Create feature dictionary with cleaned values
    features = {
        "file": os.path.basename(file_path),
        "label": label,
        **{f"mfcc{i+1}_mean": clean_value(mfcc_mean[i]) for i in range(n_mfcc)},
        **{f"mfcc{i+1}_std": clean_value(mfcc_std[i]) for i in range(n_mfcc)},
        "rms_mean": clean_value(rms_mean),
        "rms_std": clean_value(rms_std),
        "zcr_mean": clean_value(zcr_mean),
        "zcr_std": clean_value(zcr_std),
        "spectral_centroid_mean": clean_value(sc_mean),
        "spectral_centroid_std": clean_value(sc_std),
        "spectral_rolloff_mean": clean_value(sr_mean),
        "spectral_rolloff_std": clean_value(sr_std),
    }
    
    return features


all_features_data = []

# Extract from scream files
print(f"\nExtracting features from scream files...")
for i, file in enumerate(scream_audio_files, 1):
    if i % 100 == 0 or i == len(scream_audio_files):
        print(f"\tProgress: {i}/{len(scream_audio_files)}")
    all_features_data.append(extract_all_features(file, "scream"))

# Extract from non-scream files
print(f"\nExtracting features from non-scream files...")
for i, file in enumerate(non_scream_audio_files, 1):
    if i % 100 == 0 or i == len(non_scream_audio_files):
        print(f"\tProgress: {i}/{len(non_scream_audio_files)}")
    all_features_data.append(extract_all_features(file, "non_scream"))

# Extract from ESC-50 files (as non_scream)
print(f"\nExtracting features from ESC-50 files...")
for i, file in enumerate(esc50_audio_files, 1):
    if i % 200 == 0 or i == len(esc50_audio_files):
        print(f"\tProgress: {i}/{len(esc50_audio_files)}")
    try:
        all_features_data.append(extract_all_features(file, "non_scream"))
    except Exception as e:
        print(f"\tError processing {file}: {e}")

# Save to CSV
df_all_features = pd.DataFrame(all_features_data)

print(f"\n{'='*60}")
print(f"FEATURE EXTRACTION COMPLETE")
print(f"{'='*60}")
print(f"Total samples: {len(df_all_features)}")
print(f"Feature dimensions: {df_all_features.shape}")
print(f"\nClass distribution:")
print(df_all_features['label'].value_counts())
print(f"{'='*60}\n")

# Save ML features
df_all_features.to_csv("audio_features.csv", index=False)
print("Saved features to audio_features.csv")

# ===== GENERATE SPECTROGRAMS FOR DL MODELS =====
print(f"\n{'='*60}")
print(f"GENERATING SPECTROGRAMS FOR DL MODELS")
print(f"{'='*60}")

# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')
    print("Created data/ directory")

def generate_spectrogram(file_path, sr=22050, n_fft=2048, hop_length=512):
    """Generate regular spectrogram with cleaned audio"""
    y, sr_orig = librosa.load(file_path, sr=sr)
    y_clean = clean_audio(y, sr)
    
    # Generate spectrogram
    D = librosa.stft(y_clean, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Resize to fixed size (128x128)
    from scipy.ndimage import zoom
    if S_db.shape != (128, 128):
        zoom_factors = (128 / S_db.shape[0], 128 / S_db.shape[1])
        S_db_resized = zoom(S_db, zoom_factors, order=1)
    else:
        S_db_resized = S_db
    
    return S_db_resized

def generate_mel_spectrogram(file_path, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
    """Generate mel-spectrogram with cleaned audio"""
    y, sr_orig = librosa.load(file_path, sr=sr)
    y_clean = clean_audio(y, sr)
    
    # Generate mel-spectrogram
    S = librosa.feature.melspectrogram(y=y_clean, sr=sr, n_mels=n_mels, 
                                       n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    
    # Resize to fixed size (128x128)
    from scipy.ndimage import zoom
    if S_db.shape != (128, 128):
        zoom_factors = (128 / S_db.shape[0], 128 / S_db.shape[1])
        S_db_resized = zoom(S_db, zoom_factors, order=1)
    else:
        S_db_resized = S_db
    
    return S_db_resized

# Collect all files
all_audio_files = scream_audio_files + non_scream_audio_files + esc50_audio_files
all_labels = (['scream'] * len(scream_audio_files) + 
              ['non_scream'] * len(non_scream_audio_files) + 
              ['non_scream'] * len(esc50_audio_files))

print(f"\nGenerating spectrograms for {len(all_audio_files)} audio files...")

spectrograms_list = []
mel_spectrograms_list = []
labels_list = []

for i, (file, label) in enumerate(zip(all_audio_files, all_labels), 1):
    if i % 500 == 0 or i == len(all_audio_files):
        print(f"\tProgress: {i}/{len(all_audio_files)}")
    
    try:
        # Generate both types of spectrograms
        spec = generate_spectrogram(file)
        mel_spec = generate_mel_spectrogram(file)
        
        spectrograms_list.append(spec)
        mel_spectrograms_list.append(mel_spec)
        labels_list.append(label)
    except Exception as e:
        print(f"\tError processing {file}: {e}")

# Convert to numpy arrays
spectrograms = np.array(spectrograms_list)
mel_spectrograms = np.array(mel_spectrograms_list)
labels = np.array(labels_list)

print(f"\nSpectrogram shapes:")
print(f"\tRegular spectrograms: {spectrograms.shape}")
print(f"\tMel-spectrograms: {mel_spectrograms.shape}")
print(f"\tLabels: {labels.shape}")

# Save to data directory
print(f"\nSaving spectrogram datasets...")
np.save("data/cnn_spectrograms.npy", spectrograms)
np.save("data/cnn_mel_spectrograms.npy", mel_spectrograms)
np.save("data/cnn_labels.npy", labels)

print("\tSaved data/cnn_spectrograms.npy")
print("\tSaved data/cnn_mel_spectrograms.npy")
print("\tSaved data/cnn_labels.npy")

print(f"\n{'='*60}")
print(f"ALL DATASETS GENERATED SUCCESSFULLY")
print(f"{'='*60}")
print("\nML Dataset:")
print(f"  • audio_features.csv ({len(df_all_features)} samples, {df_all_features.shape[1]} features)")
print("\nDL Datasets:")
print(f"  • data/cnn_spectrograms.npy ({spectrograms.shape[0]} samples, {spectrograms.shape[1]}x{spectrograms.shape[2]})")
print(f"  • data/cnn_mel_spectrograms.npy ({mel_spectrograms.shape[0]} samples, {mel_spectrograms.shape[1]}x{mel_spectrograms.shape[2]})")
print(f"  • data/cnn_labels.npy ({labels.shape[0]} labels)")
print(f"{'='*60}")