"""
Audio Feature Extraction Pipeline
Extracts comprehensive audio features for machine learning classification
"""

import librosa
import numpy as np
import pandas as pd
import os
from glob import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Audio processing parameters
SAMPLE_RATE = 22050
DURATION = 3.0  # seconds
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

print("="*60)
print("AUDIO FEATURE EXTRACTION PIPELINE")
print("="*60)

def clean_audio(audio, sr):
    """Clean audio data before feature extraction"""
    # Remove silence
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    # Normalize
    audio = librosa.util.normalize(audio)
    
    # Ensure minimum length
    target_length = int(DURATION * sr)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
    else:
        audio = audio[:target_length]
    
    return audio

def extract_comprehensive_features(file_path):
    """Extract comprehensive audio features for classification with data cleaning"""
    try:
        # Load audio
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        audio = clean_audio(audio, sr)
        
        features = {}
        
        # ===== MFCC FEATURES =====
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        
        # MFCC means and standard deviations
        for i in range(N_MFCC):
            features[f'mfcc{i+1}_mean'] = np.mean(mfcc[i])
            features[f'mfcc{i+1}_std'] = np.std(mfcc[i])
        
        # MFCC delta features
        mfcc_delta = librosa.feature.delta(mfcc)
        for i in range(N_MFCC):
            features[f'mfcc{i+1}_delta_mean'] = np.mean(mfcc_delta[i])
            features[f'mfcc{i+1}_delta_std'] = np.std(mfcc_delta[i])
        
        # ===== SPECTRAL FEATURES =====
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_std'] = np.std(spectral_centroid)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # ===== ENERGY FEATURES =====
        # RMS energy
        rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        # ===== CHROMA FEATURES =====
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        for i in range(12):
            features[f'chroma{i}_mean'] = np.mean(chroma[i])
            features[f'chroma{i}_std'] = np.std(chroma[i])
        
        # ===== TEMPORAL FEATURES =====
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        features['tempo'] = tempo
        
        # ===== ADDITIONAL FEATURES =====
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        for i in range(spectral_contrast.shape[0]):
            features[f'spectral_contrast{i}_mean'] = np.mean(spectral_contrast[i])
            features[f'spectral_contrast{i}_std'] = np.std(spectral_contrast[i])
        
        # Tonnetz
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
        for i in range(tonnetz.shape[0]):
            features[f'tonnetz{i}_mean'] = np.mean(tonnetz[i])
            features[f'tonnetz{i}_std'] = np.std(tonnetz[i])
        
        return features
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

# ===== MAIN EXTRACTION PROCESS =====

print(f"\nOriginal dataset:")
print(f"- Sample rate: {SAMPLE_RATE} Hz")
print(f"- Duration: {DURATION} seconds")
print(f"- MFCC coefficients: {N_MFCC}")
print(f"- FFT size: {N_FFT}")
print(f"- Hop length: {HOP_LENGTH}")

# Check for audio data
try:
    scream_audio_files = glob('C:/Users/joels/Desktop/Programming/Scream Detection ML and DL/Converted_Separately/scream/*.wav')
    non_scream_audio_files = glob('C:/Users/joels/Desktop/Programming/Scream Detection ML and DL/Converted_Separately/non_scream/*.wav')
    
    print(f"\nFound {len(scream_audio_files)} scream audio files")
    print(f"Found {len(non_scream_audio_files)} non-scream audio files")
    
    if len(scream_audio_files) == 0 or len(non_scream_audio_files) == 0:
        print("\n[ERROR] No audio files found in Converted_Separately/")
        print("Please ensure you have:")
        print("- Converted_Separately/scream/ with .wav files")
        print("- Converted_Separately/non_scream/ with .wav files")
        exit(1)
        
except Exception as e:
    print(f"\n[ERROR] Error accessing audio files: {e}")
    print("Please ensure Converted_Separately/ directory exists with proper structure")
    exit(1)

# Extract features
print("\nExtracting features...")
all_features = []
labels = []
file_paths = []

# Process scream files
print(f"\nProcessing scream files...")
for file_path in tqdm(scream_audio_files, desc="Scream files"):
    features = extract_comprehensive_features(file_path)
    if features:
        all_features.append(features)
        labels.append('scream')
        file_paths.append(file_path)

# Process non-scream files
print(f"\nProcessing non-scream files...")
for file_path in tqdm(non_scream_audio_files, desc="Non-scream files"):
    features = extract_comprehensive_features(file_path)
    if features:
        all_features.append(features)
        labels.append('non_scream')
        file_paths.append(file_path)

# Create DataFrame
print("\nCreating feature dataset...")
df_features = pd.DataFrame(all_features)
df_features['label'] = labels
df_features['file_path'] = file_paths

print(f"[OK] Extracted {len(df_features)} samples")
print(f"[OK] {df_features.shape[1]-2} features per sample")
print(f"[OK] Features: {list(df_features.columns[:-2])}")

# Save features
print("\nSaving features...")
df_features.to_csv('audio_features.csv', index=False)
print("[OK] Features saved to audio_features.csv")

# Display summary
print("\n" + "="*60)
print("FEATURE EXTRACTION COMPLETE")
print("="*60)
print(f"[OK] Total samples: {len(df_features)}")
print(f"[OK] Scream samples: {len(df_features[df_features['label'] == 'scream'])}")
print(f"[OK] Non-scream samples: {len(df_features[df_features['label'] == 'non_scream'])}")
print(f"[OK] Features extracted: {df_features.shape[1]-2}")
print(f"[OK] Data saved to: audio_features.csv")
print("\nNext steps:")
print("- Run train_ml.py to train ML models")
print("- Run analyze_audio.py to visualize features")