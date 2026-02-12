"""
Audio Analysis and Visualization for CrimeAlert
Analyzes scream vs non-scream audio characteristics
Generates visualizations for frequency, spectrograms, MFCCs, etc.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import librosa 
import librosa.display 
from glob import glob
from scipy.fftpack import dct
from sklearn.preprocessing import StandardScaler

print("\n" + "="*60)
print("AUDIO ANALYSIS & VISUALIZATION")
print("="*60)

# Load audio files
scream_audio_files = glob('Converted_Separately/scream/*.wav')
non_scream_audio_files = glob('Converted_Separately/non_scream/*.wav')

print(f"\nDataset:")
print(f"\tScream files: {len(scream_audio_files)}")
print(f"\tNon-scream files: {len(non_scream_audio_files)}")

# Load example files for analysis
print("\nLoading example files for visualization...")
y1, sr1 = librosa.load(scream_audio_files[0])  # Scream example
y2, sr2 = librosa.load(non_scream_audio_files[0])  # Non-scream example

print(f"\tScream example: {scream_audio_files[0].split('/')[-1]}")
print(f"\tNon-scream example: {non_scream_audio_files[0].split('/')[-1]}")


# ===== 1. FREQUENCY/WAVEFORM ANALYSIS =====
print("\n" + "="*60)
print("1. WAVEFORM COMPARISON")
print("="*60)

sns.set_style("whitegrid")
plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)
pd.Series(y1).plot(lw=1, title="Scream Audio - Waveform", color='red')
plt.xlabel("Sample")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
pd.Series(y2).plot(lw=1, title="Non-Scream Audio - Waveform", color='blue')
plt.xlabel("Sample")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.savefig('analysis_waveforms.png', dpi=150, bbox_inches='tight')
print("Saved: analysis_waveforms.png")
plt.show()


# ===== 2. SPECTROGRAM ANALYSIS =====
print("\n" + "="*60)
print("2. SPECTROGRAM COMPARISON")
print("="*60)

D1 = librosa.stft(y1)
S_db1 = librosa.amplitude_to_db(np.abs(D1), ref=np.max)
D2 = librosa.stft(y2) 
S_db2 = librosa.amplitude_to_db(np.abs(D2), ref=np.max)

fig_spec, axes_spec = plt.subplots(1, 2, figsize=(18, 6))
img1 = librosa.display.specshow(S_db1, x_axis="time", y_axis="log", ax=axes_spec[0], cmap='viridis')
axes_spec[0].set_title("Scream Audio - Spectrogram", fontsize=16)
fig_spec.colorbar(img1, ax=axes_spec[0], format='%+2.0f dB')

img2 = librosa.display.specshow(S_db2, x_axis="time", y_axis="log", ax=axes_spec[1], cmap='viridis')
axes_spec[1].set_title("Non-Scream Audio - Spectrogram", fontsize=16)
fig_spec.colorbar(img2, ax=axes_spec[1], format='%+2.0f dB')

plt.tight_layout()
plt.savefig('analysis_spectrograms.png', dpi=150, bbox_inches='tight')
print("Saved: analysis_spectrograms.png")
plt.show()


# ===== 3. MEL-SPECTROGRAM ANALYSIS =====
print("\n" + "="*60)
print("3. MEL-SPECTROGRAM COMPARISON")
print("="*60)

S1 = librosa.feature.melspectrogram(y=y1, sr=sr1, n_mels=128)
S_db_mel1 = librosa.amplitude_to_db(S1, ref=np.max)
S2 = librosa.feature.melspectrogram(y=y2, sr=sr2, n_mels=128)
S_db_mel2 = librosa.amplitude_to_db(S2, ref=np.max)

fig_mel, axes_mel = plt.subplots(1, 2, figsize=(18, 6))
img_mel1 = librosa.display.specshow(S_db_mel1, x_axis="time", y_axis="mel", sr=sr1, ax=axes_mel[0], cmap='viridis')
axes_mel[0].set_title("Scream Audio - Mel-Spectrogram", fontsize=16)
fig_mel.colorbar(img_mel1, ax=axes_mel[0], format='%+2.0f dB')

img_mel2 = librosa.display.specshow(S_db_mel2, x_axis="time", y_axis="mel", sr=sr2, ax=axes_mel[1], cmap='viridis')
axes_mel[1].set_title("Non-Scream Audio - Mel-Spectrogram", fontsize=16)
fig_mel.colorbar(img_mel2, ax=axes_mel[1], format='%+2.0f dB')

plt.tight_layout()
plt.savefig('analysis_mel_spectrograms.png', dpi=150, bbox_inches='tight')
print("Saved: analysis_mel_spectrograms.png")
plt.show()


# ===== 4. MFCC ANALYSIS =====
print("\n" + "="*60)
print("4. MFCC (MEL-FREQUENCY CEPSTRAL COEFFICIENTS) ANALYSIS")
print("="*60)

mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13)
mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=13)

# Statistical analysis
mfcc1_mean = np.mean(mfcc1, axis=1)
mfcc1_std = np.std(mfcc1, axis=1)
mfcc2_mean = np.mean(mfcc2, axis=1)
mfcc2_std = np.std(mfcc2, axis=1)

# Plot MFCC statistics
fig_stats, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(13)

# Mean comparison
axes[0].bar(x - 0.2, mfcc1_mean, 0.4, label='Scream', alpha=0.7, color='red')
axes[0].bar(x + 0.2, mfcc2_mean, 0.4, label='Non-Scream', alpha=0.7, color='blue')
axes[0].set_title('Mean MFCC Coefficients')
axes[0].set_xlabel('MFCC Index')
axes[0].set_ylabel('Mean Value')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Std deviation comparison
axes[1].bar(x - 0.2, mfcc1_std, 0.4, label='Scream', alpha=0.7, color='red')
axes[1].bar(x + 0.2, mfcc2_std, 0.4, label='Non-Scream', alpha=0.7, color='blue')
axes[1].set_title('MFCC Standard Deviation')
axes[1].set_xlabel('MFCC Index')
axes[1].set_ylabel('Std Deviation')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis_mfcc_statistics.png', dpi=150, bbox_inches='tight')
print("Saved: analysis_mfcc_statistics.png")
plt.show()

# Print MFCC statistics
print("\nMFCC Statistics:")
print(f"\nScream Audio:")
print(f"\tMean range: [{mfcc1_mean.min():.2f}, {mfcc1_mean.max():.2f}]")
print(f"\tStd range: [{mfcc1_std.min():.2f}, {mfcc1_std.max():.2f}]")
print(f"\nNon-Scream Audio:")
print(f"\tMean range: [{mfcc2_mean.min():.2f}, {mfcc2_mean.max():.2f}]")
print(f"\tStd range: [{mfcc2_std.min():.2f}, {mfcc2_std.max():.2f}]")


# ===== 5. ADDITIONAL AUDIO FEATURES =====
print("\n" + "="*60)
print("5. ADDITIONAL AUDIO FEATURES - EXTRACTION & STATISTICS")
print("="*60)

# Extract all features for both audio samples
print("\nExtracting features...")

# RMS Energy
rms1 = librosa.feature.rms(y=y1)[0]
rms2 = librosa.feature.rms(y=y2)[0]

# Zero Crossing Rate
zcr1 = librosa.feature.zero_crossing_rate(y1)[0]
zcr2 = librosa.feature.zero_crossing_rate(y2)[0]

# Spectral Centroid
sc1 = librosa.feature.spectral_centroid(y=y1, sr=sr1)[0]
sc2 = librosa.feature.spectral_centroid(y=y2, sr=sr2)[0]

# Spectral Rolloff
sr_1 = librosa.feature.spectral_rolloff(y=y1, sr=sr1)[0]
sr_2 = librosa.feature.spectral_rolloff(y=y2, sr=sr2)[0]

# Spectral Bandwidth
sb1 = librosa.feature.spectral_bandwidth(y=y1, sr=sr1)[0]
sb2 = librosa.feature.spectral_bandwidth(y=y2, sr=sr2)[0]

# Chroma Features
chroma1 = librosa.feature.chroma_stft(y=y1, sr=sr1)
chroma2 = librosa.feature.chroma_stft(y=y2, sr=sr2)

# Spectral Contrast
contrast1 = librosa.feature.spectral_contrast(y=y1, sr=sr1)
contrast2 = librosa.feature.spectral_contrast(y=y2, sr=sr2)

print("Features extracted successfully!")

# ===== FEATURE STATISTICS =====
print("\n" + "="*60)
print("FEATURE STATISTICS (Mean ± Std)")
print("="*60)

# Create comprehensive feature dictionary
features_scream = {
    'RMS Energy': (np.mean(rms1), np.std(rms1)),
    'Zero Crossing Rate': (np.mean(zcr1), np.std(zcr1)),
    'Spectral Centroid': (np.mean(sc1), np.std(sc1)),
    'Spectral Rolloff': (np.mean(sr_1), np.std(sr_1)),
    'Spectral Bandwidth': (np.mean(sb1), np.std(sb1)),
    'Chroma Mean': (np.mean(chroma1), np.std(chroma1)),
    'Spectral Contrast Mean': (np.mean(contrast1), np.std(contrast1))
}

features_non_scream = {
    'RMS Energy': (np.mean(rms2), np.std(rms2)),
    'Zero Crossing Rate': (np.mean(zcr2), np.std(zcr2)),
    'Spectral Centroid': (np.mean(sc2), np.std(sc2)),
    'Spectral Rolloff': (np.mean(sr_2), np.std(sr_2)),
    'Spectral Bandwidth': (np.mean(sb2), np.std(sb2)),
    'Chroma Mean': (np.mean(chroma2), np.std(chroma2)),
    'Spectral Contrast Mean': (np.mean(contrast2), np.std(contrast2))
}

print("\nSCREAM AUDIO FEATURES:")
for feature_name, (mean_val, std_val) in features_scream.items():
    print(f"\t{feature_name:25s}: {mean_val:10.4f} ± {std_val:10.4f}")

print("\nNON-SCREAM AUDIO FEATURES:")
for feature_name, (mean_val, std_val) in features_non_scream.items():
    print(f"\t{feature_name:25s}: {mean_val:10.4f} ± {std_val:10.4f}")

# ===== DATA CLEANING =====
print("\n" + "="*60)
print("\tDATA CLEANING")
print("="*60)

# Combine features for cleaning
feature_matrix_scream = np.array([
    [np.mean(rms1), np.std(rms1)],
    [np.mean(zcr1), np.std(zcr1)],
    [np.mean(sc1), np.std(sc1)],
    [np.mean(sr_1), np.std(sr_1)],
    [np.mean(sb1), np.std(sb1)],
    [np.mean(chroma1), np.std(chroma1)],
    [np.mean(contrast1), np.std(contrast1)]
])

feature_matrix_non_scream = np.array([
    [np.mean(rms2), np.std(rms2)],
    [np.mean(zcr2), np.std(zcr2)],
    [np.mean(sc2), np.std(sc2)],
    [np.mean(sr_2), np.std(sr_2)],
    [np.mean(sb2), np.std(sb2)],
    [np.mean(chroma2), np.std(chroma2)],
    [np.mean(contrast2), np.std(contrast2)]
])

# Combine both for cleaning
all_features = np.vstack([feature_matrix_scream, feature_matrix_non_scream])

print("\nBefore cleaning:")
print(f"  Feature matrix shape: {all_features.shape}")
print(f"  NaN values: {np.isnan(all_features).sum()}")
print(f"  Inf values: {np.isinf(all_features).sum()}")

# Handle NaN and Inf values
all_features_cleaned = all_features.copy()
all_features_cleaned[np.isnan(all_features_cleaned)] = 0
all_features_cleaned[np.isinf(all_features_cleaned)] = 0

# Remove outliers using IQR method
from scipy import stats
z_scores = np.abs(stats.zscore(all_features_cleaned, axis=0, nan_policy='omit'))
outlier_threshold = 3
outliers_mask = z_scores > outlier_threshold
outliers_count = np.sum(outliers_mask)

# Cap outliers at threshold instead of removing (preserves data)
for i in range(all_features_cleaned.shape[1]):
    q1 = np.percentile(all_features_cleaned[:, i], 25)
    q3 = np.percentile(all_features_cleaned[:, i], 75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    
    all_features_cleaned[:, i] = np.clip(all_features_cleaned[:, i], lower_bound, upper_bound)

print("\nAfter cleaning:")
print(f"\tOutliers detected: {outliers_count}")
print(f"\tOutliers capped using IQR method")
print(f"\tNaN values: {np.isnan(all_features_cleaned).sum()}")
print(f"\tInf values: {np.isinf(all_features_cleaned).sum()}")

# ===== FEATURE STANDARDIZATION =====
print("\n" + "="*60)
print("7. FEATURE STANDARDIZATION")
print("="*60)

# Standardize cleaned features
scaler = StandardScaler()
standardized_features = scaler.fit_transform(all_features_cleaned)

# Split back into scream and non-scream
standardized_scream = standardized_features[:7]
standardized_non_scream = standardized_features[7:]

print("\nStandardized Features (z-scores):")
feature_names = ['RMS', 'ZCR', 'Spectral Centroid', 'Spectral Rolloff', 
                 'Spectral Bandwidth', 'Chroma', 'Spectral Contrast']

print("\nSCREAM (Standardized):")
for i, name in enumerate(feature_names):
    print(f"  {name:25s}: Mean={standardized_scream[i,0]:8.4f}, Std={standardized_scream[i,1]:8.4f}")

print("\nNON-SCREAM (Standardized):")
for i, name in enumerate(feature_names):
    print(f"  {name:25s}: Mean={standardized_non_scream[i,0]:8.4f}, Std={standardized_non_scream[i,1]:8.4f}")

# ===== FEATURE VISUALIZATION (SUBPLOT PAIRS) =====
print("\n" + "="*60)
print("8. FEATURE VISUALIZATION (SUBPLOT PAIRS)")
print("="*60)

# Set style
sns.set_style("whitegrid")

# 1. RMS Energy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

# Subplot 1: Feature time series
ax1.plot(rms1, label='Scream', color='#e74c3c', alpha=0.8, linewidth=2)
ax1.plot(rms2, label='Non-Scream', color='#3498db', alpha=0.8, linewidth=2)
ax1.set_title('RMS Energy Over Time', fontsize=14, fontweight='bold')
ax1.set_xlabel('Frame', fontsize=11)
ax1.set_ylabel('RMS Energy', fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='--')

# Subplot 2: Mean and std as separate bars
x_pos = np.arange(2)
width = 0.35
scream_mean = np.mean(rms1)
scream_std = np.std(rms1)
non_scream_mean = np.mean(rms2)
non_scream_std = np.std(rms2)

ax2.bar(x_pos[0] - width/2, scream_mean, width, label='Scream Mean', alpha=0.8, color='#e74c3c', edgecolor='black')
ax2.bar(x_pos[0] + width/2, scream_std, width, label='Scream Std', alpha=0.8, color='#c0392b', edgecolor='black')
ax2.bar(x_pos[1] - width/2, non_scream_mean, width, label='Non-Scream Mean', alpha=0.8, color='#3498db', edgecolor='black')
ax2.bar(x_pos[1] + width/2, non_scream_std, width, label='Non-Scream Std', alpha=0.8, color='#2980b9', edgecolor='black')
ax2.set_title('RMS Energy Statistics', fontsize=14, fontweight='bold')
ax2.set_ylabel('Value', fontsize=11)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(['Scream', 'Non-Scream'], fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

plt.tight_layout()
plt.savefig('feature_rms_energy.png', dpi=150, bbox_inches='tight')
print("Saved: feature_rms_energy.png")
plt.show()
plt.close()

# 2. Zero Crossing Rate
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

ax1.plot(zcr1, label='Scream', color='#e74c3c', alpha=0.8, linewidth=2)
ax1.plot(zcr2, label='Non-Scream', color='#3498db', alpha=0.8, linewidth=2)
ax1.set_title('Zero Crossing Rate Over Time', fontsize=14, fontweight='bold')
ax1.set_xlabel('Frame', fontsize=11)
ax1.set_ylabel('Zero Crossing Rate', fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='--')

scream_mean = np.mean(zcr1)
scream_std = np.std(zcr1)
non_scream_mean = np.mean(zcr2)
non_scream_std = np.std(zcr2)

ax2.bar(x_pos[0] - width/2, scream_mean, width, label='Scream Mean', alpha=0.8, color='#e74c3c', edgecolor='black')
ax2.bar(x_pos[0] + width/2, scream_std, width, label='Scream Std', alpha=0.8, color='#c0392b', edgecolor='black')
ax2.bar(x_pos[1] - width/2, non_scream_mean, width, label='Non-Scream Mean', alpha=0.8, color='#3498db', edgecolor='black')
ax2.bar(x_pos[1] + width/2, non_scream_std, width, label='Non-Scream Std', alpha=0.8, color='#2980b9', edgecolor='black')
ax2.set_title('Zero Crossing Rate Statistics', fontsize=14, fontweight='bold')
ax2.set_ylabel('Value', fontsize=11)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(['Scream', 'Non-Scream'], fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

plt.tight_layout()
plt.savefig('feature_zero_crossing_rate.png', dpi=150, bbox_inches='tight')
print("Saved: feature_zero_crossing_rate.png")
plt.show()
plt.close()

# 3. Spectral Centroid
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

ax1.plot(sc1, label='Scream', color='#e74c3c', alpha=0.8, linewidth=2)
ax1.plot(sc2, label='Non-Scream', color='#3498db', alpha=0.8, linewidth=2)
ax1.set_title('Spectral Centroid Over Time', fontsize=14, fontweight='bold')
ax1.set_xlabel('Frame', fontsize=11)
ax1.set_ylabel('Frequency (Hz)', fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='--')

scream_mean = np.mean(sc1)
scream_std = np.std(sc1)
non_scream_mean = np.mean(sc2)
non_scream_std = np.std(sc2)

ax2.bar(x_pos[0] - width/2, scream_mean, width, label='Scream Mean', alpha=0.8, color='#e74c3c', edgecolor='black')
ax2.bar(x_pos[0] + width/2, scream_std, width, label='Scream Std', alpha=0.8, color='#c0392b', edgecolor='black')
ax2.bar(x_pos[1] - width/2, non_scream_mean, width, label='Non-Scream Mean', alpha=0.8, color='#3498db', edgecolor='black')
ax2.bar(x_pos[1] + width/2, non_scream_std, width, label='Non-Scream Std', alpha=0.8, color='#2980b9', edgecolor='black')
ax2.set_title('Spectral Centroid Statistics', fontsize=14, fontweight='bold')
ax2.set_ylabel('Value (Hz)', fontsize=11)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(['Scream', 'Non-Scream'], fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

plt.tight_layout()
plt.savefig('feature_spectral_centroid.png', dpi=150, bbox_inches='tight')
print("Saved: feature_spectral_centroid.png")
plt.show()
plt.close()

# 4. Spectral Rolloff
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

ax1.plot(sr_1, label='Scream', color='#e74c3c', alpha=0.8, linewidth=2)
ax1.plot(sr_2, label='Non-Scream', color='#3498db', alpha=0.8, linewidth=2)
ax1.set_title('Spectral Rolloff Over Time', fontsize=14, fontweight='bold')
ax1.set_xlabel('Frame', fontsize=11)
ax1.set_ylabel('Frequency (Hz)', fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='--')

scream_mean = np.mean(sr_1)
scream_std = np.std(sr_1)
non_scream_mean = np.mean(sr_2)
non_scream_std = np.std(sr_2)

ax2.bar(x_pos[0] - width/2, scream_mean, width, label='Scream Mean', alpha=0.8, color='#e74c3c', edgecolor='black')
ax2.bar(x_pos[0] + width/2, scream_std, width, label='Scream Std', alpha=0.8, color='#c0392b', edgecolor='black')
ax2.bar(x_pos[1] - width/2, non_scream_mean, width, label='Non-Scream Mean', alpha=0.8, color='#3498db', edgecolor='black')
ax2.bar(x_pos[1] + width/2, non_scream_std, width, label='Non-Scream Std', alpha=0.8, color='#2980b9', edgecolor='black')
ax2.set_title('Spectral Rolloff Statistics', fontsize=14, fontweight='bold')
ax2.set_ylabel('Value (Hz)', fontsize=11)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(['Scream', 'Non-Scream'], fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

plt.tight_layout()
plt.savefig('feature_spectral_rolloff.png', dpi=150, bbox_inches='tight')
print("Saved: feature_spectral_rolloff.png")
plt.show()
plt.close()

# 5. Spectral Bandwidth
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

ax1.plot(sb1, label='Scream', color='#e74c3c', alpha=0.8, linewidth=2)
ax1.plot(sb2, label='Non-Scream', color='#3498db', alpha=0.8, linewidth=2)
ax1.set_title('Spectral Bandwidth Over Time', fontsize=14, fontweight='bold')
ax1.set_xlabel('Frame', fontsize=11)
ax1.set_ylabel('Frequency (Hz)', fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='--')

scream_mean = np.mean(sb1)
scream_std = np.std(sb1)
non_scream_mean = np.mean(sb2)
non_scream_std = np.std(sb2)

ax2.bar(x_pos[0] - width/2, scream_mean, width, label='Scream Mean', alpha=0.8, color='#e74c3c', edgecolor='black')
ax2.bar(x_pos[0] + width/2, scream_std, width, label='Scream Std', alpha=0.8, color='#c0392b', edgecolor='black')
ax2.bar(x_pos[1] - width/2, non_scream_mean, width, label='Non-Scream Mean', alpha=0.8, color='#3498db', edgecolor='black')
ax2.bar(x_pos[1] + width/2, non_scream_std, width, label='Non-Scream Std', alpha=0.8, color='#2980b9', edgecolor='black')
ax2.set_title('Spectral Bandwidth Statistics', fontsize=14, fontweight='bold')
ax2.set_ylabel('Value (Hz)', fontsize=11)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(['Scream', 'Non-Scream'], fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

plt.tight_layout()
plt.savefig('feature_spectral_bandwidth.png', dpi=150, bbox_inches='tight')
print("Saved: feature_spectral_bandwidth.png")
plt.show()
plt.close()

# 6. Chroma Features
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

ax1.plot(np.mean(chroma1, axis=0), label='Scream', color='#e74c3c', alpha=0.8, linewidth=2)
ax1.plot(np.mean(chroma2, axis=0), label='Non-Scream', color='#3498db', alpha=0.8, linewidth=2)
ax1.set_title('Chroma Features Over Time', fontsize=14, fontweight='bold')
ax1.set_xlabel('Frame', fontsize=11)
ax1.set_ylabel('Chroma Mean', fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='--')

scream_mean = np.mean(chroma1)
scream_std = np.std(chroma1)
non_scream_mean = np.mean(chroma2)
non_scream_std = np.std(chroma2)

ax2.bar(x_pos[0] - width/2, scream_mean, width, label='Scream Mean', alpha=0.8, color='#e74c3c', edgecolor='black')
ax2.bar(x_pos[0] + width/2, scream_std, width, label='Scream Std', alpha=0.8, color='#c0392b', edgecolor='black')
ax2.bar(x_pos[1] - width/2, non_scream_mean, width, label='Non-Scream Mean', alpha=0.8, color='#3498db', edgecolor='black')
ax2.bar(x_pos[1] + width/2, non_scream_std, width, label='Non-Scream Std', alpha=0.8, color='#2980b9', edgecolor='black')
ax2.set_title('Chroma Features Statistics', fontsize=14, fontweight='bold')
ax2.set_ylabel('Value', fontsize=11)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(['Scream', 'Non-Scream'], fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

plt.tight_layout()
plt.savefig('feature_chroma.png', dpi=150, bbox_inches='tight')
print("Saved: feature_chroma.png")
plt.show()
plt.close()

# 7. Spectral Contrast
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

ax1.plot(np.mean(contrast1, axis=0), label='Scream', color='#e74c3c', alpha=0.8, linewidth=2)
ax1.plot(np.mean(contrast2, axis=0), label='Non-Scream', color='#3498db', alpha=0.8, linewidth=2)
ax1.set_title('Spectral Contrast Over Time', fontsize=14, fontweight='bold')
ax1.set_xlabel('Frame', fontsize=11)
ax1.set_ylabel('Contrast Mean', fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='--')

scream_mean = np.mean(contrast1)
scream_std = np.std(contrast1)
non_scream_mean = np.mean(contrast2)
non_scream_std = np.std(contrast2)

ax2.bar(x_pos[0] - width/2, scream_mean, width, label='Scream Mean', alpha=0.8, color='#e74c3c', edgecolor='black')
ax2.bar(x_pos[0] + width/2, scream_std, width, label='Scream Std', alpha=0.8, color='#c0392b', edgecolor='black')
ax2.bar(x_pos[1] - width/2, non_scream_mean, width, label='Non-Scream Mean', alpha=0.8, color='#3498db', edgecolor='black')
ax2.bar(x_pos[1] + width/2, non_scream_std, width, label='Non-Scream Std', alpha=0.8, color='#2980b9', edgecolor='black')
ax2.set_title('Spectral Contrast Statistics', fontsize=14, fontweight='bold')
ax2.set_ylabel('Value', fontsize=11)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(['Scream', 'Non-Scream'], fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

plt.tight_layout()
plt.savefig('feature_spectral_contrast.png', dpi=150, bbox_inches='tight')
print("Saved: feature_spectral_contrast.png")
plt.show()
plt.close()

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print("\nGenerated Visualizations:")
print("  • feature_rms_energy.png")
print("  • feature_zero_crossing_rate.png")
print("  • feature_spectral_centroid.png")
print("  • feature_spectral_rolloff.png")
print("  • feature_spectral_bandwidth.png")
print("  • feature_chroma.png")
print("  • feature_spectral_contrast.png")
print("="*60)

