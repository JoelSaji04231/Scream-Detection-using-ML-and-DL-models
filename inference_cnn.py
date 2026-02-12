import pandas as pd
import numpy as np
import librosa
import joblib
import sounddevice as sd
from scipy.io.wavfile import write
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import time
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ===== DEFINE CNN MODEL ARCHITECTURE (must match dl.py) =====
class SpectrogramCNN(nn.Module):
    def __init__(self):
        super(SpectrogramCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.25)
        
        # Fourth convolutional block
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout2d(0.25)
        
        # Calculate size after conv layers (128x128 -> 64 -> 32 -> 16 -> 8)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout5 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.dropout6 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(256, 1)
    
    def forward(self, x):
        # Conv blocks
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout4(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense layers
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout5(x)
        
        x = F.relu(self.bn6(self.fc2(x)))
        x = self.dropout6(x)
        
        x = self.fc3(x)
        return x


# Load the trained CNN model (BEST MODEL: CNN-Spectrogram)
print("Loading trained CNN model...")
cnn_model = SpectrogramCNN().to(device)
cnn_model.load_state_dict(torch.load("models/cnn_spectrogram.pth", map_location=device))
cnn_model.eval()
print("CNN Spectrogram model loaded successfully!")

# Load label encoder
label_encoder = joblib.load("models/cnn_spec_label_encoder.pkl")
print(f"Label encoding: {dict(enumerate(label_encoder.classes_))}")


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


def extract_spectrogram(file_path, n_fft=2048, hop_length=512, target_shape=(128, 128)):
    """Extract spectrogram from audio file with data cleaning (matches training)"""
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=22050)
        
        # CRITICAL: Clean audio BEFORE generating spectrogram (same as training)
        y_clean = clean_audio(y, sr)
        
        # Compute STFT (Short-Time Fourier Transform)
        stft = librosa.stft(y_clean, n_fft=n_fft, hop_length=hop_length)
        
        # Convert to magnitude spectrogram
        spec = np.abs(stft)
        
        # Convert to log scale (dB)
        spec_db = librosa.amplitude_to_db(spec, ref=np.max)
        
        # Resize to target shape
        if spec_db.shape != target_shape:
            # Use librosa's resize function for better quality
            import scipy.ndimage
            zoom_factors = (target_shape[0] / spec_db.shape[0], target_shape[1] / spec_db.shape[1])
            spec_db = scipy.ndimage.zoom(spec_db, zoom_factors, order=1)
        
        # Normalize to [0, 1]
        spec_normalized = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min() + 1e-8)
        
        return spec_normalized
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def quick_predict(file_path):
    """Predict scream/non-scream from an audio file"""
    
    start_time = time.time()
    
    # Extract spectrogram
    spec = extract_spectrogram(file_path)
    if spec is None:
        print(f"Error processing {file_path}")
        return None
    
    # Add batch and channel dimensions: (1, 1, height, width)
    spec_tensor = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = cnn_model(spec_tensor)
        probability = torch.sigmoid(output).item()
        prediction = 1 if probability > 0.5 else 0
    
    prediction_time = time.time() - start_time
    
    result = {
        'file': file_path,
        'prediction': 'SCREAM' if prediction == 1 else 'NON-SCREAM',
        'confidence': probability if prediction == 1 else (1 - probability),
        'scream_probability': probability,
        'prediction_time': prediction_time
    }
    
    return result
