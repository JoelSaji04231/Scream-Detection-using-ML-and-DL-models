import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import joblib
import time
import os

# Define the CNN architecture (must match training)
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
        
        # Fully connected layers
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

def create_spectrogram(audio_path, target_shape=(128, 128)):
    """Create spectrogram from audio file"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050, duration=5.0)
        
        # Ensure consistent length (5 seconds)
        target_length = 5 * sr
        if len(y) > target_length:
            y = y[:target_length]
        else:
            y = np.pad(y, (0, target_length - len(y)))
        
        # Create spectrogram
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        
        # Resize to target shape
        from PIL import Image
        img = Image.fromarray(spectrogram_db)
        img = img.resize((target_shape[1], target_shape[0]), Image.LANCZOS)
        spectrogram_resized = np.array(img)
        
        # Normalize
        spectrogram_normalized = (spectrogram_resized - spectrogram_resized.min()) / (spectrogram_resized.max() - spectrogram_resized.min())
        
        return spectrogram_normalized
        
    except Exception as e:
        print(f"Error creating spectrogram: {e}")
        return None

def quick_predict(audio_path):
    """Quick prediction function for CNN model"""
    try:
        start_time = time.time()
        
        # Create spectrogram
        spectrogram = create_spectrogram(audio_path)
        if spectrogram is None:
            return {'prediction': 'ERROR', 'confidence': 0.0, 'prediction_time': 0.0}
        
        # Load CNN model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cnn_model = SpectrogramCNN().to(device)
        
        # Load state dict
        if not os.path.exists('models/cnn_spectrogram.pth'):
            return {'prediction': 'MODEL NOT FOUND', 'confidence': 0.0, 'prediction_time': 0.0}
            
        state_dict = torch.load('models/cnn_spectrogram.pth', map_location=device)
        cnn_model.load_state_dict(state_dict)
        cnn_model.eval()
        
        # Prepare input
        input_tensor = torch.FloatTensor(spectrogram).unsqueeze(0).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = cnn_model(input_tensor)
            # Binary classification with 1 output (logits)
            probability = torch.sigmoid(outputs).item()
            
            # Label mapping: 0 = non_scream, 1 = scream (based on check_le.py results)
            if probability > 0.5:
                prediction_label = "scream"
                confidence = probability
            else:
                prediction_label = "non_scream"
                confidence = 1.0 - probability
        
        prediction_time = time.time() - start_time
        
        return {
            'prediction': prediction_label.upper(),
            'confidence': float(confidence),
            'prediction_time': float(prediction_time)
        }
        
    except Exception as e:
        print(f"CNN prediction error: {e}")
        return {'prediction': 'ERROR', 'confidence': 0.0, 'prediction_time': 0.0}

def test_inference():
    """Test the CNN inference module"""
    print("Testing CNN Inference Module")
    print("="*50)
    
    # Test with sample audio (if available)
    import glob
    
    # Look for test audio files
    test_files = []
    test_files.extend(glob.glob("Converted_Separately/scream/*.wav")[:2])
    test_files.extend(glob.glob("Converted_Separately/non_scream/*.wav")[:2])
    
    if test_files:
        print(f"\nTesting with {len(test_files)} sample files...")
        
        for audio_file in test_files:
            print(f"\nTesting: {audio_file}")
            
            result = quick_predict(audio_file)
            
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Prediction time: {result['prediction_time']*1000:.2f} ms")
    else:
        print("\nNo test audio files found.")
        print("Please ensure Converted_Separately/ directory has audio files.")
    
    print("\n✓ CNN inference test complete!")

if __name__ == "__main__":
    test_inference()