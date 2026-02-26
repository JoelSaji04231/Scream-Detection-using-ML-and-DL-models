import torch
import librosa
import numpy as np
import joblib
import time
import os

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
        cnn_model = torch.load('models/cnn_spectrogram.pth', map_location='cpu')
        cnn_model.eval()
        
        # Prepare input
        input_tensor = torch.FloatTensor(spectrogram).unsqueeze(0).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = cnn_model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get label
        label_encoder = joblib.load('models/cnn_spec_label_encoder.pkl')
        prediction_label = label_encoder.inverse_transform(predicted.numpy())[0]
        
        prediction_time = time.time() - start_time
        
        return {
            'prediction': prediction_label.upper(),
            'confidence': confidence.item(),
            'prediction_time': prediction_time
        }
        
    except Exception as e:
        print(f"CNN prediction error: {e}")
        return {'prediction': 'ERROR', 'confidence': 0.0, 'prediction_time': 0.0}