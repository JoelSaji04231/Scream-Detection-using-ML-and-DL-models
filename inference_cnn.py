"""
CNN Inference Module
Real-time scream detection using trained CNN model
"""

import numpy as np
import librosa
import torch
import torch.nn.functional as F
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

# Import the CNN architecture from train_cnn
from train_cnn import SpectrogramCNN

class CNNInference:
    def __init__(self, model_path="models/cnn_spectrogram.pth", device='cpu'):
        """Initialize CNN inference with trained model"""
        self.model_path = model_path
        self.device = device
        self.model = None
        self.is_loaded = False
        
        # Audio processing parameters
        self.sample_rate = 22050
        self.duration = 3.0
        self.n_mels = 128
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 128
        
        self.load_model()
    
    def load_model(self):
        """Load trained CNN model"""
        try:
            self.model = SpectrogramCNN().to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            self.is_loaded = True
            print(f"✓ CNN model loaded from {self.model_path}")
        except FileNotFoundError:
            print(f"✗ Model file not found: {self.model_path}")
            print("Please run train_cnn.py first to train the model")
            self.is_loaded = False
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            self.is_loaded = False
    
    def clean_audio(self, audio):
        """Clean audio signal"""
        # Remove silence
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Normalize
        audio = librosa.util.normalize(audio)
        
        # Ensure minimum length
        target_length = int(self.duration * self.sample_rate)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
        
        return audio
    
    def generate_spectrogram(self, audio):
        """Generate mel-spectrogram from audio"""
        # Generate mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sample_rate, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        # Ensure consistent shape (128, 129)
        target_shape = (self.n_mels, 129)
        if mel_spec_norm.shape != target_shape:
            # Pad or truncate as needed
            if mel_spec_norm.shape[1] < target_shape[1]:
                pad_width = target_shape[1] - mel_spec_norm.shape[1]
                mel_spec_norm = np.pad(mel_spec_norm, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mel_spec_norm = mel_spec_norm[:, :target_shape[1]]
        
        return mel_spec_norm
    
    def predict_from_file(self, audio_file):
        """Predict from audio file"""
        if not self.is_loaded:
            return None, None, "Model not loaded"
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_file, sr=self.sample_rate, duration=self.duration)
            audio = self.clean_audio(audio)
            
            # Generate spectrogram
            spectrogram = self.generate_spectrogram(audio)
            
            # Prepare input tensor
            input_tensor = torch.FloatTensor(spectrogram).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Predict
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                prediction = torch.argmax(outputs, dim=1)
            inference_time = (time.time() - start_time) * 1000  # ms
            
            # Get results
            predicted_class = prediction.item()
            confidence = probabilities[0][predicted_class].item()
            
            # Convert to label
            if predicted_class == 0:
                predicted_label = "non_scream"
            else:
                predicted_label = "scream"
            
            return predicted_label, confidence, inference_time
            
        except Exception as e:
            return None, None, f"Error processing audio: {str(e)}"
    
    def predict_from_audio(self, audio, sr):
        """Predict from audio array"""
        if not self.is_loaded:
            return None, None, "Model not loaded"
        
        try:
            # Resample if necessary
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            
            audio = self.clean_audio(audio)
            
            # Generate spectrogram
            spectrogram = self.generate_spectrogram(audio)
            
            # Prepare input tensor
            input_tensor = torch.FloatTensor(spectrogram).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Predict
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                prediction = torch.argmax(outputs, dim=1)
            inference_time = (time.time() - start_time) * 1000  # ms
            
            # Get results
            predicted_class = prediction.item()
            confidence = probabilities[0][predicted_class].item()
            
            # Convert to label
            if predicted_class == 0:
                predicted_label = "non_scream"
            else:
                predicted_label = "scream"
            
            return predicted_label, confidence, inference_time
            
        except Exception as e:
            return None, None, f"Error processing audio: {str(e)}"
    
    def batch_predict(self, audio_files):
        """Batch prediction for multiple files"""
        results = []
        
        for audio_file in audio_files:
            label, confidence, inference_time = self.predict_from_file(audio_file)
            results.append({
                'file': audio_file,
                'prediction': label,
                'confidence': confidence,
                'inference_time_ms': inference_time
            })
        
        return results
    
    def get_model_info(self):
        """Get model information"""
        if not self.is_loaded:
            return {"error": "Model not loaded"}
        
        # Count model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_type': 'CNN',
            'architecture': 'SpectrogramCNN',
            'device': self.device,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'input_shape': f"(1, {self.n_mels}, 129)"
        }

# ===== MAIN EXECUTION =====

def test_inference():
    """Test the CNN inference module"""
    print("Testing CNN Inference Module")
    print("="*50)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize inference
    inference = CNNInference(device=device)
    
    if not inference.is_loaded:
        print("✗ Model loading failed")
        return
    
    # Get model info
    model_info = inference.get_model_info()
    print(f"Model Type: {model_info['model_type']}")
    print(f"Device: {model_info['device']}")
    print(f"Parameters: {model_info['total_parameters']:,}")
    print(f"Input Shape: {model_info['input_shape']}")
    
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
            
            prediction, confidence, inference_time = inference.predict_from_file(audio_file)
            
            if prediction:
                print(f"Prediction: {prediction}")
                print(f"Confidence: {confidence:.4f}")
                print(f"Inference time: {inference_time:.2f} ms")
            else:
                print(f"Error: {inference_time}")
    else:
        print("\nNo test audio files found.")
        print("Please ensure Converted_Separately/ directory has audio files.")
    
    print("\n✓ CNN inference test complete!")

if __name__ == "__main__":
    test_inference()