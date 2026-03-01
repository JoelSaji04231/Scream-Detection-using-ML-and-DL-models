"""
SVM Inference Module
Real-time scream detection using trained SVM model
"""

import numpy as np
import librosa
import joblib
import time
import os
import warnings
warnings.filterwarnings('ignore')

class SVMInference:
    def __init__(self, model_path="models/svm_esc50_pipeline.pkl"):
        """Initialize SVM inference with trained model"""
        self.model_path = model_path
        self.pipeline = None
        self.scaler = None
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        self.is_loaded = False
        
        # Audio processing parameters
        self.sample_rate = 22050
        self.duration = 3.0
        self.n_mfcc = 13
        self.n_fft = 2048
        self.hop_length = 512
        
        self.load_model()
    
    def load_model(self):
        """Load trained SVM pipeline"""
        try:
            self.pipeline = joblib.load(self.model_path)
            
            # Check if it's a dict or a Pipeline object
            if isinstance(self.pipeline, dict):
                self.scaler = self.pipeline['scaler']
                self.model = self.pipeline['model']
                self.label_encoder = self.pipeline.get('label_encoder')
                self.feature_names = self.pipeline.get('feature_names')
            else:
                # It's a Pipeline object
                self.scaler = self.pipeline.named_steps['scaler']
                self.model = self.pipeline.named_steps['classifier']
                self.label_encoder = None
                self.feature_names = None
            
            # If label_encoder or feature_names are missing, try to load from train_test_split
            if self.label_encoder is None or self.feature_names is None:
                try:
                    # Try to find the matching split file
                    split_path = "models/train_test_split_esc50.pkl"
                    if not os.path.exists(split_path):
                        split_path = "models/train_test_split.pkl"
                    
                    if os.path.exists(split_path):
                        data = joblib.load(split_path)
                        if self.label_encoder is None:
                            self.label_encoder = data.get('label_encoder')
                        if self.feature_names is None:
                            self.feature_names = data.get('feature_names')
                except Exception as e:
                    print(f"⚠ Could not load auxiliary model data: {e}")
            
            # Final check for label encoder (default to scream/non_scream if missing)
            if self.label_encoder is None:
                from sklearn.preprocessing import LabelEncoder
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(['non_scream', 'scream'])
                
            self.is_loaded = True
            print(f"✓ SVM model loaded from {self.model_path}")
        except FileNotFoundError:
            print(f"✗ Model file not found: {self.model_path}")
            print("Please run train_ml.py first to train the model")
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
    
    def extract_features(self, audio):
        """Extract features from audio signal"""
        features = {}
        
        # MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc, 
                                   n_fft=self.n_fft, hop_length=self.hop_length)
        
        for i in range(self.n_mfcc):
            features[f'mfcc{i+1}_mean'] = np.mean(mfcc[i])
            features[f'mfcc{i+1}_std'] = np.std(mfcc[i])
        
        # MFCC delta features
        mfcc_delta = librosa.feature.delta(mfcc)
        for i in range(self.n_mfcc):
            features[f'mfcc{i+1}_delta_mean'] = np.mean(mfcc_delta[i])
            features[f'mfcc{i+1}_delta_std'] = np.std(mfcc_delta[i])
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate, 
                                                             n_fft=self.n_fft, hop_length=self.hop_length)
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_std'] = np.std(spectral_centroid)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate, 
                                                           n_fft=self.n_fft, hop_length=self.hop_length)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate, 
                                                               n_fft=self.n_fft, hop_length=self.hop_length)
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # RMS energy
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate, 
                                           n_fft=self.n_fft, hop_length=self.hop_length)
        for i in range(12):
            features[f'chroma{i}_mean'] = np.mean(chroma[i])
            features[f'chroma{i}_std'] = np.std(chroma[i])
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
        features['tempo'] = tempo
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate, 
                                                             n_fft=self.n_fft, hop_length=self.hop_length)
        for i in range(spectral_contrast.shape[0]):
            features[f'spectral_contrast{i}_mean'] = np.mean(spectral_contrast[i])
            features[f'spectral_contrast{i}_std'] = np.std(spectral_contrast[i])
        
        # Tonnetz
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=self.sample_rate)
        for i in range(tonnetz.shape[0]):
            features[f'tonnetz{i}_mean'] = np.mean(tonnetz[i])
            features[f'tonnetz{i}_std'] = np.std(tonnetz[i])
        
        return features
    
    def predict_from_file(self, audio_file):
        """Predict from audio file"""
        if not self.is_loaded:
            return None, None, "Model not loaded"
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_file, sr=self.sample_rate, duration=self.duration)
            audio = self.clean_audio(audio)
            
            # Extract features
            features = self.extract_features(audio)
            
            # Convert to DataFrame
            import pandas as pd
            feature_df = pd.DataFrame([features])
            
            # Predict
            start_time = time.time()
            
            # Handle both Pipeline and dict formats
            if not isinstance(self.pipeline, dict):
                # If it's a Pipeline, we use it directly
                # If we have feature names, ensure correct order
                if self.feature_names is not None:
                    # Filter out any extra features that might have been added
                    feature_df = feature_df[self.feature_names]
                
                prediction = self.pipeline.predict(feature_df)[0]
                probability = self.pipeline.predict_proba(feature_df)[0]
            else:
                # It's a dict with separate components
                if self.feature_names is not None:
                    feature_df = feature_df[self.feature_names]
                
                X_scaled = self.scaler.transform(feature_df)
                prediction = self.model.predict(X_scaled)[0]
                probability = self.model.predict_proba(X_scaled)[0]
                
            inference_time = (time.time() - start_time) * 1000  # ms
            
            # Decode prediction
            if isinstance(prediction, str):
                predicted_label = prediction
            elif self.label_encoder:
                predicted_label = self.label_encoder.inverse_transform([prediction])[0]
            else:
                predicted_label = "scream" if prediction == 1 else "non_scream"
                
            confidence = max(probability)
            
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
            
            # Extract features
            features = self.extract_features(audio)
            
            # Convert to DataFrame
            import pandas as pd
            feature_df = pd.DataFrame([features])
            
            # Predict
            start_time = time.time()
            
            # Handle both Pipeline and dict formats
            if not isinstance(self.pipeline, dict):
                if self.feature_names is not None:
                    feature_df = feature_df[self.feature_names]
                
                prediction = self.pipeline.predict(feature_df)[0]
                probability = self.pipeline.predict_proba(feature_df)[0]
            else:
                if self.feature_names is not None:
                    feature_df = feature_df[self.feature_names]
                
                X_scaled = self.scaler.transform(feature_df)
                prediction = self.model.predict(X_scaled)[0]
                probability = self.model.predict_proba(X_scaled)[0]
                
            inference_time = (time.time() - start_time) * 1000  # ms
            
            # Decode prediction
            if isinstance(prediction, str):
                predicted_label = prediction
            elif self.label_encoder:
                predicted_label = self.label_encoder.inverse_transform([prediction])[0]
            else:
                predicted_label = "scream" if prediction == 1 else "non_scream"
                
            confidence = max(probability)
            
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
        
        return {
            'model_type': 'SVM',
            'kernel': getattr(self.model, 'kernel', 'unknown'),
            'C': getattr(self.model, 'C', 'unknown'),
            'gamma': getattr(self.model, 'gamma', 'unknown'),
            'classes': list(self.label_encoder.classes_) if self.label_encoder else 'unknown',
            'n_features': len(self.feature_names) if self.feature_names is not None else 'unknown',
            'sample_rate': self.sample_rate,
            'duration': self.duration
        }

def quick_predict(audio_path):
    """Quick prediction function for SVM model"""
    try:
        start_time = time.time()
        
        # Initialize inference
        inference = SVMInference()
        if not inference.is_loaded:
            return {'prediction': 'ERROR', 'confidence': 0.0, 'prediction_time': 0.0}
        
        # Predict from file
        prediction, confidence, _ = inference.predict_from_file(audio_path)
        
        if prediction is None:
            return {'prediction': 'ERROR', 'confidence': 0.0, 'prediction_time': 0.0}
            
        prediction_time = (time.time() - start_time)
        
        return {
            'prediction': str(prediction).upper(),
            'confidence': float(confidence),
            'prediction_time': float(prediction_time)
        }
        
    except Exception as e:
        print(f"SVM prediction error: {e}")
        return {'prediction': 'ERROR', 'confidence': 0.0, 'prediction_time': 0.0}

# ===== MAIN EXECUTION =====

def test_inference():
    """Test the inference module"""
    print("Testing SVM Inference Module")
    print("="*50)
    
    # Initialize inference
    inference = SVMInference()
    
    if not inference.is_loaded:
        print("✗ Model loading failed")
        return
    
    # Get model info
    model_info = inference.get_model_info()
    print(f"Model Type: {model_info['model_type']}")
    print(f"Kernel: {model_info['kernel']}")
    print(f"Classes: {model_info['classes']}")
    print(f"Features: {model_info['n_features']}")
    
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
    
    print("\n✓ SVM inference test complete!")

if __name__ == "__main__":
    test_inference()