import pandas as pd
import numpy as np
import librosa
import joblib
import warnings
import time
warnings.filterwarnings('ignore')

# Load the trained SVM pipeline (trained with ESC-50)
print("Loading SVM (ESC-50) pipeline...")
svm_pipeline = joblib.load("models/svm_esc50_pipeline.pkl")
print("SVM (ESC-50) pipeline loaded successfully!")

# Load train-test split to get feature names
train_test_data = joblib.load("models/train_test_split_esc50.pkl")
feature_names = train_test_data['feature_names']

# Extract components
scaler = svm_pipeline.named_steps['scaler']
svm_model = svm_pipeline.named_steps['classifier']

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

def extract_features(file_path):
    """Extract comprehensive audio features matching the training dataset with data cleaning"""
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=22050)
        
        # Clean audio data BEFORE feature extraction
        y_clean = clean_audio(y, sr)
        
        features = {}
        
        # MFCC features (13 coefficients, mean and std)
        mfcc = librosa.feature.mfcc(y=y_clean, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc{i+1}_mean'] = np.mean(mfcc[i]) 
            features[f'mfcc{i+1}_std'] = np.std(mfcc[i])
        
        # RMS Energy features
        rms = librosa.feature.rms(y=y_clean)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        # Zero Crossing Rate features
        zcr = librosa.feature.zero_crossing_rate(y_clean)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # Spectral Centroid features
        spectral_centroid = librosa.feature.spectral_centroid(y=y_clean, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_std'] = np.std(spectral_centroid)
        
        # Spectral Rolloff features
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y_clean, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Handle NaN/Inf in extracted features
        for key in features:
            if np.isnan(features[key]) or np.isinf(features[key]):
                features[key] = 0.0
        
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def quick_predict(file_path):
    
    start_time = time.time()
    
    features = extract_features(file_path)
    if features is None:
        print(f"Error processing {file_path}")
        return None
    
    feature_df = pd.DataFrame([features])
    
    for feature in feature_names:
        if feature not in feature_df.columns:
            feature_df[feature] = 0
    
    feature_df = feature_df[feature_names]
    
    features_scaled = scaler.transform(feature_df)
    
    prediction = svm_model.predict(features_scaled)[0]
    probabilities = svm_model.predict_proba(features_scaled)[0]
    
    prediction_time = time.time() - start_time
    
    result = {
        'file': file_path,
        'prediction': 'SCREAM' if prediction == 'scream' else 'NON-SCREAM',
        'confidence': max(probabilities),
        'scream_probability': probabilities[1] if prediction == 'scream' else probabilities[0],
        'prediction_time': prediction_time
    }
    
    return result
 
