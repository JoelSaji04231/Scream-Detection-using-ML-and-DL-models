"""
Basic Model Comparison
Simple comparison between SVM and CNN models
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import joblib
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("BASIC MODEL COMPARISON")
print("Simple comparison between SVM and CNN models")
print("="*60)

def load_svm_model():
    """Load SVM model and data"""
    try:
        svm_pipeline = joblib.load("models/svm_esc50_pipeline.pkl")
        train_test_data = joblib.load("models/train_test_split_esc50.pkl")
        
        X_test = train_test_data['X_test']
        y_test = train_test_data['y_test']
        
        return svm_pipeline, X_test, y_test
    except FileNotFoundError:
        print("✗ SVM model or data not found. Please run train_ml.py first.")
        return None, None, None

def load_cnn_model():
    """Load CNN model and data"""
    try:
        # Import CNN architecture
        from train_cnn import SpectrogramCNN
        
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cnn_model = SpectrogramCNN().to(device)
        cnn_model.load_state_dict(torch.load("models/cnn_spectrogram.pth", map_location=device))
        cnn_model.eval()
        
        # Load test data
        X_test = np.load("data/cnn_spectrograms.npy")
        y_test = np.load("data/cnn_labels.npy")
        
        return cnn_model, X_test, y_test, device
    except FileNotFoundError:
        print("✗ CNN model or data not found. Please run train_cnn.py first.")
        return None, None, None, None

def evaluate_svm(svm_pipeline, X_test, y_test):
    """Evaluate SVM model"""
    print("\nEvaluating SVM model...")
    
    # Extract components
    scaler = svm_pipeline['scaler']
    model = svm_pipeline['model']
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Predict
    start_time = time.time()
    y_pred = model.predict(X_test_scaled)
    svm_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✓ SVM Accuracy: {accuracy:.4f}")
    print(f"✓ SVM Inference time: {svm_time:.4f}s")
    print(f"✓ SVM Time per sample: {svm_time/len(X_test)*1000:.4f}ms")
    
    return accuracy, svm_time

def evaluate_cnn(cnn_model, X_test, y_test, device):
    """Evaluate CNN model"""
    print("\nEvaluating CNN model...")
    
    # Convert to tensor
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # Create data loader for batch processing
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Predict
    start_time = time.time()
    all_predictions = []
    
    with torch.no_grad():
        for batch_X, _ in test_loader:
            outputs = cnn_model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
    
    cnn_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, all_predictions)
    
    print(f"✓ CNN Accuracy: {accuracy:.4f}")
    print(f"✓ CNN Inference time: {cnn_time:.4f}s")
    print(f"✓ CNN Time per sample: {cnn_time/len(X_test)*1000:.4f}ms")
    
    return accuracy, cnn_time

def compare_models():
    """Compare SVM and CNN models"""
    print("\nLoading models and data...")
    
    # Load SVM
    svm_pipeline, svm_X_test, svm_y_test = load_svm_model()
    if svm_pipeline is None:
        return
    
    # Load CNN
    cnn_model, cnn_X_test, cnn_y_test, device = load_cnn_model()
    if cnn_model is None:
        return
    
    print(f"\nDataset sizes:")
    print(f"SVM test set: {len(svm_X_test)} samples")
    print(f"CNN test set: {len(cnn_X_test)} samples")
    
    # Evaluate models
    svm_accuracy, svm_time = evaluate_svm(svm_pipeline, svm_X_test, svm_y_test)
    cnn_accuracy, cnn_time = evaluate_cnn(cnn_model, cnn_X_test, cnn_y_test, device)
    
    # Comparison summary
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    print(f"\nAccuracy Comparison:")
    print(f"SVM:  {svm_accuracy:.4f}")
    print(f"CNN:  {cnn_accuracy:.4f}")
    print(f"Difference: {abs(svm_accuracy - cnn_accuracy):.4f}")
    
    print(f"\nSpeed Comparison:")
    print(f"SVM:  {svm_time/len(svm_X_test)*1000:.4f}ms per sample")
    print(f"CNN:  {cnn_time/len(cnn_X_test)*1000:.4f}ms per sample")
    
    # Determine winner
    if svm_accuracy > cnn_accuracy:
        winner = "SVM"
        accuracy_diff = svm_accuracy - cnn_accuracy
    else:
        winner = "CNN"
        accuracy_diff = cnn_accuracy - svm_accuracy
    
    print(f"\n🏆 Winner: {winner} (by {accuracy_diff:.4f} accuracy)")
    
    # Speed comparison
    svm_speed = svm_time/len(svm_X_test)*1000
    cnn_speed = cnn_time/len(cnn_X_test)*1000
    
    if svm_speed < cnn_speed:
        speed_winner = "SVM"
        speed_diff = cnn_speed - svm_speed
    else:
        speed_winner = "CNN"
        speed_diff = svm_speed - cnn_speed
    
    print(f"⚡ Faster: {speed_winner} (by {speed_diff:.4f}ms per sample)")
    
    # Recommendations
    print(f"\n📊 Recommendations:")
    if winner == "SVM" and speed_winner == "SVM":
        print("✅ SVM is both more accurate and faster - recommended for production")
    elif winner == "CNN" and speed_winner == "CNN":
        print("✅ CNN is both more accurate and faster - recommended for production")
    elif winner == "SVM":
        print("⚖️ SVM is more accurate but CNN is faster - choose based on your priorities")
    else:
        print("⚖️ CNN is more accurate but SVM is faster - choose based on your priorities")
    
    print(f"\n💡 For real-time applications: Choose {speed_winner}")
    print(f"💡 For accuracy-critical applications: Choose {winner}")

if __name__ == "__main__":
    compare_models()