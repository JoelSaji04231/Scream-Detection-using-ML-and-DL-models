import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, matthews_corrcoef, confusion_matrix
)
import pandas as pd

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== MODEL ARCHITECTURES =====

class SpectrogramCNN(nn.Module):
    def __init__(self):
        super(SpectrogramCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.25)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout2d(0.25)
        
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout5 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.dropout6 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(256, 1)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout4(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout5(x)
        x = F.relu(self.bn6(self.fc2(x)))
        x = self.dropout6(x)
        x = self.fc3(x)
        return x


class CNN_LSTM(nn.Module):
    def __init__(self, input_channels=1, num_classes=1):
        super(CNN_LSTM, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.25)
        
        self.lstm = nn.LSTM(
            input_size=128 * 16,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        self.fc1 = nn.Linear(256 * 2, 128)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        
        batch_size = x.size(0)
        x = x.view(batch_size, x.size(1), x.size(2), x.size(3))
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous().view(batch_size, x.size(1), -1)
        
        lstm_out, (hidden, cell) = self.lstm(x)
        lstm_last = lstm_out[:, -1, :]
        
        x = F.relu(self.fc1(lstm_last))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x


def evaluate_dl_model(model, test_loader):
    """Evaluate a deep learning model on test data"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs.squeeze()).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


def evaluate_ml_model(pipeline, X_test, y_test):
    """Evaluate a machine learning model (sklearn pipeline)"""
    y_pred = pipeline.predict(X_test)
    
    # Get probabilities if available
    if hasattr(pipeline, 'predict_proba'):
        y_proba = pipeline.predict_proba(X_test)
        # For binary classification, get probability of positive class
        y_prob = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba[:, 0]
    else:
        # For models without predict_proba, use decision function
        y_prob = pipeline.decision_function(X_test)
    
    return y_pred, y_prob


# ===== MAIN COMPARISON =====
if __name__ == '__main__':
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL COMPARISON: ML vs DL")
    print("="*60)
    
    results = []
    
    # ===== PART 1: EVALUATE ML MODELS =====
    print("\n" + "="*60)
    print("PART 1: MACHINE LEARNING MODELS")
    print("="*60)
    
    try:
        print("\nLoading ML test data...")
        train_test_data = joblib.load("models/train_test_split_esc50.pkl")
        X_test_ml = train_test_data['X_test']
        y_test_ml = train_test_data['y_test']
        
        # Convert labels to binary (0: non_scream, 1: scream)
        y_test_ml_binary = (y_test_ml == 'scream').astype(int)
        
        print(f"ML Test set size: {len(y_test_ml)} samples")
        
        # Evaluate Random Forest
        print("\n1. Evaluating Random Forest (ESC-50)...")
        rf_pipeline = joblib.load("models/random_forest_esc50_pipeline.pkl")
        y_pred_rf, y_prob_rf = evaluate_ml_model(rf_pipeline, X_test_ml, y_test_ml)
        y_pred_rf_binary = (y_pred_rf == 'scream').astype(int)
        
        acc_rf = accuracy_score(y_test_ml_binary, y_pred_rf_binary)
        prec_rf, rec_rf, f1_rf, _ = precision_recall_fscore_support(
            y_test_ml_binary, y_pred_rf_binary, average='weighted'
        )
        roc_rf = roc_auc_score(y_test_ml_binary, y_prob_rf)
        mcc_rf = matthews_corrcoef(y_test_ml_binary, y_pred_rf_binary)
        
        results.append({
            'Model': 'Random Forest',
            'Type': 'ML',
            'Accuracy': acc_rf * 100,
            'Precision': prec_rf * 100,
            'Recall': rec_rf * 100,
            'F1-Score': f1_rf * 100,
            'ROC-AUC': roc_rf,
            'MCC': mcc_rf
        })
        print(f"\tAccuracy: {acc_rf*100:.2f}%")
        
        # Evaluate SVM
        print("\n2. Evaluating SVM (ESC-50)...")
        svm_pipeline = joblib.load("models/svm_esc50_pipeline.pkl")
        y_pred_svm, y_prob_svm = evaluate_ml_model(svm_pipeline, X_test_ml, y_test_ml)
        y_pred_svm_binary = (y_pred_svm == 'scream').astype(int)
        
        acc_svm = accuracy_score(y_test_ml_binary, y_pred_svm_binary)
        prec_svm, rec_svm, f1_svm, _ = precision_recall_fscore_support(
            y_test_ml_binary, y_pred_svm_binary, average='weighted'
        )
        roc_svm = roc_auc_score(y_test_ml_binary, y_prob_svm)
        mcc_svm = matthews_corrcoef(y_test_ml_binary, y_pred_svm_binary)
        
        results.append({
            'Model': 'SVM',
            'Type': 'ML',
            'Accuracy': acc_svm * 100,
            'Precision': prec_svm * 100,
            'Recall': rec_svm * 100,
            'F1-Score': f1_svm * 100,
            'ROC-AUC': roc_svm,
            'MCC': mcc_svm
        })
        print(f"\tAccuracy: {acc_svm*100:.2f}%")
        
        # Evaluate Logistic Regression
        print("\n3. Evaluating Logistic Regression (ESC-50)...")
        log_pipeline = joblib.load("models/logistic_esc50_pipeline.pkl")
        y_pred_log, y_prob_log = evaluate_ml_model(log_pipeline, X_test_ml, y_test_ml)
        y_pred_log_binary = (y_pred_log == 'scream').astype(int)
        
        acc_log = accuracy_score(y_test_ml_binary, y_pred_log_binary)
        prec_log, rec_log, f1_log, _ = precision_recall_fscore_support(
            y_test_ml_binary, y_pred_log_binary, average='weighted'
        )
        roc_log = roc_auc_score(y_test_ml_binary, y_prob_log)
        mcc_log = matthews_corrcoef(y_test_ml_binary, y_pred_log_binary)
        
        results.append({
            'Model': 'Logistic Regression',
            'Type': 'ML',
            'Accuracy': acc_log * 100,
            'Precision': prec_log * 100,
            'Recall': rec_log * 100,
            'F1-Score': f1_log * 100,
            'ROC-AUC': roc_log,
            'MCC': mcc_log
        })
        print(f"\tAccuracy: {acc_log*100:.2f}%")
        
    except FileNotFoundError as e:
        print(f"\nML models not found: {e}")
        print("\tSkipping ML model evaluation...")
    
    
    # ===== PART 2: EVALUATE DL MODELS =====
    print("\n" + "="*60)
    print("PART 2: DEEP LEARNING MODELS")
    print("="*60)
    
    try:
        print("\nLoading DL test datasets...")
        spectrograms = np.load("data/cnn_spectrograms.npy")
        mel_spectrograms = np.load("data/cnn_mel_spectrograms.npy")
        labels = np.load("data/cnn_labels.npy")
        
        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(labels)
        
        # Split data (same split for fair comparison)
        _, X_test_spec, _, y_test_dl = train_test_split(
            spectrograms, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        _, X_test_mel, _, _ = train_test_split(
            mel_spectrograms, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Add channel dimension
        X_test_spec = X_test_spec[:, np.newaxis, :, :]
        X_test_mel = X_test_mel[:, np.newaxis, :, :]
        
        print(f"DL Test set size: {len(y_test_dl)} samples")
        
        # Evaluate CNN Spectrogram
        print("\n1. Evaluating CNN Spectrogram...")
        cnn_spec_model = SpectrogramCNN().to(device)
        cnn_spec_model.load_state_dict(torch.load("models/cnn_spectrogram.pth", map_location=device))
        test_dataset_spec = TensorDataset(torch.FloatTensor(X_test_spec), torch.LongTensor(y_test_dl))
        test_loader_spec = DataLoader(test_dataset_spec, batch_size=32)
        
        y_pred_spec, y_prob_spec, _ = evaluate_dl_model(cnn_spec_model, test_loader_spec)
        
        acc_spec = accuracy_score(y_test_dl, y_pred_spec)
        prec_spec, rec_spec, f1_spec, _ = precision_recall_fscore_support(
            y_test_dl, y_pred_spec, average='weighted'
        )
        roc_spec = roc_auc_score(y_test_dl, y_prob_spec)
        mcc_spec = matthews_corrcoef(y_test_dl, y_pred_spec)
        
        results.append({
            'Model': 'CNN Spectrogram',
            'Type': 'DL',
            'Accuracy': acc_spec * 100,
            'Precision': prec_spec * 100,
            'Recall': rec_spec * 100,
            'F1-Score': f1_spec * 100,
            'ROC-AUC': roc_spec,
            'MCC': mcc_spec
        })
        print(f"\tAccuracy: {acc_spec*100:.2f}%")
        
        # Evaluate CNN Mel-Spectrogram
        print("\n2. Evaluating CNN Mel-Spectrogram...")
        cnn_mel_model = SpectrogramCNN().to(device)
        cnn_mel_model.load_state_dict(torch.load("models/cnn_mel_spectrogram.pth", map_location=device))
        test_dataset_mel = TensorDataset(torch.FloatTensor(X_test_mel), torch.LongTensor(y_test_dl))
        test_loader_mel = DataLoader(test_dataset_mel, batch_size=32)
        
        y_pred_mel, y_prob_mel, _ = evaluate_dl_model(cnn_mel_model, test_loader_mel)
        
        acc_mel = accuracy_score(y_test_dl, y_pred_mel)
        prec_mel, rec_mel, f1_mel, _ = precision_recall_fscore_support(
            y_test_dl, y_pred_mel, average='weighted'
        )
        roc_mel = roc_auc_score(y_test_dl, y_prob_mel)
        mcc_mel = matthews_corrcoef(y_test_dl, y_pred_mel)
        
        results.append({
            'Model': 'CNN Mel-Spectrogram',
            'Type': 'DL',
            'Accuracy': acc_mel * 100,
            'Precision': prec_mel * 100,
            'Recall': rec_mel * 100,
            'F1-Score': f1_mel * 100,
            'ROC-AUC': roc_mel,
            'MCC': mcc_mel
        })
        print(f"\tAccuracy: {acc_mel*100:.2f}%")
        
        # Evaluate CNN-LSTM
        print("\n3. Evaluating CNN-LSTM...")
        cnn_lstm_model = CNN_LSTM().to(device)
        cnn_lstm_model.load_state_dict(torch.load("models/cnn_lstm_model.pth", map_location=device))
        
        y_pred_lstm, y_prob_lstm, _ = evaluate_dl_model(cnn_lstm_model, test_loader_mel)
        
        acc_lstm = accuracy_score(y_test_dl, y_pred_lstm)
        prec_lstm, rec_lstm, f1_lstm, _ = precision_recall_fscore_support(
            y_test_dl, y_pred_lstm, average='weighted'
        )
        roc_lstm = roc_auc_score(y_test_dl, y_prob_lstm)
        mcc_lstm = matthews_corrcoef(y_test_dl, y_pred_lstm)
        
        results.append({
            'Model': 'CNN-LSTM',
            'Type': 'DL',
            'Accuracy': acc_lstm * 100,
            'Precision': prec_lstm * 100,
            'Recall': rec_lstm * 100,
            'F1-Score': f1_lstm * 100,
            'ROC-AUC': roc_lstm,
            'MCC': mcc_lstm
        })
        print(f"\tAccuracy: {acc_lstm*100:.2f}%")
        
    except FileNotFoundError as e:
        print(f"\nDL models not found: {e}")
        print("\tSkipping DL model evaluation...")
    
    
    # ===== DISPLAY RESULTS =====
    print("\n" + "="*60)
    print("COMPREHENSIVE COMPARISON RESULTS")
    print("="*60)
    
    df_results = pd.DataFrame(results)
    
    # Display results grouped by type
    if 'Type' in df_results.columns:
        print("\n--- MACHINE LEARNING MODELS ---")
        ml_results = df_results[df_results['Type'] == 'ML']
        if not ml_results.empty:
            print(ml_results.drop('Type', axis=1).to_string(index=False))
        else:
            print("No ML models evaluated")
        
        print("\n--- DEEP LEARNING MODELS ---")
        dl_results = df_results[df_results['Type'] == 'DL']
        if not dl_results.empty:
            print(dl_results.drop('Type', axis=1).to_string(index=False))
        else:
            print("No DL models evaluated")
        
        print("\n--- ALL MODELS RANKED BY F1-SCORE ---")
        df_sorted = df_results.sort_values('F1-Score', ascending=False)
        print(df_sorted.to_string(index=False))
    else:
        print(df_results.to_string(index=False))
    
    # Find best models
    print("\n" + "="*60)
    print("BEST MODELS BY METRIC")
    print("="*60)
    
    best_acc = df_results.loc[df_results['Accuracy'].idxmax()]
    best_f1 = df_results.loc[df_results['F1-Score'].idxmax()]
    best_roc = df_results.loc[df_results['ROC-AUC'].idxmax()]
    best_mcc = df_results.loc[df_results['MCC'].idxmax()]
    
    print(f"\nBest Accuracy:\t\t{best_acc['Model']} ({best_acc['Accuracy']:.2f}%)")
    print(f"Best F1-Score:\t\t{best_f1['Model']} ({best_f1['F1-Score']:.2f}%)")
    print(f"Best ROC-AUC:\t\t{best_roc['Model']} ({best_roc['ROC-AUC']:.4f})")
    print(f"Best MCC:\t\t{best_mcc['Model']} ({best_mcc['MCC']:.4f})")
