"""
Deep Learning Model: CNN on Mel-Spectrograms
- Same CNN architecture but uses mel-spectrograms
- Mel scale better represents human perception of sound
- Good for comparison with regular spectrograms
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, precision_recall_fscore_support, matthews_corrcoef
)

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== CNN ARCHITECTURE (Mel-Spectrogram) =====
class MelSpectrogramCNN(nn.Module):
    def __init__(self):
        super(MelSpectrogramCNN, self).__init__()
        
        # Same architecture as SpectrogramCNN
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


# === TRAINING FUNCTION (with class weighting) ===
def train_model(model, train_loader, val_loader, class_weight=None, epochs=30, lr=0.001):
    # Use weighted loss if class weights are provided
    if class_weight is not None:
        pos_weight = torch.tensor([class_weight]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using class weighting - scream weight: {class_weight:.4f}")
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else torch.amp.GradScaler('cpu', enabled=False)
    
    if use_amp:
        print("Using Automatic Mixed Precision (AMP)")
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels.float())
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels.float())
                
                val_loss += loss.item() * inputs.size(0)
                predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).long()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(best_model_state)
    return model


def evaluate_model(model, test_loader):
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


# ===== MAIN TRAINING =====
if __name__ == '__main__':
    print("\n" + "="*60)
    print("CNN MEL-SPECTROGRAM MODEL TRAINING")
    print("="*60)
    
    print("\nLoading datasets...")
    mel_spectrograms = np.load("data/cnn_mel_spectrograms.npy")
    cnn_labels = np.load("data/cnn_labels.npy")
    
    print(f"Mel-Spectrograms shape: {mel_spectrograms.shape}")
    print(f"Labels shape: {cnn_labels.shape}")
    
    # Check class distribution
    unique, counts = np.unique(cnn_labels, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"\nClass distribution:")
    for label, count in class_dist.items():
        print(f"  {label}: {count} ({count/len(cnn_labels)*100:.2f}%)")
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(cnn_labels)
    print(f"\nLabel encoding: {dict(enumerate(label_encoder.classes_))}")
    
    # Calculate class weights for imbalanced dataset
    n_neg = np.sum(y_encoded == 0)  # non_scream
    n_pos = np.sum(y_encoded == 1)  # scream
    class_weight = n_neg / n_pos
    print(f"\nClass imbalance ratio: 1:{class_weight:.2f} (scream:non-scream)")
    print(f"Applying pos_weight={class_weight:.4f} to balance the loss")
    
    X_train, X_test, y_train, y_test = train_test_split(
        mel_spectrograms, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    X_train = X_train[:, np.newaxis, :, :]
    X_test = X_test[:, np.newaxis, :, :]
    
    print(f"\nTrain shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True,
                              pin_memory=use_cuda, num_workers=2 if use_cuda else 0)
    val_loader = DataLoader(val_subset, batch_size=32,
                            pin_memory=use_cuda, num_workers=2 if use_cuda else 0)
    test_loader = DataLoader(test_dataset, batch_size=32,
                             pin_memory=use_cuda, num_workers=2 if use_cuda else 0)
    
    print("\nCreating CNN Mel-Spectrogram model...")
    model = MelSpectrogramCNN().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nTraining CNN Mel-Spectrogram model with class weighting...")
    print("-" * 60)
    model = train_model(model, train_loader, val_loader, class_weight=class_weight)
    
    print("\n" + "="*60)
    print("EVALUATING CNN MEL-SPECTROGRAM MODEL")
    print("="*60)
    
    y_pred, y_pred_proba, _ = evaluate_model(model, test_loader)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1-Score: {f1*100:.2f}%")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"MCC: {mcc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    import joblib
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    torch.save(model.state_dict(), os.path.join(models_dir, 'cnn_mel_spectrogram.pth'))
    joblib.dump(label_encoder, os.path.join(models_dir, 'cnn_mel_label_encoder.pkl'))
    
    print(f"\nModel saved to {os.path.join(models_dir, 'cnn_mel_spectrogram.pth')}")
    print("="*60)
