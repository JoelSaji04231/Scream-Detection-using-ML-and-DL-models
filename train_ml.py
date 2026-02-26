"""
ML Model Training with ESC-50 Dataset Integration
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, precision_recall_fscore_support, matthews_corrcoef
)
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("="*60)
print("MACHINE LEARNING MODELS TRAINING")
print("="*60)

# Load the extracted features
print("\nLoading extracted features...")
try:
    df = pd.read_csv('audio_features.csv')
    print(f"[OK] Features loaded: {df.shape[0]} samples, {df.shape[1]} features")
except FileNotFoundError:
    print("[ERROR] audio_features.csv not found!")
    print("Please run extract_features.py first to generate features with ESC-50 dataset")
    exit(1)

# Separate features and target
print("\nPreparing data...")
feature_columns = [col for col in df.columns if col != 'label']
X = df[feature_columns]
y = df['label']

print(f"✓ Features: {X.shape[1]} dimensions")
print(f"✓ Classes: {sorted(y.unique())}")
print(f"✓ Class distribution:")
print(y.value_counts())

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

# Scale features
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled")

# Compute class weights for imbalanced dataset
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))
print(f"✓ Class weights computed: {class_weight_dict}")

# Store training data for other analyses
train_test_data = {
    'X_train': X_train_scaled,
    'X_test': X_test_scaled,
    'y_train': y_train,
    'y_test': y_test,
    'feature_names': feature_columns,
    'label_encoder': label_encoder,
    'scaler': scaler
}

# Ensure models directory exists
models_dir = "models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

joblib.dump(train_test_data, os.path.join(models_dir, 'train_test_split_esc50.pkl'))
print("✓ Training data saved for other analyses")

print("\n" + "="*60)
print("TRAINING MACHINE LEARNING MODELS")
print("="*60)

# ===== SUPPORT VECTOR MACHINE =====
print("\n1. Training Support Vector Machine...")
svm_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight='balanced',
    probability=True,
    random_state=42
)

svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
y_pred_proba_svm = svm_model.predict_proba(X_test_scaled)[:, 1]

# SVM Performance
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_auc = roc_auc_score(y_test, y_pred_proba_svm)
svm_precision, svm_recall, svm_f1, _ = precision_recall_fscore_support(y_test, y_pred_svm, average='weighted')
svm_mcc = matthews_corrcoef(y_test, y_pred_svm)

print(f"✓ SVM Accuracy: {svm_accuracy:.4f}")
print(f"✓ SVM AUC: {svm_auc:.4f}")
print(f"✓ SVM F1-Score: {svm_f1:.4f}")
print(f"✓ SVM MCC: {svm_mcc:.4f}")

# ===== RANDOM FOREST =====
print("\n2. Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

# Random Forest Performance
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_auc = roc_auc_score(y_test, y_pred_proba_rf)
rf_precision, rf_recall, rf_f1, _ = precision_recall_fscore_support(y_test, y_pred_rf, average='weighted')
rf_mcc = matthews_corrcoef(y_test, y_pred_rf)

print(f"✓ Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"✓ Random Forest AUC: {rf_auc:.4f}")
print(f"✓ Random Forest F1-Score: {rf_f1:.4f}")
print(f"✓ Random Forest MCC: {rf_mcc:.4f}")

# ===== LOGISTIC REGRESSION =====
print("\n3. Training Logistic Regression...")
lr_model = LogisticRegression(
    class_weight='balanced',
    random_state=42,
    max_iter=1000
)

lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

# Logistic Regression Performance
lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_auc = roc_auc_score(y_test, y_pred_proba_lr)
lr_precision, lr_recall, lr_f1, _ = precision_recall_fscore_support(y_test, y_pred_lr, average='weighted')
lr_mcc = matthews_corrcoef(y_test, y_pred_lr)

print(f"✓ Logistic Regression Accuracy: {lr_accuracy:.4f}")
print(f"✓ Logistic Regression AUC: {lr_auc:.4f}")
print(f"✓ Logistic Regression F1-Score: {lr_f1:.4f}")
print(f"✓ Logistic Regression MCC: {lr_mcc:.4f}")

print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

comparison_df = pd.DataFrame({
    'Model': ['SVM', 'Random Forest', 'Logistic Regression'],
    'Accuracy': [svm_accuracy, rf_accuracy, lr_accuracy],
    'AUC': [svm_auc, rf_auc, lr_auc],
    'F1-Score': [svm_f1, rf_f1, lr_f1],
    'MCC': [svm_mcc, rf_mcc, lr_mcc]
})

print(comparison_df.round(4))

# Save models
print("\n" + "="*60)
print("SAVING MODELS")
print("="*60)

# Create pipeline for SVM (best performing typically)
svm_pipeline = {
    'scaler': scaler,
    'model': svm_model,
    'label_encoder': label_encoder,
    'feature_names': feature_columns
}

joblib.dump(svm_pipeline, os.path.join(models_dir, 'svm_esc50_pipeline.pkl'))
print("✓ SVM pipeline saved")

joblib.dump(rf_model, os.path.join(models_dir, 'random_forest_model.pkl'))
print("✓ Random Forest model saved")

joblib.dump(lr_model, os.path.join(models_dir, 'logistic_regression_model.pkl'))
print("✓ Logistic Regression model saved")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"✓ All models trained with ESC-50 dataset!")
print(f"✓ Models saved to {models_dir}/ directory")
print(f"✓ Training data saved for cross-validation and analysis")
print("\nNext steps:")
print("- Run master_analysis.py for comprehensive analysis")
print("- Run main.py for real-time prediction")
print("- Check models/ directory for saved models")