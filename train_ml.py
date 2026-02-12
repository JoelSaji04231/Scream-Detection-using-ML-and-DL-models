"""
ML Model Training with ESC-50 Dataset Integration
Trains Random Forest, SVM, and Logistic Regression with:
- ESC-50 environmental sounds as additional non-scream examples
- Class weighting to handle imbalance
- Cross-validation for better evaluation
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


print("\n" + "="*60)
print("ML TRAINING WITH ESC-50 DATASET")
print("="*60)

# Check if ESC-50 features exist
import glob
esc50_audio_files = glob.glob('ESC-50-master/audio/*.wav')
print(f"\nFound {len(esc50_audio_files)} ESC-50 audio files")

# Load features with ESC-50
csv_file = "audio_features.csv"

if not os.path.exists(csv_file):
    print(f"\n{csv_file} not found!")
    print("Please run extract_features.py first to generate features with ESC-50 dataset")
    print("\nOr run:")
    print("  python extract_features.py")
    exit(1)

# Load features
df = pd.read_csv(csv_file)

print(f"\nDataset loaded: {len(df)} samples")
print(f"\nClass distribution:")
print(df['label'].value_counts())
print(f"Class ratio: {df['label'].value_counts()['non_scream'] / df['label'].value_counts()['scream']:.2f}:1 (non-scream:scream)")

features = df.drop(columns=["label", "file"]).columns.tolist()
X = df[features]
y = df["label"]

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain/Test split:")
print(f"  X_train shape: {X_train.shape}")
print(f"  X_test shape: {X_test.shape}")
print(f"\n  Train class distribution:")
print(y_train.value_counts())

# Create models directory
models_dir = "models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Save train-test split
joblib.dump({
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
    'feature_names': features
}, os.path.join(models_dir, 'train_test_split_esc50.pkl'))
print("\nTrain-test split saved")

# Train Random Forest with class weighting
print("\n" + "="*60)
print("Training Random Forest with ESC-50 + Class Weighting...")
print("="*60)

rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

# Cross-validation
rf_cv_scores = cross_val_score(rf_pipeline, X_train, y_train, cv=5, scoring='f1_weighted', n_jobs=-1)
print(f"Cross-validation F1: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std() * 2:.4f})")

rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_test)
joblib.dump(rf_pipeline, os.path.join(models_dir, 'random_forest_esc50_pipeline.pkl'))
print("\tRandom Forest (ESC-50) trained and saved")

# Train SVM with class weighting
print("\n" + "="*60)
print("Training SVM with ESC-50 + Class Weighting...")
print("="*60)

svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(
        kernel="rbf",
        class_weight='balanced',
        C=1.0,
        gamma='scale',
        random_state=42,
        probability=True
    ))
])

# Cross-validation
svm_cv_scores = cross_val_score(svm_pipeline, X_train, y_train, cv=5, scoring='f1_weighted', n_jobs=-1)
print(f"Cross-validation F1: {svm_cv_scores.mean():.4f} (+/- {svm_cv_scores.std() * 2:.4f})")

svm_pipeline.fit(X_train, y_train)
svm_pred = svm_pipeline.predict(X_test)
joblib.dump(svm_pipeline, os.path.join(models_dir, 'svm_esc50_pipeline.pkl'))
print("\tSVM (ESC-50) trained and saved")

# Train Logistic Regression with class weighting
print("\n" + "="*60)
print("Training Logistic Regression with ESC-50 + Class Weighting...")
print("="*60)

log_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(
        penalty='l2',
        C=1.0,
        class_weight='balanced',
        random_state=42,
        max_iter=1000,
        solver='lbfgs',
        n_jobs=-1
    ))
])

# Cross-validation
log_cv_scores = cross_val_score(log_pipeline, X_train, y_train, cv=5, scoring='f1_weighted', n_jobs=-1)
print(f"Cross-validation F1: {log_cv_scores.mean():.4f} (+/- {log_cv_scores.std() * 2:.4f})")

log_pipeline.fit(X_train, y_train)
log_pred = log_pipeline.predict(X_test)
joblib.dump(log_pipeline, os.path.join(models_dir, 'logistic_esc50_pipeline.pkl'))
print("\tLogistic Regression (ESC-50) trained and saved")

# Print results
print("\n" + "="*60)
print("MODEL PERFORMANCE SUMMARY (WITH ESC-50)")
print("="*60)

models_data = {
    'Random Forest (ESC-50)': rf_pred,
    'SVM (ESC-50)': svm_pred,
    'Logistic Regression (ESC-50)': log_pred
}

for model_name, y_pred in models_data.items():
    print(f"\n{model_name.upper()}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Check for overfitting
    train_pred = models_data.get(model_name.replace(" (ESC-50)", ""))
    if model_name == 'SVM (ESC-50)':
        pipeline = svm_pipeline
    elif model_name == 'Random Forest (ESC-50)':
        pipeline = rf_pipeline
    else:
        pipeline = log_pipeline
    
    train_acc = pipeline.score(X_train, y_train)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\nTrain Accuracy: {train_acc*100:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Gap: {(train_acc - test_acc)*100:.2f}%", end="")
    if train_acc - test_acc > 0.1:
        print("\tPotential overfitting")
    else:
        print("\tGood generalization")
    print("-" * 60)

print("\nAll models trained with ESC-50 dataset!")
print("\nSaved models:")
print("  - models/random_forest_esc50_pipeline.pkl")
print("  - models/svm_esc50_pipeline.pkl")
print("  - models/logistic_esc50_pipeline.pkl")
