import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display 
from glob import glob 
import IPython.display as ipd
from scipy.fftpack import dct
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

scream_audio_files = glob('/kaggle/input/audio-dataset-of-scream-and-non-scream/Converted_Separately/scream/*.wav')
non_scream_audio_files = glob('/kaggle/input/audio-dataset-of-scream-and-non-scream/Converted_Separately/non_scream/*.wav')

# Play audio files
print("Scream based audio example file:")
display(ipd.Audio(scream_audio_files[0]))

print("Non-Scream based audio example file:")
display(ipd.Audio(non_scream_audio_files[0]))

#Creating a CSV sheet containing all MFCC valeus extracted from the rest of the files.
def extract_features(file_path, label, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    # Create feature names
    feature_names = [f"mfcc{i+1}_mean" for i in range(n_mfcc)] + [f"mfcc{i+1}_std" for i in range(n_mfcc)]
    features = np.hstack([mfcc_mean, mfcc_std])
    
    return dict(zip(feature_names, features)) | {
        "label": label,
        "file": os.path.basename(file_path)
    }

# Build dataset
data = []

for file in scream_audio_files:
    data.append(extract_features(file, "scream"))

for file in non_scream_audio_files:
    data.append(extract_features(file, "non_scream"))

# Convert to DataFrame
df = pd.DataFrame(data)

print("Dataset shape:", df.shape)
print(df.head())
print(df.tail())

#Creating a CSV file to store the database
df.to_csv("audio_features.csv", index=False)

# Train-Test data for ML model
df=pd.read_csv("audio_features.csv")

features = df.drop(columns=["label", "file"]).columns.tolist()
X = df[features]
y = df["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)

# Evaluating ML models

#Load saved Train-Test data
X_train, X_test, y_train, y_test = joblib.load("/kaggle/working/train_test_split.pkl")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Train and evaluate Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Train and evaluate SVM
svm_model = SVC(kernel="rbf", random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

# Train and evaluate Logistic Regression
log_model = LogisticRegression(random_state=42, max_iter=1000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

# Print accuracy scores
print(f"Random Forest Accuracy (0 to 1): {accuracy_score(y_test, rf_pred):.2f}")
print(f"SVM Accuracy (0 to 1): {accuracy_score(y_test, svm_pred):.2f}")
print(f"Logistic Regression Accuracy (0 to 1): {accuracy_score(y_test, log_pred):.2f}")

# Print classification reports
print("\nRandom Forest Classification Report:\n", classification_report(y_test, rf_pred))
print("\nSVM Classification Report:\n", classification_report(y_test, svm_pred))
print("\nLogistic Regression Classification Report:\n", classification_report(y_test, log_pred))