# CrimeAlert

**CrimeAlert** is a Python-based machine learning system designed to classify audio or event-based data to determine whether it represents a **crime** or **non-crime** situation.

It demonstrates the application of supervised learning techniques to help build intelligent safety systems that can automatically detect and report suspicious activities based on structured input data.

The goal is to detect whether a given instance represents a crime or not, using features extracted from real-world sources such as audio signals, event logs, or contextual indicators.

---

## Project Structure

The project is organized into **two main files**:

1. **Analytics File**
   - Performs detailed analysis of the audio dataset.
   - Generates and visualizes:
     - Frequency of **scream** and **non-scream** events
     - Spectrograms for both scream and non-scream audio
     - Mel-spectrograms
     - MFCC (Mel-Frequency Cepstral Coefficients)
     - Mean, standard deviation, and absolute differences of MFCCs  
   - Helps in understanding the audio characteristics before model training.

2. **ML File**
   - Focuses on **machine learning workflow**:
     - Data preprocessing using `StandardScaler`
     - Feature extraction from audio signals (MFCCs, mean, std, etc.)
     - Model training using:
       - Random Forest
       - Support Vector Machine (SVM)
       - Logistic Regression
     - Performance evaluation using:
       - `accuracy_score`
       - `classification_report`  
   - Provides the final predictions and metrics for crime detection.

---

## Project Workflow

### 1. Data Preprocessing
- Clean, normalize, and scale the input features using `StandardScaler`.

### 2. Feature Extraction
- Extract meaningful audio features:
  - MFCCs
  - Mean and standard deviation
  - Absolute differences between MFCCs
  - Spectrogram and Mel-spectrogram representations

### 3. Model Training
- Train multiple classifiers on the processed dataset:
  - Random Forest
  - Support Vector Machine (SVM)
  - Logistic Regression

### 4. Evaluation
- Measure model performance on test data using:
  - Accuracy
  - Precision, recall, and F1-score (via `classification_report`)

### 5. Visualization
- Plot and visualize:
  - Frequency of scream vs non-scream
  - Spectrograms and Mel-spectrograms
  - MFCC features and their statistics

---

## Requirements

- Python 3.8+
- Libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `librosa`

---
