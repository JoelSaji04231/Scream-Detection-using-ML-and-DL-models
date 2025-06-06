# CrimeAlert

**CrimeAlert** is a Python-based machine learning system designed to classify audio or event-based data to determine whether it represents a **crime** or **non-crime** situation.

It demonstrates the application of supervised learning techniques to help build intelligent safety systems that can automatically detect and report suspicious activities based on structured input data.

The goal is to detect whether a given instance represents a crime or not, using features extracted from real-world sources such as audio signals, event logs, or contextual indicators.  
The project includes **data preprocessing**, **dimensionality reduction**, **model training**, and **performance evaluation**.

---

## Project Workflow

### 1. Data Preprocessing
- Clean and scale the input data using `StandardScaler`.

### 2. Dimensionality Reduction
- Apply **Principal Component Analysis (PCA)** to reduce noise and improve model performance.

### 3. Model Training
Train and evaluate three classifiers:
- Random Forest
- Support Vector Machine (SVM)
- Logistic Regression

### 4. Evaluation
- Use `accuracy_score` and `classification_report` to evaluate each model’s performance on test data.

### 5. Visualization
- Use `matplotlib` to plot:
  - PCA-transformed feature distributions
  - Model performance metrics

