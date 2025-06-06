#CrimeAlert
This project is a Python-based machine learning system designed to classify audio or event-based data, determining whether it represents a crime or non-crime situation. It demonstrates the application of supervised learning techniques to help build intelligent safety systems that can automatically detect and report suspicious activities based on input data.
The goal is to detect whether a given instance represents a crime or not, using structured features extracted from real-world data such as audio signals, event logs, or contextual indicators. The project includes data preprocessing, dimensionality reduction, model training, and performance evaluation.

Data Preprocessing
  Clean and scale the input data using StandardScaler

Dimensionality Reduction
  Apply PCA to reduce noise and improve model performance

Model Training
  Train and evaluate three classifiers:
  Random Forest
  Support Vector Machine (SVM)
  Logistic Regression

Evaluation
  Use accuracy_score and classification_report to evaluate each model

Visualization
  Plot results using matplotlib to analyze PCA and classification output
