import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

#Load Dataset
df=pd.read_csv("C:\\Users\\joels\\OneDrive\\Desktop\\Programming\\Python\\CSV file\\audio_data_eda.csv")

print(df.info())
print(df.head())

#Encode Label
df["Label"] = df["Label"].replace({"crime-scene": 1, "non-crime": 0})

# Convert MFCC string to numerical array
df["MFCC"] = df["MFCC"].apply(lambda x: np.array(x.strip("[]").split(), dtype=np.float32) if isinstance(x, str) else np.nan)
df=df.dropna(subset=['MFCC'])
mfcc_matrix=np.stack(df['MFCC'].values)

# Apply PCA to MFCC features
pca=PCA(n_components=1)
mfcc_pca=pca.fit_transform(mfcc_matrix)
mfcc_pca_df=pd.DataFrame(mfcc_pca,columns=['MFCC_PCA'])

# Merge PCA-transformed data
df = df.reset_index(drop=True)
mfcc_pca_df = mfcc_pca_df.reset_index(drop=True)
df = pd.concat([df, mfcc_pca_df], axis=1)

# Define feature columns
features = ["RMS", "ZCR", "MFCC_PCA", "Spectral Centroid", "Spectral Flux", "Energy"]
df[features]=df[features].apply(pd.to_numeric,errors='coerce')

#Features for training
x = df[features]
y = df["Label"]

# Standardizing features
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)

#Splitting into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=42)

# Train and evaluate Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)
rf_pred = rf_model.predict(x_test)

# Train and evaluate SVM
svm_model = SVC(kernel="rbf", random_state=42)
svm_model.fit(x_train, y_train)
svm_pred = svm_model.predict(x_test)

# Train and evaluate Logistic Regression
log_model = LogisticRegression(random_state=42, max_iter=1000)
log_model.fit(x_train, y_train)
log_pred = log_model.predict(x_test)

# Print accuracy scores
print(f"Random Forest Accuracy (0 to 1): {accuracy_score(y_test, rf_pred):.2f}")
print(f"SVM Accuracy (0 to 1): {accuracy_score(y_test, svm_pred):.2f}")
print(f"Logistic Regression Accuracy (0 to 1): {accuracy_score(y_test, log_pred):.2f}")

# Print classification reports
print("\nRandom Forest Classification Report:\n", classification_report(y_test, rf_pred))
print("\nSVM Classification Report:\n", classification_report(y_test, svm_pred))
print("\nLogistic Regression Classification Report:\n", classification_report(y_test, log_pred))

#Graphs based on accuracy
models=["Random Forest","SVM","Logistic Regression"]
accuracies=[accuracy_score(y_test,rf_pred),
            accuracy_score(y_test,svm_pred),
            accuracy_score(y_test,log_pred)]
plt.bar(models,accuracies)
plt.ylim(0,1)
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.show()