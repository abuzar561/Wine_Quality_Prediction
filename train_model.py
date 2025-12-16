import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# 1. LOAD DATA
# Ensure 'QualityPrediction.csv' is in the same folder
filename = 'QualityPrediction.csv'
df = pd.read_csv(filename)

print("Data Loaded Successfully!")
print(f"Original shape: {df.shape}")

# 2. DATA CLEANING
# Remove duplicates
df = df.drop_duplicates()
print(f"Shape after removing duplicates: {df.shape}")

# Separate features and target
X = df.drop('quality', axis=1)
y = df['quality']

# 3. PREPROCESSING
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the data (Important for SVC and helpful for Random Forest)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. TRAIN TWO MODELS
print("\nTraining Models...")

# Model A: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_acc:.4f}")

# Model B: Support Vector Machine (SVC)
svc_model = SVC(random_state=42)
svc_model.fit(X_train_scaled, y_train)
svc_pred = svc_model.predict(X_test_scaled)
svc_acc = accuracy_score(y_test, svc_pred)
print(f"SVC Accuracy: {svc_acc:.4f}")

# 5. COMPARE AND SAVE THE BEST MODEL
if rf_acc > svc_acc:
    best_model = rf_model
    best_name = "Random Forest"
    print(f"\nWinner: Random Forest ({(rf_acc - svc_acc)*100:.2f}% better)")
else:
    best_model = svc_model
    best_name = "SVC"
    print(f"\nWinner: SVC ({(svc_acc - rf_acc)*100:.2f}% better)")

# Save the best model and the scaler to files
joblib.dump(best_model, 'wine_quality_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print(f"\nSuccess! The {best_name} model and scaler have been saved to your folder.")