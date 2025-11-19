import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ===============================================
# 1. Load dataset (Crop Recommendation dataset)
# ===============================================
data = pd.read_csv("Crop_recommendation.csv")

# Features and target
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

# ===============================================
# 2. Split dataset
# ===============================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================================
# 3. Scale features
# ===============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================================
# 4. Train model
# ===============================================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# ===============================================
# 5. Save model and scaler
# ===============================================
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('standscaler.pkl', 'wb'))

print("✅ Model and Scaler saved successfully!")
