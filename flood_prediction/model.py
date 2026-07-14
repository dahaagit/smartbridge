import pandas as pd
import numpy as np
import kagglehub
import os
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Downloading dataset...")
path = kagglehub.dataset_download("s3programmer/flood-risk-in-india")
csv_file = None
for f in os.listdir(path):
    if f.endswith('.csv'):
        csv_file = os.path.join(path, f)
        break

print("Reading dataset...")
df = pd.read_csv(csv_file)

# 1. Clean column names
print("Original Columns:", df.columns.tolist())
df.columns = [
    'latitude', 'longitude', 'rainfall', 'temperature', 'humidity', 
    'river_discharge', 'water_level', 'elevation', 'land_cover', 
    'soil_type', 'population_density', 'infrastructure', 
    'historical_floods', 'flood_occurred'
]
print("Cleaned Columns:", df.columns.tolist())

# 2. Pre-processing
# Handle missing values (if any)
df.fillna(df.mean(numeric_only=True), inplace=True)
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Encode categorical values
categorical_cols = ['land_cover', 'soil_type', 'infrastructure']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop('flood_occurred', axis=1)
y = df['flood_occurred']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Building
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

best_acc = 0
best_model = None
best_model_name = ""

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    
    # We enforce saving XGBoost as per requirements, but let's track the actual best too
    if name == 'XGBoost':
        xgboost_model = model

print("\n--- Final Selection ---")
print("Selecting XGBoost as the final model as per project requirements.")
best_model = xgboost_model

# 4. Save Model and Scaler/Encoders
print("Saving model, scaler, and encoders to floods.save...")
artifacts = {
    'model': best_model,
    'scaler': scaler,
    'encoders': label_encoders,
    'features': X.columns.tolist()
}
joblib.dump(artifacts, 'floods.save')
print("Model saved successfully as floods.save.")
