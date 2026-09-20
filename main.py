import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, classification_report

# 1. Load Dataset
data_path = 'Student_performance_data _.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError("CSV file 'Student_performance_data _.csv' not found.")

df = pd.read_csv(data_path)
print(f"Dataset Loaded Successfully: {df.shape[0]} rows, {df.shape[1]} columns.")

# 2. Data Preprocessing
X = df.drop(columns=['StudentID', 'GPA', 'GradeClass'])
y_gpa = df['GPA']
y_class = df['GradeClass']

# Split Data
X_train, X_test, y_train_gpa, y_test_gpa = train_test_split(X, y_gpa, test_size=0.2, random_state=42)
_, _, y_train_cls, y_test_cls = train_test_split(X, y_class, test_size=0.2, random_state=42)

# 3. Model Building & Training (GPA Regression)
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

print("\n" + "="*50)
print("REGRESSION MODEL EVALUATION RESULTS (GPA Prediction)")
print("="*50)

results = []
for name, model in models.items():
    model.fit(X_train, y_train_gpa)
    preds = model.predict(X_test)
    r2 = r2_score(y_test_gpa, preds)
    rmse = np.sqrt(mean_squared_error(y_test_gpa, preds))
    mae = mean_absolute_error(y_test_gpa, preds)
    results.append({'Model': name, 'R2 Score': r2, 'RMSE': rmse, 'MAE': mae})
    print(f"{name:<20} | R2: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

# 4. Save Best Model (Linear Regression coefficients for JavaScript Web GUI)
best_model = LinearRegression()
best_model.fit(X, y_gpa)

model_payload = {
    'intercept': float(best_model.intercept_),
    'coefficients': dict(zip(X.columns, best_model.coef_))
}

print("\n" + "="*50)
print("MODEL COEFFICIENTS FOR WEB GUI:")
print("="*50)
print(f"Intercept: {model_payload['intercept']:.4f}")
for feature, coef in model_payload['coefficients'].items():
    print(f"  - {feature:<20}: {coef:+.4f}")

# Save pickle file
with open('student_gpa_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("\nSaved trained model as 'student_gpa_model.pkl'.")