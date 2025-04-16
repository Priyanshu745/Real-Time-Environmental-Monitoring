import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import joblib

# Load the dataset
data = pd.read_csv('environmental_data.csv')

# Convert 'Timestamp' to datetime if present
if 'Timestamp' in data.columns:
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')

    # Feature Engineering: extract useful time features
    data['Minutes Since Start'] = (data['Timestamp'] - data['Timestamp'].min()).dt.total_seconds() / 60
    data['Hour'] = data['Timestamp'].dt.hour
    data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
    data['Month'] = data['Timestamp'].dt.month

# Handle missing values
print("Missing Values in the dataset:")
print(data.isnull().sum())
data.fillna(data.mean(numeric_only=True), inplace=True)

# Define features
features = [
    'Temperature (°C)', 'Humidity (%)', 'CO (ppm)', 'NO2 (ppm)', 'O3 (ppm)', 
    'PM10 (µg/m³)', 'Precipitation (mm)', 'Wind Speed (m/s)', 
    'Deforestation Index (%)', 'Minutes Since Start', 'Hour', 'DayOfWeek', 'Month'
]

# Define target
target = 'PM2.5 (µg/m³)'

# Split data
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize XGBoost model
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train_scaled, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predict
y_pred = best_model.predict(X_test_scaled)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Save model and scaler
joblib.dump(best_model, 'trained_environmental_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Tuned XGBoost model and scaler have been saved successfully.")
