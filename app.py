import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from xgboost import plot_importance

st.title("🌍 Real-Time Environmental Monitoring - PM2.5 Prediction")
st.write("This application predicts PM2.5 concentration levels using environmental data. Upload your dataset to see predictions and insights.")
# Load trained model and scaler
model = joblib.load('trained_environmental_model.joblib')
scaler = joblib.load('scaler.joblib')

# Streamlit UI
st.title("🌍 Real-Time Environmental Monitoring - PM2.5 Prediction")
st.write("Upload your environmental dataset to get predictions using the trained XGBoost model.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    # Load data
    data = pd.read_csv(uploaded_file)

    # Check Timestamp and apply feature engineering
    if 'Timestamp' in data.columns:
        data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
        data['Minutes Since Start'] = (data['Timestamp'] - data['Timestamp'].min()).dt.total_seconds() / 60
        data['Hour'] = data['Timestamp'].dt.hour
        data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
        data['Month'] = data['Timestamp'].dt.month

    # Define features
    features = [
        'Temperature (°C)', 'Humidity (%)', 'CO (ppm)', 'NO2 (ppm)', 'O3 (ppm)', 
        'PM10 (µg/m³)', 'Precipitation (mm)', 'Wind Speed (m/s)', 
        'Deforestation Index (%)', 'Minutes Since Start', 'Hour', 'DayOfWeek', 'Month'
    ]

    # Check for missing values
    data.fillna(data.mean(numeric_only=True), inplace=True)

    # Predict
    X = data[features]
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    data['Predicted PM2.5'] = predictions

    # Display results
    st.subheader("📋 Predicted PM2.5 Values")
    st.dataframe(data[['Timestamp', 'Predicted PM2.5']] if 'Timestamp' in data.columns else data[['Predicted PM2.5']])

    # Plot predictions (if ground truth exists)
    if 'PM2.5 (µg/m³)' in data.columns:
        st.subheader("📊 Actual vs Predicted PM2.5")
        fig, ax = plt.subplots()
        ax.plot(data['PM2.5 (µg/m³)'].values, label="Actual", alpha=0.7)
        ax.plot(data['Predicted PM2.5'].values, label="Predicted", alpha=0.7)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.legend()
        st.pyplot(fig)

    # Plot feature importance
    st.subheader("🔍 Feature Importance (from XGBoost)")
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_importance(model, ax=ax)
    st.pyplot(fig)

    # Option to download results
    csv_download = data.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Predictions CSV", csv_download, "predicted_environment.csv", "text/csv")

else:
    st.info("Please upload a valid environmental dataset CSV to begin.")
