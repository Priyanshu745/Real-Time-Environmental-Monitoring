import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Function to generate realistic synthetic data for environmental factors
def generate_synthetic_data(num_rows=10000):
    # Seed for reproducibility
    np.random.seed(42)

    # Generate timestamps
    start_date = datetime(2025, 1, 1)
    timestamps = [start_date + timedelta(minutes=15*i) for i in range(num_rows)]
    
    # Generate environmental data based on assumed distributions
    # Temperature (°C) - Random data between 15 and 35 °C (realistic for many regions)
    temperature = np.random.normal(loc=25, scale=5, size=num_rows)  # mean=25, std=5
    
    # Humidity (%) - Random data between 30 and 90 % (average humidity range)
    humidity = np.random.normal(loc=60, scale=15, size=num_rows)  # mean=60, std=15clear
    
    
    # Carbon Monoxide (CO) - Concentration in ppm, realistic range from 0 to 5 ppm
    CO = np.random.uniform(0, 5, num_rows)  # Uniform between 0 and 5 ppm
    
    # Nitrogen Dioxide (NO2) - Concentration in ppm, realistic range from 0 to 0.1 ppm
    NO2 = np.random.uniform(0, 0.1, num_rows)  # Uniform between 0 and 0.1 ppm
    
    # Ozone (O3) - Concentration in ppm, realistic range from 0 to 0.15 ppm
    O3 = np.random.uniform(0, 0.15, num_rows)  # Uniform between 0 and 0.15 ppm
    
    # PM2.5 (µg/m³) - Realistic range from 0 to 300 µg/m³
    PM2_5 = np.random.normal(loc=35, scale=30, size=num_rows)  # mean=35, std=30
    
    # PM10 (µg/m³) - Realistic range from 0 to 500 µg/m³
    PM10 = PM2_5 + np.random.normal(loc=20, scale=50, size=num_rows)  # PM10 > PM2.5
    
    # Precipitation (mm) - Random data between 0 and 20 mm (for rain events)
    precipitation = np.random.uniform(0, 20, num_rows)  # Uniform between 0 and 20 mm
    
    # Wind Speed (m/s) - Realistic range from 0 to 15 m/s
    wind_speed = np.random.uniform(0, 15, num_rows)  # Uniform between 0 and 15 m/s
    
    # Deforestation Index - Simulated as a percentage change in forest cover (0 to 100%)
    deforestation = np.random.uniform(0, 0.1, num_rows)  # Simulated as 0 to 0.1% change per period
    
    # Create the DataFrame
    data = pd.DataFrame({
        'Timestamp': timestamps,
        'Temperature (°C)': temperature,
        'Humidity (%)': humidity,
        'CO (ppm)': CO,
        'NO2 (ppm)': NO2,
        'O3 (ppm)': O3,
        'PM2.5 (µg/m³)': PM2_5,
        'PM10 (µg/m³)': PM10,
        'Precipitation (mm)': precipitation,
        'Wind Speed (m/s)': wind_speed,
        'Deforestation Index (%)': deforestation
    })
    
    # Save to CSV
    data.to_csv('environmental.csv', index=False)
    print("Dataset generated and saved as 'synthetic_environmental_data.csv'")

# Generate 10,000 rows of data
generate_synthetic_data(num_rows=1000000)
