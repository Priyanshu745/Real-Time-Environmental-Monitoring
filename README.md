# Environmental Monitoring and Climate Change Prediction App

This project leverages machine learning models, particularly XGBoost, to predict environmental factors such as PM2.5 concentrations based on various environmental indicators. It uses real-time environmental data to make predictions and visualize the importance of different features.

## Features:
- **Real-time data input**: The app uses real environmental data, including factors such as temperature, humidity, CO, NO2, O3, PM10, wind speed, deforestation, etc.
- **Machine Learning Model**: Trained using XGBoost, a powerful gradient boosting algorithm, to predict PM2.5 levels.
- **Data Visualization**: Visualizes the feature importance to understand how different environmental factors affect PM2.5 concentration.
- **Streamlit App**: Interactive web-based interface that allows users to input environmental data and see predictions instantly.

## Prerequisites

To run this project, you need the following installed on your machine:

- **Python** 3.9 or higher
- **pip** (Python package installer)

### Install Dependencies

1. Clone this repository or download the project files.

2. Navigate to the project folder in the terminal.

3. Create and activate a Python virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Mac/Linux
