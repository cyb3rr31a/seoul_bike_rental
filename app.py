import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Seoul Bike Rental Prediction",
    page_icon="🚴‍♂️",
    layout="wide",
)

st.title("🚴‍♂️Seoul Bike Rental Prediction")
st.write("Want to predict the number of bike rentals in Seoul based on weather and time data? Input your parameters and find out!")

# --- LOAD MODEL ---
with open('model/xgb-model.pkl', 'rb') as file:
    model = pickle.load(file)

with open("model/feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)   

st.header("Input Parameters")
st.write("Adjust the parameters below to see how they affect the bike rental predictions.")

# --- FORM INPUTS ---
with st.form(key='input_form'):
    st.subheader("Date and Time")
    hour = st.number_input("Hour of the Day", min_value=0, max_value=23, value=12)
    date = st.date_input("Date")
    seasons = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])
    holiday = st.selectbox("Holiday", ["No Holiday", "Holiday"])
    functional_day = st.selectbox("Functional Day", ["No", "Yes"])

    # Weather Conditions
    st.subheader("Weather Conditions")
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=40.0, value=20.0, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=50)
    wind_speed = st.number_input("Wind Speed (m/s)", min_value=0, max_value=20, value=5)
    visibility = st.number_input("Visibility (10m)", min_value=0, max_value=20000, value=10000)
    dew_point = st.number_input("Dew Point (°C)", min_value=-10, max_value=30, value=10)
    solar_radiation = st.number_input("Solar Radiation (MJ/m²)", min_value=0, max_value=30, value=15)
    rainfall = st.number_input("Rainfall (mm)", min_value=0, max_value=50, value=0)
    snowfall = st.number_input("Snowfall (cm)", min_value=0, max_value=50, value=0)

    # Form submission button
    submit_button = st.form_submit_button(label="Predict Number of Rentals")

# --- PROCESSS PREDICTION ---
if submit_button:
    input_data = pd.DataFrame({
        'Hour': [hour],
        'Temperature(°C)': [temperature],
        'Humidity(%)': [humidity],
        'Wind speed (m/s)': [wind_speed],
        'Visibility (10m)': [visibility],
        'Dew point temperature(°C)': [dew_point],
        'Solar Radiation (MJ/m2)': [solar_radiation],
        'Rainfall(mm)': [rainfall],
        'Snowfall (cm)': [snowfall],
        'Seasons': [seasons],
        'Holiday': [holiday],
        'Functioning Day': [functional_day]
    })
    
    # 2. Generate the prediction
    # One-hot encode
    input_data = pd.get_dummies(input_data)

    # Match the training columns exactly
    input_data = input_data.reindex(columns=feature_cols, fill_value=0)

    prediction = model.predict(input_data)[0]
    predictions = np.clip(prediction, 0, None)

    # 3. Present the results to the user
    st.success(f"### Calculated Number of Rentals: {predictions:,.0f}")
    