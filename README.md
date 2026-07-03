# 🚴‍♂️ Seoul Bike Rental Prediction

Predicts the number of bike rentals in Seoul for a given hour, based on time-of-day, date, and weather conditions. The project includes an end-to-end modeling notebook and an interactive [Streamlit](https://streamlit.io/) app for making predictions.

## Overview

This project uses the [Seoul Bike Sharing Demand dataset](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand) (UCI Machine Learning Repository) to train a regression model that predicts hourly bike rental counts from weather and calendar features. A trained model is served through a simple web app where you can adjust inputs — temperature, humidity, hour, season, etc. — and get a live prediction.

## Repository Structure

| File | Description |
|---|---|
| `seoul_bikes.ipynb` | Data exploration, feature engineering, and model training/evaluation |
| `app.py` | Streamlit web app that loads the trained model and serves predictions |
| `.gitignore` | Excludes generated `.pkl` (model) and `.csv` (data) files from version control |

> **Note:** The dataset (`.csv`) and trained model artifacts (`.pkl`) are not committed to this repo (see `.gitignore`). You'll need to download the dataset and run the notebook to generate them before the app will run — see [Setup](#setup) below.

## Features Used

The model predicts rental count from the following inputs:

- **Time**: Hour of day, Date, Season, Holiday, Functioning Day
- **Weather**: Temperature (°C), Humidity (%), Wind Speed (m/s), Visibility (10m), Dew Point Temperature (°C), Solar Radiation (MJ/m²), Rainfall (mm), Snowfall (cm)

Categorical features (Season, Holiday, Functioning Day) are one-hot encoded before being passed to the model.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/cyb3rr31a/seoul_bike_rental.git
cd seoul_bike_rental
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn xgboost streamlit jupyter
```

### 3. Get the dataset

Download the [Seoul Bike Sharing Demand dataset](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand) from the UCI Machine Learning Repository and place the CSV file in the project directory (see the notebook for the expected filename/path).

### 4. Train the model

Run through `seoul_bikes.ipynb` to explore the data and train the model. The notebook should export:

- `model/xgb-model.pkl` — the trained model
- `model/feature_cols.pkl` — the feature column order used at training time (required so the app's one-hot-encoded inputs line up correctly with the model)

### 5. Run the app

```bash
streamlit run app.py
```

This opens the app in your browser, where you can input weather and time parameters and get a predicted rental count.

## How It Works

1. The user fills out a form with date/time and weather inputs.
2. `app.py` one-hot encodes the categorical inputs and reindexes the resulting DataFrame to match the exact feature columns the model was trained on.
3. The trained model generates a prediction, which is clipped at zero (rental counts can't be negative) and displayed to the user.

## Tech Stack

- **Python** — pandas, numpy
- **Modeling** — scikit-learn / XGBoost
- **App** — Streamlit
- **Notebook** — Jupyter

## Acknowledgments

- Dataset: Sathishkumar V E, Jangwoo Park, Yongyun Cho, *"Using data mining techniques for bike sharing demand prediction in metropolitan city,"* Computer Communications, Vol.153, pp.353-366, March 2020, via the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand).

## License

MIT License