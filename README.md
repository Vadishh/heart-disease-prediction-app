# Heart Disease Prediction App

A machine learning web application that predicts whether a person is likely to have heart disease based on medical input features.

The app is built using **Python**, **Scikit-learn**, and **Streamlit**.

---

## Project Overview

This project uses a trained Machine Learning classification model to predict heart disease presence.

The user enters patient details such as age, chest pain type, cholesterol level, blood pressure, maximum heart rate, exercise angina, and ST slope. The app then returns a prediction:

- **0**: No Heart Disease
- **1**: Heart Disease Detected

---

## Features

- Interactive web UI built with Streamlit
- Takes medical input features from the user
- Uses a trained K-Nearest Neighbors model
- Shows prediction result clearly
- Displays model confidence for the heart disease class
- Uses saved scaler and column structure for correct preprocessing

---

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

---

## Files in This Repository

```text
app.py              # Streamlit app code
KNN_heart.pkl       # Trained machine learning model
scaler.pkl          # Saved scaler used during preprocessing
columns.pkl         # Saved column order after encoding
requirements.txt    # Required Python libraries
