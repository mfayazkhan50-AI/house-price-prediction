# 🏠 House Price Prediction App

This project uses a trained Random Forest model to estimate house prices and wraps everything in a simple Streamlit interface. It’s a small end-to-end demo showing how an ML model can be turned into an interactive web app.

## 🚀 Live Demo

https://smart-house-price-app.streamlit.app/

## 📌 What’s Inside

Predicts house prices based on user-provided inputs

Straightforward Streamlit UI

Uses a pre-trained model (housepriceprediction.pkl)

Supports encoded categorical features

Runs quickly and doesn’t require heavy hardware

## 🧠 Model Info

Model: Random Forest (with tuning)

Accuracy: ~88%

Files:

housepriceprediction.pkl

feature_names.pkl

## 📂 Project Structure
app.py
requirements.txt
housepriceprediction.pkl
feature_names.pkl

## ▶️ Running the Project Locally
pip install -r requirements.txt
streamlit run app.py

## 📜 License

MIT License
