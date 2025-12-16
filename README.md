# 🍷 Wine Quality Prediction System

A Machine Learning web application that predicts the quality of red wine based on physicochemical tests. Built with Python, Scikit-Learn, and Streamlit.

## 📌 Project Overview
The goal of this project is to predict the quality of wine on a scale of 1-10 (mapped to 3-8 in the dataset) given a set of features such as fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, etc.

It helps to understand how different chemical compositions affect the perceived quality of wine.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Random Forest, SVM)
* **Web Framework:** Streamlit
* **Model Persistence:** Joblib

## 📂 Project Structure
* `QualityPrediction.csv` - The dataset file.
* `train_model.py` - Script to clean data, train models, compare accuracy, and save the best model.
* `app.py` - The Streamlit web application for user interaction.
* `wine_quality_model.pkl` - The saved trained model.
* `scaler.pkl` - The saved standard scaler for data normalization.

## 🚀 How to Run

### 1. Install Dependencies
Open your terminal and install the required libraries:
```bash
pip install pandas numpy scikit-learn streamlit joblib matplotlib seaborn