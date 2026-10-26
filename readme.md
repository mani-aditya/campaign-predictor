# 🏦 Term Deposit Subscription Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://campaign-predictor-maniaditya.streamlit.app)

**Live Dashboard:** [https://campaign-predictor-maniaditya.streamlit.app](https://campaign-predictor-maniaditya.streamlit.app)

## 📌 Overview
This project is an end-to-end machine learning pipeline designed to predict the probability of a bank client subscribing to a term deposit based on demographic, financial, and historical marketing campaign data. 

The predictive engine is powered by an **Extreme Gradient Boosting (XGBoost)** classifier. To handle the severe class imbalance inherent in conversion datasets, the architecture utilizes **SMOTE** (Synthetic Minority Over-sampling Technique) inside an isolated pipeline to prevent data leakage. The model's decision boundary was mathematically optimized using a Precision-Recall curve to maximize the F1 score.

## ⚙️ Model Engineering Pipeline
1. **Data Preprocessing:** Utilized `scikit-learn`'s `ColumnTransformer` to apply `StandardScaler` and `RobustScaler` to numerical continuous features, and `OneHotEncoder` to sparse categorical variables.
2. **Imbalance Handling:** Implemented an `imblearn.pipeline.Pipeline` to strictly isolate synthetic oversampling (SMOTE) to the training folds.
3. **Threshold Calibration:** Extracted probability distributions and mapped Precision-Recall curves to lower the default decision threshold from 0.50 to an optimal ~0.258, effectively capturing rare positive conversions and doubling the F1 score from baseline linear models.
4. **Deployment:** Serialized the optimized pipeline and threshold using `joblib` and deployed an interactive inference application via Streamlit.

## 💻 Tech Stack
* **Language:** Python
* **Machine Learning:** `scikit-learn`, `xgboost`, `imbalanced-learn`
* **Data Manipulation:** `pandas`, `numpy`
* **Web Framework:** `streamlit`
* **Serialization:** `joblib`

## 🚀 Running the App Locally

To run this application on your local machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mani-aditya/campaign-predictor.git
   cd campaign-predictor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit server:**
   ```bash
   streamlit run app.py
   ```