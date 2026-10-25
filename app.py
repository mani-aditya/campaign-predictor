# Q5
import streamlit as st
import pandas as pd
import joblib

pipeline = joblib.load('xgb_model_pipeline.pkl')
threshold = joblib.load('optimal_threshold.pkl')

st.title("Campaign Conversion Predictor")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
job = st.selectbox("Job", ["admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student", "blue-collar", "self-employed", "retired", "technician", "services"])
marital = st.selectbox("Marital Status", ["married", "divorced", "single"])
education = st.selectbox("Education", ["unknown", "secondary", "primary", "tertiary"])
default = st.selectbox("Has Credit in Default?", ["yes", "no"])
balance = st.number_input("Average Yearly Balance", value=0)
housing = st.selectbox("Has Housing Loan?", ["yes", "no"])
loan = st.selectbox("Has Personal Loan?", ["yes", "no"])
contact = st.selectbox("Contact Communication Type", ["unknown", "telephone", "cellular"])
day = st.number_input("Last Contact Day of Month", min_value=1, max_value=31, value=15)
month = st.selectbox("Last Contact Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
duration = st.number_input("Last Contact Duration (seconds)", value=0)
campaign = st.number_input("Number of Contacts Performed During This Campaign", min_value=1, value=1)
pdays = st.number_input("Number of Days Passed After Last Contact", value=-1)
previous = st.number_input("Number of Contacts Performed Before This Campaign", value=0)
poutcome = st.selectbox("Outcome of Previous Campaign", ["unknown", "other", "failure", "success"])

if st.button("Predict Conversion"):
    input_data = pd.DataFrame([[age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome]], 
                              columns=['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome'])
    
    probability = pipeline.predict_proba(input_data)[0, 1]
    prediction = 1 if probability >= threshold else 0
    
    st.write(f"Conversion Probability: {probability:.4f}")
    st.write(f"Prediction: {'Success' if prediction == 1 else 'Failure'} (Threshold: {threshold:.4f})")