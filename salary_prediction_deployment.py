import streamlit as st
import pandas as pd
import joblib

model = joblib.load("salary_prediction_rfr_model.pkl")
encoder = joblib.load("label_encoder_Salary.pkl")

st.title("Salary Prediction Model")

age =st.number_input("Enter your age", 18.65)
gender = st.selectbox("Select Your Gender", encoder["Gender"].classes_)
education = st.selectbox("Select Your Education", encoder["Education Level"].classes_)
job_title = st.selectbox("Select Your Job Title", encoder["Job_Title"].classes_)
experience = st.number_input("Enter Your Experience (in years)", 0.50)

df = pd.DataFrame({
    "Enter your age":[age],
    "Select Your Gender":[gender],
    "Select Your Education":[education],
    "Select Your Job Title":[job_title],
    "Enter Your Experience (in years)":[experience]
})

if st.button("Predict Salary"):
  for col in encoder:
    df[col] = encoder[col].transform(df[col])

    prediction = model.predict(df)
    st.success(f"Predicted Salary : {prediction[0]:}")
