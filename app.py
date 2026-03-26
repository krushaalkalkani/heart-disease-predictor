import streamlit as st
import numpy as np
import joblib

# import pandas as pd
# df = pd.read_csv('heart_attack_prediction_dataset.csv')
# print(df[df['Heart Attack Risk'] == 1].head(1).to_string())
# print(df[df['Heart Attack Risk'] == 0].head(1).to_string())

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('💔 Heart Disease Predictor')
st.write('Enter patient data to predict heart attack risk')

# first four input
st.subheader('👨🏼‍⚕️ Patient Information')
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age: ", 18, 100, 50)
    heart_rate = st.number_input("Resting Heart Rate (bpm): ", 0, 400, 70)
    blood_pressure_systolic = st.number_input(
        "Systolic Resting Blood Pressure (mm Hg): ", 0, 300, 120)
    bmi = st.number_input("Body Mass Index (BMI): ", 0.0, 100.0, 25.0)
with col2:
    sex = st.selectbox("Sex: ", ["Male", "Female"])
    cholesterol = st.number_input("Cholesterol (mg/dl): ", 0, 400, 200)
    blood_pressure_diastolic = st.number_input(
        "Diastolic Resting Blood Pressure (mm Hg): ", 0, 200, 80)
    triglycerides = st.number_input("Triglycerides (mg/dl): ", 0, 1000, 150)

st.subheader('🏥 Medical History')
col3, col4 = st.columns(2)

with col3:
    diabetes = st.selectbox("Diabetes: ", ["No", "Yes"])
    family_history = st.selectbox(
        "Family History of Heart Disease: ", ["No", "Yes"])
    medication = st.selectbox("Medication: ", ["No", "Yes"])

with col4:
    smoking = st.selectbox("Smoking: ", ["No", "Yes"])
    alcohol_consumption = st.selectbox("Alcohol Consumption: ", ["No", "Yes"])
    obesity = st.selectbox("Obesity: ", ["No", "Yes"])
    previous_heart_problems = st.selectbox(
        "Previous Heart Problems: ", ["No", "Yes"])

st.subheader('🏃🏼‍♂️ Lifestyle Information')
col5, col6 = st.columns(2)

with col5:
    physical_activity = st.number_input(
        "Physical Activity Day per Week: ", 0, 7, 5)
    sleep_duration = st.number_input("Sleep Duration (hours): ", 0, 24, 7)
    exercise_hours_per_week = st.number_input(
        "Exercise Hours Per Week: ", 0.0, 40.0, 3.0)

with col6:
    stress_level = st.slider("Stress Level:", 1, 10, 5)
    diet = st.selectbox("Diet Quality: ", ["Healthy", "Unhealthy", "Average"])
    sedentary_hours_per_day = st.number_input(
        "Sedentary Hours Per Day: ", 0.0, 24.0, 8.0)
    income = st.number_input("Income: ", 0.0, 100000000.0, 50000.0)

st.divider()
# button

if st.button("Predict Heart Attack Risk"):
    diabetes_encoded = int(diabetes == "Yes")
    sex_encoded = int(sex == "Male")
    family_history_encoded = int(family_history == "Yes")
    medication_encoded = int(medication == "Yes")
    smoking_encoded = int(smoking == "Yes")
    alcohol_consumption_encoded = int(alcohol_consumption == "Yes")
    obesity_encoded = int(obesity == "Yes")
    previous_heart_problems_encoded = int(previous_heart_problems == "Yes")

    if diet == "Healthy":
        diet_healthy = 1
        diet_unhealthy = 0
    elif diet == "Unhealthy":
        diet_unhealthy = 1
        diet_healthy = 0
    else:
        diet_unhealthy = 0
        diet_healthy = 0

    input_array = np.array([[age, sex_encoded, cholesterol, heart_rate,
                             diabetes_encoded, family_history_encoded, smoking_encoded, obesity_encoded, alcohol_consumption_encoded, exercise_hours_per_week, previous_heart_problems_encoded, medication_encoded, stress_level, sedentary_hours_per_day, income, bmi, triglycerides, physical_activity, sleep_duration, blood_pressure_systolic, blood_pressure_diastolic, diet_healthy, diet_unhealthy]])

    scale_input = scaler.transform(input_array)
    prediction = model.predict(scale_input)

    if prediction[0] == 1:
        st.warning("High Risk ⚠️")
        # st.balloons()
        # st.metric(label="Heart Attack Risk", value="High Risk", delta="⚠️")
        st.toast("Prediction complete!")
    else:
        st.success("Low Risk ✅")
        # st.balloons()
        # st.metric(label="Heart Attack Risk", value="High Risk", delta="⚠️")
        st.toast("Prediction complete!")
