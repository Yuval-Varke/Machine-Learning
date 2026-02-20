import streamlit as st
import pickle
import pandas as pd

# Load the trained XGBoost model
try:
    with open('xgb_tuned_model.pkl', 'rb') as f:
        xgb_tuned = pickle.load(f)
    st.success("XGBoost model loaded successfully!")
except FileNotFoundError:
    st.error("Error: 'xgb_tuned_model.pkl' not found. Please ensure the model file is in the same directory.")
    st.stop()

# Load the label encoders
try:
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    st.success("Label encoders loaded successfully!")
except FileNotFoundError:
    st.error("Error: 'label_encoders.pkl' not found. Please ensure the encoder file is in the same directory.")
    st.stop()

st.title("Exam Score Prediction App")
st.write("Enter the student's details to predict their exam score.")

# Input features (based on the X_train used in model training)
# Numerical features
study_hours = st.slider("Study Hours", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
class_attendance = st.slider("Class Attendance (%)", min_value=0.0, max_value=100.0, value=75.0, step=0.1)
sleep_hours = st.slider("Sleep Hours", min_value=0.0, max_value=12.0, value=7.0, step=0.1)

# Categorical features
sleep_quality_options = list(label_encoders['sleep_quality'].classes_)
selected_sleep_quality = st.selectbox("Sleep Quality", options=sleep_quality_options)

study_method_options = list(label_encoders['study_method'].classes_)
selected_study_method = st.selectbox("Study Method", options=study_method_options)

facility_rating_options = list(label_encoders['facility_rating'].classes_)
selected_facility_rating = st.selectbox("Facility Rating", options=facility_rating_options)

if st.button("Predict Exam Score"):
    # Preprocess inputs
    encoded_sleep_quality = label_encoders['sleep_quality'].transform([selected_sleep_quality])[0]
    encoded_study_method = label_encoders['study_method'].transform([selected_study_method])[0]
    encoded_facility_rating = label_encoders['facility_rating'].transform([selected_facility_rating])[0]

    # Create a DataFrame for prediction
    input_data = pd.DataFrame([{
        'study_hours': study_hours,
        'class_attendance': class_attendance,
        'sleep_hours': sleep_hours,
        'sleep_quality': encoded_sleep_quality,
        'study_method': encoded_study_method,
        'facility_rating': encoded_facility_rating
    }])

    # Make prediction
    prediction = xgb_tuned.predict(input_data)[0]

    st.success(f"Predicted Exam Score: {prediction:.2f}")

st.write("\n--- Notes ---")
