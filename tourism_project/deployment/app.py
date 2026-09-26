import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism Package Purchase Prediction")
st.write("""
This application predicts whether a customer will purchase the Wellness Tourism Package
based on their demographic and interaction data. Please fill in the details below.
""")

# User input fields based on the tourism dataset features
st.header("Customer Information")
age = st.slider("Age", min_value=18, max_value=90, value=30)
type_of_contact = st.selectbox("Type of Contact", ['Company Invited', 'Self Inquiry'])
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer', 'Government Sector'])
gender = st.selectbox("Gender", ['Male', 'Female', 'Fe Male']) # 'Fe Male' is handled in prep.py, but kept here for user input flexibility
num_person_visiting = st.number_input("Number of Persons Visiting", min_value=0, max_value=10, value=2)
preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
marital_status = st.selectbox("Marital Status", ['Married', 'Single', 'Divorced'])
num_trips = st.number_input("Number of Trips Annually", min_value=0, max_value=20, value=5)
passport = st.checkbox("Has Passport?")
own_car = st.checkbox("Owns a Car?")
num_children_visiting = st.number_input("Number of Children Visiting (below 5)", min_value=0, max_value=5, value=0)
designation = st.selectbox("Designation", ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP', 'Director'])
monthly_income = st.number_input("Monthly Income", min_value=0.0, max_value=500000.0, value=50000.0, step=1000.0)

st.header("Customer Interaction Data")
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
product_pitched = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])
num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=5, max_value=60, value=20)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': type_of_contact,
    'CityTier': city_tier,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': num_person_visiting,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': num_trips,
    'Passport': 1 if passport else 0,
    'OwnCar': 1 if own_car else 0,
    'NumberOfChildrenVisiting': num_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'ProductPitched': product_pitched,
    'NumberOfFollowups': num_followups,
    'DurationOfPitch': duration_of_pitch
}])

# Predict button
if st.button("Predict Purchase"):  
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"Customer is likely to purchase the package! (Probability: {prediction_proba[1]:.2f})")
    else:
        st.warning(f"Customer is unlikely to purchase the package. (Probability: {prediction_proba[0]:.2f})")

    st.write("--- Debug Information ---")
    st.write("Input Data (pre-processed by model pipeline):")
    st.dataframe(model.named_steps['columntransformer'].transform(input_data)) # Display preprocessed input
