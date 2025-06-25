import streamlit as st
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

st.title("🔍 Financial Inclusion - Bank Account Predictor")

# Form inputs
country = st.selectbox("Country", ["Kenya", "Rwanda", "Tanzania", "Uganda"])
year = st.selectbox("Year", [2016, 2017, 2018])
location_type = st.selectbox("Location Type", ["Urban", "Rural"])
cellphone_access = st.selectbox("Cellphone Access", ["Yes", "No"])
household_size = st.slider("Household Size", 1, 20, 3)
age_of_respondent = st.slider("Age", 18, 100, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
relationship_with_head = st.selectbox("Relationship with Head", [
    "Head of Household", "Spouse", "Child", "Parent", "Other relative", "Other non-relatives"
])
marital_status = st.selectbox("Marital Status", [
    "Married/Living together", "Single/Never Married", "Widowed", "Divorced/Seperated", "Don't know"
])
education_level = st.selectbox("Education Level", [
    "No formal education", "Primary education", "Secondary education",
    "Tertiary education", "Vocational/Specialised training"
])
job_type = st.selectbox("Job Type", [
    "Self employed", "Formally employed Private", "Formally employed Government",
    "Informally employed", "Farming and Fishing", "Remittance Dependent",
    "Government Dependent", "Other Income", "No Income", "Don't Know/Refuse to answer"
])

# Handle prediction
if st.button("Predict"):
    input_dict = {
        'year': year,
        'household_size': household_size,
        'age_of_respondent': age_of_respondent,
        'country_' + country: 1,
        'location_type_' + location_type: 1,
        'cellphone_access_Yes': 1 if cellphone_access == 'Yes' else 0,
        'gender_of_respondent_' + gender: 1,
        'relationship_with_head_' + relationship_with_head: 1,
        'marital_status_' + marital_status: 1,
        'education_level_' + education_level: 1,
        'job_type_' + job_type: 1
    }

    # Create full dummy column template based on training data
    all_possible_cols = model.feature_names_in_
    input_data = pd.DataFrame(columns=all_possible_cols)
    input_data.loc[0] = 0  # initialize all values to zero
    for col in input_dict:
        if col in input_data.columns:
            input_data.at[0, col] = input_dict[col]
        else:
            pass  # some categories may not exist in training set and should be skipped

    prediction = model.predict(input_data)[0]
    result = "✅ Has Bank Account" if prediction == 1 else "❌ No Bank Account"
    st.success(f"Prediction: {result}")
