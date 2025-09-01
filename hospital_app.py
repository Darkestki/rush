import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained Linear Regression model
try:
    with open('lr.sav', 'rb') as f:
        lr = pickle.load(f)
except FileNotFoundError:
    st.error("Model file 'lr.sav' not found. Please ensure the model is saved in the correct directory.")
    st.stop()

# Load the hospital data for preprocessing consistency
try:
    df = pd.read_csv('/content/hospital.csv')
except FileNotFoundError:
    st.error("Data file 'hospital.csv' not found. Please ensure the data is in the correct directory.")
    st.stop()

# Preprocessing function (consistent with training)
def preprocess_input(input_df):
    # Impute missing numerical values with the mean from the training data
    for col in ['BP_HIGH', 'BP_LOW', 'HB', 'UREA', 'CREATININE']:
         # Calculate mean from the original data for consistency
        mean_val = df[col].mean()
        input_df[col].fillna(mean_val, inplace=True)


    # Impute missing categorical values with the mode from the training data
    for col in ['KEY_COMPLAINTS_CODE', 'PAST_MEDICAL_HISTORY_CODE', 'MODE_OF_ARRIVAL', 'STATE_AT_THE_TIME_OF_ARRIVAL', 'TYPE_OF_ADMSN', 'IMPLANT_USED_', 'GENDER', 'MARITAL_STATUS']:
        # Calculate mode from the original data for consistency
        mode_val = df[col].mode()[0]
        input_df[col].fillna(mode_val, inplace=True)


    # One-hot encode categorical features
    categorical_cols = ['GENDER', 'MARITAL_STATUS', 'KEY_COMPLAINTS_CODE', 'PAST_MEDICAL_HISTORY_CODE', 'MODE_OF_ARRIVAL', 'STATE_AT_THE_TIME_OF_ARRIVAL', 'TYPE_OF_ADMSN', 'IMPLANT_USED_']
    input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

    # Ensure all columns from training are present, fill missing with 0
    # This is crucial for consistent feature order and presence
    trained_cols = ['AGE',
                    'BODY_WEIGHT',
                    'BODY_HEIGHT',
                    'HR_PULSE',
                    'BP_HIGH',
                    'BP_LOW',
                    'RR',
                    'HB',
                    'UREA',
                    'CREATININE',
                    'TOTAL_AMOUNT_BILLED_TO_THE_PATIENT',
                    'CONCESSION',
                    'TOTAL_LENGTH_OF_STAY',
                    'LENGTH_OF_STAY_ICU',
                    'LENGTH_OF_STAY_WARD',
                    'COST_OF_IMPLANT',
                    'GENDER_M',
                    'MARITAL_STATUS_UNMARRIED',
                    'TYPE_OF_ADMSN_EMERGENCY',
                    'IMPLANT_USED__Y'] # Ensure this matches the selected features in training

    for col in trained_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match the training data
    input_df = input_df[trained_cols]

    return input_df

# Prediction function
def predict_cost(model, preprocessed_data):
    prediction = model.predict(preprocessed_data)
    return prediction[0]

# Streamlit App
st.title('Hospital Cost Prediction')

st.sidebar.header('Input Features')

# Define input fields for each feature used in the model
# AGE, BODY_WEIGHT, BODY_HEIGHT, HR_PULSE, BP_HIGH, BP_LOW, RR, HB, UREA, CREATININE,
# TOTAL_AMOUNT_BILLED_TO_THE_PATIENT, CONCESSION, TOTAL_LENGTH_OF_STAY,
# LENGTH_OF_STAY_ICU, LENGTH_OF_STAY_WARD, COST_OF_IMPLANT
# GENDER_M (True/False), MARITAL_STATUS_UNMARRIED (True/False),
# TYPE_OF_ADMSN_EMERGENCY (True/False), IMPLANT_USED__Y (True/False)

age = st.sidebar.number_input('AGE', min_value=0.0, max_value=100.0, value=30.0)
body_weight = st.sidebar.number_input('BODY_WEIGHT', min_value=0.0, max_value=200.0, value=70.0)
body_height = st.sidebar.number_input('BODY_HEIGHT', min_value=0.0, max_value=300.0, value=170.0)
hr_pulse = st.sidebar.number_input('HR_PULSE', min_value=0.0, max_value=200.0, value=80.0)
bp_high = st.sidebar.number_input('BP_HIGH', min_value=0.0, max_value=300.0, value=120.0)
bp_low = st.sidebar.number_input('BP_LOW', min_value=0.0, max_value=200.0, value=80.0)
rr = st.sidebar.number_input('RR', min_value=0.0, max_value=100.0, value=20.0)
hb = st.sidebar.number_input('HB', min_value=0.0, max_value=30.0, value=14.0)
urea = st.sidebar.number_input('UREA', min_value=0.0, max_value=200.0, value=20.0)
creatinine = st.sidebar.number_input('CREATININE', min_value=0.0, max_value=10.0, value=1.0)
total_amount_billed = st.sidebar.number_input('TOTAL_AMOUNT_BILLED_TO_THE_PATIENT', min_value=0.0, value=300000.0)
concession = st.sidebar.number_input('CONCESSION', min_value=0.0, value=10000.0)
total_length_of_stay = st.sidebar.number_input('TOTAL_LENGTH_OF_STAY', min_value=0.0, value=10.0)
length_of_stay_icu = st.sidebar.number_input('LENGTH_OF_STAY_ICU', min_value=0.0, value=5.0)
length_of_stay_ward = st.sidebar.number_input('LENGTH_OF_STAY_WARD', min_value=0.0, value=5.0)
cost_of_implant = st.sidebar.number_input('COST_OF_IMPLANT', min_value=0.0, value=50000.0)
gender = st.sidebar.selectbox('GENDER', ['M', 'F'])
marital_status = st.sidebar.selectbox('MARITAL_STATUS', ['MARRIED', 'UNMARRIED'])
type_of_admsn = st.sidebar.selectbox('TYPE_OF_ADMSN', ['EMERGENCY', 'ELECTIVE'])
implant_used = st.sidebar.selectbox('IMPLANT_USED_', ['Y', 'N'])

# Convert boolean inputs to string representation for consistent preprocessing
gender_str = 'M' if gender == 'M' else 'F'
marital_status_str = 'UNMARRIED' if marital_status == 'UNMARRIED' else 'MARRIED'
type_of_admsn_str = 'EMERGENCY' if type_of_admsn == 'EMERGENCY' else 'ELECTIVE'
implant_used_str = 'Y' if implant_used == 'Y' else 'N'


# Create a dictionary with the input values
input_data = {
    'AGE': age,
    'BODY_WEIGHT': body_weight,
    'BODY_HEIGHT': body_height,
    'HR_PULSE': hr_pulse,
    'BP_HIGH': bp_high,
    'BP_LOW': bp_low,
    'RR': rr,
    'HB': hb,
    'UREA': urea,
    'CREATININE': creatinine,
    'TOTAL_AMOUNT_BILLED_TO_THE_PATIENT': total_amount_billed,
    'CONCESSION': concession,
    'TOTAL_LENGTH_OF_STAY': total_length_of_stay,
    'LENGTH_OF_STAY_ICU': length_of_stay_icu,
    'LENGTH_OF_STAY_WARD': length_of_stay_ward,
    'COST_OF_IMPLANT': cost_of_implant,
    'GENDER': gender_str,
    'MARITAL_STATUS': marital_status_str,
    'TYPE_OF_ADMSN': type_of_admsn_str,
    'IMPLANT_USED_': implant_used_str,
    'KEY_COMPLAINTS_CODE': df['KEY_COMPLAINTS_CODE'].mode()[0], # Use mode for simplicity in app
    'PAST_MEDICAL_HISTORY_CODE': df['PAST_MEDICAL_HISTORY_CODE'].mode()[0], # Use mode for simplicity in app
    'MODE_OF_ARRIVAL': df['MODE_OF_ARRIVAL'].mode()[0], # Use mode for simplicity in app
    'STATE_AT_THE_TIME_OF_ARRIVAL': df['STATE_AT_THE_TIME_OF_ARRIVAL'].mode()[0] # Use mode for simplicity in app
}


# Create a DataFrame from the input data
input_df = pd.DataFrame([input_data])


# Prediction button
if st.sidebar.button('Predict Cost'):
    # Preprocess the input data
    preprocessed_input = preprocess_input(input_df.copy())

    # Make prediction
    predicted_cost = predict_cost(lr, preprocessed_input)

    # Display the prediction
    st.subheader('Predicted Total Cost to Hospital')
    st.write(f'${predicted_cost:,.2f}')
