import streamlit as st
import pandas as pd
import joblib

# Load model and features
model = joblib.load('decision_tree_model.pkl')
features = joblib.load('features.pkl')

st.set_page_config(page_title="Bank Term Deposit Prediction", layout="centered")
st.title("💰 Bank Marketing Prediction")
st.markdown("Predict if a customer will **subscribe** to a term deposit.")

# 👉 Input sliders for numeric features
age = st.slider("Age", min_value=18, max_value=100, value=30)
balance = st.slider("Account Balance", min_value=-2000, max_value=100000, value=1000, step=100)
day = st.slider("Day of Contact", min_value=1, max_value=31, value=15)
duration = st.slider("Call Duration (seconds)", min_value=0, max_value=3000, value=150)
campaign = st.slider("Number of Contacts in Campaign", min_value=1, max_value=50, value=1)
pdays = st.slider("Days Passed Since Last Contact", min_value=-1, max_value=999, value=-1)
previous = st.slider("Number of Contacts Before", min_value=0, max_value=20, value=0)

# 👉 Selectboxes for categorical features
job = st.selectbox("Job", ['blue-collar', 'technician', 'admin.', 'management', 'retired'])
marital = st.selectbox("Marital Status", ['single', 'married', 'divorced'])
education = st.selectbox("Education Level", ['primary', 'secondary', 'tertiary'])
default = st.selectbox("Credit in Default?", ['no', 'yes'])
housing = st.selectbox("Has Housing Loan?", ['no', 'yes'])
loan = st.selectbox("Has Personal Loan?", ['no', 'yes'])
contact = st.selectbox("Contact Type", ['cellular', 'telephone'])
month = st.selectbox("Month of Contact", ['may', 'jul', 'aug', 'nov', 'jan'])
poutcome = st.selectbox("Previous Outcome", ['unknown', 'success', 'failure'])

# 👉 Prepare input data
input_data = {
    'age': age,
    'job': job,
    'marital': marital,
    'education': education,
    'default': default,
    'balance': balance,
    'housing': housing,
    'loan': loan,
    'contact': contact,
    'day': day,
    'month': month,
    'duration': duration,
    'campaign': campaign,
    'pdays': pdays,
    'previous': previous,
    'poutcome': poutcome
}

# 👉 Prediction
if st.button("🔍 Predict"):
    input_df = pd.DataFrame([input_data])
    result = model.predict(input_df)[0]

    if result == 1:
        st.success("🎯 The customer is **likely to subscribe** ")
    else:
        st.warning("📉 The customer is **unlikely to subscribe**")
