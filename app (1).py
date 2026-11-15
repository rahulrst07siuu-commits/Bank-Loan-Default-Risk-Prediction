import os
import joblib
import pandas as pd
import streamlit as st

MODEL_FILE = "loan_default_model.pkl"
DATA_FILE = "credit_risk_dataset.csv"

st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="🏦",
    layout="centered"
)

st.title("Bank Loan Default Risk Predictor")
st.caption("Enter borrower and loan details to estimate default risk.")

if not os.path.exists(MODEL_FILE):
    st.error("Model file not found. Add loan_default_model.pkl to the same GitHub repository as app.py.")
    st.stop()

if not os.path.exists(DATA_FILE):
    st.error("Dataset file not found. Add credit_risk_dataset.csv to the same GitHub repository as app.py.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

@st.cache_data
def load_dropdown_data():
    return pd.read_csv(DATA_FILE)

model = load_model()
data = load_dropdown_data()

home_options = sorted(data["person_home_ownership"].dropna().astype(str).unique())
intent_options = sorted(data["loan_intent"].dropna().astype(str).unique())
grade_options = sorted(data["loan_grade"].dropna().astype(str).unique())
default_options = sorted(data["cb_person_default_on_file"].dropna().astype(str).unique())

st.subheader("Borrower and Loan Information")

with st.form("loan_prediction_form"):
    left, right = st.columns(2)

    with left:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        income = st.number_input("Annual income", min_value=0, value=60000, step=1000)
        home = st.selectbox("Home ownership", home_options)
        employment = st.number_input("Employment length (years)", min_value=0.0, max_value=60.0, value=5.0, step=1.0)
        intent = st.selectbox("Loan purpose", intent_options)
        grade = st.selectbox("Loan grade", grade_options)

    with right:
        amount = st.number_input("Loan amount", min_value=0, value=10000, step=500)
        interest = st.number_input("Interest rate (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.25)
        loan_income = st.number_input("Loan-to-income ratio", min_value=0.0, max_value=5.0, value=0.20, step=0.01, format="%.2f")
        previous_default = st.selectbox("Previous default on file", default_options)
        credit_history = st.number_input("Credit history length (years)", min_value=0, max_value=60, value=5, step=1)

    submitted = st.form_submit_button("Predict Loan Default", use_container_width=True)

if submitted:
    input_row = pd.DataFrame([{
        "person_age": age,
        "person_income": income,
        "person_home_ownership": home,
        "person_emp_length": employment,
        "loan_intent": intent,
        "loan_grade": grade,
        "loan_amnt": amount,
        "loan_int_rate": interest,
        "loan_percent_income": loan_income,
        "cb_person_default_on_file": previous_default,
        "cb_person_cred_hist_length": credit_history,
    }])

    predicted_class = int(model.predict(input_row)[0])
    default_probability = float(model.predict_proba(input_row)[0][1])

    st.subheader("Prediction Result")
    st.metric("Estimated default probability", f"{default_probability:.1%}")

    if predicted_class == 1:
        st.error("Higher predicted default risk")
    else:
        st.success("Lower predicted default risk")

    st.caption("Educational estimate only. This is not a credit approval or lending decision.")
