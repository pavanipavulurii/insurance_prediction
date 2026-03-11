import streamlit as st
from src.prediction import InsurancePrediction

st.title("Insurance Prediction App")
st.write("Predict insurance premium using machine learning")

Age = st.number_input("Enter Age")
Annual_Income_LPA = st.number_input("Enter Annual Income (LPA)")
Policy_Term_Years = st.number_input("Enter Policy Term (Years)")
Sum_Assured_Lakhs = st.number_input("Enter Sum Assured (Lakhs)")

if st.button("Predict"):
    model = InsurancePrediction()
    result = model.prediction(
        Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs
    )

    st.success(f"Predicted Insurance Premium: {result}")