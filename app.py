import streamlit as st
import joblib
import pandas as pd

st.title("Heart Disease Prediction")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model=load_model()

fields=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
vals={}
for f in fields:
    vals[f]=st.number_input(f,value=0.0)
if st.button("Predict"):
    X=pd.DataFrame([vals])
    pred=model.predict(X)[0]
    st.success("Heart Disease Detected" if pred else "No Heart Disease")
