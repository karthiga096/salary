import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ----------------------------
# App Title
# ----------------------------
st.set_page_config(page_title="Salary Prediction App", layout="centered")
st.title("💼 Salary Prediction App")
st.write("Predict salary based on Years of Experience using Linear Regression")

# ----------------------------
# Upload CSV (Optional)
# ----------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Salary Data CSV (Optional)",
    type=["csv"]
)

# ----------------------------
# Load Data
# ----------------------------
if uploaded_file is not None:
    saldf = pd.read_csv(uploaded_file)
else:
    # Default sample dataset
    saldf = pd.DataFrame({
        "Years of Experience": [0.5, 1, 2, 3, 4, 5, 6],
        "Salary": [25000, 30000, 40000, 50000, 60000, 75000, 90000]
    })

saldf = saldf.dropna()

# ----------------------------
# Dataset Preview
# ----------------------------
st.subheader("📊 Dataset Preview")
st.dataframe(saldf)
