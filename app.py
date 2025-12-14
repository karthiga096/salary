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
# Upload CSV
# ----------------------------
uploaded_file = st.file_uploader("📂 Upload Salary Data CSV", type=["csv"])

if uploaded_file is None:
    st.info("Please upload the Salary Data CSV file to continue.")
    st.stop()

# ----------------------------
# Load & Clean Data
# ----------------------------
saldf = pd.read_csv(uploaded_file)
saldf = saldf.dropna()

# ----------------------------
# Dataset Preview
# ----------------------------
st.subheader("📊 Dataset Preview")
st.dataframe(saldf.head())

# ----------------------------
# Train Model
# ----------------------------
X = saldf[['Years of Experience']]
y = saldf['Salary']

model = LinearRegression()
model.fit(X, y)

# ----------------------------
# User Input
# ----------------------------
st.subheader("🧮 Enter Years of Experience")
experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=50.0,
    step=0.5
)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Salary"):
    input_df = pd.DataFrame([[experience]], columns=['Years of Experience'])
    prediction = model.predict(input_df)
    st.success(f"💰 Predicted Salary: ₹ {prediction[0]:,.2f}")

# ----------------------------
# Visualization
# ----------------------------
st.subheader("📈 Experience vs Salary")

fig, ax = plt.subplots()
ax.scatter(X, y)
ax.plot(X, model.predict(X))
ax.set_xlabel("Years of Experience")
ax.set_ylabel("Salary")
ax.set_title("Salary Prediction using Linear Regression")

st.pyplot(fig)

