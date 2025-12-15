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
# Default Dataset (No Upload)
# ----------------------------
saldf = pd.DataFrame({
    "Years of Experience": [0.5, 1, 2, 3, 4, 5, 6],
    "Salary": [25000, 30000, 40000, 50000, 60000, 75000, 90000]
})

# ----------------------------
# Dataset Preview
# ----------------------------
st.subheader("📊 Dataset Preview")
st.dataframe(saldf)

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
    prediction = model.predict([[experience]])
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
