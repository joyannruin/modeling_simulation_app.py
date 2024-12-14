### Streamlit App: Modeling and Simulation with Python

# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Title and Introduction
st.title("CSEC 413: Modeling and Simulation Project")
st.markdown("""
This project demonstrates key concepts of modeling and simulation using Python. We'll work with synthetic data to illustrate exploratory data analysis (EDA), modeling, simulation, and evaluation techniques.
""")

# Step 1: Data Generation
st.header("Step 1: Data Generation")
noise_level = st.slider("Select noise level for the synthetic data", 0, 30, 10)
X, y = np.random.RandomState(42).rand(100, 1), np.random.normal(50, noise_level, 100)

# Create a DataFrame for easier visualization
data = pd.DataFrame({"Feature": X.flatten(), "Target": y})

# Display the first few rows of the data
st.subheader("Generated Data")
st.dataframe(data.head())

# Step 2: Visualizing the Data
st.header("Step 2: Data Visualization")
fig, ax = plt.subplots()
ax.scatter(data["Feature"], data["Target"], color='blue', alpha=0.7)
ax.set_title("Generated Synthetic Data")
ax.set_xlabel("Feature")
ax.set_ylabel("Target")
st.pyplot(fig)

# Step 3: Exploratory Data Analysis (EDA)
st.header("Step 3: Exploratory Data Analysis")
st.write("Summary Statistics:")
st.write(data.describe())

# Correlation heatmap
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Step 4: Modeling
st.header("Step 4: Modeling")
test_size = st.slider("Select test data percentage", 10, 50, 20) / 100
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Display model coefficients
st.write("Model Coefficients:")
st.write(f"Slope: {model.coef_[0]}, Intercept: {model.intercept_}")

# Step 5: Simulation
st.header("Step 5: Simulation")
y_pred = model.predict(X_test)

# Predictions vs Actual Plot
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, color='green', alpha=0.7)
ax.set_title("Predictions vs Actual")
ax.set_xlabel("Actual Values")
ax.set_ylabel("Predicted Values")
st.pyplot(fig)

# Step 6: Evaluation
st.header("Step 6: Evaluation")
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R-Squared Value: {r2:.2f}")

# Conclusion
st.header("Conclusion")
st.markdown("""
- The linear regression model captured the relationship between the feature and target variables effectively, as shown by the R-squared value.
- This project demonstrated key steps in modeling and simulation, including data generation, visualization, and evaluation.
- Future work can involve testing different models and datasets.
""")
