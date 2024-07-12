import pandas as pd
import streamlit as st
import pickle
import numpy as np


st.header('Linear Regression', divider="rainbow")
st.header("Linear Regression Analysis by Kirubhakaran **")

# Uncomment this line if you want to display the dataset
# df = pd.read_csv("https://d2beiqkhq929f0.cloudfront.net/public_assets/assets/000/001/839/original/Jamboree_Admission.csv")
# st.dataframe(df)

# User inputs
CGPA = st.slider("CGPA", 0.0, 10.0, step=0.1)
university_ratings = st.selectbox("University Ratings", [1, 2, 3, 4, 5])

# Function to predict using the loaded model
def model_predict(CGPA, university_ratings):
    st.write("Loading model...")
    with open("linear1.pkl", "rb") as file:
        reg_model = pickle.load(file)
        st.write("Model loaded.")
        # Create an input array for the prediction
        input_array = np.array([[1, 337, 118, university_ratings, 4, 4.5, CGPA, 1]])  # Replace the dummy values with actual inputs if needed
        prediction = reg_model.predict(input_array)
        st.write("Prediction made.")
        return prediction

# Predict button
if st.button("Predict"):
    st.write("Predict button clicked.")
    prediction = model_predict(CGPA, university_ratings)
    st.write("Prediction received.")
    st.text(f"Predicted Chance of Admit: {prediction[0][0]:.2f}")