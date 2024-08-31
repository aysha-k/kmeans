# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('kmeans_model.pkl', 'rb') as file:
    kmeans_model = pickle.load(file)

# Title of the web app
st.title('Iris Clustering hi78 Prediction App')

# Instructions
st.write("""
## Predict Cluster for Iris Data
Enter the sepal and petal measurements to see which cluster the data point belongs to.
""")

# Input features
sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, step=0.1, format="%.2f")
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, step=0.1, format="%.2f")
petal_length = st.number_input('Petal Length (cm)', min_value=0.0, step=0.1, format="%.2f")
petal_width = st.number_input('Petal Width (cm)', min_value=0.0, step=0.1, format="%.2f")

# Predict button
if st.button('Predict Cluster'):
    # Prepare the input data for prediction
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Predict the cluster
    cluster_label = kmeans_model.predict(input_data)[0]

    # Display the predicted cluster
    st.write(f"The predicted cluster for the input data is: **Cluster {cluster_label}**")
