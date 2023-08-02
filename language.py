import streamlit as st
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

# Add a title to the web app
st.title("LANGUAGE PREDICTION")

# Load the data from the CSV file in the same directory
var = pd.read_csv("dataset2.csv")

# Divide data into input and output
x = var.Text.tolist()  # Convert DataFrame column to a list of strings
y = var.language  # output

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create the pipeline with CountVectorizer and MultinomialNB
from sklearn.pipeline import make_pipeline
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Fit the model on the training data
model.fit(x_train, y_train)

# Input review
x_review = st.text_input('ENTER THE TEXT ')
if x_review:
    # Predict the output
    y_pred = model.predict([x_review])

    # Print the predicted output
    st.title(f"Predicted Language: {y_pred[0]}")
