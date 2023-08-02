import streamlit as st
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Add a title to the web app
st.title("LANGUAGE PREDICTION")

# Load the data from the CSV file in the same directory
var = pd.read_csv("dataset2.csv")

# Divide data into input and output
x = var.Text.tolist()  # Convert DataFrame column to a list of strings
y = var.language      # output

from sklearn.pipeline import make_pipeline
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(x, y)

# Input review
x_review = st.text_input('ENTER THE TEXT ')
if x_review:
    # Predict the output
    y_pred = model.predict([x_review])

    # Print the predicted output
    st.title(y_pred[0])

