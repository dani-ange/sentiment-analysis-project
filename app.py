import streamlit as st
from transformers import pipeline

# Load trained model
classifier = pipeline("text-classification", model="./models")

st.title("Sentiment Analysis with DistilBERT")
text = st.text_area("Enter a text")

if st.button("Predict"):
    result = classifier(text)
    st.write(result)
