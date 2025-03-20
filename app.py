import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

# Load trained model & tokenizer
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("models/sentiment_model")
    tokenizer = AutoTokenizer.from_pretrained("models/sentiment_model")
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit UI
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="wide")

st.title("💬 Sentiment Analyzer")
st.write("Analyze the sentiment of any text! Enter a sentence below and get an instant analysis.")

user_input = st.text_area("Enter your text:", "")

if st.button("Analyze Sentiment"):
    if user_input:
        with st.spinner("Analyzing..."):
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            sentiment_index = torch.argmax(probs).item()
            confidence = round(probs[0][sentiment_index].item() * 100, 2)

        # Map index to label
        labels = ["Negative", "Neutral", "Positive"]  # Adjust this based on your training labels
        sentiment = labels[sentiment_index]

        # Display result
        st.subheader("🔍 Result")
        if sentiment == "Positive":
            st.success(f"😊 **Positive Sentiment** ({confidence}%)")
        elif sentiment == "Negative":
            st.error(f"😠 **Negative Sentiment** ({confidence}%)")
        else:
            st.warning(f"😐 **Neutral Sentiment** ({confidence}%)")

    else:
        st.warning("⚠️ Please enter some text.")

st.markdown("---")
st.markdown("🔗 Built with Streamlit | Model: DistilBERT (Fine-tuned)")

