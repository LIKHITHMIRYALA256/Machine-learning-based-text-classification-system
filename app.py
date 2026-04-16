import streamlit as st
import pickle
from preprocessing import clean_text

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📨 SMS Spam Detection System")
st.write("AI/ML Engineer Intern Assignment – Ardentix")

text = st.text_area("Enter SMS text")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter a message")
    else:
        cleaned = clean_text(text)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == "spam":
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Ham (Not Spam)")
