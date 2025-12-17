import streamlit as st
import pickle
import re

# Load model & vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Show logo at top
#st.image("logo.jpg", width=50)

# Title
st.markdown(
    '<h1 style="color:green; text-align:center; &nbsp=20;">📊Sentiment Analysis</h1>',
    unsafe_allow_html=True
)
st.markdown(
    '<h1 style="color:green; text-align:center;">😒😂😢😡</h1>',
    unsafe_allow_html=True
)

st.markdown(
    '<h3 style="color:black;">Enter a review below to predict its sentiment:</h3>',
    unsafe_allow_html=True
)

# Background color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffd1dc;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    textarea {
        background-color: #ffffff !important;   /* background */
        color: #000000 !important;              /* typed text color */
        font-size: 16px;
        border-radius: 10px;
        border: 2px solid #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    textarea::placeholder {
        color: #888888;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    label {
        color: black !important;
        font-size: 18px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)



def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

def predict_sentiment(review):
    review_clean = clean_text(review)
    vector = vectorizer.transform([review_clean])
    pred = model.predict(vector)[0]
    return "Positive" if pred == 1 else "Negative"

# Input
user_input = st.text_area("Enter your review here:")

# Prediction
if st.button("Predict Sentiment"):
    if user_input:
        sentiment = predict_sentiment(user_input)

        if sentiment == "Positive":
            st.markdown(
                f'<p style="color:green; font-size:20px;">Predicted Sentiment: {sentiment}</p>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<p style="color:red; font-size:20px;">Predicted Sentiment: {sentiment}</p>',
                unsafe_allow_html=True
            )
