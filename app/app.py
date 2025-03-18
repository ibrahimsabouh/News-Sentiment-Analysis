import re
import pickle
import os
import numpy as np
import streamlit as st
import nltk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Page configuration
st.set_page_config(
    page_title="News Sentiment Analysis",
    page_icon="📰",
    layout="centered"
)

# Download required NLTK resources (if needed)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Preprocessing function (same as used during training)
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Get model paths from environment or use defaults
base_path = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(base_path, '../model/sentiment_model.h5'))
TOKENIZER_PATH = os.environ.get('TOKENIZER_PATH', os.path.join(base_path, '../model/tokenizer.pickle'))

# Use caching to load model and tokenizer once
@st.cache_resource
def load_sentiment_model():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_sentiment_model()

# Maximum sequence length (must match training)
maxlen = 100

# ---- Custom CSS Styling ----
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #e0f7fa, #ffffff);
        padding: 20px;
    }
    .title {
        text-align: center;
        color: #006064;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .subtitle {
        text-align: center;
        color: #004d40;
        font-size: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        background-color: #00796B;
        color: white;
        width: 100%;
    }
    .prediction {
        font-size: 1.5rem;
        font-weight: bold;
        color: #27AE60;
    }
    .negative {
        color: #E74C3C;
    }
    .footer {
        text-align: center;
        font-size: 0.8rem;
        color: #7F8C8D;
        margin-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ---- Main Container ----
st.markdown("<h1 class='title'>News Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enter news text below and click <strong>Analyze</strong> to get the sentiment prediction.</p>", unsafe_allow_html=True)

# Text input area with a placeholder
user_input = st.text_area("News Text", placeholder="Paste news article or text here...", height=150)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("Analyze Sentiment")

if analyze_button:
    if user_input.strip():
        with st.spinner("Analyzing sentiment..."):
            processed_text = preprocess_text(user_input)
            seq = tokenizer.texts_to_sequences([processed_text])
            padded_seq = pad_sequences(seq, maxlen=maxlen, padding='post', truncating='post')
            prediction = model.predict(padded_seq)[0][0]
            sentiment = "Positive" if prediction > 0.5 else "Negative"
            probability = float(prediction)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h4>Prediction Results:</h4>", unsafe_allow_html=True)
        
        # Display results
        if sentiment == "Positive":
            st.success(f"Sentiment: {sentiment} (Confidence: {probability:.2f})")
            # Display a green progress bar for positive sentiment
            st.progress(probability)
        else:
            st.error(f"Sentiment: {sentiment} (Confidence: {1-probability:.2f})")
            # Display a red progress bar for negative sentiment
            st.progress(1-probability)
            
        # Add more detailed analysis
        st.markdown("### Text Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Text Length: {len(user_input)} characters")
        with col2:
            st.info(f"Word Count: {len(user_input.split())}")
    else:
        st.error("Please enter some text to analyze.")

st.markdown("<div class='footer'>Created by Ibrahim Sabouh</div>", unsafe_allow_html=True)