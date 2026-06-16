# Import required libraries
import streamlit as st
import re
import string
import nltk
import joblib
import numpy as np
import torch
import transformers
import lightgbm as lgb

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models.phrases import Phraser
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel

# Set up page config
st.set_page_config(
    page_title="Store Stockout Classifier", 
    layout="centered"
)

# Professional styling
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 0.75rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* Card styling */
    .info-card {
        background: #ffffff;
        border: 1px solid #e8ecf1;
        border-radius: 10px;
        padding: 1.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .info-card h3 {
        color: #1e3a5f;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #e8ecf1;
    }
    .info-card p, .info-card li {
        color: #4a5568;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    .info-card ul {
        margin: 0;
        padding-left: 1.25rem;
    }
    .info-card li {
        margin-bottom: 0.5rem;
    }
    
    /* Metrics row */
    .metrics-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        flex: 1;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.25rem;
        text-align: center;
    }
    .metric-box .label {
        color: #64748b;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    .metric-box .value {
        color: #1e3a5f;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    /* Input section */
    .input-section {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.75rem;
        margin-top: 1.5rem;
    }
    .input-section h3 {
        color: #1e3a5f;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0 0 1rem 0;
    }
    
    /* Result styling */
    .result-container {
        margin-top: 1.5rem;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .result-outage {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 1px solid #fecaca;
    }
    .result-no-outage {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #bbf7d0;
    }
    .result-label {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .result-outage .result-label { color: #dc2626; }
    .result-no-outage .result-label { color: #16a34a; }
    .result-prob {
        font-size: 0.9rem;
        color: #64748b;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 500;
        border-radius: 8px;
        width: 100%;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(30, 58, 95, 0.3);
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        font-size: 0.95rem;
    }
    .stTextArea textarea:focus {
        border-color: #2d5a87;
        box-shadow: 0 0 0 2px rgba(45, 90, 135, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Download required NLTK resources (suppress output)
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define TF-IDF preprocessing function 
def preprocess_text_for_phrasing(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return tokens

# Load pickled objects from notebook
@st.cache_resource
def load_models():
    with open('phraser.pkl', 'rb') as f:
        phraser = joblib.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = joblib.load(f)
    with open('bert_scaler.pkl', 'rb') as f:
        bert_scaler = joblib.load(f)
    with open('lgb_model.pkl', 'rb') as f:
        final_model = joblib.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = AutoModel.from_pretrained('bert-base-uncased')
    bert_model.eval()
    
    return phraser, tfidf_vectorizer, bert_scaler, final_model, tokenizer, bert_model

phraser, tfidf_vectorizer, bert_scaler, final_model, tokenizer, bert_model = load_models()

# Define the best threshold found during model evaluation
best_thresh = 0.5

# Function to extract BERT [CLS] embedding and scale it
def get_bert_embedding(text, max_length=128):
    encoded = tokenizer(
        [text],
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to('cpu')

    with torch.no_grad():
        outputs = bert_model(**encoded)
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    return bert_scaler.transform(cls_embedding)

# Header
st.markdown("""
<div class="main-header">
    <h1>Store Stockout Classifier</h1>
    <p>Identifying Store Inventory Stockouts Through Uber Eats Customer Reviews</p>
</div>
""", unsafe_allow_html=True)

# Executive Summary Card
st.markdown("""
<div class="info-card">
    <h3>Objective</h3>
    <p>Recover lost revenue and protect customer trust by identifying undetected store-level stockouts. 
    Items frequently appear available on Uber Eats but cannot be fulfilled—gaps not captured by existing inventory systems. 
    Analysis shows ~5% of Uber Eats reviews reference missing or unfulfilled items, highlighting a significant operational blind spot.</p>
    <p style="margin-top: 1rem;">This classifier turns real-time customer feedback into immediate operational action: 
    proactive stockout detection, faster store response, root cause visibility, and revenue recovery at scale.</p>
</div>
""", unsafe_allow_html=True)

# Metrics Row
st.markdown("""
<div class="metrics-row">
    <div class="metric-box">
        <div class="label">Model Type</div>
        <div class="value">TF-IDF + Logistic Regression</div>
    </div>
    <div class="metric-box">
        <div class="label">Classification</div>
        <div class="value">Binary (Stockout/No Stockout)</div>
    </div>
    <div class="metric-box">
        <div class="label">Data Source</div>
        <div class="value">Starbucks Uber Eats Reviews</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Business Context Card
st.markdown("""
<div class="info-card">
    <h3>Business Impact</h3>
    <ul>
        <li><strong>Problem:</strong> Menu items show as available when stores can't fulfill them—customers receive missing items or substitutions</li>
        <li><strong>Solution:</strong> Use customer feedback to fix inventory gaps</li>
        <li><strong>Value:</strong> Enable faster response to stockouts, rebuild customer trust, regain lost revenue, and prioritize high-impact issues by store or product</li>
        <li><strong>Application:</strong> Real-time alerts enable immediate menu corrections and replenishment actions, while classified data supports analytics and pattern detection across stores and products</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Input Section
st.markdown('<div class="input-section"><h3>Analyze Uber Eats Review</h3></div>', unsafe_allow_html=True)
user_input = st.text_area(
    "Enter customer text to classify:",
    placeholder="Example: 'Received bag with note saying it's out of stock'\nExample: 'Love the strawberry refresher'",
    height=120,
    label_visibility="collapsed"
)

# Predict Button
if st.button('Classify'):
    if user_input:
        with st.spinner('Analyzing sentiment...'):
            # TF-IDF preprocessing
            processed_text = preprocess_text_for_phrasing(user_input)
            phrased_text = phraser[processed_text]
            tfidf_text = tfidf_vectorizer.transform([' '.join(phrased_text)]).toarray()

            # BERT embedding
            bert_text = get_bert_embedding(user_input)

            # Combine features
            combined_features = np.concatenate((bert_text, tfidf_text), axis=1)

            # Predict
            proba = final_model.predict_proba(combined_features)[:, 1]
            prediction = (proba >= best_thresh).astype(int)[0]

            # Display results
            if prediction == 1:
                st.markdown(f"""
                <div class="result-container result-outage">
                    <div class="result-label">Potential Product Stockout Detected</div>
                    <div class="result-prob">Confidence: {proba[0]*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-container result-no-outage">
                    <div class="result-label">No Product Stockout Indicated</div>
                    <div class="result-prob">Confidence: {(1-proba[0])*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Please enter customer feedback text to analyze.")
