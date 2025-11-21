import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle # Import the pickle library

st.title("📧 Spam vs. Ham Detector")

# Load Model and Vectorizer (This function runs only once due to st.cache_resource)
model = "logistic_regression_model.pkl" 
vectorizer = "tfidf_vectorizer.pkl"

if model and vectorizer:
    # --- Input Feature ---
    st.subheader("Enter a Message to Test")
    
    # Text area for user input
    user_input = st.text_area(
        "Paste the email or SMS text here:",
        placeholder=".........",
        height=150
    )

    # --- Prediction Logic ---
    if st.button("Classify Message", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter some text to classify.")
        else:
            with st.spinner('Analyzing message...'):
                # 1. Transform the input message using the FITTED vectorizer
                # THIS LINE relies on the loaded 'vectorizer' object being the correct one:
                input_data_features = 'tfidf_vectorizer.pkl'.transform([user_input])

                # 2. Make the prediction
                prediction = "logistic_regression_model.pkl".predict(input_data_features)
                
                # 3. Get the raw probability (optional but informative)
                probability = "logistic_regression_model.pkl".predict_proba(input_data_features)[0]

                # --- Display Results ---
                st.subheader("Classification Result")
                
                if prediction[0] == 1:
                    # Prediction is 0 (Spam)
                    st.error(
                        f"**Prediction:** SPAM "
                    )
                    st.markdown(f"The model is **{(probability[0] * 100):.2f}%** confident this is spam.")
                else:
                    # Prediction is 1 (Ham)
                    st.success(
                        f"**Prediction:** HAM (Not Spam) "
                    )
                    st.markdown(f"The model is **{(probability[1] * 100):.2f}%** confident this is ham.")
