import streamlit as st
import requests
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


def download_file(url):
    """Helper function to download a file from a specified URL."""
    response = requests.get(url, stream=True)
    response.raise_for_status()  # This will check for HTTP errors
    return response.content


# Load CSS styles

def load_css():
    url = 'https://raw.githubusercontent.com/AnanyaThyagarajan/Spam-Detection/main/style.css'
    response = requests.get(url)
    if response.status_code == 200:
        css_content = response.text
        st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)
    else:
        st.error('Failed to download CSS')

def main():
    load_css()
    st.title('Spam Detection System')

    # URL for the vectorizer and model on GitHub
    vectorizer_url = 'https://github.com/AnanyaThyagarajan/Spam-Detection/blob/main/tfidf_vectorizer.pkl?raw=true'
    model_url = 'https://github.com/AnanyaThyagarajan/Spam-Detection/blob/main/spam_svm_model.pkl?raw=true'

    # Download and load the vectorizer and model
    try:
        vectorizer_data = download_file(vectorizer_url)
        vectorizer = pickle.loads(vectorizer_data)
        model_data = download_file(model_url)
        model = pickle.loads(model_data)
        st.success("Model and vectorizer loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load files: {str(e)}")
        return

    # Text area for user input
    message = st.text_area("Enter the message or email content here:", height=150)
    
    if st.button("Predict"):
        message_tfidf = vectorizer.transform([message])
        result = model.predict(message_tfidf)
        
        # Display results
        if result[0] == 1:
            st.error("This is a SPAM message!")
        else:
            st.success("This is NOT SPAM.")

if __name__ == '__main__':
    main()