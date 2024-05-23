import streamlit as st
import requests
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to download the model from GitHub
def download_model(url):
    response = requests.get(url)
    if response.status_code == 200:
        model = pickle.loads(response.content)
        return model
    else:
        st.error("Failed to download model.")
        return None

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

   

    # Model URL (replace 'yourusername/yourrepo' with your actual GitHub repository path)
    model_url = 'https://github.com/AnanyaThyagarajan/Spam-Detection/blob/main/spam_svm_model.pkl?raw=true'
    model = download_model(model_url)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)  # Assuming the vectorizer settings

    # User input
    message = st.text_area("Enter the message or email content here:", height=150)

    if st.button("Predict"):
        if model:
            message_tfidf = vectorizer.transform([message])
            result = model.predict(message_tfidf)
            
            # Display results
            if result[0] == 1:
                st.error("This is a SPAM message!")
            else:
                st.success("This is NOT SPAM.")

if __name__ == '__main__':
    main()
