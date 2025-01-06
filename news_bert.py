import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import pickle

# Download stopwords and punkt (for tokenization)
nltk.download('stopwords')
nltk.download('punkt')

# Load Dataset and Embeddings
df = pd.read_csv('books datasets/Articles.csv', encoding='ISO-8859-1')
# Load the pre-trained Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can use other models if you prefer
embeddings = pickle.load(open('books datasets/embeddings.pkl', 'rb'))


# Initialize the PorterStemmer and stopwords
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
# Preprocessing function to clean text (removes punctuation, converts to lowercase, removes stopwords, applies stemming)
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize the text
    words = text.split()

    # Remove stopwords and apply stemming
    processed_words = [stemmer.stem(word) for word in words if word not in stop_words]

    # Join the words back into a string
    return " ".join(processed_words)


# Function to recommend top N articles based on a search query
def recommend_articles_from_search(query, df, model,embeddings, num_recommendations=5):
    # Step 0: Clean Query
    query = preprocess_text(query)
    # Step 1: Get the embedding for the search query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Step 2: Compute cosine similarity between the query and all article embeddings
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)

    # Step 3: Get the indices of the top N most similar articles
    similarities = similarities.flatten()  # Flatten the 2D array to 1D
    top_indices = similarities.argsort()[-num_recommendations:][::-1]  # Get indices of top N

    # Step 4: Retrieve the corresponding articles and their similarity scores
    recommended_articles = df.iloc[top_indices][['Heading', 'NewsType', 'Article', 'Date']]
    return recommended_articles


# UI APP
st.title("Context-Aware Recommendation System")
query = st.text_input("Enter News Heading Here..........")
if query:
    recommended_articles = recommend_articles_from_search(query, df, model,embeddings, num_recommendations=5)

    # Display
    st.subheader("\n Top News Articles")
    st.dataframe(recommended_articles)
else:
    st.write("Enter News Heading...........")
