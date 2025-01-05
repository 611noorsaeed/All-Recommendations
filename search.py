import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load the saved DataFrame and necessary files
df = pd.read_csv('books datasets/top_books_final.csv')  # For popularity-based recommendations
df1 = pd.read_csv("books datasets/final_df.csv")  # For search-based recommendations
tfidf = pickle.load(open("books datasets/tfidf.pkl", 'rb'))
tfidf_matrix = pickle.load(open("books datasets/tfidf_matrix.pkl", 'rb'))

# Define the search-based recommendation function
def search_based_recommendation(book_title, df, tfidf, tfidf_matrix, top_n=10):
    """
    Recommend books based on similarity of titles using TF-IDF and cosine similarity.
    """
    query_vector = tfidf.transform([book_title])
    cosine_sim = cosine_similarity(query_vector, tfidf_matrix).flatten()
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    book_indices = [i[0] for i in sim_scores]
    return df[['Book-Title', 'Book-Rating', 'Image-URL-M']].iloc[book_indices].reset_index(drop=True)

# Streamlit App UI
st.title('Book Recommendation System')

# Search Input
book_to_search = st.text_input("Enter a book title to search for recommendations:", "")

if book_to_search:  # Show search-based recommendations if a title is entered
    st.subheader(f"Recommendations for: {book_to_search}")
    recommendations = search_based_recommendation(book_to_search, df1, tfidf, tfidf_matrix)
    if not recommendations.empty:
        cols = st.columns(5)
        for idx, row in recommendations.iterrows():
            col = cols[idx % 5]
            col.image(row['Image-URL-M'], width=100)
            col.markdown(f"**{row['Book-Title']}**")
    else:
        st.write("No recommendations found. Please try another title.")
else:  # Show default popularity-based recommendations
    st.subheader("Top 10 Popular Books")
    cols = st.columns(5)
    for i, col in enumerate(cols):
        if i < len(df):
            col.image(df.loc[i, 'Image-URL-M'], width=100)
            col.markdown(f"**{df.loc[i, 'Book-Title']}**")
    for i, col in enumerate(cols):
        if i + 5 < len(df):
            col.image(df.loc[i + 5, 'Image-URL-M'], width=100)
            col.markdown(f"**{df.loc[i + 5, 'Book-Title']}**")
