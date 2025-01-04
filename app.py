import streamlit as st
import pandas as pd

# Load the saved DataFrame
df = pd.read_csv('books datasets/top_books_final.csv')  # Make sure to replace this with the actual file path

# Display the title of the app
st.title('Popularity Base Recommendation System')
st.subheader("Top 10 Popular Books")
# First Row - Displaying first 5 books with their images and details
col1, col2, col3, col4, col5 = st.columns(5)
# Display books in the first row
for i, col in enumerate([col1, col2, col3, col4, col5]):
    if i < len(df):
        col.image(df.loc[i, 'Image-URL-M'], width=100)
        col.markdown(f"**{df.loc[i, 'Book-Title']}**")

# Second Row - Displaying next 5 books with their images and details

col6, col7, col8, col9, col10 = st.columns(5)
# Display books in the second row
for i, col in enumerate([col6, col7, col8, col9, col10]):
    if i + 5 < len(df):
        col.image(df.loc[i + 5, 'Image-URL-M'], width=100)
        col.markdown(f"**{df.loc[i + 5, 'Book-Title']}**")

