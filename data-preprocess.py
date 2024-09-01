import pandas as pd
import numpy as np
import re

# Load datasets
books = pd.read_csv("data-raw/BX-Books.csv", sep=";", encoding='ISO-8859-1', on_bad_lines='skip')
users = pd.read_csv("data-raw/BX-Users.csv", sep=";", encoding='ISO-8859-1', on_bad_lines='skip')
ratings = pd.read_csv("data-raw/BX-Book-Ratings.csv", sep=";", encoding='ISO-8859-1', on_bad_lines='skip')

# Clean and preprocess Books data
books.dropna(inplace=True)
books_merge = pd.merge(books, ratings, on='ISBN')
books_merge.dropna(inplace=True)
books_merge.reset_index(drop=True, inplace=True)
books_merge.drop(index=books_merge[books_merge["Book-Rating"] == 0].index, inplace=True)
books_merge["Book-Title"] = books_merge["Book-Title"].apply(lambda x: re.sub(r"\bamp\b", "&", x))  # Replace 'amp' with '&'
# Correct possessive form and preserve apostrophes
books_merge["Book-Title"] = books_merge["Book-Title"].apply(lambda x: re.sub(r"([^\s\w]|_)+", "", x).strip())
books_merge.head(100000).to_csv('data/books-ratings.csv') 
# Clean and preprocess Users data
users.dropna(inplace=True)
users_merge = pd.merge(users, ratings, on='User-ID', how="inner")
users_merge.dropna(inplace=True)
users_merge.reset_index(drop=True, inplace=True)
users_merge.drop(index=users_merge[users_merge["Book-Rating"] == 0].index, inplace=True)
users_merge["Age"] = users_merge["Age"].apply(lambda x: int(x) if pd.notnull(x) else x)  # Convert float age to int
merged_data = pd.merge(users_merge, books, on="ISBN", how="inner")
merged_data["Book-Title"] = merged_data["Book-Title"].apply(lambda x: re.sub(r"\bamp\b", "&", x))  # Replace 'amp' with '&'
merged_data["Book-Title"] = merged_data["Book-Title"].apply(lambda x: re.sub(r"([^\s\w]|_)+", "", x).strip())  # Preserve apostrophes in titles
merged_data.head(100000).to_csv('data/users-ratings.csv')
