import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv("products.csv")

# Combine text for similarity
df["text"] = df["name"] + " " + df["category"] + " " + df["description"]

# TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["text"])

# Cosine similarity
similarity_matrix = cosine_similarity(tfidf_matrix)

# Normalize functions
def normalize_price(price, min_price, max_price):
    return (max_price - price) / (max_price - min_price)

def normalize_rating(rating):
    return rating / 5.0

def normalize_popularity(views, max_views):
    return views / max_views

# Recommendation function
def recommend(product_name):
    if product_name not in df["name"].values:
        print("Product not found")
        return

    idx = df[df["name"] == product_name].index[0]

    min_price = df["price"].min()
    max_price = df["price"].max()
    max_views = df["views"].max()

    scores = []

    for i in range(len(df)):
        if i == idx:
            continue

        sim = similarity_matrix[idx][i]
        price_score = normalize_price(df.iloc[i]["price"], min_price, max_price)
        rating_score = normalize_rating(df.iloc[i]["rating"])
        pop_score = normalize_popularity(df.iloc[i]["views"], max_views)

        # Weights
        final_score = (0.4 * sim) + (0.3 * price_score) + (0.2 * rating_score) + (0.1 * pop_score)

        scores.append((df.iloc[i]["name"], final_score))

    # Sort
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print(f"\nRecommended for {product_name}:")
    for item in scores[:5]:
        print(item[0], "-> Score:", round(item[1], 3))


# TEST
recommend("Gaming Mouse")