import os
from datetime import datetime, timedelta
from urllib.parse import quote

import mysql.connector
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
app = Flask(__name__)

CACHE_TTL_MINUTES = 30
WEIGHT_MAP = {"high": 0.5, "medium": 0.3, "low": 0.2}
REQUIRED_COLUMNS = [
    "id",
    "name",
    "category",
    "price",
    "rating",
    "views",
    "description",
    "image",
]

_cache = {
    "df": None,
    "similarity": None,
    "last_loaded": None,
}


def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
    )


def read_products_from_db(conn):
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM products")
        rows = cursor.fetchall()
    finally:
        cursor.close()
    return pd.DataFrame(rows)


def guess_image_path(product_name):
    name = str(product_name).lower()
    if "keyboard" in name:
        return "keyboard.jpg"
    if "mouse" in name:
        return "mouse.jpg"
    return "mouse.jpg"


def build_image_url(image_value, product_name):
    image = str(image_value or "").strip().replace("\\", "/")
    if image.startswith(("http://", "https://", "/")):
        return image

    filename = image or guess_image_path(product_name)
    if filename.startswith("static/"):
        return f"/{filename}"
    if filename.startswith("images/"):
        return f"/static/{filename}"
    return f"/static/images/{filename}"


def normalize_query(query):
    return " ".join(str(query or "").strip().lower().split())


def normalize_label(value):
    return normalize_query(value)


def text_matches_query(text, query):
    cleaned_query = normalize_query(query)
    if not cleaned_query:
        return True

    haystack = str(text or "").lower()
    terms = [term for term in cleaned_query.split() if term]
    return all(term in haystack for term in terms)


def product_matches_filters(product, query="", category=""):
    query_ok = text_matches_query(
        f"{product.get('name', '')} {product.get('category', '')} {product.get('description', '')}",
        query,
    )
    category_value = normalize_label(product.get("category", ""))
    selected_category = normalize_label(category)
    category_ok = not selected_category or category_value == selected_category
    return query_ok and category_ok


def ensure_columns(df):
    df = df.copy()
    defaults = {
        "id": range(1, len(df) + 1),
        "name": "",
        "category": "",
        "price": 0,
        "rating": 0,
        "views": 0,
        "description": "",
        "image": "",
    }

    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            df[column] = defaults[column]

    df["name"] = df["name"].fillna("").astype(str)
    df["category"] = df["category"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)
    df["views"] = pd.to_numeric(df["views"], errors="coerce").fillna(0)
    df["image"] = df["image"].fillna("").astype(str)
    df.loc[df["image"].str.strip() == "", "image"] = df["name"].apply(guess_image_path)
    df["image_url"] = [
        build_image_url(image, name) for image, name in zip(df["image"], df["name"])
    ]

    return df.reset_index(drop=True)


def load_data():
    now = datetime.now()
    if _cache["df"] is not None and _cache["last_loaded"] is not None:
        if now - _cache["last_loaded"] <= timedelta(minutes=CACHE_TTL_MINUTES):
            return _cache["df"], _cache["similarity"]

    df = None
    conn = None

    try:
        conn = get_db_connection()
        df = read_products_from_db(conn)
    except Exception:
        csv_path = os.path.join(os.path.dirname(__file__), "products.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            raise
    finally:
        if conn is not None and conn.is_connected():
            conn.close()

    df = ensure_columns(df)
    df["text"] = df["name"] + " " + df["category"] + " " + df["description"]

    if df.empty:
        similarity = []
    else:
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(df["text"])
        similarity = cosine_similarity(tfidf)

    _cache["df"] = df
    _cache["similarity"] = similarity
    _cache["last_loaded"] = now
    return df, similarity


def norm_price(price, min_price, max_price):
    if max_price == min_price:
        return 1.0
    return (max_price - price) / (max_price - min_price)


def norm_rating(rating):
    return rating / 5 if rating else 0


def norm_pop(views, max_views):
    if max_views == 0:
        return 0
    return views / max_views


def get_weight(level):
    return WEIGHT_MAP.get(level, WEIGHT_MAP["low"])


def recommend(product_name, w_price, w_rating, w_pop, query="", category=""):
    df, similarity = load_data()
    if df.empty or product_name not in df["name"].values:
        return []

    idx = df.index[df["name"] == product_name][0]
    min_price = df["price"].min()
    max_price = df["price"].max()
    max_views = df["views"].max()

    weights = {
        "price": get_weight(w_price),
        "rating": get_weight(w_rating),
        "pop": get_weight(w_pop),
    }

    total = sum(weights.values()) or 1
    weights = {key: value / total for key, value in weights.items()}

    results = []
    used_products = set()

    for i in range(len(df)):
        if i == idx:
            continue

        sim_score = float(similarity[idx][i]) if len(similarity) else 0.0
        price_score = norm_price(df.iloc[i]["price"], min_price, max_price)
        rating_score = norm_rating(df.iloc[i]["rating"])
        pop_score = norm_pop(df.iloc[i]["views"], max_views)

        user_score = (
            weights["price"] * price_score
            + weights["rating"] * rating_score
            + weights["pop"] * pop_score
        )
        final_score = 0.5 * sim_score + 0.5 * user_score

        product = df.iloc[i][REQUIRED_COLUMNS].to_dict()
        if not product_matches_filters(product, query=query, category=category):
            continue

        product_key = (product.get("id"), product.get("name"))
        if product_key in used_products:
            continue

        used_products.add(product_key)

        reasons = []
        if sim_score >= 0.6:
            reasons.append("Highly similar")
        elif sim_score >= 0.3:
            reasons.append("Moderately similar")
        if rating_score >= 0.8:
            reasons.append("High rating")
        if price_score >= 0.7:
            reasons.append("Affordable")
        if pop_score >= 0.5:
            reasons.append("Popular")

        product["score"] = round(final_score, 3)
        product["price_score"] = round(price_score, 2)
        product["rating_score"] = round(rating_score, 2)
        product["pop_score"] = round(pop_score, 2)
        product["similarity"] = round(sim_score, 2)
        product["reason"] = ", ".join(reasons) if reasons else "Recommended"
        product["image_url"] = build_image_url(product["image"], product["name"])
        product["product_link"] = f"/product/{quote(str(product['name']))}"
        results.append(product)

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:5]


@app.route("/")
def home():
    df, _ = load_data()
    products = df.to_dict(orient="records")
    categories = sorted({str(item["category"]).strip() for item in products if str(item["category"]).strip()})
    selected_query = normalize_query(request.args.get("query", ""))
    selected_category = request.args.get("category", "").strip()
    selected_product = request.args.get("product", "").strip()

    filtered_products = [
        item for item in products if product_matches_filters(item, selected_query, selected_category)
    ]
    visible_products = filtered_products if filtered_products else products
    visible_names = {item["name"] for item in visible_products}
    if selected_product not in visible_names:
        selected_product = visible_products[0]["name"] if visible_products else ""

    return render_template(
        "index.html",
        products=products,
        filtered_products=filtered_products,
        visible_products=visible_products,
        categories=categories,
        selected_query=selected_query,
        selected_category=selected_category,
        selected_product=selected_product,
    )


@app.route("/recommend", methods=["POST"])
def rec():
    query = normalize_query(request.form.get("query", ""))
    category = request.form.get("category", "").strip()
    recommendations = recommend(
        request.form["product"],
        request.form["price"],
        request.form["rating"],
        request.form["popularity"],
        query=query,
        category=category,
    )
    return render_template(
        "result.html",
        recommendations=recommendations,
        selected=request.form["product"],
        selected_query=query,
        selected_category=category,
    )


@app.route("/product/<path:product_name>")
def product_detail(product_name):
    decoded_name = product_name
    df, _ = load_data()
    match = df[df["name"] == decoded_name]
    if match.empty:
        return render_template(
            "product.html",
            product=None,
            recommendations=[],
        ), 404

    product = match.iloc[0].to_dict()
    product["image_url"] = build_image_url(product.get("image"), product.get("name"))
    recommendations = recommend(product["name"], "medium", "high", "medium")
    for item in recommendations:
        item["product_link"] = f"/product/{quote(item['name'])}"

    return render_template(
        "product.html",
        product=product,
        recommendations=recommendations,
    )


if __name__ == "__main__":
    app.run(debug=True)
