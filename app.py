import os
from datetime import datetime, timedelta
from urllib.parse import quote

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, render_template, request
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

load_dotenv()
app = Flask(__name__)

CACHE_TTL_MINUTES = 30
REQUIRED_COLUMNS = ["id", "name", "category", "price", "rating", "views", "description", "image"]

PERSONALITY_PROFILES = {
    "performance": {"price": 0.1, "rating": 0.6, "pop": 0.1, "similarity": 0.2},
    "budget":      {"price": 0.6, "rating": 0.1, "pop": 0.1, "similarity": 0.2},
    "trend":       {"price": 0.1, "rating": 0.2, "pop": 0.6, "similarity": 0.1},
    "precision":   {"price": 0.1, "rating": 0.1, "pop": 0.1, "similarity": 0.7},
}

PERSONALITY_LABELS = {
    "performance": "🏆 Performance Seeker",
    "budget":      "💰 Budget Hunter",
    "trend":       "🔥 Trend Follower",
    "precision":   "🎯 Precision Matcher",
}

WEIGHT_MAP = {"high": 0.5, "medium": 0.3, "low": 0.2}

_cache = {
    "df": None, "similarity": None,
    "mf_similarity": None, "last_loaded": None,
}


def get_db_connection():
    if not MYSQL_AVAILABLE:
        raise Exception("mysql.connector not available")
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"), database=os.getenv("DB_NAME"),
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
    if "keyboard" in name: return "keyboard1.jpg"
    if "laptop" in name:   return "laptop1.jpg"
    if "mouse" in name:    return "mouse1.jpg"
    return "mouse1.jpg"


def build_image_url(image_value, product_name):
    image = str(image_value or "").strip().replace("\\", "/")
    if image.startswith(("http://", "https://", "/")): return image
    filename = image or guess_image_path(product_name)
    if filename.startswith("static/"): return f"/{filename}"
    if filename.startswith("images/"): return f"/static/{filename}"
    return f"/static/images/{filename}"


def normalize_query(query):
    return " ".join(str(query or "").strip().lower().split())


def normalize_label(value):
    return normalize_query(value)


def text_matches_query(text, query):
    cleaned_query = normalize_query(query)
    if not cleaned_query: return True
    haystack = str(text or "").lower()
    terms = [t for t in cleaned_query.split() if t]
    return all(t in haystack for t in terms)


def product_matches_filters(product, query="", category=""):
    query_ok = text_matches_query(
        f"{product.get('name','')} {product.get('category','')} {product.get('description','')}", query)
    category_value = normalize_label(product.get("category", ""))
    selected_category = normalize_label(category)
    category_ok = not selected_category or category_value == selected_category
    return query_ok and category_ok


def ensure_columns(df):
    df = df.copy()
    defaults = {"id": range(1, len(df)+1), "name": "", "category": "", "price": 0,
                "rating": 0, "views": 0, "description": "", "image": ""}
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = defaults[col]
    df["name"] = df["name"].fillna("").astype(str)
    df["category"] = df["category"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)
    df["views"] = pd.to_numeric(df["views"], errors="coerce").fillna(0)
    df["image"] = df["image"].fillna("").astype(str)
    df.loc[df["image"].str.strip() == "", "image"] = df["name"].apply(guess_image_path)
    df["image_url"] = [build_image_url(img, name) for img, name in zip(df["image"], df["name"])]
    return df.reset_index(drop=True)


def build_mf_matrix(df):
    scaler = MinMaxScaler()
    features = df[["price", "rating", "views"]].values.astype(float)
    features_scaled = scaler.fit_transform(features)
    categories = pd.get_dummies(df["category"]).values.astype(float)
    product_matrix = np.hstack([features_scaled, categories])
    n_components = max(2, min(8, product_matrix.shape[0]-1, product_matrix.shape[1]-1))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    latent_matrix = svd.fit_transform(product_matrix)
    return cosine_similarity(latent_matrix)


def load_data():
    now = datetime.now()
    if _cache["df"] is not None and _cache["last_loaded"] is not None:
        if now - _cache["last_loaded"] <= timedelta(minutes=CACHE_TTL_MINUTES):
            return _cache["df"], _cache["similarity"], _cache["mf_similarity"]

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
        if conn is not None and hasattr(conn, 'is_connected') and conn.is_connected():
            conn.close()

    df = ensure_columns(df)
    df["text"] = df["name"] + " " + df["category"] + " " + df["description"]

    if df.empty:
        similarity = []
        mf_similarity = []
    else:
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(df["text"])
        similarity = cosine_similarity(tfidf)
        mf_similarity = build_mf_matrix(df)

    _cache["df"] = df
    _cache["similarity"] = similarity
    _cache["mf_similarity"] = mf_similarity
    _cache["last_loaded"] = now
    return df, similarity, mf_similarity


def norm_price(price, min_price, max_price):
    if max_price == min_price: return 1.0
    return (max_price - price) / (max_price - min_price)

def norm_rating(rating):
    return rating / 5 if rating else 0

def norm_pop(views, max_views):
    if max_views == 0: return 0
    return views / max_views

def get_weight(level):
    return WEIGHT_MAP.get(level, WEIGHT_MAP["low"])


def compute_nova_score(sim_score, mf_score, price_score, rating_score, pop_score, personality, w_price, w_rating, w_pop):
    """
    NovaScore™ — proprietary hybrid scoring formula.
    Blends TF-IDF content similarity, SVD latent factors,
    and personality-calibrated weights into one final score.
    """
    profile = PERSONALITY_PROFILES.get(personality, PERSONALITY_PROFILES["precision"])

    manual_price  = get_weight(w_price)
    manual_rating = get_weight(w_rating)
    manual_pop    = get_weight(w_pop)
    manual_total  = manual_price + manual_rating + manual_pop

    # 60% personality, 40% manual blend
    blended_price  = 0.6 * profile["price"]  + 0.4 * (manual_price / manual_total)
    blended_rating = 0.6 * profile["rating"] + 0.4 * (manual_rating / manual_total)
    blended_pop    = 0.6 * profile["pop"]    + 0.4 * (manual_pop / manual_total)
    blended_sim    = profile["similarity"]

    total = blended_price + blended_rating + blended_pop + blended_sim
    wp  = blended_price  / total
    wr  = blended_rating / total
    wpo = blended_pop    / total
    ws  = blended_sim    / total

    hybrid_sim = 0.5 * sim_score + 0.5 * mf_score
    nova = ws * hybrid_sim + wp * price_score + wr * rating_score + wpo * pop_score

    # Confidence: how many of the 5 signals are strong (>=0.5)
    signals = [sim_score, mf_score, price_score, rating_score, pop_score]
    high_signals = sum(1 for s in signals if s >= 0.5)
    confidence = round((high_signals / len(signals)) * 100)

    # Score breakdown string for tooltip
    breakdown = (
        f"{ws:.2f}×similarity({sim_score:.2f}) + "
        f"{wp:.2f}×price({price_score:.2f}) + "
        f"{wr:.2f}×rating({rating_score:.2f}) + "
        f"{wpo:.2f}×popularity({pop_score:.2f}) = {nova:.3f}"
    )

    return round(nova, 3), confidence, breakdown


def compute_personality_match(sim_score, mf_score, price_score, rating_score, pop_score, personality):
    """
    Personality Match % — how well this product's scores align
    with the chosen buyer personality profile weights.
    """
    profile = PERSONALITY_PROFILES.get(personality, PERSONALITY_PROFILES["precision"])
    scores = {
        "similarity": (sim_score + mf_score) / 2,
        "price":      price_score,
        "rating":     rating_score,
        "pop":        pop_score,
    }
    weighted_sum = sum(profile[k] * scores[k] for k in profile)
    total_weight = sum(profile.values())
    match_pct = round((weighted_sum / total_weight) * 100)
    return min(match_pct, 100)


def is_unexpected_discovery(sim_score, mf_score, selected_category, item_category):
    """
    Unexpected Discovery — MF found a strong latent connection
    but TF-IDF text similarity is low, OR it's cross-category.
    """
    cross_category = selected_category != item_category
    latent_strong  = mf_score >= 0.7
    text_weak      = sim_score < 0.3
    return (latent_strong and text_weak) or (cross_category and latent_strong)


def build_reason(product_name, item_row, sim_score, price_score, pop_score, df, personality):
    reasons = []
    profile_label = PERSONALITY_LABELS.get(personality, "")
    if profile_label:
        reasons.append(f"Matched your '{profile_label}' buyer profile")

    if sim_score >= 0.6:
        reasons.append(f"Highly similar to '{product_name}' based on content analysis")
    elif sim_score >= 0.3:
        reasons.append(f"Moderately similar to '{product_name}' in product features")
    else:
        reasons.append(f"Loosely related to '{product_name}' — discovered via latent pattern analysis")

    selected_rows = df[df["name"] == product_name]
    if not selected_rows.empty:
        selected_price = selected_rows.iloc[0]["price"]
        item_price = item_row["price"]
        price_diff = abs(item_price - selected_price)
        if item_price < selected_price:
            reasons.append(f"More affordable — RM {price_diff:.2f} cheaper than '{product_name}'")
        elif item_price == selected_price:
            reasons.append(f"Same price range as '{product_name}' (RM {item_price:.2f})")
        else:
            reasons.append(f"RM {price_diff:.2f} higher than '{product_name}' but may offer better specs")

    item_rating = item_row["rating"]
    if item_rating >= 4.5:   reasons.append(f"Excellent customer rating of {item_rating}/5")
    elif item_rating >= 4.0: reasons.append(f"Good customer rating of {item_rating}/5")
    else:                    reasons.append(f"Rated {item_rating}/5")

    item_views = int(item_row["views"])
    if pop_score >= 0.7:   reasons.append(f"Very popular with {item_views:,} views")
    elif pop_score >= 0.4: reasons.append(f"Moderately popular with {item_views:,} views")
    else:                  reasons.append(f"Niche pick with {item_views:,} views")

    item_category = item_row["category"]
    if not selected_rows.empty:
        sel_cat = selected_rows.iloc[0]["category"]
        if item_category == sel_cat:
            reasons.append(f"Same category ({item_category}) as your selected product")
        else:
            reasons.append(f"Cross-category discovery from {item_category} — complementary to your selection")

    return " • ".join(reasons)


def build_why_not(product_name, excluded, df, personality):
    profile = PERSONALITY_PROFILES.get(personality, PERSONALITY_PROFILES["precision"])
    profile_label = PERSONALITY_LABELS.get(personality, "your profile")
    why_not_list = []

    for item in excluded[:3]:
        reasons = []
        if profile["rating"] >= 0.4 and item["rating_score"] < 0.5:
            reasons.append(f"rating ({item['rating']}/5) was too low for {profile_label}")
        if profile["price"] >= 0.4 and item["price_score"] < 0.4:
            reasons.append(f"price was outside the budget range preferred by {profile_label}")
        if profile["pop"] >= 0.4 and item["pop_score"] < 0.3:
            reasons.append(f"not popular enough for {profile_label}")
        if profile["similarity"] >= 0.4 and item["similarity"] < 0.2:
            reasons.append(f"similarity to '{product_name}' was too low for {profile_label}")
        if item["nova_score"] < 0.3:
            reasons.append(f"overall NovaScore™ ({item['nova_score']}) was below the threshold")
        if not reasons:
            reasons.append(f"scored lower than the top 5 picks under {profile_label} weighting")

        why_not_list.append({
            "name": item["name"],
            "nova_score": item["nova_score"],
            "image_url": item["image_url"],
            "reason": " • ".join(reasons),
        })

    return why_not_list


def recommend(product_name, w_price, w_rating, w_pop, personality="precision", query="", category=""):
    df, similarity, mf_similarity = load_data()
    if df.empty or product_name not in df["name"].values:
        return [], []

    idx = df.index[df["name"] == product_name][0]
    min_price  = df["price"].min()
    max_price  = df["price"].max()
    max_views  = df["views"].max()
    sel_category = df.iloc[idx]["category"]

    all_results = []
    used_products = set()

    for i in range(len(df)):
        if i == idx: continue

        sim_score    = float(similarity[idx][i])    if len(similarity)    else 0.0
        mf_score     = float(mf_similarity[idx][i]) if len(mf_similarity) else 0.0
        price_score  = norm_price(df.iloc[i]["price"], min_price, max_price)
        rating_score = norm_rating(df.iloc[i]["rating"])
        pop_score    = norm_pop(df.iloc[i]["views"], max_views)

        nova_score, confidence, breakdown = compute_nova_score(
            sim_score, mf_score, price_score, rating_score, pop_score,
            personality, w_price, w_rating, w_pop
        )

        personality_match = compute_personality_match(
            sim_score, mf_score, price_score, rating_score, pop_score, personality
        )

        unexpected = is_unexpected_discovery(
            sim_score, mf_score, sel_category, df.iloc[i]["category"]
        )

        product = df.iloc[i][REQUIRED_COLUMNS].to_dict()
        product_key = (product.get("id"), product.get("name"))
        if product_key in used_products: continue
        used_products.add(product_key)

        product["nova_score"]       = nova_score
        product["score"]            = nova_score
        product["confidence"]       = confidence
        product["nova_breakdown"]   = breakdown
        product["personality_match"]= personality_match
        product["unexpected"]       = unexpected
        product["sim_score"]        = round(sim_score, 2)
        product["mf_score"]         = round(mf_score, 2)
        product["price_score"]      = round(price_score, 2)
        product["rating_score"]     = round(rating_score, 2)
        product["pop_score"]        = round(pop_score, 2)
        product["similarity"]       = round(sim_score, 2)
        product["image_url"]        = build_image_url(product["image"], product["name"])
        product["product_link"]     = f"/product/{quote(str(product['name']))}"
        product["reason"]           = build_reason(product_name, df.iloc[i], sim_score, price_score, pop_score, df, personality)
        all_results.append(product)

    all_results.sort(key=lambda x: x["nova_score"], reverse=True)
    filtered = [p for p in all_results if product_matches_filters(p, query=query, category=category)]

    top5     = filtered[:5]
    excluded = [p for p in filtered[5:] if not any(p["name"] == t["name"] for t in top5)]
    why_not  = build_why_not(product_name, excluded, df, personality)

    return top5, why_not


@app.route("/")
def home():
    df, _, __ = load_data()
    products = df.to_dict(orient="records")
    categories = sorted({str(item["category"]).strip() for item in products if str(item["category"]).strip()})
    selected_query    = normalize_query(request.args.get("query", ""))
    selected_category = request.args.get("category", "").strip()
    selected_product  = request.args.get("product", "").strip()

    filtered_products = [item for item in products if product_matches_filters(item, selected_query, selected_category)]
    visible_products  = filtered_products if filtered_products else products
    visible_names     = {item["name"] for item in visible_products}
    if selected_product not in visible_names:
        selected_product = visible_products[0]["name"] if visible_products else ""

    return render_template("index.html",
        products=products, filtered_products=filtered_products,
        visible_products=visible_products, categories=categories,
        selected_query=selected_query, selected_category=selected_category,
        selected_product=selected_product,
    )


@app.route("/recommend", methods=["POST"])
def rec():
    query        = normalize_query(request.form.get("query", ""))
    category     = request.form.get("category", "").strip()
    product_name = request.form["product"]
    personality  = request.form.get("personality", "precision")

    recommendations, why_not = recommend(
        product_name, request.form["price"], request.form["rating"],
        request.form["popularity"], personality=personality, query=query, category=category,
    )

    return render_template("result.html",
        recommendations=recommendations, why_not=why_not,
        selected=product_name, selected_query=query,
        selected_category=category, personality=personality,
        personality_label=PERSONALITY_LABELS.get(personality, ""),
    )


@app.route("/product/<path:product_name>")
def product_detail(product_name):
    df, _, __ = load_data()
    match = df[df["name"] == product_name]
    if match.empty:
        return render_template("product.html", product=None, recommendations=[]), 404

    product = match.iloc[0].to_dict()
    product["image_url"] = build_image_url(product.get("image"), product.get("name"))
    recommendations, _ = recommend(product["name"], "medium", "high", "medium", personality="precision")
    for item in recommendations:
        item["product_link"] = f"/product/{quote(item['name'])}"

    return render_template("product.html", product=product, recommendations=recommendations)


if __name__ == "__main__":
    app.run(debug=True)