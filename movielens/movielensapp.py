"""
MovieLens-1M Recommendation App (Strict No-Padding Version)
============================================================

Fixes applied:
✔ Removed padding completely (model does NOT support padding)
✔ Fixed recommendation length condition
✔ Strict MIN_SEQ_LEN enforcement
✔ Stable training + inference
✔ Model auto-load
✔ DeepInfra LLM reranking added
"""

import os
import random
import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request, session, render_template
from dotenv import load_dotenv
import requests

from movielens_sasrec import (
    EnhancedSASRec,
    CatalogFeatureBuilder,
    MIN_SEQ_LEN,
)

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

MAX_SEQ_LEN = 50
MODEL_PATH = "movielens_enhanced_sasrec_full.pth"

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "fallback-secret")

DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
MOVIELENS_API_KEY = os.getenv("MOVIELENS_API_KEY")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------
# GLOBAL STATE
# -------------------------------------------------------

MOVIES_DF = None
RATINGS_DF = None
CATALOG = None
MODEL = None

MOVIES_BY_ID = {}
USER_SESSIONS = defaultdict(list)
CACHED_USER_SEQUENCES = None

def rerank_with_llm(history_titles, candidates):

    if MOVIELENS_API_KEY is None:
        return candidates

    url = "https://api.deepinfra.com/v1/openai/chat/completions"

    history_text = ", ".join(history_titles[-10:])

    movie_list = "\n".join(
        [f"{i+1}. {m['Title']} ({m['Genres']})" for i, m in enumerate(candidates)]
    )

    prompt = f"""
User previously watched these movies:
{history_text}

Here are recommended movies:

{movie_list}

Reorder the movies from MOST relevant to LEAST relevant.

Return ONLY movie numbers separated by commas.

Example:
3,1,5,2,4
"""

    payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 50
    }

    headers = {
        "Authorization": f"Bearer {MOVIELENS_API_KEY}",
        "Content-Type": "application/json"
    }

    try:

        r = requests.post(url, json=payload, headers=headers, timeout=20)

        if r.status_code != 200:
            print("DeepInfra HTTP Error:", r.status_code, r.text)
            return candidates

        data = r.json()

        if "choices" not in data:
            print("Unexpected LLM response:", data)
            return candidates

        text = data["choices"][0]["message"]["content"]

        order = [
            int(x.strip()) - 1
            for x in text.replace("\n", "").split(",")
            if x.strip().isdigit()
        ]

        reordered = [
            candidates[i] for i in order if i < len(candidates)
        ]

        if len(reordered) > 0:
            return reordered

        return candidates

    except Exception as e:
        print("LLM reranking failed:", e)
        return candidates

# -------------------------------------------------------
# LOAD DATA (.DAT FORMAT)
# -------------------------------------------------------

def load_movielens_data():

    global MOVIES_DF, RATINGS_DF, CATALOG
    global MOVIES_BY_ID, CACHED_USER_SEQUENCES

    base = os.path.dirname(__file__)

    MOVIES_DF = pd.read_csv(
        os.path.join(base, "movies.dat"),
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["MovieID", "Title", "Genres"],
    )

    RATINGS_DF = pd.read_csv(
        os.path.join(base, "ratings.dat"),
        sep="::",
        engine="python",
        names=["UserID", "MovieID", "Rating", "Timestamp"],
    )

    MOVIES_DF["main_category"] = MOVIES_DF["Genres"].str.split("|").str[0]

    MOVIES_BY_ID = MOVIES_DF.set_index("MovieID").to_dict("index")

    CATALOG = CatalogFeatureBuilder(MOVIES_DF, RATINGS_DF)

    ratings_sorted = RATINGS_DF.sort_values("Timestamp")

    user_groups = (
        ratings_sorted.groupby("UserID")["MovieID"]
        .apply(list)
        .to_dict()
    )

    CACHED_USER_SEQUENCES = {
        uid: seq
        for uid, seq in user_groups.items()
        if len(seq) >= MIN_SEQ_LEN + 1
    }

    print(f"[OK] Loaded {len(MOVIES_DF)} movies")
    print(f"[OK] Loaded {len(RATINGS_DF)} ratings")
    print(f"[OK] {len(CACHED_USER_SEQUENCES)} usable user sequences")


# -------------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------------

def train_model(epochs=5):

    global MODEL

    sequences = list(CACHED_USER_SEQUENCES.values())

    MODEL = EnhancedSASRec(CATALOG).to(device)

    optimizer = torch.optim.Adam(MODEL.parameters(), lr=1e-3)

    MODEL.train()

    for epoch in range(epochs):

        random.shuffle(sequences)

        total_loss = 0

        for seq in sequences:

            seq = seq[-MAX_SEQ_LEN:]

            if len(seq) < MIN_SEQ_LEN + 1:
                continue

            input_seq = seq[:-1]
            target = seq[-1]

            if len(input_seq) < MIN_SEQ_LEN:
                continue

            pids = torch.tensor([input_seq], dtype=torch.long, device=device)

            scores = MODEL(pids)

            target_idx = torch.tensor(
                [CATALOG.pid_to_idx[target]],
                device=device,
            )

            loss = F.cross_entropy(scores, target_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss {total_loss:.4f}")

    torch.save(MODEL.state_dict(), MODEL_PATH)

    MODEL.eval()

    print("[INFO] Training complete.")


# -------------------------------------------------------
# LOAD OR TRAIN
# -------------------------------------------------------

def load_or_train_model():

    global MODEL

    if os.path.exists(MODEL_PATH):

        print("[INFO] Loading saved model...")

        MODEL = EnhancedSASRec(CATALOG).to(device)

        MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=device))

        MODEL.eval()

    else:

        print("[INFO] No saved model found. Training...")

        train_model(epochs=5)


# -------------------------------------------------------
# SESSION MANAGEMENT
# -------------------------------------------------------

def get_session_user():

    if "user_id" not in session:

        session["user_id"] = hashlib.sha256(
            str(random.random()).encode()
        ).hexdigest()[:16]

    return session["user_id"]


# -------------------------------------------------------
# RECORD WATCH
# -------------------------------------------------------

@app.route("/watch", methods=["POST"])
def record_watch():

    user_id = get_session_user()

    movie_id = int(request.json.get("movie_id"))

    if movie_id not in MOVIES_BY_ID:

        return jsonify({"error": "Invalid MovieID"}), 400

    if movie_id not in USER_SESSIONS[user_id]:

        USER_SESSIONS[user_id].append(movie_id)

    return jsonify({"status": "recorded"})


# -------------------------------------------------------
# USER HISTORY
# -------------------------------------------------------

@app.route("/history", methods=["GET"])
def get_history():

    user_id = get_session_user()

    history = USER_SESSIONS.get(user_id, [])

    movies = []

    for pid in history:

        movie = MOVIES_BY_ID.get(pid)

        if movie:

            movies.append({
                "MovieID": int(pid),
                "Title": movie["Title"],
                "Genres": movie["Genres"]
            })

    return jsonify(movies)


# -------------------------------------------------------
# HOME
# -------------------------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# -------------------------------------------------------
# RECOMMEND
# -------------------------------------------------------

@app.route("/recommend", methods=["GET"])
def recommend():

    user_id = get_session_user()

    history = USER_SESSIONS.get(user_id, [])

    if MODEL is None:
        return jsonify({"error": "Model not available"}), 500

    if len(history) < MIN_SEQ_LEN:
        return trending_movies()

    history = history[-MAX_SEQ_LEN:]

    valid_history = [h for h in history if h in CATALOG.pid_to_idx]

    if len(valid_history) < MIN_SEQ_LEN:
        return jsonify({"error": "Invalid history data"}), 400

    pids = torch.tensor([valid_history], dtype=torch.long, device=device)

    MODEL.eval()

    with torch.no_grad():

        scores = MODEL(pids)[0].cpu().numpy()

    ranked_idx = np.argsort(-scores)

    recs = []

    for idx in ranked_idx:

        pid = CATALOG.idx_to_pid[idx]

        if pid in history:
            continue

        movie = MOVIES_BY_ID.get(pid)

        if movie is None:
            continue

        recs.append({
            "MovieID": int(pid),
            "Title": movie["Title"],
            "Genres": movie["Genres"]
        })

        if len(recs) >= 20:
            break

    history_titles = [
        MOVIES_BY_ID[h]["Title"]
        for h in history
        if h in MOVIES_BY_ID
    ]

    if len(history_titles) >= 10:

        recs = rerank_with_llm(history_titles, recs)

    return jsonify(recs[:10])


# -------------------------------------------------------
# RANDOM MOVIES
# -------------------------------------------------------

@app.route("/movies", methods=["GET"])
def list_movies():

    sample = MOVIES_DF.sample(20)

    movies = []

    for _, row in sample.iterrows():

        movies.append({
            "MovieID": int(row["MovieID"]),
            "Title": row["Title"],
            "Genres": row["Genres"]
        })

    return jsonify(movies)


# -------------------------------------------------------
# TRENDING MOVIES
# -------------------------------------------------------

@app.route("/trending", methods=["GET"])
def trending_movies():

    top = (
        RATINGS_DF.groupby("MovieID")
        .size()
        .sort_values(ascending=False)
        .head(20)
        .index
    )

    movies = []

    for pid in top:

        movie = MOVIES_BY_ID.get(pid)

        if movie:

            movies.append({
                "MovieID": int(pid),
                "Title": movie["Title"],
                "Genres": movie["Genres"]
            })

    return jsonify(movies)


# -------------------------------------------------------
# TOP RATED MOVIES
# -------------------------------------------------------

@app.route("/toprated", methods=["GET"])
def top_rated():

    avg = (
        RATINGS_DF.groupby("MovieID")["Rating"]
        .mean()
        .sort_values(ascending=False)
        .head(20)
        .index
    )

    movies = []

    for pid in avg:

        movie = MOVIES_BY_ID.get(pid)

        if movie:

            movies.append({
                "MovieID": int(pid),
                "Title": movie["Title"],
                "Genres": movie["Genres"]
            })

    return jsonify(movies)


# -------------------------------------------------------
# SEARCH
# -------------------------------------------------------

@app.route("/search", methods=["GET"])
def search_movies():

    query = request.args.get("q", "").lower()

    if query == "":
        return jsonify([])

    results = MOVIES_DF[
        MOVIES_DF["Title"].str.lower().str.contains(query, na=False)
    ].head(20)

    movies = []

    for _, row in results.iterrows():

        movies.append({
            "MovieID": int(row["MovieID"]),
            "Title": row["Title"],
            "Genres": row["Genres"]
        })

    return jsonify(movies)


# -------------------------------------------------------
# TRAIN ENDPOINT
# -------------------------------------------------------

@app.route("/train", methods=["POST"])
def train():

    epochs = int(request.json.get("epochs", 5))

    train_model(epochs=epochs)

    return jsonify({"status": "training complete"})


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

if __name__ == "__main__":

    load_movielens_data()
    load_or_train_model()

    app.run(debug=True)
