"""
CineFind - Data Preprocessing Module
=====================================
Handles loading, cleaning, parsing, and preparing the TMDB 5000 movie dataset
for the Information Retrieval engine.
"""

import ast
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# Ensure required NLTK data is available
# ---------------------------------------------------------------------------
for resource in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)


# ── helpers for parsing JSON‑like columns ──────────────────────────────────

def _safe_literal_eval(val):
    """Safely parse a stringified Python literal (list of dicts, etc.)."""
    if pd.isna(val) or val == "":
        return []
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []


def _extract_names(data, max_items=None):
    """Return a list of 'name' values from a list of dicts."""
    items = _safe_literal_eval(data) if isinstance(data, str) else data
    if not isinstance(items, list):
        return []
    names = [d["name"] for d in items if isinstance(d, dict) and "name" in d]
    if max_items:
        names = names[:max_items]
    return names


def _extract_director(crew_data):
    """Extract the director's name from the crew JSON column."""
    crew = _safe_literal_eval(crew_data) if isinstance(crew_data, str) else crew_data
    if not isinstance(crew, list):
        return ""
    for member in crew:
        if isinstance(member, dict) and member.get("job") == "Director":
            return member.get("name", "")
    return ""


def _extract_top_actors(cast_data, top_n=3):
    """Return the top‑N actor names from the cast JSON column."""
    cast = _safe_literal_eval(cast_data) if isinstance(cast_data, str) else cast_data
    if not isinstance(cast, list):
        return []
    return [m["name"] for m in cast[:top_n] if isinstance(m, dict) and "name" in m]


# ── text preprocessing ─────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Lowercase → tokenize → remove stopwords & punctuation → rejoin.
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)


# ── main loading pipeline ──────────────────────────────────────────────────

def load_and_prepare_data(movies_path: str = "data/tmdb_5000_movies.csv",
                          credits_path: str = "data/tmdb_5000_credits.csv") -> pd.DataFrame:
    """
    Load, merge, clean and feature‑engineer the TMDB dataset.

    Returns a DataFrame with columns ready for the IR engine:
        title, genres, keywords, overview, actors, director,
        release_year, vote_average, popularity, combined_text,
        processed_text
    """

    # 1. Load raw CSVs
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    # 2. Merge on movie id
    df = movies.merge(credits, left_on="id", right_on="movie_id", how="inner",
                      suffixes=("", "_credits"))

    # 3. Keep the movies.title (drop credits duplicate if present)
    if "title_credits" in df.columns:
        df.drop(columns=["title_credits"], inplace=True)

    # 4. Handle missing values
    df["overview"] = df["overview"].fillna("")
    df["release_date"] = df["release_date"].fillna("")
    df["genres"] = df["genres"].fillna("[]")
    df["keywords"] = df["keywords"].fillna("[]")
    df["cast"] = df["cast"].fillna("[]")
    df["crew"] = df["crew"].fillna("[]")

    # 5. Drop exact duplicates on movie id
    df.drop_duplicates(subset="id", keep="first", inplace=True)

    # 6. Extract structured fields
    df["genres_list"] = df["genres"].apply(lambda x: _extract_names(x))
    df["keywords_list"] = df["keywords"].apply(lambda x: _extract_names(x))
    df["actors"] = df["cast"].apply(lambda x: _extract_top_actors(x, top_n=3))
    df["director"] = df["crew"].apply(_extract_director)

    # Readable string versions
    df["genres_str"] = df["genres_list"].apply(lambda g: " ".join(g))
    df["keywords_str"] = df["keywords_list"].apply(lambda k: " ".join(k))
    df["actors_str"] = df["actors"].apply(lambda a: " ".join(a))

    # 7. Release year
    df["release_year"] = pd.to_numeric(
        df["release_date"].str[:4], errors="coerce"
    ).fillna(0).astype(int)

    # 8. Build combined searchable text
    df["combined_text"] = (
        df["title"].fillna("") + " " +
        df["genres_str"] + " " +
        df["keywords_str"] + " " +
        df["overview"].fillna("") + " " +
        df["actors_str"] + " " +
        df["director"].fillna("")
    )

    # 9. Preprocessed (cleaned) version for TF‑IDF
    df["processed_text"] = df["combined_text"].apply(preprocess_text)

    # 10. Keep only useful columns (plus id for keying)
    keep_cols = [
        "id", "title", "genres_list", "genres_str",
        "keywords_list", "keywords_str",
        "overview", "actors", "actors_str",
        "director", "release_year",
        "vote_average", "vote_count", "popularity",
        "combined_text", "processed_text",
    ]
    df = df[[c for c in keep_cols if c in df.columns]].reset_index(drop=True)

    return df
