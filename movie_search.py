"""
CineFind - Movie Search Engine Module
=======================================
TF-IDF vectorization and cosine-similarity based search & recommendation.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_preprocessing import preprocess_text


class MovieSearchEngine:
    """
    Information Retrieval engine built on TF-IDF and Cosine Similarity.

    The TF-IDF matrix is computed **once** at construction time and reused
    for every subsequent search / recommendation call.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Parameters
        ----------
        df : pd.DataFrame
            The preprocessed movie DataFrame produced by
            ``data_preprocessing.load_and_prepare_data()``.
        """
        self.df = df.copy().reset_index(drop=True)

        # Build TF-IDF matrix once
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df["processed_text"]
        )

    # ── search ──────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_n: int = 30,
        genre_filter: list | None = None,
        year_range: tuple | None = None,
        min_rating: float = 0.0,
        sort_by: str = "relevance",
    ) -> pd.DataFrame:
        """
        Search movies by a free-text query.

        Returns a DataFrame of matching movies sorted by the chosen
        criterion, with an added ``relevance_score`` column that always
        holds the genuine cosine-similarity value (filters never
        overwrite it).
        """
        cleaned = preprocess_text(query)
        if not cleaned.strip():
            return pd.DataFrame()

        query_vec = self.vectorizer.transform([cleaned])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Attach scores
        results = self.df.copy()
        results["relevance_score"] = scores

        # Keep only rows with similarity > 0
        results = results[results["relevance_score"] > 0]

        if results.empty:
            return results

        # Apply filters (these never touch relevance_score)
        if genre_filter:
            lf = [g.lower() for g in genre_filter]
            results = results[
                results["genres_list"].apply(
                    lambda gl: any(g.lower() in lf for g in gl)
                )
            ]
        if year_range:
            lo, hi = year_range
            results = results[
                (results["release_year"] >= lo) & (results["release_year"] <= hi)
            ]
        if min_rating > 0:
            results = results[results["vote_average"] >= min_rating]

        # Sorting
        if sort_by == "rating":
            results = results.sort_values(
                ["vote_average", "relevance_score"], ascending=[False, False]
            )
        elif sort_by == "year":
            results = results.sort_values(
                ["release_year", "relevance_score"], ascending=[False, False]
            )
        else:  # relevance (default)
            results = results.sort_values("relevance_score", ascending=False)

        return results.head(top_n).reset_index(drop=True)

    # ── similar movies ──────────────────────────────────────────────────

    def get_similar_movies(self, title: str, top_n: int = 5) -> pd.DataFrame:
        """
        Recommend movies most similar to the selected title.

        Returns up to ``top_n`` movies (excluding the query movie itself)
        with a ``similarity_score`` column showing the actual cosine
        similarity.
        """
        matches = self.df[self.df["title"].str.lower() == title.lower()]
        if matches.empty:
            return pd.DataFrame()

        idx = matches.index[0]
        sim_scores = cosine_similarity(
            self.tfidf_matrix[idx], self.tfidf_matrix
        ).flatten()

        results = self.df.copy()
        results["similarity_score"] = sim_scores

        # Drop the query movie itself
        results = results.drop(index=idx)

        # Drop zero-similarity results
        results = results[results["similarity_score"] > 0]

        results = results.sort_values("similarity_score", ascending=False)
        return results.head(top_n).reset_index(drop=True)

    # ── helpers ─────────────────────────────────────────────────────────

    def get_all_genres(self) -> list[str]:
        """Return a sorted list of unique genre names."""
        genres = set()
        for gl in self.df["genres_list"]:
            genres.update(gl)
        return sorted(genres)

    def get_year_range(self) -> tuple[int, int]:
        """Return (min_year, max_year) in the dataset."""
        valid = self.df[self.df["release_year"] > 0]["release_year"]
        return int(valid.min()), int(valid.max())

    def get_movie_details(self, title: str) -> dict | None:
        """Return a dict of details for a single movie, or None."""
        row = self.df[self.df["title"].str.lower() == title.lower()]
        if row.empty:
            return None
        r = row.iloc[0]
        return {
            "title": r["title"],
            "overview": r["overview"],
            "genres": r["genres_list"],
            "keywords": r["keywords_list"],
            "actors": r["actors"],
            "director": r["director"],
            "release_year": int(r["release_year"]),
            "vote_average": float(r["vote_average"]),
            "popularity": float(r["popularity"]),
            "vote_count": int(r["vote_count"]),
        }

    def get_all_titles(self) -> list[str]:
        """Sorted list of every movie title."""
        return sorted(self.df["title"].dropna().unique().tolist())
