import ast
import re
import joblib
import numpy as np
import pandas as pd
import time
import os

from pathlib import Path
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent

DATA_PATH = ROOT_DIR / "data" / "processed" / "movies_catalog.csv"
MODEL_DIR = ROOT_DIR / "data" / "models" / "tmdb"

CONTENT_WEIGHT = 0.75
GENRE_WEIGHT = 0.20
QUALITY_WEIGHT = 0.05

CANDIDATE_COUNT = 200
MIN_RELEASE_YEAR = 1990



# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

def parse_list(value):

    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    try:
        parsed = ast.literal_eval(str(value))

        if isinstance(parsed, list):
            return parsed

    except (ValueError, SyntaxError):
        pass

    return [
        item.strip()
        for item in str(value).split(",")
        if item.strip()
    ]


def load_movies():

    df = pd.read_csv(DATA_PATH, low_memory=False)

    df["genre_list"] = df["genre_list"].apply(parse_list)
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    df = df[df["release_year"] >= MIN_RELEASE_YEAR].reset_index(drop=True)

    return df


# --------------------------------------------------
# TEXT FEATURE ENGINEERING
# --------------------------------------------------

def clean_text(value):

    if pd.isna(value):
        return ""

    return re.sub(r"\s+", " ", str(value)).strip()




def build_features(df):

    print("Building movie features...", flush=True)

    text_columns = [
        "overview",
        "tagline",
        "keywords",
        "user_tags",
        "director",
        "cast_names",
        "collection_name"
    ]

    for column in text_columns:
        if column not in df.columns:
            df[column] = ""

        df[column] = df[column].apply(
            lambda x: " ".join(
                str(item) for item in x
            ) if isinstance(x, list)
            else "" if pd.isna(x)
            else str(x)
        )

    # Weighted feature construction
    df["combined_features"] = (
        (df["overview"] + " ") * 3 +
        (df["keywords"] + " ") * 3 +
        (df["user_tags"] + " ") * 2 +
        (df["tagline"] + " ") * 2 +
        (df["director"] + " ") * 2 +
        df["cast_names"] + " " +
        (df["collection_name"] + " ") * 2 +
        (" ".join(df["genre_list"].apply(
            lambda x: x if isinstance(x, list) else []
        ).explode().dropna().astype(str).tolist())
         if False else "")
    )

    # Append genres per movie
    genre_text = df["genre_list"].apply(
        lambda genres: " ".join(genres)
        if isinstance(genres, list)
        else ""
    )

    df["combined_features"] += " " + genre_text

    df["combined_features"] = (
        df["combined_features"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    print("Feature construction complete.", flush=True)

    return df




# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------


def train_model():

    start = time.time()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("STEP 1: Loading dataset...", flush=True)

    df = load_movies()

    print(f"Loaded {len(df)} movies.", flush=True)

    print("STEP 2: Building features...", flush=True)

    df = build_features(df)

    print("STEP 3: Initializing TF-IDF...", flush=True)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        dtype=np.float32
    )

    print("STEP 4: Fitting TF-IDF matrix...", flush=True)

    tfidf_matrix = vectorizer.fit_transform(
        df["combined_features"]
    )

    print("STEP 5: TF-IDF completed.", flush=True)
    print("Matrix shape:", tfidf_matrix.shape)
    print("Non-zero values:", tfidf_matrix.nnz)
    print(
        "Matrix memory:",
        round(
            (tfidf_matrix.data.nbytes +
             tfidf_matrix.indices.nbytes +
             tfidf_matrix.indptr.nbytes) / (1024 ** 2),
            2
        ),
        "MB",
        flush=True
    )

    print("STEP 6: Saving artifacts...", flush=True)

    save_npz(
        MODEL_DIR / "tmdb_tfidf_matrix.npz",
        tfidf_matrix
    )

    joblib.dump(
        vectorizer,
        MODEL_DIR / "tmdb_vectorizer.pkl"
    )

    df.to_pickle(
        MODEL_DIR / "tmdb_movies.pkl"
    )

    print("MODEL TRAINING COMPLETE!", flush=True)
    print("Time taken:", round(time.time() - start, 2), "seconds")
    print("Saved to:", MODEL_DIR)

    return df, tfidf_matrix


# --------------------------------------------------
# GENRE SCORING
# --------------------------------------------------

def calculate_genre_score(movie_genres, selected_genres):

    movie_genres = {
        str(g).strip().lower()
        for g in movie_genres
    }

    selected_genres = {
        str(g).strip().lower()
        for g in selected_genres
    }

    if not selected_genres:
        return 1.0

    matched = movie_genres.intersection(selected_genres)

    return len(matched) / len(selected_genres)


# --------------------------------------------------
# QUALITY SCORING
# --------------------------------------------------

def get_quality_score(movie, global_mean, min_votes):

    rating_column = (
        "rating" if "rating" in movie.index
        else "vote_average"
    )

    count_column = (
        "rating_count" if "rating_count" in movie.index
        else "vote_count"
    )

    if rating_column not in movie.index:
        return 0.5

    rating = pd.to_numeric(
        movie.get(rating_column), errors="coerce"
    )

    votes = pd.to_numeric(
        movie.get(count_column, 0), errors="coerce"
    )

    if pd.isna(rating):
        return 0.5

    if pd.isna(votes):
        votes = 0

    # Bayesian weighted rating
    weighted_rating = (
        (votes / (votes + min_votes)) * rating
        + (min_votes / (votes + min_votes)) * global_mean
    )

    # Normalize ratings from 0-10 to 0-1
    return float(np.clip(weighted_rating / 10, 0, 1))


# --------------------------------------------------
# RECOMMENDATION ENGINE
# --------------------------------------------------

class MovieRecommendationEngine:

    def __init__(self):

        self.df = None
        self.matrix = None
        self.neighbor_model = None

    def load_model(self):

        self.df = pd.read_pickle(
            MODEL_DIR / "tmdb_movies.pkl"
        )

        self.matrix = load_npz(
            MODEL_DIR / "tmdb_tfidf_matrix.npz"
        )

        self.neighbor_model = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_jobs=2
        )

        self.neighbor_model.fit(self.matrix)

        print("Recommendation engine loaded.", flush=True)

    
    def recommend(
        self,
        tmdb_id,
        selected_genres=None,
        top_n=15
    ):

        # ------------------------------------------
        # 1. LOAD MODEL IF NOT ALREADY LOADED
        # ------------------------------------------

        if self.df is None or self.matrix is None:
            self.load_model()

        if self.neighbor_model is None:
            self.neighbor_model = NearestNeighbors(
                metric="cosine",
                algorithm="brute",
                n_jobs=2
            )
            self.neighbor_model.fit(self.matrix)

        # ------------------------------------------
        # 2. FIND SELECTED MOVIE
        # ------------------------------------------

        movie_ids = pd.to_numeric(
            self.df["id"],
            errors="coerce"
        )

        movie_indices = self.df.index[
            movie_ids == int(tmdb_id)
        ].tolist()

        if not movie_indices:
            raise ValueError(
                f"Movie with TMDB ID {tmdb_id} not found"
            )

        movie_index = movie_indices[0]

        # By default, treat the selected film's genres as the preference.
        # Callers can pass an explicit list to personalize the ranking.
        if selected_genres is None:
            selected_genres = self.df.iloc[movie_index]["genre_list"]

        # ------------------------------------------
        # 3. RETRIEVE SIMILAR MOVIES
        # ------------------------------------------

        n_neighbors = min(
            CANDIDATE_COUNT + 1,
            self.matrix.shape[0]
        )

        distances, indices = self.neighbor_model.kneighbors(
            self.matrix[movie_index],
            n_neighbors=n_neighbors
        )

        candidate_indices = indices[0]
        candidate_scores = 1 - distances[0]

        # Exclude selected movie
        valid = candidate_indices != movie_index

        candidate_indices = candidate_indices[valid]
        candidate_scores = candidate_scores[valid]

        # ------------------------------------------
        # 4. CREATE CANDIDATE DATAFRAME
        # ------------------------------------------

        candidates = self.df.iloc[
            candidate_indices
        ].copy()

        candidates["content_score"] = candidate_scores

        # ------------------------------------------
        # 5. CALCULATE GENRE COMPATIBILITY
        # ------------------------------------------

        candidates["genre_score"] = candidates[
            "genre_list"
        ].apply(
            lambda genres: calculate_genre_score(
                genres,
                selected_genres
            )
        )

        # ------------------------------------------
        # 6. CALCULATE QUALITY SCORE
        # ------------------------------------------

        rating_col = (
            "rating"
            if "rating" in candidates.columns
            else "vote_average"
        )

        count_col = (
            "rating_count"
            if "rating_count" in candidates.columns
            else "vote_count"
        )

        if rating_col in self.df.columns:

            ratings = pd.to_numeric(
                self.df[rating_col],
                errors="coerce"
            )

            global_mean = ratings.mean()

            if pd.isna(global_mean):
                global_mean = 5.0

        else:
            global_mean = 5.0

        min_votes = 100

        candidates["quality_score"] = candidates.apply(
            lambda row: get_quality_score(
                row,
                global_mean,
                min_votes
            ),
            axis=1
        )

        # ------------------------------------------
        # 7. CALCULATE FINAL RANKING
        # ------------------------------------------

        candidates["final_score"] = (
            CONTENT_WEIGHT * candidates["content_score"]
            + GENRE_WEIGHT * candidates["genre_score"]
            + QUALITY_WEIGHT * candidates["quality_score"]
        )

        # ------------------------------------------
        # 8. SORT AND RETURN TOP RECOMMENDATIONS
        # ------------------------------------------

        candidates = candidates.sort_values(
            by="final_score",
            ascending=False
        )

        return candidates.head(top_n).to_dict(
            orient="records"
        )

