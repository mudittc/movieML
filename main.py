
import pandas as pd
import re
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import save_npz


# ==========================================
# 1. LOAD DATA
# ==========================================

movies = pd.read_csv("data/movies.csv")
tags = pd.read_csv("data/tags.csv")


# ==========================================
# 2. TEXT NORMALIZATION
# ==========================================

synonym_map = {
    "science fiction": "science_fiction",
    "sci-fi": "science_fiction",
    "scifi": "science_fiction",
    "time travel": "time_travel",
    "time-travel": "time_travel",
    "black holes": "black_hole",
    "black hole": "black_hole",
    "space exploration": "space_exploration",
    "space-exploration": "space_exploration"
}

def normalize_text(text):

    text = str(text).lower()

    for original, replacement in synonym_map.items():

        pattern = r"(?<!\w)" + re.escape(original) + r"(?!\w)"

        text = re.sub(pattern, replacement, text)

    return text


# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================

tags_grouped = tags.groupby("movieId")["tag"].apply(
    lambda x: " ".join(
        x.dropna().astype(str).str.lower().unique()
    )
).reset_index()

tags_grouped.columns = ["movieId", "combined_tags"]

movie_data = movies.merge(
    tags_grouped,
    on="movieId",
    how="left"
)

movie_data["combined_tags"] = (
    movie_data["combined_tags"].fillna("")
)

movie_data["features"] = (
    movie_data["genres"].str.replace("|", " ", regex=False)
    + " "
    + movie_data["combined_tags"]
)

movie_data["features"] = movie_data["features"].apply(
    normalize_text
)


# ==========================================
# 4. TF-IDF
# ==========================================

custom_stopwords = {
    "nolan",
    "christopher",
    "oscar",
    "nominee",
    "good",
    "bad",
    "interesting",
    "great",
    "masterpiece",
    "beautiful",
    "pg",
    "imax",
    "rated"
}

all_stopwords = list(
    ENGLISH_STOP_WORDS.union(custom_stopwords)
)

tfidf = TfidfVectorizer(
    stop_words=all_stopwords,
    min_df=50,
    max_df=0.85,
    max_features=None,
    sublinear_tf=True,
    token_pattern=r"(?u)\b[^\W\d_][\w'-]+\b"
    )



tfidf_matrix = tfidf.fit_transform(
    movie_data["features"]
)
save_npz("data/tfidf_matrix.npz", tfidf_matrix)
np.save(
    "data/tfidf_feature_names.npy",
    tfidf.get_feature_names_out().astype(str)
)

# ==========================================
# 5. NEAREST NEIGHBOURS MODEL
# ==========================================
valid_indices = tfidf_matrix.getnnz(axis=1) >= 10

valid_matrix = tfidf_matrix[valid_indices]
valid_movie_indices = movie_data.index[valid_indices].to_numpy()

model = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=11
)

model.fit(valid_matrix)

print("Movies available for recommendation:", len(valid_indices.nonzero()[0]))


# ==========================================
# 6. RECOMMENDATION FUNCTION
# ==========================================

def get_recommendations(selected_index):

    query_vector = tfidf_matrix[selected_index]

    if query_vector.nnz < 10:
        return []

    distances, indices = model.kneighbors(query_vector)

    selected_features = tfidf_matrix[selected_index]
    feature_names = tfidf.get_feature_names_out()

    recommendations = []

    for i in range(len(indices[0])):

        recommended_index = valid_movie_indices[indices[0][i]]

        if recommended_index == selected_index:
            continue

        recommended_features = tfidf_matrix[recommended_index]

        common_features = selected_features.multiply(
            recommended_features
        )

        feature_scores = list(zip(
            feature_names,
            common_features.toarray().flatten()
        ))

        feature_scores.sort(
            key=lambda x: x[1],
            reverse=True
        )

        top_features = [
            word for word, score in feature_scores[:5]
            if score > 0
        ]

        recommendations.append({
            "title": movie_data.iloc[recommended_index]["title"],
            "similarity": float(round(1 - distances[0][i], 3)),
            "features": top_features
        })

        if len(recommendations) == 10:
            break

    return recommendations


ratings_preview = pd.read_csv(
    "data/ratings.csv",
    nrows=5
)
