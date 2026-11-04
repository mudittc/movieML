import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from scipy.sparse import load_npz
from sklearn.neighbors import NearestNeighbors
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import ast


# ==========================================
# 1. INITIALIZE FASTAPI
# ==========================================

app = FastAPI(
    title="MovieMatch API",
    description="A content-based movie recommendation engine",
    version="1.0"
)
templates = Jinja2Templates(directory="templates")

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# ==========================================
# 2. LOAD DATASETS
# ==========================================

movies = pd.read_csv("data/movielens/movies.csv")
movie_metadata = pd.read_csv("data/movielens/movie_metadata.csv")
# New TMDB dataset
tmdb_movies = pd.read_csv(
    "data/processed/movies_features.csv",
    low_memory=False
)

def parse_list(value):
    if pd.isna(value):
        return []

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []

links = pd.read_csv(
    "data/processed/links_clean.csv",
    low_memory=False
)

links["movieId"] = pd.to_numeric(
    links["movieId"], errors="coerce"
)

links["tmdbId"] = pd.to_numeric(
    links["tmdbId"], errors="coerce"
)

links = links.dropna(subset=["movieId"]).copy()
links["movieId"] = links["movieId"].astype("int64")

# Build MovieLens ID -> TMDB ID mapping

valid_links = links.dropna(subset=["tmdbId"]).copy()

# Identify MovieLens IDs with conflicting TMDB mappings
mapping_counts = valid_links.groupby("movieId")["tmdbId"].nunique()

ambiguous_ids = mapping_counts[mapping_counts > 1].index

valid_links = valid_links[
    ~valid_links["movieId"].isin(ambiguous_ids)
]

movie_id_to_tmdb = (
    valid_links.drop_duplicates(subset="movieId")
    .set_index("movieId")["tmdbId"]
    .astype("int64")
    .to_dict()
)

print("MovieLens-to-TMDB mappings:", len(movie_id_to_tmdb))
print("Ambiguous mappings excluded:", len(ambiguous_ids))
    
# ==========================================
# 3. LOAD RECOMMENDATION MODEL
# ==========================================

tfidf_matrix = load_npz("data/models/tfidf_matrix.npz")
tfidf_feature_names = np.load("data/models/tfidf_feature_names.npy")

valid_indices = tfidf_matrix.getnnz(axis=1) >= 10

valid_matrix = tfidf_matrix[valid_indices]

valid_movie_indices = movies.index[valid_indices].to_numpy()

model = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=11
)

model.fit(valid_matrix)

print("Recommendation model loaded!")

#MOVIE LOOKUP


def find_movie(movie_id=None, title=None):

    # Exactly one reference must be provided
    if (movie_id is None) == (title is None):
        raise HTTPException(
            status_code=400,
            detail="Provide either movie_id or title, not both."
        )

    # Search using MovieLens ID
    if movie_id is not None:

        matches = movies[movies["movieId"] == movie_id]

    # Search using movie title
    else:

        matches = movies[
            movies["title"].str.lower() == title.strip().lower()
        ]

    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail="Movie not found."
        )

    # Handle duplicate movie titles
    if len(matches) > 1:
        return {
            "message": "Multiple movies found. Please use movie_id.",
            "matches": matches[["movieId", "title"]].to_dict(
                orient="records"
            )
        }

    # Return the movie's DataFrame row

    movie_index = movies.index[
    movies["movieId"] == movie_id
    ][0]

    movie_vector = tfidf_matrix[movie_index].toarray().flatten()

    top_indices = movie_vector.argsort()[::-1][:10]

    keywords = [
        {
            "word": tfidf_feature_names[i],
            "score": round(float(movie_vector[i]), 3)
        }
        for i in top_indices
        if movie_vector[i] > 0
    ]
    return matches.iloc[0]



# ==========================================
# 4. API ENDPOINTS
# ==========================================

@app.get("/")
def home():
    return {
        "message": "Welcome to MovieMatch!",
        "status": "running"
    }

@app.get("/ui")
def home_page(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.get("/movies/search")
def search_movies(query: str):

    matches = movies[
        movies["title"].str.contains(
            query,
            case=False,
            na=False,
            regex=False
        )
    ].head(15)

    results = []

    for _, movie in matches.iterrows():

        movie_id = int(movie["movieId"])
        tmdb_id = movie_id_to_tmdb.get(movie_id)

        results.append({
            "movieId": movie_id,
            "tmdb_id": tmdb_id,
            "title": movie["title"]
        })

    return results

@app.get("/movies/lookup")
def lookup_movie(movie_id: int = None, title: str = None):

    movie = find_movie(movie_id, title)

    if isinstance(movie, dict):
        return movie

    return {
        "movieId": int(movie["movieId"]),
        "title": movie["title"]
    }


@app.get("/movies/recommend")
def recommend_movie(movie_id: int = None, title: str = None):

    # Identify the selected movie
    movie = find_movie(movie_id, title)

    if isinstance(movie, dict):
        return movie

    selected_index = movie.name
    selected_movie_id = int(movie["movieId"])

    query_vector = tfidf_matrix[selected_index]

    if query_vector.nnz < 10:
        raise HTTPException(
            status_code=422,
            detail="This movie has insufficient descriptive features."
        )

    distances, indices = model.kneighbors(query_vector)

    recommendations = []

    for i in range(len(indices[0])):

        recommended_index = valid_movie_indices[indices[0][i]]

        if recommended_index == selected_index:
            continue

        recommended_movie = movies.iloc[recommended_index]
        recommended_movie_id = int(recommended_movie["movieId"])

        recommendations.append({
            "movieId": recommended_movie_id,
            "tmdb_id": movie_id_to_tmdb.get(recommended_movie_id),
            "title": recommended_movie["title"],
            "similarity": float(round(1 - distances[0][i], 3))
        })

        if len(recommendations) == 10:
            break

    return {
        "selected_movie": {
            "movieId": selected_movie_id,
            "tmdb_id": movie_id_to_tmdb.get(selected_movie_id),
            "title": movie["title"]
        },
        "recommendations": recommendations
    }


@app.get("/movies/details")
def movie_details(tmdb_id: int):

    # Find movie using TMDB ID
    matches = tmdb_movies[
        tmdb_movies["id"] == tmdb_id
    ]

    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail="Movie not found."
        )

    movie = matches.iloc[0]

    # Extract IMDb ID
    imdb_id = movie["imdb_id"]

    # Build response
    return {
        "tmdb_id": int(movie["id"]),
        "title": movie["title"],
        "overview": movie["overview"],

        "genres": parse_list(movie["genre_list"]),
        "director": parse_list(movie["director"]),
        "cast": parse_list(movie["cast_names"]),

        "collection": (
            movie["collection_name"]
            if pd.notna(movie["collection_name"])
            else None
        ),

        "release_year": (
            int(movie["release_year"])
            if pd.notna(movie["release_year"])
            else None
        ),

        "language": movie["language"],

        "runtime": (
            int(movie["runtime"])
            if pd.notna(movie["runtime"])
            else None
        ),

        "rating": (
            float(movie["vote_average"])
            if pd.notna(movie["vote_average"])
            else None
        ),

        "rating_count": (
            int(movie["vote_count"])
            if pd.notna(movie["vote_count"])
            else 0
        ),

        "imdb_id": (
            imdb_id if pd.notna(imdb_id) else None
        ),

        "imdb_url": (
            f"https://www.imdb.com/title/{imdb_id}/"
            if pd.notna(imdb_id) else None
        ),

        "homepage": (
            movie["homepage"]
            if pd.notna(movie["homepage"])
            else None
        )
    }

