import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from scipy.sparse import load_npz
from sklearn.neighbors import NearestNeighbors
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles


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

movies = pd.read_csv("data/movies.csv")
movie_metadata = pd.read_csv("data/movie_metadata.csv")

# ==========================================
# 3. LOAD RECOMMENDATION MODEL
# ==========================================

tfidf_matrix = load_npz("data/tfidf_matrix.npz")
tfidf_feature_names = np.load("data/tfidf_feature_names.npy")

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
    ]

    results = matches[["movieId", "title"]].head(15)

    return results.to_dict(orient="records")

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

    # Identify the movie
    movie = find_movie(movie_id, title)

    if isinstance(movie, dict):
        return movie

    # Get the internal DataFrame index
    selected_index = movie.name

    # Retrieve its TF-IDF vector
    query_vector = tfidf_matrix[selected_index]

    if query_vector.nnz < 10:
        raise HTTPException(
            status_code=422,
            detail="This movie has insufficient descriptive features."
        )

    # Find similar movies
    distances, indices = model.kneighbors(query_vector)

    recommendations = []

    for i in range(len(indices[0])):

        recommended_index = valid_movie_indices[indices[0][i]]

        if recommended_index == selected_index:
            continue

        recommended_movie = movies.iloc[recommended_index]

        recommendations.append({
            "movieId": int(recommended_movie["movieId"]),
            "title": recommended_movie["title"],
            "similarity": float(round(1 - distances[0][i], 3))
        })

        if len(recommendations) == 10:
            break

    return {
        "selected_movie": {
            "movieId": int(movie["movieId"]),
            "title": movie["title"]
        },
        "recommendations": recommendations
    }


@app.get("/movies/details")
def movie_details(movie_id: int):

    matches = movie_metadata[
        movie_metadata["movieId"] == movie_id
    ]

    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail="Movie not found."
        )

    movie = matches.iloc[0]

    # Find the movie's original row index
    movie_index = movies.index[
        movies["movieId"] == movie_id
    ][0]

    # Extract its TF-IDF scores
    movie_vector = tfidf_matrix[movie_index].toarray().flatten()

    # Get the 10 highest-scoring features
    top_indices = movie_vector.argsort()[::-1][:10]

    keywords = [
        {
            "word": tfidf_feature_names[i],
            "score": round(float(movie_vector[i]), 3)
        }
        for i in top_indices
        if movie_vector[i] > 0
    ]

    return {
        "movieId": int(movie["movieId"]),
        "title": movie["title"],
        "genres": movie["genres"].split("|"),
        "tags": movie["combined_tags"].split(" || "),
        "avg_rating": float(movie["avg_rating"]),
        "rating_count": int(movie["rating_count"]),
        "imdbId": str(movie["imdbId"]),
        "tmdbId": str(movie["tmdbId"]),
        "keywords": keywords
    }
