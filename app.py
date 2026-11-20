from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from rec_engine import MovieRecommendationEngine


ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "processed" / "movies_catalog.csv"

app = FastAPI(
    title="MovieMatch API",
    description="A content-based movie recommendation engine",
    version="2.0",
)
templates = Jinja2Templates(directory=str(ROOT_DIR / "templates"))
app.mount(
    "/static",
    StaticFiles(directory=str(ROOT_DIR / "static")),
    name="static",
)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Movie data not found: {DATA_PATH}. Follow the data preparation steps in README.md."
    )

tmdb_movies = pd.read_csv(DATA_PATH, low_memory=False)
tmdb_movies["id"] = pd.to_numeric(tmdb_movies["id"], errors="coerce")
tmdb_movies["release_year"] = pd.to_numeric(
    tmdb_movies["release_year"], errors="coerce"
)
tmdb_movies = tmdb_movies[
    tmdb_movies["release_year"] >= 1990
].reset_index(drop=True)
_engine: MovieRecommendationEngine | None = None


def parse_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return []
    if isinstance(value, dict):
        return [value]
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def get_engine() -> MovieRecommendationEngine:
    global _engine
    if _engine is None:
        try:
            _engine = MovieRecommendationEngine()
            _engine.load_model()
        except FileNotFoundError as exc:
            _engine = None
            raise HTTPException(
                status_code=503,
                detail=(
                    "The recommendation model is not built yet. Run "
                    "`python -c 'from rec_engine import train_model; train_model()'` "
                    "from the project folder, then restart the app."
                ),
            ) from exc
    return _engine


def find_movie(tmdb_id: int | None = None, title: str | None = None):
    if (tmdb_id is None) == (title is None):
        raise HTTPException(
            status_code=400,
            detail="Provide either tmdb_id or title, not both.",
        )

    if tmdb_id is not None:
        matches = tmdb_movies[tmdb_movies["id"] == int(tmdb_id)]
    else:
        normalized_title = title.strip().casefold()
        matches = tmdb_movies[
            tmdb_movies["title"].fillna("").str.strip().str.casefold()
            == normalized_title
        ]

    matches = matches.dropna(subset=["id"])
    if matches.empty:
        raise HTTPException(status_code=404, detail="Movie not found.")
    if len(matches) > 1:
        return {
            "message": "Multiple movies found. Please choose a movie by TMDB ID.",
            "matches": [
                {"tmdb_id": int(row.id), "title": row.title}
                for row in matches[["id", "title"]].itertuples(index=False)
            ],
        }
    return matches.iloc[0]


@app.get("/")
def home():
    return {"message": "Welcome to MovieMatch!", "status": "running"}


@app.get("/ui")
def home_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/movies/search")
def search_movies(query: str = Query(min_length=1, max_length=100)):
    matches = tmdb_movies[
        tmdb_movies["title"].fillna("").str.contains(
            query, case=False, na=False, regex=False
        )
    ].dropna(subset=["id"]).head(15)
    return [
        {"tmdb_id": int(row.id), "title": row.title}
        for row in matches[["id", "title"]].itertuples(index=False)
    ]


@app.get("/movies/lookup")
def lookup_movie(tmdb_id: int | None = None, title: str | None = None):
    movie = find_movie(tmdb_id, title)
    if isinstance(movie, dict):
        return movie
    return {"tmdb_id": int(movie["id"]), "title": movie["title"]}


@app.get("/movies/recommend")
def recommend_movie(
    tmdb_id: int | None = None,
    title: str | None = None,
    top_n: int = Query(default=15, ge=1, le=50),
    selected_genres: list[str] | None = Query(default=None),
):
    movie = find_movie(tmdb_id, title)
    if isinstance(movie, dict):
        return movie

    selected_id = int(movie["id"])
    engine = get_engine()
    try:
        recommendations = engine.recommend(
            tmdb_id=selected_id,
            selected_genres=selected_genres,
            top_n=top_n,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "selected_movie": {"tmdb_id": selected_id, "title": movie["title"]},
        "recommendations": [
            {
                "tmdb_id": int(result["id"]),
                "title": result["title"],
                "similarity": float(result["final_score"]),
                "final_score": float(result["final_score"]),
                "content_score": float(result["content_score"]),
                "genre_score": float(result["genre_score"]),
                "quality_score": float(result["quality_score"]),
            }
            for result in recommendations
        ],
    }


@app.get("/movies/details")
def movie_details(tmdb_id: int):
    matches = tmdb_movies[tmdb_movies["id"] == tmdb_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail="Movie not found.")

    movie = matches.iloc[0]
    imdb_id = movie.get("imdb_id")
    imdb_id = imdb_id if pd.notna(imdb_id) else None

    def optional_int(column: str):
        value = movie.get(column)
        return int(value) if pd.notna(value) else None

    def optional_float(column: str):
        value = movie.get(column)
        return float(value) if pd.notna(value) else None

    def optional_text(column: str):
        value = movie.get(column)
        return value if pd.notna(value) and str(value).strip() else None

    return {
        "tmdb_id": int(movie["id"]),
        "title": optional_text("title") or "Untitled",
        "overview": optional_text("overview"),
        "genres": parse_list(movie.get("genre_list")),
        "director": parse_list(movie.get("director")),
        "cast": parse_list(movie.get("cast_names")),
        "collection": optional_text("collection_name"),
        "release_year": optional_int("release_year"),
        "language": optional_text("language"),
        "runtime": optional_int("runtime"),
        "rating": optional_float("vote_average"),
        "rating_count": optional_int("vote_count") or 0,
        "imdb_id": imdb_id,
        "imdb_url": f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else None,
        "homepage": optional_text("homepage"),
    }
