# MovieMatch

MovieMatch recommends films released in 1990 or later from a TMDB-derived catalog. The current app uses the V2 engine in `rec_engine.py`: TF-IDF content similarity (75%), genre fit (20%), and Bayesian rating quality (5%). It returns 15 recommendations by default. The FastAPI app and browser UI use TMDB IDs throughout.

## Requirements

- Python 3.11 or later
- The MovieLens TMDB-style source files and processed movie data described below

Create and activate a virtual environment, then install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Prepare the data

The raw MovieLens/TMDB data files are not included in Git. Put `movies_metadata.csv`, `keywords.csv`, `credits.csv`, and `links.csv` in `data/movies_dataset/`. Then run the existing preparation pipeline from the project folder:

```bash
python scripts/clean_data.py
python scripts/build_master.py
python scripts/extract_features.py
python scripts/augment_movie_catalog.py
```

The catalog builder joins the local MovieLens 32M catalog and its user-applied tags to TMDB records through `links.csv`. It adds linked titles absent from the older TMDB feature file and writes `data/processed/movies_catalog.csv`. The app and recommender use that combined catalog; MovieLens tags contribute to content similarity.

## Build the recommendation model

Model artifacts are local and ignored by Git. Build them from the prepared feature file:

```bash
python -c "from rec_engine import train_model; train_model()"
```

This writes the model files under `data/models/tmdb/`. Rebuild after changing catalog contents or feature construction.

## Run MovieMatch

```bash
uvicorn app:app --reload
```

Open <http://127.0.0.1:8000/ui>. API documentation is at <http://127.0.0.1:8000/docs>.

The UI searches TMDB movie records, gets recommendations from the V2 engine, shows the score breakdown, and opens movie details. The API also accepts `tmdb_id` or an exact `title` for lookup and recommendations. Optional genre preferences can be passed to `/movies/recommend` as repeated `selected_genres` query parameters.

## API endpoints

- `GET /movies/search?query=inception`
- `GET /movies/lookup?tmdb_id=27205` or `GET /movies/lookup?title=Inception`
- `GET /movies/recommend?tmdb_id=27205` (15 results by default; set `top_n` to change it)
- `GET /movies/details?tmdb_id=27205`

## Project layout

```text
app.py                 FastAPI routes and TMDB catalog loading
rec_engine.py          TF-IDF recommendation and ranking engine
scripts/                Data cleaning and feature preparation
static/                 Browser UI behavior and styling
templates/              Browser UI page
data/                   Local datasets and generated models (not tracked)
```
