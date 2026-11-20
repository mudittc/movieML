"""Merge the local MovieLens catalog and user tags into the TMDB features.

Run after scripts/extract_features.py. Output is a local generated catalog and
remains under the ignored data/ directory.
"""
from pathlib import Path
import re

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT_DIR / "data" / "processed" / "movies_features.csv"
MOVIES_PATH = ROOT_DIR / "data" / "movielens" / "movies.csv"
LINKS_PATH = ROOT_DIR / "data" / "movielens" / "links.csv"
TAGS_PATH = ROOT_DIR / "data" / "movielens" / "tags.csv"
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "movies_catalog.csv"
MIN_RELEASE_YEAR = 1990


def normalise_tags(values):
    # Deduplicate case-insensitively while retaining a readable tag spelling.
    values = values.dropna().astype(str).str.strip()
    values = values[values.ne("")]
    unique = {}
    for value in values:
        key = " ".join(value.split()).casefold()
        if key:
            unique.setdefault(key, " ".join(value.split()))
    return " ".join(unique[key] for key in sorted(unique))


def main():
    for path in (FEATURES_PATH, MOVIES_PATH, LINKS_PATH, TAGS_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required input file missing: {path}")

    print("Loading existing movie features...", flush=True)
    features = pd.read_csv(FEATURES_PATH, low_memory=False)
    features["id"] = pd.to_numeric(features["id"], errors="coerce")
    features = features.dropna(subset=["id"]).copy()
    features["id"] = features["id"].astype("int64")
    features = features.drop_duplicates(subset=["id"], keep="first")

    print("Loading MovieLens titles and TMDB links...", flush=True)
    ml_movies = pd.read_csv(MOVIES_PATH)
    ml_movies["release_year"] = pd.to_numeric(
        ml_movies["title"].str.extract(r"\((\d{4})\)$", expand=False),
        errors="coerce",
    )
    ml_movies = ml_movies.dropna(subset=["release_year"]).copy()
    ml_movies["release_year"] = ml_movies["release_year"].astype(int)
    ml_movies = ml_movies[ml_movies["release_year"] >= MIN_RELEASE_YEAR]
    ml_movies["clean_title"] = ml_movies["title"].str.replace(
        r"\s+\(\d{4}\)$", "", regex=True
    ).str.strip()

    links = pd.read_csv(LINKS_PATH, usecols=["movieId", "tmdbId"])
    links["movieId"] = pd.to_numeric(links["movieId"], errors="coerce")
    links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce")
    links = links.dropna(subset=["movieId", "tmdbId"]).copy()
    links["movieId"] = links["movieId"].astype("int64")
    links["tmdbId"] = links["tmdbId"].astype("int64")
    ambiguous = links.groupby("movieId")["tmdbId"].nunique()
    ambiguous_ids = ambiguous[ambiguous > 1].index
    links = links[~links["movieId"].isin(ambiguous_ids)]
    links = links.drop_duplicates(subset=["movieId"])

    linked = ml_movies.merge(links, on="movieId", how="inner")
    print(
        f"MovieLens titles after 1990 with TMDB IDs: {linked['tmdbId'].nunique():,}",
        flush=True,
    )

    print("Aggregating MovieLens user tags...", flush=True)
    tags = pd.read_csv(TAGS_PATH, usecols=["movieId", "tag"])
    tags["movieId"] = pd.to_numeric(tags["movieId"], errors="coerce")
    tags = tags.dropna(subset=["movieId", "tag"]).copy()
    tags["movieId"] = tags["movieId"].astype("int64")
    tags["tag"] = tags["tag"].astype(str).str.strip()
    tags = tags[tags["tag"].ne("")]
    tags = tags.merge(links, on="movieId", how="inner")
    tags_by_tmdb = tags.groupby("tmdbId")["tag"].agg(normalise_tags)

    features["user_tags"] = features["id"].map(tags_by_tmdb).fillna("")
    features["user_tags"] = features["user_tags"].astype(str)
    existing_ids = set(features["id"].tolist())

    records = []
    for tmdb_id, group in linked.groupby("tmdbId", sort=False):
        if int(tmdb_id) in existing_ids:
            continue
        genres = sorted({
            genre.strip()
            for value in group["genres"].dropna().astype(str)
            for genre in value.split("|")
            if genre.strip() and genre.strip() != "(no genres listed)"
        })
        first = group.iloc[0]
        record = {column: None for column in features.columns}
        record.update({
            "id": int(tmdb_id),
            "title": first["clean_title"],
            "overview": "",
            "tagline": "",
            "genres": "|".join(genres),
            "genre_list": genres,
            "keywords": "",
            "user_tags": tags_by_tmdb.get(int(tmdb_id), ""),
            "cast": "[]",
            "crew": "[]",
            "director": [],
            "cast_names": [],
            "collection_name": None,
            "release_year": int(group["release_year"].max()),
            "language": None,
            "runtime": None,
            "vote_average": None,
            "vote_count": None,
            "imdb_id": None,
            "homepage": None,
        })
        records.append(record)

    if records:
        additions = pd.DataFrame.from_records(records, columns=features.columns)
        features = pd.concat([features, additions], ignore_index=True)

    features["release_year"] = pd.to_numeric(
        features["release_year"], errors="coerce"
    )
    features = features.drop_duplicates(subset=["id"], keep="last")
    features.to_csv(OUTPUT_PATH, index=False)

    tagged_count = int(features["user_tags"].fillna("").astype(str).str.len().gt(0).sum())
    recent_count = int((features["release_year"] >= 2018).sum())
    print(f"Added MovieLens-only movies: {len(records):,}", flush=True)
    print(f"Movies with MovieLens tags: {tagged_count:,}", flush=True)
    print(f"Catalog movies from 2018 onward: {recent_count:,}", flush=True)
    print(f"Saved catalog: {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
