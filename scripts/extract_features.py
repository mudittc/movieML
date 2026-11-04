
import pandas as pd
import ast
from pathlib import Path

DATA_PATH = Path("data/processed")

# --------------------------------------------------
# 1. LOAD MASTER DATASET
# --------------------------------------------------

movies = pd.read_csv(
    DATA_PATH / "movies_master.csv",
    low_memory=False
)


# --------------------------------------------------
# 2. PARSE JSON-LIKE STRINGS
# --------------------------------------------------

def parse_data(value):
    """Convert a string representation of a Python structure into data."""

    if pd.isna(value) or value == "":
        return []

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


# --------------------------------------------------
# 3. EXTRACT GENRES
# --------------------------------------------------

def extract_genres(value):
    data = parse_data(value)

    return [
        item["name"]
        for item in data
        if isinstance(item, dict) and "name" in item
    ]


movies["genre_list"] = movies["genres"].apply(extract_genres)


# --------------------------------------------------
# 4. EXTRACT DIRECTOR
# --------------------------------------------------

def extract_director(value):
    crew = parse_data(value)

    return [
        person["name"]
        for person in crew
        if isinstance(person, dict)
        and person.get("job") == "Director"
    ]


movies["director"] = movies["crew"].apply(extract_director)


# --------------------------------------------------
# 5. EXTRACT CAST
# --------------------------------------------------

def extract_cast(value):
    cast = parse_data(value)

    # Keep the first 10 credited cast members
    return [
        person["name"]
        for person in cast[:10]
        if isinstance(person, dict) and "name" in person
    ]


movies["cast_names"] = movies["cast"].apply(extract_cast)


# --------------------------------------------------
# 6. EXTRACT COLLECTION
# --------------------------------------------------

def extract_collection(value):
    collection = parse_data(value)

    if isinstance(collection, dict):
        return collection.get("name")

    return None


movies["collection_name"] = movies[
    "belongs_to_collection"
].apply(extract_collection)


# --------------------------------------------------
# 7. EXTRACT RELEASE YEAR
# --------------------------------------------------

movies["release_year"] = pd.to_datetime(
    movies["release_date"],
    errors="coerce"
).dt.year


# --------------------------------------------------
# 8. EXTRACT LANGUAGE
# --------------------------------------------------

movies["language"] = movies["original_language"]


# --------------------------------------------------
# 9. EXTRACT RUNTIME
# --------------------------------------------------

movies["runtime"] = pd.to_numeric(
    movies["runtime"],
    errors="coerce"
)


# --------------------------------------------------
# 10. SAVE
# --------------------------------------------------

movies.to_csv(
    DATA_PATH / "movies_features.csv",
    index=False
)


# --------------------------------------------------
# 11. VALIDATION
# --------------------------------------------------

print("\nFEATURE EXTRACTION COMPLETE")

print("Total movies:", len(movies))

print("\nSAMPLE EXTRACTED FEATURES")

print(
    movies[
        [
            "title",
            "genre_list",
            "director",
            "cast_names",
            "collection_name",
            "release_year",
            "language",
            "runtime"
        ]
    ].head(10).to_string(index=False)
)

print("\nMISSING FEATURE COUNTS")

for column in [
    "genre_list",
    "director",
    "cast_names",
    "collection_name",
    "release_year",
    "language",
    "runtime"
]:
    print(column, ":", movies[column].isna().sum())
