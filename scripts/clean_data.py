
import pandas as pd
import json
from pathlib import Path

# --------------------------------------------------
# PATHS
# --------------------------------------------------

RAW_PATH = Path("data/movies_dataset")
OUTPUT_PATH = Path("data/processed")

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def normalize_id(df, column):
    """Convert IDs to numeric values and remove invalid IDs."""
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[column]).copy()
    df[column] = df[column].astype("int64")
    return df


def parse_json(value):
    """Safely parse JSON-like strings from the dataset."""
    if pd.isna(value) or value == "":
        return []

    try:
        return json.loads(value.replace("'", '"'))
    except (ValueError, AttributeError):
        try:
            import ast
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []


# --------------------------------------------------
# 1. MOVIE METADATA
# --------------------------------------------------

print("Loading movie metadata...")

metadata = pd.read_csv(
    RAW_PATH / "movies_metadata.csv",
    low_memory=False
)

metadata = normalize_id(metadata, "id")

# Remove exact duplicate rows
metadata = metadata.drop_duplicates()

# Prefer the most complete record for conflicting IDs
metadata["_completeness"] = metadata.notna().sum(axis=1)

metadata = (
    metadata.sort_values("_completeness", ascending=False)
    .drop_duplicates(subset="id", keep="first")
    .drop(columns="_completeness")
)

metadata = metadata.reset_index(drop=True)

print("Clean metadata records:", len(metadata))


# --------------------------------------------------
# 2. KEYWORDS
# --------------------------------------------------

print("\nLoading keywords...")

keywords = pd.read_csv(RAW_PATH / "keywords.csv")

keywords = normalize_id(keywords, "id")
keywords = keywords.drop_duplicates()

def combine_keywords(series):
    combined = {}

    for value in series:
        for item in parse_json(value):
            if isinstance(item, dict):
                keyword_id = item.get("id")
                name = item.get("name")

                if name:
                    combined[keyword_id or name.lower()] = name

    return list(combined.values())


keywords = (
    keywords.groupby("id")["keywords"]
    .apply(combine_keywords)
    .reset_index()
)

print("Clean keyword records:", len(keywords))


# --------------------------------------------------
# 3. CREDITS
# --------------------------------------------------

print("\nLoading credits...")

credits = pd.read_csv(RAW_PATH / "credits.csv")

credits = normalize_id(credits, "id")
credits = credits.drop_duplicates()


def combine_credits(series, identifier):
    combined = {}

    for value in series:
        for person in parse_json(value):
            if not isinstance(person, dict):
                continue

            person_id = person.get("credit_id") or person.get("id")

            if person_id is not None:
                combined[person_id] = person

    return list(combined.values())


credits_grouped = []

for movie_id, group in credits.groupby("id"):
    cast = combine_credits(group["cast"], "cast")
    crew = combine_credits(group["crew"], "crew")

    credits_grouped.append({
        "id": movie_id,
        "cast": cast,
        "crew": crew
    })

credits = pd.DataFrame(credits_grouped)

print("Clean credits records:", len(credits))


# --------------------------------------------------
# 4. LINKS
# --------------------------------------------------

print("\nLoading links...")

links = pd.read_csv(RAW_PATH / "links.csv")

links["tmdbId"] = pd.to_numeric(
    links["tmdbId"], errors="coerce"
)

links["movieId"] = pd.to_numeric(
    links["movieId"], errors="coerce"
)

links = links.dropna(subset=["movieId"]).copy()
links["movieId"] = links["movieId"].astype("int64")

# Keep multiple MovieLens IDs mapping to the same TMDB ID.
links = links.drop_duplicates()

print("Clean link records:", len(links))


# --------------------------------------------------
# 5. SAVE CLEANED DATA
# --------------------------------------------------

metadata.to_csv(
    OUTPUT_PATH / "movies_metadata_clean.csv",
    index=False
)

keywords.to_csv(
    OUTPUT_PATH / "keywords_clean.csv",
    index=False
)

credits.to_csv(
    OUTPUT_PATH / "credits_clean.csv",
    index=False
)

links.to_csv(
    OUTPUT_PATH / "links_clean.csv",
    index=False
)

print("\nCleaning complete!")
print("Output directory:", OUTPUT_PATH)
