
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed")

# Load cleaned datasets
movies = pd.read_csv(
    DATA_PATH / "movies_metadata_clean.csv",
    low_memory=False
)
keywords = pd.read_csv(DATA_PATH / "keywords_clean.csv")
credits = pd.read_csv(DATA_PATH / "credits_clean.csv")

# Make sure the join keys have the same type
for df in [movies, keywords, credits]:
    df["id"] = pd.to_numeric(df["id"], errors="raise").astype("int64")

# Check that each dataset has unique movie IDs
assert movies["id"].is_unique, "Duplicate IDs in metadata"
assert keywords["id"].is_unique, "Duplicate IDs in keywords"
assert credits["id"].is_unique, "Duplicate IDs in credits"

# Merge keywords and credits into movie metadata
master = movies.merge(
    keywords,
    on="id",
    how="left"
)

master = master.merge(
    credits,
    on="id",
    how="left"
)

# Check the merged dataset
print("\nMASTER DATASET")
print("Total movies:", len(master))
print("Unique movie IDs:", master["id"].nunique())

print("\nMISSING DATA")
print("Missing keywords:", master["keywords"].isna().sum())
print("Missing cast:", master["cast"].isna().sum())
print("Missing crew:", master["crew"].isna().sum())
print("Missing overview:", master["overview"].isna().sum())
print("Missing genres:", master["genres"].isna().sum())

print("\nMOVIES WITH MISSING DETAILS")

missing_details = master[
    master["keywords"].isna()
    | master["cast"].isna()
    | master["crew"].isna()
]

print(
    missing_details[
        ["id", "title", "keywords", "cast", "crew"]
    ].to_string(index=False)
)

# Save the master dataset
master.to_csv(DATA_PATH / "movies_master.csv", index=False)

print("\nMaster dataset saved successfully!")
