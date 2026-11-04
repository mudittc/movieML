
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/movies_dataset")

metadata = pd.read_csv(
    DATA_PATH / "movies_metadata.csv",
    low_memory=False
)

keywords = pd.read_csv(DATA_PATH / "keywords.csv")
credits = pd.read_csv(DATA_PATH / "credits.csv")
links = pd.read_csv(DATA_PATH / "links.csv")


def inspect_ids(name, df, column):
    ids = pd.to_numeric(df[column], errors="coerce")

    print(f"\n--- {name} ---")
    print("Total rows:", len(df))
    print("Invalid IDs:", ids.isna().sum())
    print("Duplicate IDs:", ids.duplicated().sum())
    print("Unique valid IDs:", ids.nunique())


inspect_ids("MOVIES METADATA", metadata, "id")
inspect_ids("KEYWORDS", keywords, "id")
inspect_ids("CREDITS", credits, "id")
inspect_ids("LINKS - TMDB", links, "tmdbId")

print("\n--- LINK COVERAGE ---")

metadata_ids = set(
    pd.to_numeric(metadata["id"], errors="coerce").dropna()
)

keyword_ids = set(
    pd.to_numeric(keywords["id"], errors="coerce").dropna()
)

credit_ids = set(
    pd.to_numeric(credits["id"], errors="coerce").dropna()
)

link_ids = set(
    pd.to_numeric(links["tmdbId"], errors="coerce").dropna()
)

print("Metadata without keywords:",
      len(metadata_ids - keyword_ids))

print("Metadata without credits:",
      len(metadata_ids - credit_ids))

print("Metadata without links:",
      len(metadata_ids - link_ids))

print("\nValidation complete!")
