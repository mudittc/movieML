
import pandas as pd

# ==========================================
# 1. LOAD MOVIES
# ==========================================

movies = pd.read_csv("data/movies.csv")

# ==========================================
# 2. COMBINE TAGS
# ==========================================

tags = pd.read_csv("data/tags.csv")

tags_grouped = tags.groupby("movieId")["tag"].apply(
    lambda x: " || ".join(
        x.dropna().astype(str).str.strip().unique()
    )
).reset_index()

tags_grouped.columns = ["movieId", "combined_tags"]

# ==========================================
# 3. AGGREGATE RATINGS IN CHUNKS
# ==========================================

rating_parts = []

for chunk in pd.read_csv(
    "data/ratings.csv",
    chunksize=500000
):

    partial = chunk.groupby("movieId")["rating"].agg(
        ["sum", "count"]
    )

    rating_parts.append(partial)

    print("Processed a ratings chunk...")

# Combine the partial aggregates
rating_stats = pd.concat(rating_parts).groupby(
    level=0
).sum()

rating_stats["avg_rating"] = (
    rating_stats["sum"] / rating_stats["count"]
)

rating_stats = rating_stats.reset_index()

rating_stats = rating_stats.rename(
    columns={"count": "rating_count"}
)

rating_stats = rating_stats[
    ["movieId", "avg_rating", "rating_count"]
]

# ==========================================
# 4. LOAD EXTERNAL IDS
# ==========================================

links = pd.read_csv("data/links.csv")

# ==========================================
# 5. MERGE EVERYTHING
# ==========================================

movie_metadata = movies.merge(
    tags_grouped,
    on="movieId",
    how="left"
)

movie_metadata = movie_metadata.merge(
    rating_stats,
    on="movieId",
    how="left"
)

movie_metadata = movie_metadata.merge(
    links,
    on="movieId",
    how="left"
)

# Handle missing values
movie_metadata["combined_tags"] = (
    movie_metadata["combined_tags"].fillna("")
)

movie_metadata["avg_rating"] = (
    movie_metadata["avg_rating"].fillna(0)
)

movie_metadata["rating_count"] = (
    movie_metadata["rating_count"].fillna(0).astype(int)
)

# ==========================================
# 6. SAVE ENRICHED DATASET
# ==========================================

movie_metadata.to_csv(
    "data/movie_metadata.csv",
    index=False
)

print("\nMetadata preparation complete!")
print("Total movies:", len(movie_metadata))
print(movie_metadata.head())
print(movie_metadata.columns)
