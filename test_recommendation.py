
from rec_engine import MovieRecommendationEngine

engine = MovieRecommendationEngine()

recommendations = engine.recommend(
    tmdb_id=27205,  # Inception
    selected_genres=[
        "Science Fiction",
        "Adventure",
        "Drama",
        "Action"
    ],
    top_n=15
)

print("\nRECOMMENDATIONS FOR INCEPTION")
print("=" * 65)

for i, movie in enumerate(recommendations, 1):

    print(f"\n{i}. {movie['title']}")

    print("Genres:", movie["genre_list"])

    print(
        "Content similarity:",
        round(movie["content_score"], 3)
    )

    print(
        "Genre compatibility:",
        round(movie["genre_score"], 3)
    )

    print(
        "Quality:",
        round(movie["quality_score"], 3)
    )

    print(
        "Final score:",
        round(movie["final_score"], 3)
    )
