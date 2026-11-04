
const searchInput = document.getElementById("movie-search");
const searchResults = document.getElementById("search-results");
const selectedMovieText = document.getElementById("selected-movie");

let selectedMovie = null;

searchInput.addEventListener("input", async function () {

    const query = searchInput.value.trim();

    selectedMovie = null;
    selectedMovieText.textContent = "";
    searchResults.innerHTML = "";

    if (query.length < 2) {
        searchResults.style.display = "none";
        return;
    }

    try {

        const response = await fetch(
            `/movies/search?query=${encodeURIComponent(query)}`
        );

        const movies = await response.json();

        searchResults.innerHTML = "";

        movies.forEach(movie => {

            const item = document.createElement("div");

            item.className = "search-result";
            item.textContent = movie.title;

            item.addEventListener("click", function () {

                selectedMovie = movie;

                searchInput.value = movie.title;

                selectedMovieText.textContent =
                    `Selected: ${movie.title}`;

                searchResults.style.display = "none";
            });

            searchResults.appendChild(item);
        });

        searchResults.style.display =
            movies.length > 0 ? "block" : "none";

    } catch (error) {
        console.error("Movie search failed:", error);
    }
});


const recommendButton = document.getElementById("recommend-button");
const recommendationsContainer = document.getElementById("recommendations");
const recommendationStatus = document.getElementById("recommendation-status");

recommendButton.addEventListener("click", async function () {

    if (!selectedMovie) {
        recommendationStatus.textContent = "Please select a movie first.";
        return;
    }

    recommendationStatus.textContent = "Finding similar movies...";
    recommendationsContainer.innerHTML = "";

    try {

        const response = await fetch(
            `/movies/recommend?movie_id=${selectedMovie.movieId}`
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Something went wrong.");
        }

        recommendationStatus.textContent =
            `Movies similar to ${data.selected_movie.title}`;

        data.recommendations.forEach(movie => {

            const card = document.createElement("div");
            card.className = "movie-card";

            const title = document.createElement("h3");
            title.textContent = movie.title;

            const score = document.createElement("p");
            score.textContent =
                `${(movie.similarity * 100).toFixed(1)}% similarity`;

            card.appendChild(title);
            card.appendChild(score);

            recommendationsContainer.appendChild(card);
        });

    } catch (error) {

        recommendationStatus.textContent = error.message;
        console.error(error);

    }
});
