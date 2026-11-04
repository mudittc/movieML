
const searchInput = document.getElementById("movie-search");
const searchResults = document.getElementById("search-results");
const selectedMovieText = document.getElementById("selected-movie");

const recommendButton = document.getElementById("recommend-button");
const recommendationsContainer = document.getElementById("recommendations");
const recommendationStatus = document.getElementById("recommendation-status");

const movieModal = document.getElementById("movie-modal");
const closeModalButton = document.getElementById("close-modal");

let selectedMovie = null;


// ==========================================
// 1. MOVIE SEARCH
// ==========================================

searchInput.addEventListener("input", async function () {

    const query = searchInput.value.trim();

    selectedMovie = null;
    selectedMovieText.textContent = "";
    selectedMovieText.onclick = null;
    searchResults.innerHTML = "";

    if (query.length < 2) {
        searchResults.style.display = "none";
        return;
    }

    try {

        const response = await fetch(
            `/movies/search?query=${encodeURIComponent(query)}`
        );

        if (!response.ok) {
            throw new Error("Movie search failed.");
        }

        const movies = await response.json();

        movies.forEach(movie => {

            const item = document.createElement("div");

            item.className = "search-result";
            item.textContent = movie.title;

            item.addEventListener("click", function () {

                selectedMovie = movie;

                searchInput.value = movie.title;

                selectedMovieText.textContent =
                    `Selected: ${movie.title} — View details`;

                selectedMovieText.style.cursor = "pointer";

                selectedMovieText.onclick = function () {
                    openMovieDetails(selectedMovie.tmdb_id);
                };

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


// ==========================================
// 2. RECOMMENDATIONS
// ==========================================

recommendButton.addEventListener("click", async function () {

    if (!selectedMovie) {
        recommendationStatus.textContent =
            "Please select a movie first.";
        return;
    }

    recommendationStatus.textContent =
        "Finding similar movies...";

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
            card.style.cursor = "pointer";

            card.addEventListener("click", function () {
                openMovieDetails(movie.tmdb_id);
            });

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


// ==========================================
// 3. OPEN MOVIE DETAILS
// ==========================================

async function openMovieDetails(tmdbId) {

    if (!tmdbId) {
        console.error("This movie does not have a TMDB ID.");
        alert("Details are unavailable for this movie.");
        return;
    }

    try {

        const response = await fetch(
            `/movies/details?tmdb_id=${encodeURIComponent(tmdbId)}`
        );

        const movie = await response.json();

        if (!response.ok) {
            throw new Error(movie.detail || "Could not load movie details.");
        }

        document.getElementById("modal-title").textContent =
            movie.title || "Untitled";

        const metadata = [
            movie.release_year,
            movie.language,
            movie.runtime ? `${movie.runtime} min` : null
        ].filter(Boolean);

        document.getElementById("modal-meta").textContent =
            metadata.join(" • ");

        document.getElementById("modal-rating").textContent =
            movie.rating != null
                ? `★ ${movie.rating.toFixed(1)} / 10 (${movie.rating_count} votes)`
                : "Rating unavailable";

        document.getElementById("modal-overview").textContent =
            movie.overview || "No overview available.";

        renderModalTags("modal-genres", movie.genres);
        renderModalTags("modal-cast", movie.cast);

        document.getElementById("modal-director").textContent =
            movie.director?.length
                ? movie.director.join(", ")
                : "Not available";

        document.getElementById("modal-collection").textContent =
            movie.collection || "No collection information available.";

        const imdbLink = document.getElementById("modal-imdb");

        if (movie.imdb_url) {
            imdbLink.href = movie.imdb_url;
            imdbLink.style.display = "inline-block";
        } else {
            imdbLink.style.display = "none";
        }

        movieModal.classList.add("active");

    } catch (error) {

        console.error("Movie details failed:", error);
        alert(error.message);

    }
}


// ==========================================
// 4. RENDER GENRES AND CAST
// ==========================================

function renderModalTags(elementId, items) {

    const container = document.getElementById(elementId);
    container.innerHTML = "";

    if (!items || items.length === 0) {
        container.textContent = "Not available";
        return;
    }

    items.forEach(item => {

        const tag = document.createElement("span");

        tag.className = "modal-tag";
        tag.textContent = item;

        container.appendChild(tag);
    });
}


// ==========================================
// 5. CLOSE MODAL
// ==========================================

closeModalButton.addEventListener("click", function () {
    movieModal.classList.remove("active");
});

movieModal.addEventListener("click", function (event) {

    if (event.target === movieModal) {
        movieModal.classList.remove("active");
    }

});

document.addEventListener("keydown", function (event) {

    if (event.key === "Escape") {
        movieModal.classList.remove("active");
    }

});


console.log("MovieMatch JavaScript loaded successfully.");
console.log("Modal element:", movieModal);
