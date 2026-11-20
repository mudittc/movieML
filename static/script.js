const searchInput = document.getElementById("movie-search");
const searchResults = document.getElementById("search-results");
const selectedMovieText = document.getElementById("selected-movie");
const recommendButton = document.getElementById("recommend-button");
const recommendationsContainer = document.getElementById("recommendations");
const recommendationStatus = document.getElementById("recommendation-status");
const movieModal = document.getElementById("movie-modal");
const closeModalButton = document.getElementById("close-modal");

let selectedMovie = null;
let searchTimer = null;

searchInput.addEventListener("input", function () {
    const query = searchInput.value.trim();
    selectedMovie = null;
    selectedMovieText.textContent = "";
    selectedMovieText.onclick = null;
    searchResults.replaceChildren();
    clearTimeout(searchTimer);

    if (query.length < 2) {
        searchResults.style.display = "none";
        return;
    }

    searchTimer = setTimeout(async () => {
        try {
            const response = await fetch(`/movies/search?query=${encodeURIComponent(query)}`);
            if (!response.ok) throw new Error("Movie search failed.");
            const movies = await response.json();
            searchResults.replaceChildren();

            movies.forEach(movie => {
                const item = document.createElement("button");
                item.type = "button";
                item.className = "search-result";
                item.textContent = movie.title;
                item.addEventListener("click", () => {
                    selectedMovie = movie;
                    searchInput.value = movie.title;
                    selectedMovieText.textContent = `Selected: ${movie.title} — View details`;
                    selectedMovieText.style.cursor = "pointer";
                    selectedMovieText.onclick = () => openMovieDetails(selectedMovie.tmdb_id);
                    searchResults.style.display = "none";
                });
                searchResults.appendChild(item);
            });
            searchResults.style.display = movies.length ? "block" : "none";
        } catch (error) {
            console.error("Movie search failed:", error);
        }
    }, 180);
});

recommendButton.addEventListener("click", async function () {
    if (!selectedMovie) {
        recommendationStatus.textContent = "Please select a movie first.";
        return;
    }

    recommendationStatus.textContent = "Finding similar movies...";
    recommendationsContainer.replaceChildren();
    recommendButton.disabled = true;

    try {
        const response = await fetch(`/movies/recommend?tmdb_id=${encodeURIComponent(selectedMovie.tmdb_id)}`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Something went wrong.");

        recommendationStatus.textContent = `Movies similar to ${data.selected_movie.title}`;
        data.recommendations.forEach(movie => {
            const card = document.createElement("div");
            card.className = "movie-card";
            card.style.cursor = "pointer";
            card.addEventListener("click", () => openMovieDetails(movie.tmdb_id));

            const title = document.createElement("h3");
            title.textContent = movie.title;
            const score = document.createElement("p");
            score.textContent = `Match score: ${(movie.final_score * 100).toFixed(1)}%`;
            const breakdown = document.createElement("p");
            breakdown.className = "match-breakdown";
            breakdown.textContent = `Content ${Math.round(movie.content_score * 100)}% · Genre fit ${Math.round(movie.genre_score * 100)}% · Quality ${Math.round(movie.quality_score * 100)}%`;
            card.append(title, score, breakdown);
            recommendationsContainer.appendChild(card);
        });
        if (!data.recommendations.length) recommendationStatus.textContent = "No recommendations found for this movie.";
    } catch (error) {
        recommendationStatus.textContent = error.message;
        console.error(error);
    } finally {
        recommendButton.disabled = false;
    }
});

async function openMovieDetails(tmdbId) {
    if (!tmdbId) {
        alert("Details are unavailable for this movie.");
        return;
    }

    try {
        const response = await fetch(`/movies/details?tmdb_id=${encodeURIComponent(tmdbId)}`);
        const movie = await response.json();
        if (!response.ok) throw new Error(movie.detail || "Could not load movie details.");

        document.getElementById("modal-title").textContent = movie.title || "Untitled";
        document.getElementById("modal-meta").textContent = [
            movie.release_year,
            movie.language,
            movie.runtime ? `${movie.runtime} min` : null,
        ].filter(Boolean).join(" • ");
        document.getElementById("modal-rating").textContent = movie.rating != null
            ? `★ ${movie.rating.toFixed(1)} / 10 (${movie.rating_count} votes)`
            : "Rating unavailable";
        document.getElementById("modal-overview").textContent = movie.overview || "No overview available.";
        renderModalTags("modal-genres", movie.genres);
        renderModalTags("modal-cast", movie.cast);
        document.getElementById("modal-director").textContent = movie.director?.length
            ? movie.director.join(", ") : "Not available";
        document.getElementById("modal-collection").textContent = movie.collection || "No collection information available.";

        const imdbLink = document.getElementById("modal-imdb");
        imdbLink.style.display = movie.imdb_url ? "inline-block" : "none";
        if (movie.imdb_url) imdbLink.href = movie.imdb_url;
        movieModal.classList.add("active");
    } catch (error) {
        console.error("Movie details failed:", error);
        alert(error.message);
    }
}

function renderModalTags(elementId, items) {
    const container = document.getElementById(elementId);
    container.replaceChildren();
    (items || []).slice(0, 12).forEach(item => {
        const tag = document.createElement("span");
        tag.className = "modal-tag";
        tag.textContent = item;
        container.appendChild(tag);
    });
    if (!container.childElementCount) container.textContent = "Not available";
}

closeModalButton.addEventListener("click", () => movieModal.classList.remove("active"));
movieModal.addEventListener("click", event => {
    if (event.target === movieModal) movieModal.classList.remove("active");
});
document.addEventListener("keydown", event => {
    if (event.key === "Escape") movieModal.classList.remove("active");
});
