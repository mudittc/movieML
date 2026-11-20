
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

DATA_PATH = ROOT_DIR / "data" / "processed" / "movies_features.csv"

MODEL_DIR = ROOT_DIR / "data" / "models" / "v3"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS = {
    "themes": 0.35,
    "genres": 0.25,
    "keywords": 0.20,
    "director": 0.12,
    "cast": 0.08,
}

# These weights are configurable and sum to 1.
# Missing features are automatically renormalized.


# ============================================================
# THEME TAXONOMY
# ============================================================

# Six major thematic families.
# Each family contains smaller concepts and their variations.
#
# This is a deterministic, in-house classification system.
# It does not use an LLM or external API.

THEME_TAXONOMY = {

    "psychological_existential": {

        "identity": [
            "identity",
            "self discovery",
            "self discovery",
            "alter ego",
            "dual identity",
            "personal identity",
            "identity crisis",
        ],

        "memory": [
            "memory",
            "memories",
            "amnesia",
            "forgotten past",
            "false memories",
            "recovered memories",
            "repressed memories",
        ],

        "reality_perception": [
            "reality",
            "illusion",
            "dream",
            "dreams",
            "nightmare",
            "virtual reality",
            "alternate reality",
            "perception",
            "delusion",
        ],

        "consciousness": [
            "consciousness",
            "self awareness",
            "subconscious",
            "sentience",
            "human mind",
            "psychological manipulation",
        ],

        "mental_health": [
            "psychological trauma",
            "mental illness",
            "madness",
            "insanity",
            "paranoia",
            "obsession",
            "schizophrenia",
        ],

        "existence": [
            "existence",
            "existential crisis",
            "meaning of life",
            "purpose of life",
            "absurdism",
            "nihilism",
        ],
    },

    "social_political": {

        "power_corruption": [
            "power",
            "corruption",
            "political corruption",
            "authoritarianism",
            "dictatorship",
            "political control",
        ],

        "class_inequality": [
            "class struggle",
            "social inequality",
            "wealth inequality",
            "poverty",
            "class divide",
            "social hierarchy",
        ],

        "freedom_oppression": [
            "oppression",
            "liberation",
            "freedom",
            "imprisonment",
            "tyranny",
            "captivity",
        ],

        "justice": [
            "justice",
            "injustice",
            "revenge against injustice",
            "law and order",
            "corruption of justice",
        ],

        "rebellion": [
            "rebellion",
            "revolution",
            "resistance",
            "civil disobedience",
            "social uprising",
        ],

        "conformity": [
            "social conformity",
            "conformity",
            "social pressure",
            "individualism",
            "social expectations",
        ],
    },

    "relationships_emotional": {

        "love": [
            "love",
            "romance",
            "romantic relationship",
            "unrequited love",
            "forbidden love",
            "true love",
        ],

        "family": [
            "family",
            "parenthood",
            "fatherhood",
            "motherhood",
            "siblings",
            "family relationships",
        ],

        "sacrifice": [
            "sacrifice",
            "self sacrifice",
            "devotion",
            "selflessness",
        ],

        "loneliness": [
            "loneliness",
            "isolation",
            "alienation",
            "solitude",
            "social isolation",
        ],

        "betrayal": [
            "betrayal",
            "treachery",
            "deception",
            "broken trust",
            "loyalty",
        ],

        "grief_loss": [
            "grief",
            "loss",
            "bereavement",
            "mourning",
            "separation",
        ],
    },

    "science_technology_humanity": {

        "artificial_intelligence": [
            "artificial intelligence",
            "ai",
            "android",
            "robot",
            "robotics",
            "sentient machines",
        ],

        "humanity_technology": [
            "humanity versus technology",
            "technology versus humanity",
            "human machine relationship",
            "technological dependence",
            "human evolution",
        ],

        "scientific_ethics": [
            "scientific ethics",
            "ethical experimentation",
            "genetic engineering",
            "human experimentation",
            "scientific responsibility",
        ],

        "free_will": [
            "free will",
            "determinism",
            "predestination",
            "choice versus fate",
        ],

        "space_exploration": [
            "space exploration",
            "space travel",
            "extraterrestrial life",
            "alien civilization",
        ],
    },

    "conflict_human_struggle": {

        "survival": [
            "survival",
            "human survival",
            "survival instinct",
            "survival against odds",
        ],

        "revenge": [
            "revenge",
            "vengeance",
            "retribution",
        ],

        "war": [
            "war",
            "warfare",
            "military conflict",
            "battle",
            "armed conflict",
        ],

        "resilience": [
            "resilience",
            "perseverance",
            "overcoming adversity",
            "determination",
        ],

        "ambition": [
            "ambition",
            "greed",
            "success",
            "competition",
            "obsession with success",
        ],

        "morality": [
            "morality",
            "ethics",
            "moral dilemma",
            "good versus evil",
            "ethical dilemma",
        ],
    },

    "spiritual_philosophical": {

        "faith": [
            "faith",
            "religion",
            "belief",
            "spirituality",
            "religious belief",
        ],

        "fate_destiny": [
            "destiny",
            "fate",
            "predestination",
            "divine intervention",
        ],

        "good_evil": [
            "good versus evil",
            "evil",
            "sin",
            "redemption",
            "temptation",
        ],

        "mortality": [
            "mortality",
            "death",
            "afterlife",
            "immortality",
            "fear of death",
        ],

        "human_nature": [
            "human nature",
            "humanity",
            "human condition",
            "civilization",
            "human behaviour",
        ],

        "meaning_purpose": [
            "purpose",
            "meaning",
            "meaning of existence",
            "search for meaning",
            "purpose of existence",
        ],
    },
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def clean_text(value):

    if value is None:
        return ""

    if isinstance(value, (list, dict)):
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def normalise_text(value):

    value = clean_text(value).lower()

    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def parse_list(value):
    """
    Handles:
    - Actual lists
    - Dictionaries
    - JSON strings
    - Python list strings
    - Comma-separated strings
    - Missing values
    """

    if isinstance(value, list):
        return value

    if isinstance(value, dict):
        return [value]

    if value is None:
        return []

    if isinstance(value, (int, float)):
        return []

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    if not isinstance(value, str):
        return []

    value = value.strip()

    if not value:
        return []

    try:
        parsed = json.loads(value)

        if isinstance(parsed, list):
            return parsed

        if isinstance(parsed, dict):
            return [parsed]

    except (json.JSONDecodeError, TypeError):
        pass

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return parsed

        if isinstance(parsed, dict):
            return [parsed]

    except (ValueError, SyntaxError):
        pass

    # Fallback for simple comma-separated metadata.
    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def extract_names(value):
    """
    Extracts names from TMDB metadata.
    """

    parsed = parse_list(value)

    names = []

    for item in parsed:

        if isinstance(item, dict):

            name = (
                item.get("name")
                or item.get("original_name")
                or item.get("title")
            )

            if name:
                names.append(normalise_text(name))

        elif isinstance(item, str):

            name = normalise_text(item)

            if name:
                names.append(name)

    return sorted(set(names))


def parse_directors(value):

    if isinstance(value, (list, dict)):
        return extract_names(value)

    text = normalise_text(value)

    return [text] if text else []


# ============================================================
# THEME EXTRACTION
# ============================================================

def extract_themes(overview, keywords):
    """
    Detects thematic concepts using phrase matching.

    Returns:
        themes: individual detected concepts
        families: detected major thematic families
    """

    keyword_text = " ".join(keywords)

    combined_text = normalise_text(
        overview + " " + keyword_text
    )

    detected_themes = set()
    detected_families = set()

    for family, concepts in THEME_TAXONOMY.items():

        for concept, phrases in concepts.items():

            matched = False

            for phrase in phrases:

                phrase = normalise_text(phrase)

                pattern = (
                    r"\b" +
                    re.escape(phrase) +
                    r"\b"
                )

                if re.search(pattern, combined_text):

                    detected_themes.add(concept)
                    detected_families.add(family)

                    matched = True
                    break

    return (
        sorted(detected_themes),
        sorted(detected_families)
    )


# ============================================================
# DATASET PREPARATION
# ============================================================

def prepare_dataset():

    print("Loading movie dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "id",
        "title",
        "overview",
    ]

    for column in required_columns:

        if column not in df.columns:
            raise ValueError(
                f"Required column missing: {column}"
            )

    print(f"Loaded {len(df)} movies.")

    # Clean basic fields.

    text_columns = [
        "title",
        "overview",
        "tagline",
        "director",
    ]

    for column in text_columns:

        if column not in df.columns:
            df[column] = ""

        df[column] = df[column].apply(clean_text)

    # Parse structured metadata.

    metadata_columns = [
        "genres",
        "keywords",
        "cast",
    ]

    for column in metadata_columns:

        if column not in df.columns:
            df[column] = [[] for _ in range(len(df))]

        df[f"{column}_parsed"] = (
            df[column].apply(extract_names)
        )

    # Director.

    df["director_parsed"] = (
        df["director"].apply(parse_directors)
    )

    # Story is retained as metadata, not used for scoring.

    df["story_text"] = (
        df["overview"].fillna("") + " " +
        df["tagline"].fillna("")
    ).str.strip()

    # Extract themes.

    print("Extracting thematic concepts...")

    theme_results = [
        extract_themes(overview, keywords)
        for overview, keywords in zip(
            df["overview"],
            df["keywords_parsed"]
        )
    ]

    df["themes_parsed"] = [
        result[0] for result in theme_results
    ]

    df["theme_families"] = [
        result[1] for result in theme_results
    ]

    # Validate IDs.

    df["id"] = pd.to_numeric(
        df["id"],
        errors="coerce"
    )

    df = df.dropna(subset=["id"])

    df["id"] = df["id"].astype(int)

    df = (
        df.drop_duplicates(subset=["id"])
        .reset_index(drop=True)
    )

    print(f"Prepared {len(df)} movies.")

    return df


# ============================================================
# KEYWORD TF-IDF
# ============================================================

def build_keyword_vectors(df):
    """
    Builds an independent TF-IDF representation
    for movie keywords.
    """

    print("Building keyword TF-IDF vectors...")

    keyword_documents = [
        " ".join(keywords)
        for keywords in df["keywords_parsed"]
    ]

    if not any(keyword_documents):

        print("No keyword data found.")

        return None, None

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        norm="l2",
    )

    try:

        matrix = vectorizer.fit_transform(
            keyword_documents
        )

    except ValueError:

        # Handles very small datasets or empty vocabularies.
        print("Insufficient keyword vocabulary.")

        return None, None

    print(
        f"Keyword vocabulary: {len(vectorizer.vocabulary_)}"
    )

    return vectorizer, matrix


# ============================================================
# SIMILARITY FUNCTIONS
# ============================================================


def weighted_jaccard(list_a, list_b):

    set_a = set(list_a)
    set_b = set(list_b)

    if not set_a and not set_b:
        return None

    if not set_a or not set_b:
        return 0.0

    return len(set_a & set_b) / len(set_a | set_b)


def calculate_director_similarity(movie_a, movie_b):

    directors_a = set(movie_a["director_parsed"])
    directors_b = set(movie_b["director_parsed"])

    if not directors_a and not directors_b:
        return None

    if not directors_a or not directors_b:
        return 0.0

    return 1.0 if directors_a & directors_b else 0.0


def calculate_keyword_similarity(
    keyword_matrix,
    index_a,
    index_b
):

    if keyword_matrix is None:
        return None

    row_a = keyword_matrix[index_a]
    row_b = keyword_matrix[index_b]

    if row_a.nnz == 0 and row_b.nnz == 0:
        return None

    if row_a.nnz == 0 or row_b.nnz == 0:
        return 0.0

    score = cosine_similarity(row_a, row_b)[0, 0]

    return float(np.clip(score, 0, 1))



def calculate_theme_similarity(movie_a, movie_b):
    """
    Combines:
        65% concept similarity
        35% major family similarity

    This prevents the six broad families from
    completely dominating the thematic comparison.
    """

    concept_score = weighted_jaccard(
        movie_a["themes_parsed"],
        movie_b["themes_parsed"]
    )

    family_score = weighted_jaccard(
        movie_a["theme_families"],
        movie_b["theme_families"]
    )

    available = []

    if concept_score is not None:
        available.append((concept_score, 0.65))

    if family_score is not None:
        available.append((family_score, 0.35))

    if not available:
        return None

    total_weight = sum(
        weight for _, weight in available
    )

    return sum(
        score * weight
        for score, weight in available
    ) / total_weight


def calculate_genre_similarity(movie_a, movie_b):

    return weighted_jaccard(
        movie_a["genres_parsed"],
        movie_b["genres_parsed"]
    )


def calculate_cast_similarity(movie_a, movie_b):

    return weighted_jaccard(
        movie_a["cast_parsed"],
        movie_b["cast_parsed"]
    )


# ============================================================
# FINAL WEIGHTED SCORE
# ============================================================

def calculate_final_score(scores):
    """
    Missing signals are excluded.

    Remaining weights are proportionally
    renormalized to sum to 1.
    """

    weighted_sum = 0.0
    available_weight = 0.0

    for signal, weight in WEIGHTS.items():

        score = scores.get(signal)

        if score is None:
            continue

        weighted_sum += score * weight
        available_weight += weight

    if available_weight == 0:
        return 0.0

    return weighted_sum / available_weight


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

class MovieMatchV3:

    def __init__(self):

        self.df = prepare_dataset()

        self.id_to_index = {
            int(movie_id): index
            for index, movie_id in enumerate(self.df["id"])
        }

        self.keyword_vectorizer, self.keyword_matrix = (
            build_keyword_vectors(self.df)
        )

        print("\nMovieMatch V3 ready!")

        print(
            f"Movies available: {len(self.df)}"
        )

        print(
            f"Keyword model: "
            f"{'Ready' if self.keyword_matrix is not None else 'Unavailable'}"
        )

    # --------------------------------------------------------
    # INDIVIDUAL FEATURE SCORES
    # --------------------------------------------------------

    def calculate_signal_scores(
        self,
        query_movie,
        candidate_movie,
        query_index,
        candidate_index,
    ):

        scores = {}

        # 1. Themes

        scores["themes"] = calculate_theme_similarity(
            query_movie,
            candidate_movie
        )

        # 2. Genres

        scores["genres"] = calculate_genre_similarity(
            query_movie,
            candidate_movie
        )

        # 3. Keywords

        scores["keywords"] = calculate_keyword_similarity(
            self.keyword_matrix,
            query_index,
            candidate_index
        )

        # 4. Director

        scores["director"] = calculate_director_similarity(
            query_movie,
            candidate_movie
        )

        # 5. Cast

        scores["cast"] = calculate_cast_similarity(
            query_movie,
            candidate_movie
        )

        return scores

    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    def recommend(
        self,
        movie_id,
        top_n=10,
        selected_genres=None,
    ):

        movie_id = int(movie_id)

        if movie_id not in self.id_to_index:

            raise ValueError(
                f"Movie ID {movie_id} not found."
            )

        query_index = self.id_to_index[movie_id]

        query_movie = self.df.iloc[query_index]

        results = []

        selected_genres = (
            set(normalise_text(g) for g in selected_genres)
            if selected_genres else None
        )

        for candidate_index, candidate_movie in self.df.iterrows():

            if candidate_index == query_index:
                continue

            scores = self.calculate_signal_scores(
                query_movie,
                candidate_movie,
                query_index,
                candidate_index,
            )

            final_score = calculate_final_score(scores)

            # Optional user-selected genre filtering.
            if selected_genres:

                candidate_genres = set(
                    candidate_movie["genres_parsed"]
                )

                if not candidate_genres & selected_genres:
                    continue

            results.append({

                "id": int(candidate_movie["id"]),

                "title": candidate_movie["title"],

                "genres": candidate_movie["genres_parsed"],

                "themes": candidate_movie["themes_parsed"],

                "theme_families": candidate_movie["theme_families"],

                "keywords": candidate_movie["keywords_parsed"],

                "final_score": float(final_score),

                **scores,

            })

        results.sort(
            key=lambda movie: movie["final_score"],
            reverse=True
        )

        return results[:top_n]



    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    def display_recommendations(
        self,
        movie_id,
        top_n=10,
        selected_genres=None,
    ):

        movie_id = int(movie_id)

        if movie_id not in self.id_to_index:

            raise ValueError(
                f"Movie ID {movie_id} not found."
            )

        movie_title = self.df.iloc[
            self.id_to_index[movie_id]
        ]["title"]

        print(
            f"\nRecommendations for: {movie_title}"
        )

        print("=" * 75)

        recommendations = self.recommend(
            movie_id=movie_id,
            top_n=top_n,
            selected_genres=selected_genres,
        )

        for rank, movie in enumerate(
            recommendations,
            start=1
        ):

            print(
                f"\n{rank}. {movie['title']}"
            )

            print(
                f"   Final score: {movie['final_score']:.3f}"
            )

            for signal in WEIGHTS:

                score = movie.get(signal)

                display_score = (
                    f"{score:.3f}"
                    if score is not None
                    else "N/A"
                )

                print(
                    f"   {signal.capitalize()}: {display_score}"
                )

            print(
                f"   Themes: {format_list(movie.get('themes'))}"
            )

            print(
                print(f"   Theme families: {format_list(movie.get('theme_families'))}")
            )

            print(
                print(f"   Genres: {format_list(movie.get('genres'))}")
            )

        return recommendations


# ============================================================
# RUN DIRECTLY
# ============================================================

def format_list(value):
    if value is None:
        return "None"

    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(x) for x in value if x) or "None"

    if isinstance(value, float) and np.isnan(value):
        return "None"

    if isinstance(value, str):
        return value or "None"

    return str(value)

if __name__ == "__main__":

    engine = MovieMatchV3()

    movie = engine.df[
        engine.df["id"] == 27205
    ].iloc[0]

    print("\nINCEPTION THEME ANALYSIS")
    print("Detected concepts:", movie["themes_parsed"])
    print("Detected families:", movie["theme_families"])
    print("Overview:", movie["overview"])

    # Inception
    engine.display_recommendations(
        movie_id=27205,
        top_n=10,
    )
