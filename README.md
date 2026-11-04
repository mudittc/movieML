# MovieMatch

MovieMatch is a movie recommendation system built using Python and machine learning. It uses movie genres and user-generated tags to find films similar to a movie selected by the user.

I built this project to explore content-based recommendation systems and understand how text vectorization and similarity measures can be used in a practical application.

## Tech Stack

- **Python:** Pandas, NumPy, Scikit-learn
- **Machine Learning:** TF-IDF, Cosine Similarity, Nearest Neighbors
- **Backend:** FastAPI
- **Frontend:** HTML, CSS, JavaScript

## How it works

1. Movie genres and tags are combined to create a description for each movie.
2. TF-IDF converts these descriptions into numerical vectors.
3. A Nearest Neighbors model uses cosine distance to find similar movies.
4. The API returns recommendations along with similarity scores.

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/movieML.git
cd movieML
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Download the [MovieLens Latest Full Dataset](https://grouplens.org/datasets/movielens/) and place the following files inside a `data/` directory:

- `movies.csv`
- `ratings.csv`
- `tags.csv`
- `links.csv`

**4. Prepare the data and model**

```bash
python prepare_metadata.py
python main.py
```

**5. Run the application**

```bash
uvicorn app:app --reload
```

Open http://127.0.0.1:8000/ui to use MovieMatch.

API documentation: http://127.0.0.1:8000/docs

## Project Structure

```text
movieML/
├── data/
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
├── main.py
├── app.py
├── prepare_metadata.py
├── requirements.txt
└── README.md
```

The MovieLens dataset and generated model files are not included in this repository.