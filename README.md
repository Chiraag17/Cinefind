# 🎬 CineFind — Movie Search & Recommendation Engine

A **Movie Search and Recommendation** application built with Python and Streamlit, demonstrating **Information Retrieval** using **TF-IDF Vectorization** and **Cosine Similarity**.

> Academic Information Retrieval Mini-Project

---

## 🎯 Objectives

- Build a genuine Information Retrieval (IR) system for movie search.
- Allow free-text natural language queries across titles, genres, keywords, plot overviews, actors and directors.
- Recommend similar movies based on content similarity.
- Present results through a modern, cinematic dark-themed web interface.

---

## 📊 Dataset

**TMDB 5000 Movie Dataset** from Kaggle:  
🔗 https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

| File | Records | Description |
|------|---------|-------------|
| `tmdb_5000_movies.csv` | 4 803 | Movie metadata (title, genres, keywords, overview, ratings, etc.) |
| `tmdb_5000_credits.csv` | 4 803 | Cast and crew information |

The two files are merged on `movies.id = credits.movie_id`.

---

## 🔧 Technologies Used

| Layer | Tool |
|-------|------|
| Language | Python 3.10+ |
| Web Framework | Streamlit |
| Data Processing | Pandas, NumPy |
| Natural Language Processing | NLTK (tokenization, stopword removal) |
| Information Retrieval | Scikit-learn (`TfidfVectorizer`, `cosine_similarity`) |

No external AI APIs, paid APIs, or databases are used.

---

## 🏗️ Information Retrieval Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Raw Dataset │────▶│  Preprocessing   │────▶│  TF-IDF Matrix   │
│  (CSV files) │     │  (clean, merge,  │     │  (10 000 features│
│              │     │   tokenize)      │     │   × 4 803 movies)│
└──────────────┘     └──────────────────┘     └──────────────────┘
                                                       │
                     ┌──────────────────┐              │
                     │   User Query     │              │
                     │  (preprocessed)  │──────┐       │
                     └──────────────────┘      │       │
                                               ▼       ▼
                                        ┌──────────────────┐
                                        │ Cosine Similarity │
                                        │   Calculation     │
                                        └──────────────────┘
                                               │
                                               ▼
                                        ┌──────────────────┐
                                        │  Ranked Results   │
                                        │  (filtered &      │
                                        │   sorted)         │
                                        └──────────────────┘
```

### Preprocessing Pipeline

1. **Load & merge** — movies + credits joined on movie ID.
2. **Parse JSON columns** — genres, keywords, cast, crew extracted via `ast.literal_eval`.
3. **Feature extraction** — top 3 actors, director name, genre/keyword lists.
4. **Text combination** — title + genres + keywords + overview + actors + director → single search field.
5. **Text cleaning** — lowercase → tokenize → remove stop words & punctuation.

### TF-IDF (Term Frequency – Inverse Document Frequency)

TF-IDF scores each word in a document relative to the whole corpus:

```
TF-IDF(t, d) = TF(t, d) × log(N / DF(t))
```

- **TF(t, d)** — how often term *t* appears in document *d* (with sublinear scaling: `1 + log(tf)`).
- **DF(t)** — number of documents containing *t*.
- **N** — total number of documents.

Words that are frequent in one movie but rare across the corpus receive higher weights.

### Cosine Similarity

Measures the angle between two TF-IDF vectors:

```
similarity(A, B) = (A · B) / (‖A‖ × ‖B‖)
```

- **1.0** → identical content direction.
- **0.0** → completely unrelated.

Used for both **query → movie ranking** and **movie → movie recommendations**.

---

## 📁 Project Structure

```
CineFind/
├── app.py                  # Streamlit UI and page routing
├── movie_search.py         # TF-IDF engine & cosine similarity search
├── data_preprocessing.py   # Dataset loading, cleaning, feature extraction
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── test_engine.py          # Automated test script
└── data/
    ├── tmdb_5000_movies.csv
    └── tmdb_5000_credits.csv
```

---

## ⚙️ Installation

### 1. Clone or copy the project

```bash
cd CineFind
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Ensure dataset files exist

Place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` inside the `data/` folder.

---

## 🚀 Running the Application

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**.

---

## 🧪 Testing Examples

### Search Queries

| Query | Top Result | Score |
|-------|-----------|-------|
| Science fiction movies about space | Gravity | 0.2548 |
| Action movies with superheroes | Megamind | 0.2074 |
| Romantic movies | The Notebook | 0.1774 |
| Adventure movies about the ocean | Kon-Tiki | 0.2705 |
| Movies starring Leonardo DiCaprio | Don's Plum | 0.2935 |
| Movies about artificial intelligence | Chappie | 0.2501 |

### Similar Movie Recommendations

| Selected Movie | Top Similar | Score |
|---------------|-------------|-------|
| Avatar | Guardians of the Galaxy | 0.2432 |
| Inception | Shutter Island | 0.2290 |
| The Dark Knight | Batman Begins | 0.3738 |

### Edge Cases

- **Empty query** → returns 0 results (handled gracefully).
- **Nonsense query** → returns 0 results with a friendly message.
- **Filters** → genre, year, and rating filters narrow results without altering similarity scores.

### Run automated tests

```bash
python test_engine.py
```

---

## 📄 License

Academic project — TMDB dataset is provided under its original license terms from Kaggle.
