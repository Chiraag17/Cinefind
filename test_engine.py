"""Test suite for CineFind search engine."""
import sys
sys.path.insert(0, r"c:\Users\hp\Desktop\CineFind")

from data_preprocessing import load_and_prepare_data
from movie_search import MovieSearchEngine

print("=== Loading & preprocessing dataset ===")
df = load_and_prepare_data()
print(f"Loaded {len(df)} movies")
print(f"Columns: {df.columns.tolist()}")
print(f"Sample processed_text: {df.iloc[0]['processed_text'][:120]}...")
print()

print("=== Building search engine ===")
engine = MovieSearchEngine(df)
print(f"TF-IDF matrix shape: {engine.tfidf_matrix.shape}")
print(f"Genres: {engine.get_all_genres()[:10]}...")
print(f"Year range: {engine.get_year_range()}")
print()

# Test queries
queries = [
    "Science fiction movies about space",
    "Action movies with superheroes",
    "Romantic movies",
    "Adventure movies about the ocean",
    "Movies starring Leonardo DiCaprio",
    "Movies about artificial intelligence",
]
for q in queries:
    results = engine.search(q, top_n=5)
    if results.empty:
        print(f'Query: "{q}" -> NO RESULTS')
    else:
        print(f'Query: "{q}" -> {len(results)} results')
        for _, r in results.iterrows():
            print(f'   {r["title"]:45s} score={r["relevance_score"]:.4f}  rating={r["vote_average"]}')
    print()

# Test similar movies
for title in ["Avatar", "Inception", "The Dark Knight"]:
    sim = engine.get_similar_movies(title, top_n=5)
    print(f'Similar to "{title}":')
    if sim.empty:
        print("   No similar movies found")
    else:
        for _, r in sim.iterrows():
            print(f'   {r["title"]:45s} score={r["similarity_score"]:.4f}')
    print()

# Edge cases
print("=== Edge cases ===")
r1 = engine.search("")
print(f"Empty query: {len(r1)} results (expected 0)")
r2 = engine.search("xyzzynonexistentquery12345")
print(f"Nonsense query: {len(r2)} results (expected 0)")
print()

# Test filters
print("=== Filters ===")
r3 = engine.search("action", genre_filter=["Action"], min_rating=7.0, year_range=(2010, 2016))
print(f'"action" + genre=Action, rating>=7, year 2010-2016: {len(r3)} results')
if not r3.empty:
    for _, r in r3.head(3).iterrows():
        print(f'   {r["title"]:45s} score={r["relevance_score"]:.4f} rating={r["vote_average"]} year={r["release_year"]}')
print()

# Test movie details
print("=== Movie details ===")
d = engine.get_movie_details("Avatar")
if d:
    print(f"Title: {d['title']}")
    print(f"Director: {d['director']}")
    print(f"Actors: {d['actors']}")
    print(f"Genres: {d['genres']}")
    print(f"Rating: {d['vote_average']}")
else:
    print("Avatar not found!")
print()

print("=== All tests passed ===")
