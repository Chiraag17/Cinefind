"""
CineFind — Movie Search & Recommendation Engine
=================================================
A Streamlit application that demonstrates Information Retrieval
using TF-IDF vectorization and Cosine Similarity on the TMDB 5000
movie dataset.

Run:  streamlit run app.py
"""

import hashlib
import streamlit as st
import pandas as pd

from data_preprocessing import load_and_prepare_data
from movie_search import MovieSearchEngine

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineFind – Discover Movies",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS – cinematic dark theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── global ─────────────────────────────────────────── */
:root {
    --bg-primary:   #0d1117;
    --bg-card:      #161b22;
    --bg-card-hover:#1c2333;
    --accent:       #7c3aed;
    --accent-light: #a78bfa;
    --accent-blue:  #3b82f6;
    --text-primary: #e6edf3;
    --text-muted:   #8b949e;
    --border:       #30363d;
    --gradient-1:   linear-gradient(135deg, #7c3aed 0%, #3b82f6 100%);
    --gradient-2:   linear-gradient(135deg, #1e1b4b 0%, #172554 100%);
}

.stApp {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary);
}

/* ── top nav bar ───────────────────────────────────── */
.nav-bar {
    display: flex; align-items: center; gap: 2rem;
    padding: 1rem 2rem;
    background: var(--bg-card);
    border-bottom: 1px solid var(--border);
    margin: -1rem -1rem 1.5rem -1rem;
}
.nav-brand {
    font-size: 1.55rem; font-weight: 800;
    background: var(--gradient-1); -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.nav-links { display: flex; gap: 1.2rem; margin-left: auto; }
.nav-links a {
    color: var(--text-muted); text-decoration: none;
    font-size: .92rem; font-weight: 500;
    transition: color .2s;
}
.nav-links a:hover, .nav-links a.active { color: var(--accent-light); }

/* ── hero ──────────────────────────────────────────── */
.hero {
    text-align: center; padding: 3.5rem 1rem 2rem;
    background: var(--gradient-2); border-radius: 18px;
    margin-bottom: 2rem;
}
.hero h1 {
    font-size: 2.7rem; font-weight: 800;
    background: var(--gradient-1); -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: .4rem;
}
.hero p { color: var(--text-muted); font-size: 1.1rem; }

/* ── movie card ────────────────────────────────────── */
.movie-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px; padding: 1.3rem 1.3rem 1rem;
    transition: transform .2s, box-shadow .2s;
    height: 100%;
}
.movie-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 30px rgba(124,58,237,.18);
}
.movie-card .poster-placeholder {
    width: 100%; height: 160px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 2.5rem; margin-bottom: .8rem;
}
.movie-card h3 {
    font-size: 1.05rem; font-weight: 700;
    color: var(--text-primary); margin: 0 0 .35rem;
    line-height: 1.3;
}
.movie-card .meta {
    font-size: .82rem; color: var(--text-muted);
    margin-bottom: .4rem;
}
.movie-card .overview-snippet {
    font-size: .82rem; color: var(--text-muted);
    line-height: 1.45; margin-bottom: .5rem;
    display: -webkit-box; -webkit-line-clamp: 3;
    -webkit-box-orient: vertical; overflow: hidden;
}
.score-badge {
    display: inline-block; padding: .18rem .55rem;
    border-radius: 6px; font-size: .78rem; font-weight: 700;
    background: var(--gradient-1); color: #fff;
}
.rating-badge {
    display: inline-block; padding: .18rem .55rem;
    border-radius: 6px; font-size: .78rem; font-weight: 600;
    background: #1e293b; color: #facc15;
    margin-left: .35rem;
}

/* ── detail panel ──────────────────────────────────── */
.detail-panel {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 16px; padding: 2rem;
}
.detail-panel h2 {
    font-size: 1.7rem; font-weight: 800;
    background: var(--gradient-1); -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.detail-panel .chip {
    display: inline-block; padding: .22rem .65rem;
    border-radius: 20px; font-size: .8rem; font-weight: 600;
    background: #1e1b4b; color: var(--accent-light);
    margin: .15rem .2rem;
}

/* ── section heading ───────────────────────────────── */
.section-heading {
    font-size: 1.35rem; font-weight: 700;
    background: var(--gradient-1); -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 1.5rem 0 .8rem;
}

/* ── sidebar overrides ─────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
}

/* ── search history chips ──────────────────────────── */
.history-chip {
    display: inline-block; padding: .3rem .7rem;
    border-radius: 20px; font-size: .82rem;
    background: #1e1b4b; color: var(--accent-light);
    margin: .15rem .15rem; cursor: pointer;
    border: 1px solid var(--accent);
    transition: background .2s;
}
.history-chip:hover { background: #312e81; }

/* hide default Streamlit footer & menu  */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: deterministic gradient for poster placeholder
# ─────────────────────────────────────────────────────────────────────────────
_GRADIENTS = [
    "linear-gradient(135deg,#4c1d95,#1e3a8a)",
    "linear-gradient(135deg,#701a75,#7e22ce)",
    "linear-gradient(135deg,#0f766e,#1d4ed8)",
    "linear-gradient(135deg,#9d174d,#4c1d95)",
    "linear-gradient(135deg,#1e40af,#7c3aed)",
    "linear-gradient(135deg,#065f46,#3730a3)",
    "linear-gradient(135deg,#7c2d12,#9333ea)",
    "linear-gradient(135deg,#1e3a5f,#6d28d9)",
]


def _gradient_for(title: str) -> str:
    idx = int(hashlib.md5(title.encode()).hexdigest(), 16) % len(_GRADIENTS)
    return _GRADIENTS[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Load data & engine (cached so it runs only once)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading movie database & building TF-IDF index …")
def _init_engine():
    df = load_and_prepare_data()
    engine = MovieSearchEngine(df)
    return engine

engine = _init_engine()

# ─────────────────────────────────────────────────────────────────────────────
# Session-state defaults
# ─────────────────────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None
if "search_query" not in st.session_state:
    st.session_state.search_query = ""


# ─────────────────────────────────────────────────────────────────────────────
# Navigation bar
# ─────────────────────────────────────────────────────────────────────────────
def _nav():
    active = st.session_state.page
    st.markdown(f"""
    <div class="nav-bar">
        <span class="nav-brand">🎬 CineFind</span>
        <div class="nav-links">
            <a class="{'active' if active=='home' else ''}" href="?page=home"
               onclick="return false;">Home</a>
            <a class="{'active' if active=='search' else ''}" href="?page=search"
               onclick="return false;">Search Movies</a>
            <a class="{'active' if active=='similar' else ''}" href="?page=similar"
               onclick="return false;">Similar Movies</a>
            <a class="{'active' if active=='about' else ''}" href="?page=about"
               onclick="return false;">About</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    labels = [("🏠  Home", "home"), ("🔍  Search Movies", "search"),
              ("🎯  Similar Movies", "similar"), ("ℹ️  About", "about")]
    for col, (label, page_key) in zip(cols, labels):
        with col:
            if st.button(label, use_container_width=True,
                         type="primary" if active == page_key else "secondary"):
                st.session_state.page = page_key
                st.rerun()

_nav()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – filters & search history
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Filters")

    all_genres = engine.get_all_genres()
    genre_filter = st.multiselect("Genre", all_genres)

    yr_min, yr_max = engine.get_year_range()
    year_range = st.slider("Release Year", yr_min, yr_max, (yr_min, yr_max))

    min_rating = st.slider("Minimum Rating", 0.0, 10.0, 0.0, 0.5)

    sort_options = {"Relevance": "relevance", "Highest Rating": "rating",
                    "Release Year": "year"}
    sort_label = st.selectbox("Sort By", list(sort_options.keys()))
    sort_by = sort_options[sort_label]

    if st.button("🗑️  Clear Filters", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.markdown("### 🕘 Recent Searches")
    if st.session_state.search_history:
        for q in st.session_state.search_history[-8:][::-1]:
            if st.button(f"🔁 {q}", key=f"hist_{q}", use_container_width=True):
                st.session_state.search_query = q
                st.session_state.page = "search"
                st.rerun()
        if st.button("🗑️  Clear History", use_container_width=True):
            st.session_state.search_history = []
            st.rerun()
    else:
        st.caption("No recent searches.")


# ─────────────────────────────────────────────────────────────────────────────
# Helper – render a movie card
# ─────────────────────────────────────────────────────────────────────────────
def render_card(row, score_col=None):
    title = row.get("title", "Unknown")
    genres = ", ".join(row.get("genres_list", [])[:3]) or "—"
    year = int(row.get("release_year", 0)) or "—"
    rating = row.get("vote_average", 0)
    overview = (row.get("overview", "") or "")[:200]
    grad = _gradient_for(title)
    emoji = "🎬"

    score_html = ""
    if score_col and score_col in row:
        pct = row[score_col] * 100
        score_html = f'<span class="score-badge">Match {pct:.1f}%</span>'

    st.markdown(f"""
    <div class="movie-card">
        <div class="poster-placeholder" style="background:{grad};">{emoji}</div>
        <h3>{title}</h3>
        <div class="meta">{genres} · {year}</div>
        <div class="meta">
            {score_html}
            <span class="rating-badge">⭐ {rating:.1f}</span>
        </div>
        <div class="overview-snippet">{overview}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("View Details", key=f"detail_{title}_{score_col}_{id(row)}",
                 use_container_width=True):
        st.session_state.selected_movie = title
        st.session_state.page = "details"
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────────────────────────────────────

# ── HOME ────────────────────────────────────────────────────────────────────
if st.session_state.page == "home":
    st.markdown("""
    <div class="hero">
        <h1>Discover Your Next Favourite Movie</h1>
        <p>Search thousands of movies using intelligent text retrieval
           powered by TF-IDF &amp; Cosine Similarity</p>
    </div>
    """, unsafe_allow_html=True)

    query = st.text_input("🔍  What kind of movie are you looking for?",
                          value=st.session_state.search_query,
                          placeholder="e.g. Science fiction movies about space",
                          key="home_search_input")

    if st.button("🚀  Search", type="primary", use_container_width=True):
        if query.strip():
            st.session_state.search_query = query.strip()
            # save to history
            hist = st.session_state.search_history
            if query.strip() not in hist:
                hist.append(query.strip())
                st.session_state.search_history = hist[-15:]
            st.session_state.page = "search"
            st.rerun()
        else:
            st.warning("Please enter a search query.")

    # Quick suggestions
    st.markdown('<p class="section-heading">🎯 Quick Suggestions</p>',
                unsafe_allow_html=True)
    suggestions = [
        "Science fiction movies about space",
        "Action movies with superheroes",
        "Romantic movies",
        "Adventure movies about the ocean",
        "Movies about artificial intelligence",
        "Movies starring Leonardo DiCaprio",
    ]
    cols = st.columns(3)
    for i, s in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(f"🔎 {s}", key=f"sug_{i}", use_container_width=True):
                st.session_state.search_query = s
                hist = st.session_state.search_history
                if s not in hist:
                    hist.append(s)
                    st.session_state.search_history = hist[-15:]
                st.session_state.page = "search"
                st.rerun()


# ── SEARCH ──────────────────────────────────────────────────────────────────
elif st.session_state.page == "search":
    st.markdown('<p class="section-heading">🔍 Search Movies</p>',
                unsafe_allow_html=True)

    query = st.text_input("Search query",
                          value=st.session_state.search_query,
                          placeholder="e.g. Action movies with superheroes",
                          key="search_input")

    if st.button("🚀  Search", type="primary"):
        st.session_state.search_query = query.strip()
        hist = st.session_state.search_history
        if query.strip() and query.strip() not in hist:
            hist.append(query.strip())
            st.session_state.search_history = hist[-15:]
        st.rerun()

    if st.session_state.search_query:
        results = engine.search(
            st.session_state.search_query,
            top_n=30,
            genre_filter=genre_filter or None,
            year_range=tuple(year_range) if year_range else None,
            min_rating=min_rating,
            sort_by=sort_by,
        )
        if results.empty:
            st.info("😕 No movies matched your query. Try different keywords or relax your filters.")
        else:
            st.success(f"Found **{len(results)}** movies for *\"{st.session_state.search_query}\"*")
            # 3-column grid
            cols = st.columns(3)
            for i, (_, row) in enumerate(results.iterrows()):
                with cols[i % 3]:
                    render_card(row.to_dict(), score_col="relevance_score")
    else:
        st.info("Enter a query above to search.")


# ── SIMILAR MOVIES ──────────────────────────────────────────────────────────
elif st.session_state.page == "similar":
    st.markdown('<p class="section-heading">🎯 Find Similar Movies</p>',
                unsafe_allow_html=True)

    all_titles = engine.get_all_titles()
    selected = st.selectbox("Select a movie", [""] + all_titles,
                            key="similar_select")

    if selected:
        similar = engine.get_similar_movies(selected, top_n=5)
        if similar.empty:
            st.info("No similar movies found.")
        else:
            st.success(f"Top **{len(similar)}** movies similar to *\"{selected}\"*")
            cols = st.columns(3)
            for i, (_, row) in enumerate(similar.iterrows()):
                with cols[i % 3]:
                    render_card(row.to_dict(), score_col="similarity_score")
    else:
        st.info("Select a movie title to get recommendations.")


# ── MOVIE DETAILS ───────────────────────────────────────────────────────────
elif st.session_state.page == "details":
    title = st.session_state.selected_movie
    if title:
        details = engine.get_movie_details(title)
        if details:
            grad = _gradient_for(details["title"])
            st.markdown(f"""
            <div class="detail-panel">
                <div class="poster-placeholder"
                     style="background:{grad}; width:100%; height:220px;
                            border-radius:12px; display:flex;
                            align-items:center; justify-content:center;
                            font-size:4rem; margin-bottom:1rem;">🎬</div>
                <h2>{details["title"]}</h2>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("⭐ Rating", f"{details['vote_average']:.1f} / 10")
            col2.metric("📅 Year", details["release_year"] or "—")
            col3.metric("🔥 Popularity", f"{details['popularity']:.1f}")

            st.markdown("**📖 Overview**")
            st.write(details["overview"] or "No overview available.")

            st.markdown("**🎭 Genres**")
            chips = " ".join(
                f'<span class="chip" style="display:inline-block;padding:.22rem .65rem;'
                f'border-radius:20px;font-size:.8rem;font-weight:600;'
                f'background:#1e1b4b;color:#a78bfa;margin:.15rem .2rem;">{g}</span>'
                for g in details["genres"]
            )
            st.markdown(chips, unsafe_allow_html=True)

            st.markdown("**🔑 Keywords**")
            kw_chips = " ".join(
                f'<span class="chip" style="display:inline-block;padding:.22rem .65rem;'
                f'border-radius:20px;font-size:.8rem;font-weight:600;'
                f'background:#172554;color:#93c5fd;margin:.15rem .2rem;">{k}</span>'
                for k in details["keywords"][:10]
            )
            st.markdown(kw_chips or "—", unsafe_allow_html=True)

            st.markdown("**🎭 Main Actors**")
            st.write(", ".join(details["actors"]) if details["actors"] else "—")

            st.markdown("**🎬 Director**")
            st.write(details["director"] or "—")

            st.markdown("---")
            if st.button("🎯 Find Similar Movies", type="primary"):
                st.session_state.page = "similar"
                # pre-fill the selectbox via a workaround
                st.session_state.similar_select = title
                st.rerun()

            if st.button("⬅️ Back to Search"):
                st.session_state.page = "search"
                st.rerun()
        else:
            st.error("Movie not found.")
    else:
        st.session_state.page = "home"
        st.rerun()


# ── ABOUT ───────────────────────────────────────────────────────────────────
elif st.session_state.page == "about":
    st.markdown("""
    <div class="detail-panel">
        <h2>About CineFind</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **CineFind** is an academic Information Retrieval mini-project that
    demonstrates movie searching and recommendation using **TF-IDF
    Vectorization** and **Cosine Similarity**.

    ### 🎯 Objectives
    - Build a genuine IR system — no LLM black boxes or hardcoded results.
    - Allow free-text search across titles, genres, keywords, overviews, cast
      and crew.
    - Recommend similar movies based on content similarity.

    ### 📊 Dataset
    [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
    from Kaggle — 4 803 movies with metadata and credits.

    ### 🔧 Technology Stack
    | Layer | Tool |
    |-------|------|
    | Language | Python 3 |
    | UI | Streamlit |
    | Data | Pandas, NumPy |
    | NLP | NLTK |
    | IR Engine | Scikit-learn (TF-IDF, Cosine Similarity) |

    ### 📐 How It Works

    **TF-IDF (Term Frequency – Inverse Document Frequency)**

    Each movie is represented as a document whose text combines its title,
    genres, keywords, overview, actors and director. The TF-IDF vectorizer
    converts these documents into numerical vectors that capture word
    importance relative to the entire corpus.

    """)
    st.latex(r"\text{TF-IDF}(t,d) = \text{TF}(t,d) \times \log\!\left(\frac{N}{\text{DF}(t)}\right)")
    st.markdown("""

    **Cosine Similarity**

    The angle between two TF-IDF vectors measures how similar two movies (or a
    query and a movie) are, regardless of document length.

    """)
    st.latex(r"\text{similarity}(A,B) = \frac{A \cdot B}{\|A\| \; \|B\|}")
    st.markdown("""

    ### 👨‍💻 Authors
    Academic Information Retrieval Mini-Project.

    ---
    *Built with ❤️ using Python, Streamlit & Scikit-learn*
    """)
