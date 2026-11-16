"""
CineFind — Movie Search & Recommendation Engine
=================================================
A Streamlit application that demonstrates Information Retrieval
using TF-IDF vectorization and Cosine Similarity on the TMDB 5000
movie dataset.

Run:  python -m streamlit run app.py
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
# Custom CSS – premium cinematic dark theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ═══════════════════════════════════════════════════════
   IMPORTS & VARIABLES
   ═══════════════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg-primary:    #08090d;
    --bg-secondary:  #0f1117;
    --bg-card:       #13151c;
    --bg-card-hover: #1a1d28;
    --bg-elevated:   #1c1f2e;
    --accent:        #8b5cf6;
    --accent-hover:  #a78bfa;
    --accent-glow:   rgba(139, 92, 246, 0.25);
    --accent-blue:   #3b82f6;
    --accent-cyan:   #06b6d4;
    --accent-pink:   #ec4899;
    --accent-amber:  #f59e0b;
    --text-primary:  #f1f5f9;
    --text-secondary:#94a3b8;
    --text-muted:    #64748b;
    --border:        rgba(148, 163, 184, 0.08);
    --border-hover:  rgba(139, 92, 246, 0.3);
    --gradient-main: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 50%, #06b6d4 100%);
    --gradient-warm: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
    --gradient-hero: linear-gradient(160deg, #0c0e1a 0%, #131640 40%, #1a0e2e 70%, #0c0e1a 100%);
    --shadow-card:   0 4px 24px rgba(0,0,0,0.4), 0 0 0 1px var(--border);
    --shadow-glow:   0 8px 40px rgba(139, 92, 246, 0.15);
    --radius:        16px;
    --radius-sm:     10px;
}

/* ═══════════════════════════════════════════════════════
   GLOBAL STYLES
   ═══════════════════════════════════════════════════════ */
.stApp {
    background: var(--bg-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 3px; }

/* ═══════════════════════════════════════════════════════
   TOP NAVIGATION
   ═══════════════════════════════════════════════════════ */
.top-nav {
    display: flex;
    align-items: center;
    padding: 0.8rem 1.5rem;
    background: rgba(8, 9, 13, 0.85);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--border);
    margin: -1rem -1rem 0 -1rem;
    position: sticky;
    top: 0;
    z-index: 100;
}
.brand {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    background: var(--gradient-main);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.brand-icon {
    font-size: 1.5rem;
    -webkit-text-fill-color: initial;
}

/* ═══════════════════════════════════════════════════════
   HERO SECTION
   ═══════════════════════════════════════════════════════ */
.hero-section {
    position: relative;
    text-align: center;
    padding: 5rem 2rem 4rem;
    background: var(--gradient-hero);
    border-radius: 0 0 32px 32px;
    margin: 0 -1rem 2.5rem -1rem;
    overflow: hidden;
}
.hero-section::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 40%, rgba(139, 92, 246, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 70% 60%, rgba(59, 130, 246, 0.06) 0%, transparent 50%),
                radial-gradient(circle at 50% 50%, rgba(6, 182, 212, 0.04) 0%, transparent 50%);
    animation: heroGlow 8s ease-in-out infinite alternate;
}
@keyframes heroGlow {
    0%   { transform: translate(0, 0) scale(1); }
    100% { transform: translate(-2%, -2%) scale(1.05); }
}
.hero-title {
    position: relative;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -1.5px;
    line-height: 1.15;
    background: var(--gradient-main);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 auto 0.6rem;
    max-width: 700px;
}
.hero-subtitle {
    position: relative;
    color: var(--text-secondary);
    font-size: 1.1rem;
    font-weight: 400;
    max-width: 520px;
    margin: 0 auto 0.5rem;
    line-height: 1.6;
}
.hero-badge {
    position: relative;
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--accent);
    background: rgba(139, 92, 246, 0.1);
    border: 1px solid rgba(139, 92, 246, 0.2);
    margin-bottom: 1.5rem;
}

/* ═══════════════════════════════════════════════════════
   STATS ROW
   ═══════════════════════════════════════════════════════ */
.stats-row {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin: 2rem 0;
    flex-wrap: wrap;
}
.stat-card {
    text-align: center;
    padding: 1.3rem 2rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    min-width: 150px;
    transition: all 0.3s ease;
}
.stat-card:hover {
    border-color: var(--border-hover);
    box-shadow: var(--shadow-glow);
    transform: translateY(-2px);
}
.stat-number {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    background: var(--gradient-main);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-label {
    font-size: 0.8rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.25rem;
}

/* ═══════════════════════════════════════════════════════
   SECTION HEADERS
   ═══════════════════════════════════════════════════════ */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 2rem 0 1.2rem;
}
.section-header h2 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
}
.section-header .accent-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border-hover), transparent);
}
.section-icon {
    font-size: 1.3rem;
}
.results-count {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 99px;
    font-size: 0.8rem;
    font-weight: 600;
    background: rgba(139, 92, 246, 0.12);
    color: var(--accent-hover);
    border: 1px solid rgba(139, 92, 246, 0.15);
    margin-left: 0.5rem;
}

/* ═══════════════════════════════════════════════════════
   MOVIE CARDS
   ═══════════════════════════════════════════════════════ */
.movie-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0;
    overflow: hidden;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    height: 100%;
    display: flex;
    flex-direction: column;
}
.movie-card:hover {
    transform: translateY(-6px);
    border-color: var(--border-hover);
    box-shadow: var(--shadow-glow);
}
.card-poster {
    width: 100%;
    height: 180px;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    overflow: hidden;
}
.card-poster::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 60px;
    background: linear-gradient(transparent, var(--bg-card));
}
.poster-initials {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: rgba(255,255,255,0.2);
    letter-spacing: 2px;
    text-transform: uppercase;
    z-index: 1;
}
.card-body {
    padding: 1rem 1.2rem 1.2rem;
    flex: 1;
    display: flex;
    flex-direction: column;
}
.card-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 0.5rem;
    line-height: 1.3;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.card-meta {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 0.6rem;
}
.meta-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 600;
}
.pill-genre {
    background: rgba(139, 92, 246, 0.12);
    color: var(--accent-hover);
}
.pill-year {
    background: rgba(59, 130, 246, 0.12);
    color: #93c5fd;
}
.pill-rating {
    background: rgba(245, 158, 11, 0.12);
    color: #fbbf24;
}
.card-overview {
    font-size: 0.82rem;
    color: var(--text-muted);
    line-height: 1.55;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
    margin-bottom: 0.8rem;
    flex: 1;
}
.card-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-top: 0.7rem;
    border-top: 1px solid var(--border);
}
.score-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.3rem 0.75rem;
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 700;
    background: var(--gradient-main);
    color: #fff;
}
.score-bar-container {
    flex: 1;
    margin-left: 0.6rem;
    height: 4px;
    background: rgba(255,255,255,0.06);
    border-radius: 2px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 2px;
    background: var(--gradient-main);
    transition: width 0.5s ease;
}

/* ═══════════════════════════════════════════════════════
   DETAIL PAGE
   ═══════════════════════════════════════════════════════ */
.detail-hero {
    position: relative;
    border-radius: var(--radius);
    overflow: hidden;
    margin-bottom: 2rem;
}
.detail-banner {
    width: 100%;
    height: 280px;
    display: flex;
    align-items: flex-end;
    padding: 2rem;
    position: relative;
}
.detail-banner::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(transparent 30%, var(--bg-primary) 100%);
}
.detail-banner-content {
    position: relative;
    z-index: 1;
}
.detail-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: var(--text-primary);
    margin: 0 0 0.4rem;
    line-height: 1.2;
}
.detail-tagline {
    font-size: 1rem;
    color: var(--text-secondary);
    font-style: italic;
}
.detail-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin: 1.5rem 0;
}
.detail-info-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 1.3rem;
}
.detail-info-card h4 {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-muted);
    margin: 0 0 0.6rem;
}
.detail-info-card p {
    font-size: 0.95rem;
    color: var(--text-primary);
    margin: 0;
    line-height: 1.6;
}
.detail-metric {
    text-align: center;
    padding: 1.5rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
}
.detail-metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    background: var(--gradient-main);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.detail-metric-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.3rem;
}

/* ═══════════════════════════════════════════════════════
   CHIPS / TAGS
   ═══════════════════════════════════════════════════════ */
.chip {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 99px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0.15rem;
    transition: all 0.2s;
}
.chip-purple {
    background: rgba(139, 92, 246, 0.12);
    color: var(--accent-hover);
    border: 1px solid rgba(139, 92, 246, 0.15);
}
.chip-blue {
    background: rgba(59, 130, 246, 0.12);
    color: #93c5fd;
    border: 1px solid rgba(59, 130, 246, 0.15);
}
.chip-cyan {
    background: rgba(6, 182, 212, 0.12);
    color: #67e8f9;
    border: 1px solid rgba(6, 182, 212, 0.15);
}
.chip-pink {
    background: rgba(236, 72, 153, 0.12);
    color: #f9a8d4;
    border: 1px solid rgba(236, 72, 153, 0.15);
}

/* ═══════════════════════════════════════════════════════
   SUGGESTION CARDS (Home page)
   ═══════════════════════════════════════════════════════ */
.suggestion-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 1rem 1.2rem;
    text-align: center;
    transition: all 0.3s ease;
    cursor: pointer;
}
.suggestion-card:hover {
    border-color: var(--border-hover);
    box-shadow: var(--shadow-glow);
    transform: translateY(-2px);
}
.suggestion-icon {
    font-size: 1.5rem;
    margin-bottom: 0.3rem;
}
.suggestion-text {
    font-size: 0.85rem;
    color: var(--text-secondary);
    font-weight: 500;
}

/* ═══════════════════════════════════════════════════════
   ABOUT PAGE
   ═══════════════════════════════════════════════════════ */
.about-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem;
    margin-bottom: 1.2rem;
}
.about-card h3 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.about-card p, .about-card li {
    color: var(--text-secondary);
    font-size: 0.9rem;
    line-height: 1.7;
}
.formula-box {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 1.2rem;
    text-align: center;
    margin: 1rem 0;
}
.tech-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.8rem;
    margin-top: 1rem;
}
.tech-item {
    background: var(--bg-secondary);
    border-radius: var(--radius-sm);
    padding: 0.8rem;
    text-align: center;
    border: 1px solid var(--border);
}
.tech-item .tech-name {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-primary);
}
.tech-item .tech-desc {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 0.2rem;
}

/* ═══════════════════════════════════════════════════════
   NO-RESULTS STATE
   ═══════════════════════════════════════════════════════ */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
}
.empty-state .empty-icon {
    font-size: 3.5rem;
    margin-bottom: 1rem;
    opacity: 0.5;
}
.empty-state h3 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}
.empty-state p {
    color: var(--text-muted);
    font-size: 0.9rem;
}

/* ═══════════════════════════════════════════════════════
   SIDEBAR
   ═══════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-muted);
}
.sidebar-brand {
    text-align: center;
    padding: 0.5rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1rem;
}
.sidebar-brand-text {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    background: var(--gradient-main);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ═══════════════════════════════════════════════════════
   BUTTON OVERRIDES
   ═══════════════════════════════════════════════════════ */
.stButton > button {
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    transition: all 0.2s ease !important;
    border: 1px solid var(--border) !important;
}
.stButton > button[kind="primary"] {
    background: var(--gradient-main) !important;
    border: none !important;
    color: white !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: var(--shadow-glow) !important;
    transform: translateY(-1px);
}

/* ═══════════════════════════════════════════════════════
   DIVIDER
   ═══════════════════════════════════════════════════════ */
.styled-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-hover), transparent);
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Gradient & poster helpers
# ─────────────────────────────────────────────────────────────────────────────
_GRADIENTS = [
    "linear-gradient(135deg, #4c1d95 0%, #1e3a8a 100%)",
    "linear-gradient(135deg, #701a75 0%, #7e22ce 100%)",
    "linear-gradient(135deg, #0f766e 0%, #1d4ed8 100%)",
    "linear-gradient(135deg, #9d174d 0%, #4c1d95 100%)",
    "linear-gradient(135deg, #1e40af 0%, #7c3aed 100%)",
    "linear-gradient(135deg, #065f46 0%, #3730a3 100%)",
    "linear-gradient(135deg, #7c2d12 0%, #9333ea 100%)",
    "linear-gradient(135deg, #1e3a5f 0%, #6d28d9 100%)",
    "linear-gradient(135deg, #312e81 0%, #0e7490 100%)",
    "linear-gradient(135deg, #831843 0%, #1e40af 100%)",
    "linear-gradient(135deg, #064e3b 0%, #581c87 100%)",
    "linear-gradient(135deg, #78350f 0%, #4338ca 100%)",
]

_EMOJIS = ["🎬", "🎥", "🎞️", "🎭", "🍿", "🌟", "✨", "🎪", "🎦", "📽️"]

def _gradient_for(title: str) -> str:
    idx = int(hashlib.md5(title.encode()).hexdigest(), 16) % len(_GRADIENTS)
    return _GRADIENTS[idx]

def _emoji_for(title: str) -> str:
    idx = int(hashlib.sha1(title.encode()).hexdigest(), 16) % len(_EMOJIS)
    return _EMOJIS[idx]

def _initials(title: str) -> str:
    words = title.split()
    if len(words) >= 2:
        return (words[0][0] + words[1][0]).upper()
    return title[:2].upper()


# ─────────────────────────────────────────────────────────────────────────────
# Load data & engine (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🎬 Loading movie database & building TF-IDF index …")
def _init_engine():
    df = load_and_prepare_data()
    engine = MovieSearchEngine(df)
    return engine

engine = _init_engine()


# ─────────────────────────────────────────────────────────────────────────────
# Session-state defaults
# ─────────────────────────────────────────────────────────────────────────────
for key, default in {
    "page": "home",
    "search_history": [],
    "selected_movie": None,
    "search_query": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────────
# Navigation
# ─────────────────────────────────────────────────────────────────────────────
def _render_nav():
    st.markdown("""
    <div class="top-nav">
        <div class="brand"><span class="brand-icon">🎬</span> CineFind</div>
    </div>
    """, unsafe_allow_html=True)

    active = st.session_state.page
    nav_items = [
        ("🏠  Home", "home"),
        ("🔍  Search Movies", "search"),
        ("🎯  Similar Movies", "similar"),
        ("ℹ️  About", "about"),
    ]
    cols = st.columns(len(nav_items))
    for col, (label, key) in zip(cols, nav_items):
        with col:
            btn_type = "primary" if active == key else "secondary"
            if st.button(label, use_container_width=True, type=btn_type, key=f"nav_{key}"):
                st.session_state.page = key
                st.rerun()

_render_nav()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-text">🎬 CineFind</div>
        <div style="font-size:0.7rem; color:var(--text-muted); margin-top:0.2rem;">
            Powered by TF-IDF & Cosine Similarity
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎛️ Filters")

    all_genres = engine.get_all_genres()
    genre_filter = st.multiselect("Genre", all_genres, key="filter_genre")

    yr_min, yr_max = engine.get_year_range()
    year_range = st.slider("Release Year", yr_min, yr_max, (yr_min, yr_max), key="filter_year")

    min_rating = st.slider("Minimum Rating", 0.0, 10.0, 0.0, 0.5, key="filter_rating")

    sort_options = {"Relevance": "relevance", "Highest Rating": "rating", "Release Year": "year"}
    sort_label = st.selectbox("Sort By", list(sort_options.keys()), key="filter_sort")
    sort_by = sort_options[sort_label]

    if st.button("🗑️  Clear All Filters", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.markdown("### 🕘 Recent Searches")
    if st.session_state.search_history:
        for q in st.session_state.search_history[-8:][::-1]:
            if st.button(f"🔁 {q}", key=f"hist_{q}", use_container_width=True):
                st.session_state.search_query = q
                st.session_state.page = "search"
                st.rerun()
        if st.button("🗑️  Clear History", use_container_width=True, key="clear_hist"):
            st.session_state.search_history = []
            st.rerun()
    else:
        st.caption("No recent searches yet.")


# ─────────────────────────────────────────────────────────────────────────────
# Helper — render a movie card
# ─────────────────────────────────────────────────────────────────────────────
def render_card(row, score_col=None):
    title = row.get("title", "Unknown")
    genres = row.get("genres_list", [])
    genre_str = genres[0] if genres else "—"
    year = int(row.get("release_year", 0)) or "—"
    rating = row.get("vote_average", 0)
    overview = (row.get("overview", "") or "No overview available.")[:180]
    grad = _gradient_for(title)
    initials = _initials(title)

    score_html = ""
    bar_html = ""
    if score_col and score_col in row and row[score_col] > 0:
        pct = row[score_col] * 100
        bar_width = min(pct * 3, 100)  # Scale for visual
        score_html = f'<span class="score-badge">🎯 {pct:.1f}%</span>'
        bar_html = f'''
        <div class="score-bar-container">
            <div class="score-bar-fill" style="width:{bar_width}%"></div>
        </div>'''

    st.markdown(f"""
    <div class="movie-card">
        <div class="card-poster" style="background:{grad};">
            <span class="poster-initials">{initials}</span>
        </div>
        <div class="card-body">
            <div class="card-title">{title}</div>
            <div class="card-meta">
                <span class="meta-pill pill-genre">🎭 {genre_str}</span>
                <span class="meta-pill pill-year">📅 {year}</span>
                <span class="meta-pill pill-rating">⭐ {rating:.1f}</span>
            </div>
            <div class="card-overview">{overview}</div>
            <div class="card-footer">
                {score_html}
                {bar_html}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔍 View Details", key=f"detail_{title}_{score_col}_{id(row)}",
                 use_container_width=True):
        st.session_state.selected_movie = title
        st.session_state.page = "details"
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Helper — section header
# ─────────────────────────────────────────────────────────────────────────────
def section_header(icon, text, count=None):
    count_html = f'<span class="results-count">{count}</span>' if count else ""
    st.markdown(f"""
    <div class="section-header">
        <span class="section-icon">{icon}</span>
        <h2>{text}{count_html}</h2>
        <div class="accent-line"></div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════════════════════════

# ── HOME ────────────────────────────────────────────────────────────────────
if st.session_state.page == "home":
    # Hero section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-badge">✦ Powered by TF-IDF & Cosine Similarity</div>
        <div class="hero-title">Discover Your Next<br>Favourite Movie</div>
        <div class="hero-subtitle">
            Search across 4,800+ movies using intelligent text retrieval.
            Find films by plot, genre, actors, directors, or keywords.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Search bar
    col_search, col_btn = st.columns([5, 1])
    with col_search:
        query = st.text_input(
            "search_home",
            value=st.session_state.search_query,
            placeholder="🔍  Try: science fiction movies about space, movies starring Leonardo DiCaprio ...",
            label_visibility="collapsed",
            key="home_search_input",
        )
    with col_btn:
        search_clicked = st.button("🚀 Search", type="primary", use_container_width=True, key="home_search_btn")

    if search_clicked and query.strip():
        st.session_state.search_query = query.strip()
        hist = st.session_state.search_history
        if query.strip() not in hist:
            hist.append(query.strip())
            st.session_state.search_history = hist[-15:]
        st.session_state.page = "search"
        st.rerun()

    # Stats
    n_movies = len(engine.df)
    n_genres = len(engine.get_all_genres())
    yr_lo, yr_hi = engine.get_year_range()
    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-number">{n_movies:,}</div>
            <div class="stat-label">Movies</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{n_genres}</div>
            <div class="stat-label">Genres</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{yr_lo}–{yr_hi}</div>
            <div class="stat-label">Year Range</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">10K</div>
            <div class="stat-label">TF-IDF Features</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick suggestions
    section_header("🎯", "Quick Suggestions")
    suggestions = [
        ("🚀", "Science fiction movies about space"),
        ("🦸", "Action movies with superheroes"),
        ("💕", "Romantic movies"),
        ("🌊", "Adventure movies about the ocean"),
        ("🤖", "Movies about artificial intelligence"),
        ("🎭", "Movies starring Leonardo DiCaprio"),
    ]
    cols = st.columns(3)
    for i, (icon, text) in enumerate(suggestions):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="suggestion-card">
                <div class="suggestion-icon">{icon}</div>
                <div class="suggestion-text">{text}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Search →", key=f"sug_{i}", use_container_width=True):
                st.session_state.search_query = text
                hist = st.session_state.search_history
                if text not in hist:
                    hist.append(text)
                    st.session_state.search_history = hist[-15:]
                st.session_state.page = "search"
                st.rerun()


# ── SEARCH ──────────────────────────────────────────────────────────────────
elif st.session_state.page == "search":
    section_header("🔍", "Search Movies")

    col_s, col_b = st.columns([5, 1])
    with col_s:
        query = st.text_input(
            "search_q",
            value=st.session_state.search_query,
            placeholder="🔍  What kind of movie are you looking for?",
            label_visibility="collapsed",
            key="search_input",
        )
    with col_b:
        if st.button("🚀 Search", type="primary", use_container_width=True, key="search_btn"):
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
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">🔍</div>
                <h3>No Movies Found</h3>
                <p>Try different keywords or relax your filters to see more results.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            section_header("📊", f"Results for \"{st.session_state.search_query}\"",
                           count=f"{len(results)} movies")
            st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

            cols = st.columns(3)
            for i, (_, row) in enumerate(results.iterrows()):
                with cols[i % 3]:
                    render_card(row.to_dict(), score_col="relevance_score")
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🎬</div>
            <h3>Start Searching</h3>
            <p>Enter a query above to discover movies.</p>
        </div>
        """, unsafe_allow_html=True)


# ── SIMILAR MOVIES ──────────────────────────────────────────────────────────
elif st.session_state.page == "similar":
    section_header("🎯", "Find Similar Movies")
    st.caption("Select a movie to discover films with similar content, themes, and style.")

    all_titles = engine.get_all_titles()
    selected = st.selectbox("Select a movie", [""] + all_titles,
                            key="similar_select",
                            label_visibility="collapsed")

    if selected:
        similar = engine.get_similar_movies(selected, top_n=5)

        # Show selected movie info
        details = engine.get_movie_details(selected)
        if details:
            grad = _gradient_for(selected)
            st.markdown(f"""
            <div style="background:{grad}; border-radius:var(--radius);
                        padding:1.5rem; margin:1rem 0 1.5rem; position:relative; overflow:hidden;">
                <div style="position:relative; z-index:1;">
                    <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:1px;
                                color:rgba(255,255,255,0.6); margin-bottom:0.3rem;">Finding movies similar to</div>
                    <div style="font-family:'Space Grotesk',sans-serif; font-size:1.5rem;
                                font-weight:700; color:white;">{selected}</div>
                    <div style="color:rgba(255,255,255,0.7); font-size:0.85rem; margin-top:0.3rem;">
                        {', '.join(details['genres'][:3])} · {details['release_year']} · ⭐ {details['vote_average']:.1f}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if similar.empty:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">🤔</div>
                <h3>No Similar Movies Found</h3>
                <p>This movie doesn't have enough content overlap with others in the dataset.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            section_header("🍿", f"Top {len(similar)} Similar Movies")
            cols = st.columns(3)
            for i, (_, row) in enumerate(similar.iterrows()):
                with cols[i % 3]:
                    render_card(row.to_dict(), score_col="similarity_score")
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🎯</div>
            <h3>Select a Movie</h3>
            <p>Choose a movie title from the dropdown above to get content-based recommendations.</p>
        </div>
        """, unsafe_allow_html=True)


# ── MOVIE DETAILS ───────────────────────────────────────────────────────────
elif st.session_state.page == "details":
    title = st.session_state.selected_movie
    if title:
        details = engine.get_movie_details(title)
        if details:
            grad = _gradient_for(details["title"])
            initials = _initials(details["title"])

            # Banner
            st.markdown(f"""
            <div class="detail-hero">
                <div class="detail-banner" style="background:{grad};">
                    <div class="detail-banner-content">
                        <div class="detail-title">{details["title"]}</div>
                        <div class="detail-tagline">
                            {', '.join(details['genres'])} · {details['release_year']}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""
                <div class="detail-metric">
                    <div class="detail-metric-value">⭐ {details['vote_average']:.1f}</div>
                    <div class="detail-metric-label">Rating</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="detail-metric">
                    <div class="detail-metric-value">📅 {details['release_year'] or '—'}</div>
                    <div class="detail-metric-label">Year</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="detail-metric">
                    <div class="detail-metric-value">🔥 {details['popularity']:.0f}</div>
                    <div class="detail-metric-label">Popularity</div>
                </div>""", unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class="detail-metric">
                    <div class="detail-metric-value">🗳️ {details['vote_count']:,}</div>
                    <div class="detail-metric-label">Votes</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("")

            # Info grid
            col_left, col_right = st.columns([3, 2])

            with col_left:
                # Overview
                st.markdown(f"""
                <div class="detail-info-card">
                    <h4>📖 Plot Overview</h4>
                    <p>{details['overview'] or 'No overview available.'}</p>
                </div>
                """, unsafe_allow_html=True)

                # Keywords
                kw_chips = " ".join(
                    f'<span class="chip chip-blue">{k}</span>'
                    for k in details["keywords"][:12]
                )
                st.markdown(f"""
                <div class="detail-info-card" style="margin-top:1rem;">
                    <h4>🔑 Keywords</h4>
                    <p>{kw_chips or '—'}</p>
                </div>
                """, unsafe_allow_html=True)

            with col_right:
                # Genres
                genre_chips = " ".join(
                    f'<span class="chip chip-purple">{g}</span>'
                    for g in details["genres"]
                )
                st.markdown(f"""
                <div class="detail-info-card">
                    <h4>🎭 Genres</h4>
                    <p>{genre_chips}</p>
                </div>
                """, unsafe_allow_html=True)

                # Cast & Crew
                actor_chips = " ".join(
                    f'<span class="chip chip-cyan">{a}</span>'
                    for a in details["actors"]
                )
                director_chip = f'<span class="chip chip-pink">{details["director"]}</span>' if details["director"] else "—"
                st.markdown(f"""
                <div class="detail-info-card" style="margin-top:1rem;">
                    <h4>🎭 Main Cast</h4>
                    <p>{actor_chips or '—'}</p>
                    <h4 style="margin-top:0.8rem;">🎬 Director</h4>
                    <p>{director_chip}</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

            # Action buttons
            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("🎯 Find Similar Movies", type="primary", use_container_width=True):
                    st.session_state.page = "similar"
                    st.session_state.similar_select = title
                    st.rerun()
            with b2:
                if st.button("🔍 Back to Search", use_container_width=True):
                    st.session_state.page = "search"
                    st.rerun()
            with b3:
                if st.button("🏠 Go Home", use_container_width=True):
                    st.session_state.page = "home"
                    st.rerun()
        else:
            st.error("Movie not found in the database.")
    else:
        st.session_state.page = "home"
        st.rerun()


# ── ABOUT ───────────────────────────────────────────────────────────────────
elif st.session_state.page == "about":
    st.markdown("""
    <div class="hero-section" style="padding:3rem 2rem; margin-bottom:2rem;">
        <div class="hero-badge">✦ Academic Mini-Project</div>
        <div class="hero-title" style="font-size:2.5rem;">About CineFind</div>
        <div class="hero-subtitle">
            An Information Retrieval system demonstrating movie search
            using TF-IDF Vectorization and Cosine Similarity.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="about-card">
            <h3>🎯 Objectives</h3>
            <ul>
                <li>Build a genuine IR system — no LLM black boxes or hardcoded results</li>
                <li>Free-text search across titles, genres, keywords, overviews, cast & crew</li>
                <li>Content-based movie recommendations using cosine similarity</li>
                <li>Modern, cinematic user interface with filters and search history</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card">
            <h3>📊 Dataset</h3>
            <p><strong>TMDB 5000 Movie Dataset</strong> from Kaggle — 4,803 movies
            with full metadata including title, genres, keywords, plot overview,
            cast, crew, ratings, and popularity scores.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="about-card">
            <h3>🔧 Technology Stack</h3>
            <div class="tech-grid">
                <div class="tech-item">
                    <div class="tech-name">Python</div>
                    <div class="tech-desc">Language</div>
                </div>
                <div class="tech-item">
                    <div class="tech-name">Streamlit</div>
                    <div class="tech-desc">Web UI</div>
                </div>
                <div class="tech-item">
                    <div class="tech-name">Pandas</div>
                    <div class="tech-desc">Data Processing</div>
                </div>
                <div class="tech-item">
                    <div class="tech-name">NLTK</div>
                    <div class="tech-desc">NLP</div>
                </div>
                <div class="tech-item">
                    <div class="tech-name">Scikit-learn</div>
                    <div class="tech-desc">TF-IDF / ML</div>
                </div>
                <div class="tech-item">
                    <div class="tech-name">NumPy</div>
                    <div class="tech-desc">Computation</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # How it works
    st.markdown("""
    <div class="about-card">
        <h3>📐 How It Works</h3>
        <p>Each movie is represented as a document combining its title, genres, keywords,
        overview, actors, and director. The system uses two core algorithms:</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="about-card">
            <h3>📊 TF-IDF Vectorization</h3>
            <p>Converts text into numerical vectors where each dimension represents
            a word's importance relative to the corpus. Words frequent in one movie
            but rare across all movies get higher weights.</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\text{TF-IDF}(t,d) = \text{TF}(t,d) \times \log\!\left(\frac{N}{\text{DF}(t)}\right)")

    with c2:
        st.markdown("""
        <div class="about-card">
            <h3>📏 Cosine Similarity</h3>
            <p>Measures the angle between two TF-IDF vectors to determine how similar
            two movies are, regardless of document length. A score of 1.0 means
            identical content direction; 0.0 means completely unrelated.</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\text{similarity}(A,B) = \frac{A \cdot B}{\|A\| \; \|B\|}")

    st.markdown("""
    <div class="styled-divider"></div>
    <div style="text-align:center; padding:1rem; color:var(--text-muted); font-size:0.85rem;">
        Built with ❤️ using Python, Streamlit & Scikit-learn
    </div>
    """, unsafe_allow_html=True)
