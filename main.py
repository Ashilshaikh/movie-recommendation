import os
import pickle
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv


# ========================= 
# ENV - Load API Key
# ========================= 
# Load environment variables from .env file
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    # Don't crash import-time in production if you prefer; but for you better fail early:
    raise RuntimeError("TMDB_API_KEY missing. Put it in .env as TMDB_API_KEY=xxxx")


# ========================= 
# FASTAPI APP - Initialize Server
# ========================= 
# Create FastAPI application instance
app = FastAPI(title="Movie Recommender API", version="3.0")

# Add CORS (Cross-Origin Resource Sharing) middleware
# Allows requests from any origin (needed for Streamlit frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========================= 
# PICKLE GLOBALS - Loaded on Startup
# ========================= 
# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to saved model files
DF_PATH = os.path.join(BASE_DIR, "df.pkl")  # Movies dataframe
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")  # Title to index mapping
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")  # Sparse TF-IDF matrix
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")  # TF-IDF vectorizer

# Global variables (loaded at startup)
df: Optional[pd.DataFrame] = None  # Movies dataframe with 'title' column
indices_obj: Any = None  # Title index mapping
tfidf_matrix: Any = None  # Sparse matrix for similarity calculations
tfidf_obj: Any = None  # TF-IDF vectorizer object

TITLE_TO_IDX: Optional[Dict[str, int]] = None  # Normalized title -> index mapping


# ========================= 
# MODELS - Pydantic Data Structures
# ========================= 
class TMDBMovieCard(BaseModel):
    """Lightweight movie info: ID, title, poster, release date, rating"""
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class TMDBMovieDetails(BaseModel):
    """Full movie details: overview, genres, backdrop, release date"""
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []


class TFIDFRecItem(BaseModel):
    """TF-IDF recommendation: title, similarity score, and optional TMDB card"""
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    """Complete recommendation bundle: movie details + TF-IDF recs + genre recs"""
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]


# ========================= 
# UTILS - Helper Functions
# ========================= 
def _norm_title(t: str) -> str:
    """Normalize title: strip whitespace and convert to lowercase for consistent matching"""
    return str(t).strip().lower()


def make_img_url(path: Optional[str]) -> Optional[str]:
    """Convert TMDB image path to full URL. Returns None if path is empty"""
    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"


async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Async GET request to TMDB API with error handling.
    - Adds API key to params
    - Network errors -> HTTPException 502
    - TMDB API errors -> HTTPException 502 with detail
    """
    q = dict(params)
    q["api_key"] = TMDB_API_KEY

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{TMDB_BASE}{path}", params=q)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"TMDB request error: {type(e).__name__} | {repr(e)}",
        )

    if r.status_code != 200:
        raise HTTPException(
            status_code=502, detail=f"TMDB error {r.status_code}: {r.text}"
        )

    return r.json()


async def tmdb_cards_from_results(
    results: List[dict], limit: int = 20
) -> List[TMDBMovieCard]:
    """Convert TMDB API search results to TMDBMovieCard objects (max limit items)"""
    out: List[TMDBMovieCard] = []
    for m in (results or [])[:limit]:
        out.append(
            TMDBMovieCard(
                tmdb_id=int(m["id"]),
                title=m.get("title") or m.get("name") or "",
                poster_url=make_img_url(m.get("poster_path")),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            )
        )
    return out


async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    """Fetch full movie details from TMDB API by movie ID"""
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or [],
    )


async def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    """
    Keyword search on TMDB returns MULTIPLE results.
    Used by Streamlit for suggestions and grid display.
    """
    return await tmdb_get(
        "/search/movie",
        {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": page,
        },
    )


async def tmdb_search_first(query: str) -> Optional[dict]:
    """Search TMDB and return only the FIRST result (or None if no results)"""
    data = await tmdb_search_movies(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None


# ========================= 
# TF-IDF Helpers - Local Recommendations
# ========================= 
def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    """
    Build normalized title -> index mapping from indices.pkl.
    Supports dict or pandas Series-like objects.
    """
    title_to_idx: Dict[str, int] = {}

    if isinstance(indices, dict):
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx

    # pandas Series or similar mapping
    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception:
        # last resort: if it's a list-like etc.
        raise RuntimeError(
            "indices.pkl must be dict or pandas Series-like (with .items())"
        )


def get_local_idx_by_title(title: str) -> int:
    """Get local dataset index by movie title. Raises 404 if not found"""
    global TITLE_TO_IDX
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TF-IDF index map not initialized")
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    raise HTTPException(
        status_code=404, detail=f"Title not found in local dataset: '{title}'"
    )


def tfidf_recommend_titles(
    query_title: str, top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Generate TF-IDF recommendations using cosine similarity on the TF-IDF matrix.
    Returns list of (title, similarity_score) tuples in descending order.
    """
    global df, tfidf_matrix
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")

    # Get index of query movie in local dataset
    idx = get_local_idx_by_title(query_title)

    # Calculate similarity scores: TF-IDF matrix @ query vector
    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()

    # Sort in descending order (highest similarity first)
    order = np.argsort(-scores)

    out: List[Tuple[str, float]] = []
    for i in order:
        if int(i) == int(idx):
            continue  # Skip the query movie itself
        try:
            title_i = str(df.iloc[int(i)]["title"])
        except Exception:
            continue
        out.append((title_i, float(scores[int(i)])))
        if len(out) >= top_n:
            break
    return out


async def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    """
    Search TMDB by title and fetch poster/details.
    Returns None if not found (never crashes the endpoint).
    """
    try:
        m = await tmdb_search_first(title)
        if not m:
            return None
        return TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or title,
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
    except Exception:
        return None


# ========================= 
# STARTUP: Load Pickle Files
# ========================= 
@app.on_event("startup")
def load_pickles():
    """Load all serialized data at server startup"""
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    # Load movies dataframe
    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    # Load indices (title -> index mapping)
    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)

    # Load TF-IDF sparse matrix (used for similarity calculations)
    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    # Load TF-IDF vectorizer (optional, not used directly in current endpoints)
    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    # Build normalized title -> index mapping
    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    # Validate dataframe structure
    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pkl must contain a DataFrame with a 'title' column")


# ========================= 
# ROUTES - API Endpoints
# ========================= 
@app.get("/health")
def health():
    """Health check endpoint - returns status ok"""
    return {"status": "ok"}


# ---------- HOME FEED (TMDB Popular Movies) ----------
@app.get("/home", response_model=List[TMDBMovieCard])
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),
):
    """
    Get popular movies for home feed (with posters).
    Categories: trending, popular, top_rated, upcoming, now_playing
    """
    try:
        if category == "trending":
            data = await tmdb_get("/trending/movie/day", {"language": "en-US"})
            return await tmdb_cards_from_results(data.get("results", []), limit=limit)

        if category not in {"popular", "top_rated", "upcoming", "now_playing"}:
            raise HTTPException(status_code=400, detail="Invalid category")

        data = await tmdb_get(f"/movie/{category}", {"language": "en-US", "page": 1})
        return await tmdb_cards_from_results(data.get("results", []), limit=limit)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Home route failed: {e}")


# ---------- TMDB KEYWORD SEARCH (Multiple Results) ----------
@app.get("/tmdb/search")
async def tmdb_search(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=10),
):
    """
    Search TMDB for multiple movies by keyword.
    Returns raw TMDB response with results list.
    """
    return await tmdb_search_movies(query=query, page=page)


# ---------- MOVIE DETAILS (Get Full Info) ----------
@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetails)
async def movie_details_route(tmdb_id: int):
    """Get full movie details (overview, genres, images) by TMDB ID"""
    return await tmdb_movie_details(tmdb_id)


# ---------- GENRE RECOMMENDATIONS (TMDB Movies in Same Genre) ----------
@app.get("/recommend/genre", response_model=List[TMDBMovieCard])
async def recommend_genre(
    tmdb_id: int = Query(...),
    limit: int = Query(18, ge=1, le=50),
):
    """
    Get movies in the same genre as the given movie.
    Uses TMDB discover endpoint filtered by first genre.
    """
    details = await tmdb_movie_details(tmdb_id)
    if not details.genres:
        return []

    genre_id = details.genres[0]["id"]
    discover = await tmdb_get(
        "/discover/movie",
        {
            "with_genres": genre_id,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "page": 1,
        },
    )
    cards = await tmdb_cards_from_results(discover.get("results", []), limit=limit)
    return [c for c in cards if c.tmdb_id != tmdb_id]


# ---------- TF-IDF ONLY (Local Recommendations) ----------
@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    """Get TF-IDF recommendations (local dataset only, no TMDB posters)"""
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score": s} for t, s in recs]


# ---------- BUNDLE: Details + TF-IDF + Genre Recommendations ----------
@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    """
    Complete recommendation package:
      1. Movie details from TMDB
      2. TF-IDF recommendations with TMDB posters
      3. Genre recommendations from TMDB
    """
    best = await tmdb_search_first(query)
    if not best:
        raise HTTPException(
            status_code=404, detail=f"No TMDB movie found for query: {query}"
        )

    tmdb_id = int(best["id"])
    details = await tmdb_movie_details(tmdb_id)

    # 1) TF-IDF recommendations (never crash endpoint)
    tfidf_items: List[TFIDFRecItem] = []

    recs: List[Tuple[str, float]] = []
    try:
        # try local dataset by TMDB title
        recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
    except Exception:
        # fallback to user query
        try:
            recs = tfidf_recommend_titles(query, top_n=tfidf_top_n)
        except Exception:
            recs = []

    for title, score in recs:
        card = await attach_tmdb_card_by_title(title)
        tfidf_items.append(TFIDFRecItem(title=title, score=score, tmdb=card))

    # 2) Genre recommendations (TMDB discover by first genre)
    genre_recs: List[TMDBMovieCard] = []
    if details.genres:
        genre_id = details.genres[0]["id"]
        discover = await tmdb_get(
            "/discover/movie",
            {
                "with_genres": genre_id,
                "language": "en-US",
                "sort_by": "popularity.desc",
                "page": 1,
            },
        )
        cards = await tmdb_cards_from_results(
            discover.get("results", []), limit=genre_limit
        )
        genre_recs = [c for c in cards if c.tmdb_id != details.tmdb_id]

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )