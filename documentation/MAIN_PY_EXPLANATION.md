# 🎬 Main.py - Complete Code Explanation

**A Beginner-Friendly Guide to the Movie Recommender API**

---

## 📋 Table of Contents
1. [Imports & Setup](#imports--setup)
2. [Environment Variables](#environment-variables)
3. [FastAPI App & CORS](#fastapi-app--cors)
4. [Pickle Files (Data Loading)](#pickle-files-data-loading)
5. [Data Models (Pydantic)](#data-models-pydantic)
6. [Helper Functions](#helper-functions)
7. [TF-IDF Recommendation Logic](#tfidf-recommendation-logic)
8. [Startup Function](#startup-function)
9. [API Routes (Endpoints)](#api-routes-endpoints)

---

## 🔧 Imports & Setup

### What Are Imports?
Imports are like borrowing tools from a toolbox. We import libraries (pre-written code) to use in our project.

```python
import os                           # Read environment variables and file paths
import pickle                       # Load saved machine learning models
from typing import Optional, List, Dict, Any, Tuple  # Type hints for clarity

import numpy as np                  # Math operations on numbers
import pandas as pd                 # Work with data in tables (DataFrames)
import httpx                        # Make async HTTP requests (talk to TMDB API)
from fastapi import FastAPI, HTTPException, Query  # Build the web API
from fastapi.middleware.cors import CORSMiddleware  # Allow cross-origin requests
from pydantic import BaseModel      # Validate incoming/outgoing data
from dotenv import load_dotenv      # Load .env file with secrets
```

**In Simple Terms:**
- We're gathering tools to build a web server
- These tools help us talk to the TMDB API, store data, and handle web requests

---

## 🔐 Environment Variables

### Why This Section Matters
API keys are SECRET passwords. Never hardcode them in code!

```python
# Load environment variables from .env file
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# TMDB API base URL (where the movie data lives)
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

# Safety check: fail if API key is missing
if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY missing. Put it in .env as TMDB_API_KEY=xxxx")
```

**What This Does:**
- `load_dotenv()` → Reads your `.env` file (contains secrets)
- `os.getenv("TMDB_API_KEY")` → Gets the API key from environment
- `if not TMDB_API_KEY` → Checks if key exists; crashes if not (better than crashing later!)

**Why Safe?**
- API key lives in `.env`, NOT in code
- `.env` is in `.gitignore` (won't be uploaded to GitHub)

---

## 🚀 FastAPI App & CORS

### What is FastAPI?
FastAPI is a tool to build web servers that handle HTTP requests (like when Streamlit calls our API).

```python
# Create FastAPI application instance
app = FastAPI(title="Movie Recommender API", version="3.0")

# Add CORS (Cross-Origin Resource Sharing) middleware
# This allows the Streamlit frontend (different origin) to call our API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],              # Allow requests from ANY origin
    allow_credentials=True,            # Allow cookies/auth headers
    allow_methods=["*"],               # Allow GET, POST, DELETE, etc.
    allow_headers=["*"],               # Allow any headers in requests
)
```

**What This Does:**
- Creates a web server that listens for requests
- CORS middleware = "permission slip" for Streamlit to talk to FastAPI
- `["*"]` means accept from anywhere (fine for local testing)

**Simple Analogy:**
- FastAPI = Restaurant
- CORS = Allowing customers from anywhere to enter

---

## 📦 Pickle Files (Data Loading)

### What Are Pickle Files?
Pickle files are Python's way of saving objects (like trained machine learning models) to disk.

```python
# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to saved model files
DF_PATH = os.path.join(BASE_DIR, "df.pkl")              # Movies database
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")    # Title -> Index mapping
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")  # Similarity matrix
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")        # TF-IDF vectorizer

# Global variables (will be filled when server starts)
df: Optional[pd.DataFrame] = None           # None = not loaded yet
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None  # Title -> Index lookup dictionary
```

**What These Files Contain:**

| File | What It Is | Example |
|------|-----------|---------|
| `df.pkl` | All movies + metadata | ID, title, genre, etc. |
| `indices.pkl` | Movie title to array index | "The Matrix" → 42 |
| `tfidf_matrix.pkl` | Similarity matrix | How similar movies are |
| `tfidf.pkl` | TF-IDF vectorizer | Convert text to numbers |

**Why Global Variables?**
- Load once at startup (expensive)
- Use in all requests (fast, no reloading needed)

---

## 📊 Data Models (Pydantic)

### What are Pydantic Models?
They're blueprints for data. They validate that incoming/outgoing data has the right format.

#### **TMDBMovieCard** - Lightweight Movie Info
```python
class TMDBMovieCard(BaseModel):
    """Lightweight movie info: ID, title, poster, release date, rating"""
    tmdb_id: int                           # Movie ID from TMDB
    title: str                             # Movie title
    poster_url: Optional[str] = None       # Poster image URL (may not exist)
    release_date: Optional[str] = None     # Release date like "2023-05-10"
    vote_average: Optional[float] = None   # Rating like 8.5
```

**Real Example:**
```json
{
  "tmdb_id": 550,
  "title": "Fight Club",
  "poster_url": "https://image.tmdb.org/t/p/w500/...",
  "release_date": "1999-10-15",
  "vote_average": 8.8
}
```

#### **TMDBMovieDetails** - Full Movie Info
```python
class TMDBMovieDetails(BaseModel):
    """Full movie details: overview, genres, backdrop, release date"""
    tmdb_id: int
    title: str
    overview: Optional[str] = None         # Plot description
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None     # Large background image
    genres: List[dict] = []                # List of genres like [{"id": 28, "name": "Action"}]
```

#### **TFIDFRecItem** - TF-IDF Recommendation
```python
class TFIDFRecItem(BaseModel):
    """TF-IDF recommendation: title, similarity score, and optional TMDB card"""
    title: str                             # Recommended movie title
    score: float                           # Similarity score (0-1)
    tmdb: Optional[TMDBMovieCard] = None   # Full movie card (poster, etc.)
```

**Real Example:**
```json
{
  "title": "Se7en",
  "score": 0.876,
  "tmdb": {
    "tmdb_id": 807,
    "title": "Se7en",
    "poster_url": "...",
    "vote_average": 8.6
  }
}
```

#### **SearchBundleResponse** - Complete Package
```python
class SearchBundleResponse(BaseModel):
    """Complete recommendation bundle: details + TF-IDF + genre recs"""
    query: str                                    # Original search query
    movie_details: TMDBMovieDetails               # Full movie info
    tfidf_recommendations: List[TFIDFRecItem]     # 12 similar movies
    genre_recommendations: List[TMDBMovieCard]    # 12 movies in same genre
```

---

## 🛠️ Helper Functions

### 1. `_norm_title()` - Normalize Text
```python
def _norm_title(t: str) -> str:
    """Normalize title: strip whitespace and convert to lowercase"""
    return str(t).strip().lower()
```

**Why This Matters:**
- "The Matrix" vs "the matrix" vs "  the matrix  " are the SAME
- Normalization makes them all identical for matching

**Example:**
```
Input:  "  THE MATRIX  "
Output: "the matrix"
```

### 2. `make_img_url()` - Build Image URL
```python
def make_img_url(path: Optional[str]) -> Optional[str]:
    """Convert TMDB image path to full URL"""
    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"
```

**What It Does:**
- TMDB gives partial path: `/w500/pEFRqqPPfWX9kVmRQmRtNdKH9Kv.jpg`
- We add the domain: `https://image.tmdb.org/t/p/w500/pEFRqqPPfWX9kVmRQmRtNdKH9Kv.jpg`

---

### 3. `tmdb_get()` - Talk to TMDB API (Async)
```python
async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Async GET request to TMDB API with error handling.
    - Adds API key to params
    - Network errors -> HTTPException 502
    - TMDB API errors -> HTTPException 502 with detail
    """
    q = dict(params)
    q["api_key"] = TMDB_API_KEY  # Add API key to request

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{TMDB_BASE}{path}", params=q)
    except httpx.RequestError as e:
        # Network error (connection failed, timeout, etc.)
        raise HTTPException(
            status_code=502,
            detail=f"TMDB request error: {type(e).__name__} | {repr(e)}",
        )

    # Check if TMDB responded with an error
    if r.status_code != 200:
        raise HTTPException(
            status_code=502, 
            detail=f"TMDB error {r.status_code}: {r.text}"
        )

    return r.json()  # Return parsed JSON response
```

**What This Does:**
1. Add API key to request params
2. Make async HTTP GET request (doesn't block other requests)
3. Handle network errors gracefully
4. Check response status
5. Return parsed JSON

**Why Async?**
- Multiple Streamlit users can call API simultaneously
- Async allows server to handle many requests without waiting

---

### 4. `tmdb_cards_from_results()` - Convert Search Results
```python
async def tmdb_cards_from_results(
    results: List[dict], limit: int = 20
) -> List[TMDBMovieCard]:
    """Convert TMDB API search results to TMDBMovieCard objects"""
    out: List[TMDBMovieCard] = []
    for m in (results or [])[:limit]:
        out.append(
            TMDBMovieCard(
                tmdb_id=int(m["id"]),
                title=m.get("title") or m.get("name") or "",  # Handle TV shows too
                poster_url=make_img_url(m.get("poster_path")),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            )
        )
    return out
```

**What It Does:**
- Takes raw TMDB JSON results
- Converts to nice TMDBMovieCard objects
- Limit to first 20 (or specified limit)

---

### 5. `tmdb_movie_details()` - Get Full Movie Info
```python
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
```

**What It Does:**
- Call `/movie/{id}` endpoint on TMDB API
- Extract fields and build TMDBMovieDetails object

---

### 6. `tmdb_search_movies()` - Keyword Search
```python
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
```

**What It Does:**
- Search TMDB for movies matching a keyword
- Return raw response with multiple results
- Support pagination (page 1, 2, 3...)

---

### 7. `tmdb_search_first()` - Get First Match
```python
async def tmdb_search_first(query: str) -> Optional[dict]:
    """Search TMDB and return only the FIRST result (or None)"""
    data = await tmdb_search_movies(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None
```

**What It Does:**
- Search and take only the TOP result
- Return None if no results found

---

## 🧠 TF-IDF Recommendation Logic

### Understanding TF-IDF
**TF-IDF** = Term Frequency - Inverse Document Frequency
- Converts movie plots/text into numbers
- Calculates similarity between movies
- Used for "Find similar movies" feature

### 1. `build_title_to_idx_map()` - Create Lookup Table
```python
def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    """
    Build normalized title -> index mapping from indices.pkl.
    Supports dict or pandas Series-like objects.
    
    Example:
      Input:  {"The Matrix": 42, "Inception": 100}
      Output: {"the matrix": 42, "inception": 100}
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
        raise RuntimeError(
            "indices.pkl must be dict or pandas Series-like (with .items())"
        )
```

**What It Does:**
1. Load indices from pickle file (title → array position)
2. Normalize all titles (lowercase, trim whitespace)
3. Create fast lookup dictionary

---

### 2. `get_local_idx_by_title()` - Find Movie Index
```python
def get_local_idx_by_title(title: str) -> int:
    """Get local dataset index by movie title. Raises 404 if not found"""
    global TITLE_TO_IDX
    if TITLE_TO_IDX is None:
        raise HTTPException(
            status_code=500, 
            detail="TF-IDF index map not initialized"
        )
    
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    
    # Movie not in local dataset
    raise HTTPException(
        status_code=404, 
        detail=f"Title not found in local dataset: '{title}'"
    )
```

**What It Does:**
1. Normalize the movie title
2. Look it up in TITLE_TO_IDX dictionary
3. Return the array index
4. Raise error if not found

**Example:**
```python
# If we ask for index of "The Matrix"
idx = get_local_idx_by_title("the matrix")
# Returns: 42 (position in df.pkl and tfidf_matrix)
```

---

### 3. `tfidf_recommend_titles()` - Core Recommendation Engine
```python
def tfidf_recommend_titles(
    query_title: str, top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Generate TF-IDF recommendations using cosine similarity.
    Returns list of (title, similarity_score) tuples.
    """
    global df, tfidf_matrix
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")

    # Step 1: Get index of query movie in local dataset
    idx = get_local_idx_by_title(query_title)

    # Step 2: Get the TF-IDF vector for this movie
    qv = tfidf_matrix[idx]

    # Step 3: Calculate similarity scores with ALL movies
    # Matrix multiplication: tfidf_matrix @ query_vector
    # Result: similarity score for each movie
    scores = (tfidf_matrix @ qv.T).toarray().ravel()

    # Step 4: Sort movies by similarity (highest first)
    order = np.argsort(-scores)

    # Step 5: Build output list
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
            break  # Stop when we have top_n recommendations
    
    return out
```

**How This Works (Step-by-Step):**

1. **Find Query Movie:** Get array index of "The Matrix"
2. **Get Vector:** Get TF-IDF vector for "The Matrix"
   - Vector looks like: `[0.1, 0.5, 0.3, 0.0, ..., 0.8]`
   - Each number represents importance of different words
3. **Calculate Similarity:** Multiply with ALL movie vectors
   - High similarity = similar words = similar movies
4. **Sort:** Arrange by similarity (best first)
5. **Return:** Top 10 recommendations

**Example Output:**
```
[
  ("Se7en", 0.876),
  ("Disturbing Behavior", 0.654),
  ("Omega Code", 0.534),
  ...
]
```

---

### 4. `attach_tmdb_card_by_title()` - Add Poster to Recommendation
```python
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
        return None  # Never crash, just skip if no poster found
```

**What It Does:**
- Search TMDB for movie by title
- Add poster and other details
- Return None gracefully if not found

**Why This Matters:**
- TF-IDF recommends from old local data
- But posters come from live TMDB API
- Users see current images for recommendations

---

## 🚀 Startup Function

### What Happens When Server Starts?

```python
@app.on_event("startup")
def load_pickles():
    """Load all serialized data at server startup"""
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    # Step 1: Load movies dataframe
    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    # Step 2: Load indices (title -> index mapping)
    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)

    # Step 3: Load TF-IDF sparse matrix (used for similarity calculations)
    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    # Step 4: Load TF-IDF vectorizer (optional, not used directly)
    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    # Step 5: Build normalized title -> index mapping
    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    # Step 6: Validate dataframe structure
    if df is None or "title" not in df.columns:
        raise RuntimeError(
            "df.pkl must contain a DataFrame with a 'title' column"
        )
```

**Startup Timeline:**
```
Server starts
    ↓
@app.on_event("startup") runs
    ↓
Load all pickle files into memory
    ↓
Build lookup dictionary
    ↓
Validate data
    ↓
✅ Server ready to accept requests!
```

**Why This Approach?**
- Load ONCE at startup (expensive operation)
- All subsequent requests use loaded data (fast)
- If loading fails, server won't start

---

## 📡 API Routes (Endpoints)

### Route 1: `/health` - Health Check
```python
@app.get("/health")
def health():
    """Health check endpoint - returns status ok"""
    return {"status": "ok"}
```

**What It Does:**
- Simple endpoint to check if server is running
- Called by monitoring systems
- Streamlit can use it to verify FastAPI is alive

**Example:**
```
GET http://127.0.0.1:8000/health

Response:
{
  "status": "ok"
}
```

---

### Route 2: `/home` - Popular Movies for Home Feed
```python
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

        data = await tmdb_get(
            f"/movie/{category}", 
            {"language": "en-US", "page": 1}
        )
        return await tmdb_cards_from_results(
            data.get("results", []), 
            limit=limit
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Home route failed: {e}")
```

**What It Does:**
- Get trending or popular movies from TMDB
- Return as list of TMDBMovieCard objects (with posters)
- Limit results (max 50 at a time)

**Example:**
```
GET http://127.0.0.1:8000/home?category=trending&limit=12

Response:
[
  {
    "tmdb_id": 507,
    "title": "Red Dragon",
    "poster_url": "https://...",
    "release_date": "2002-06-20",
    "vote_average": 7.1
  },
  ...
]
```

---

### Route 3: `/tmdb/search` - Search Movies
```python
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
```

**What It Does:**
- Search TMDB for movies matching keyword
- Return multiple results
- Support pagination

**Example:**
```
GET http://127.0.0.1:8000/tmdb/search?query=matrix&page=1

Response:
{
  "results": [
    {
      "id": 603,
      "title": "The Matrix",
      "poster_path": "/vfnFAQdDKAUB2AkSNkAQMXc0Bj7.jpg",
      "release_date": "1999-03-31",
      "vote_average": 8.7
    },
    ...
  ],
  "page": 1,
  "total_results": 847
}
```

---

### Route 4: `/movie/id/{tmdb_id}` - Get Movie Details
```python
@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetails)
async def movie_details_route(tmdb_id: int):
    """Get full movie details (overview, genres, images) by TMDB ID"""
    return await tmdb_movie_details(tmdb_id)
```

**What It Does:**
- Get complete info for one movie
- Includes: overview, genres, backdrop image
- Takes movie ID as URL parameter

**Example:**
```
GET http://127.0.0.1:8000/movie/id/603

Response:
{
  "tmdb_id": 603,
  "title": "The Matrix",
  "overview": "A computer programmer discovers...",
  "release_date": "1999-03-31",
  "poster_url": "https://...",
  "backdrop_url": "https://...",
  "genres": [
    {"id": 28, "name": "Action"},
    {"id": 878, "name": "Science Fiction"}
  ]
}
```

---

### Route 5: `/recommend/genre` - Genre Recommendations
```python
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
    cards = await tmdb_cards_from_results(
        discover.get("results", []), 
        limit=limit
    )
    return [c for c in cards if c.tmdb_id != tmdb_id]
```

**What It Does:**
1. Get details for movie (tmdb_id)
2. Extract first genre
3. Search TMDB for all movies in that genre
4. Return list (excluding original movie)

**Example:**
```
GET http://127.0.0.1:8000/recommend/genre?tmdb_id=603&limit=12

# For The Matrix (Action genre):
# Returns 12 other Action movies sorted by popularity
```

---

### Route 6: `/recommend/tfidf` - TF-IDF Recommendations
```python
@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    """Get TF-IDF recommendations (local dataset only, no TMDB posters)"""
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score": s} for t, s in recs]
```

**What It Does:**
- Get AI recommendations based on content similarity
- Uses local trained model (TF-IDF matrix)
- NO TMDB API calls (fast!)

**Example:**
```
GET http://127.0.0.1:8000/recommend/tfidf?title=The%20Matrix&top_n=10

Response:
[
  {"title": "The Matrix Reloaded", "score": 0.923},
  {"title": "The Matrix Revolutions", "score": 0.876},
  {"title": "Johnny Mnemonic", "score": 0.654},
  ...
]
```

---

### Route 7: `/movie/search` - Complete Search Bundle
```python
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
    # Step 1: Search TMDB for the movie
    best = await tmdb_search_first(query)
    if not best:
        raise HTTPException(
            status_code=404, 
            detail=f"No TMDB movie found for query: {query}"
        )

    tmdb_id = int(best["id"])
    
    # Step 2: Get full movie details
    details = await tmdb_movie_details(tmdb_id)

    # Step 3: Get TF-IDF recommendations with TMDB posters
    tfidf_items: List[TFIDFRecItem] = []
    recs: List[Tuple[str, float]] = []
    
    try:
        # Try using TMDB title
        recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
    except Exception:
        # Fallback to user query
        try:
            recs = tfidf_recommend_titles(query, top_n=tfidf_top_n)
        except Exception:
            recs = []

    # For each recommendation, fetch TMDB poster
    for title, score in recs:
        card = await attach_tmdb_card_by_title(title)
        tfidf_items.append(TFIDFRecItem(title=title, score=score, tmdb=card))

    # Step 4: Get genre recommendations
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
            discover.get("results", []), 
            limit=genre_limit
        )
        genre_recs = [c for c in cards if c.tmdb_id != details.tmdb_id]

    # Step 5: Return everything bundled together
    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )
```

**What It Does:**
- **One endpoint to rule them all!**
- Searches for movie on TMDB
- Gets TF-IDF recommendations
- Gets genre recommendations
- Returns everything bundled together

**Example:**
```
GET http://127.0.0.1:8000/movie/search?query=fight%20club

Response:
{
  "query": "fight club",
  "movie_details": {
    "tmdb_id": 550,
    "title": "Fight Club",
    "overview": "An insomniac office worker...",
    "genres": [{"id": 18, "name": "Drama"}],
    ...
  },
  "tfidf_recommendations": [
    {
      "title": "Se7en",
      "score": 0.876,
      "tmdb": {
        "tmdb_id": 807,
        "title": "Se7en",
        "poster_url": "https://...",
        ...
      }
    },
    ...
  ],
  "genre_recommendations": [
    {
      "tmdb_id": 278,
      "title": "The Shawshank Redemption",
      ...
    },
    ...
  ]
}
```

---

## 🎯 Summary: How Everything Works Together

```
User (Streamlit)
     ↓
Clicks "Search for Matrix"
     ↓
Streamlit calls: /movie/search?query=matrix
     ↓
FastAPI (main.py)
     ├─ tmdb_search_first() → Search TMDB for "matrix"
     ├─ tmdb_movie_details() → Get full info
     ├─ tfidf_recommend_titles() → AI recommendations from trained model
     ├─ attach_tmdb_card_by_title() → Add posters to recommendations
     └─ recommend_genre() → Genre-based recommendations
     ↓
Response sent back to Streamlit
     ↓
Streamlit displays:
  - Movie details with poster
  - 12 similar movies (TF-IDF)
  - 12 movies in same genre
```

---

## 🔧 How to Run

### Terminal 1: Start FastAPI Server
```bash
uvicorn main:app --reload
```
- Server starts on `http://127.0.0.1:8000`
- `--reload` = auto-restart on code changes

### Terminal 2: Start Streamlit Frontend
```bash
streamlit run app.py
```
- Opens browser on `http://localhost:8501`
- Calls FastAPI on localhost:8000

---

## 📝 Key Concepts Recap

| Term | Meaning |
|------|---------|
| **FastAPI** | Framework to build web servers (APIs) |
| **TMDB API** | Movie database with live data |
| **TF-IDF** | AI model for finding similar movies |
| **Pickle** | Format to save Python objects to disk |
| **Async** | Non-blocking requests (handle multiple users) |
| **CORS** | Allows frontend and backend to communicate |
| **Pydantic** | Data validation framework |
| **HTTPException** | Error responses from API |

---

## 🎓 Next Steps

1. **Understand one endpoint at a time** - Pick your favorite and trace through the code
2. **Try the API directly** - Use browser or Postman to call endpoints
3. **Modify responses** - Add fields to TMDBMovieCard, see what breaks
4. **Experiment** - Change limit values, try different categories
5. **Read TMDB docs** - Understand what data TMDB gives us

---

**Happy coding! 🚀**
