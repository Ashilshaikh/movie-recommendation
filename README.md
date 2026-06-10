# 🎬 Movie Recommender System

A **hybrid movie recommendation system** built using Machine Learning (TF-IDF + Cosine Similarity), FastAPI backend, and Streamlit frontend, enriched with TMDB API data.

---

## 🚀 Features

- 🔎 Smart Movie Search (autocomplete + keyword matching)
- 🎯 TF-IDF Based Recommendations
- 🎭 Genre-Based Recommendations
- 🌐 Live TMDB API Integration
- 🖼️ Posters + Backdrops
- ⚡ FastAPI Backend
- 🎨 Streamlit Frontend UI
- 📄 Movie Details Page

---

## 🧠 How It Works

Movies are processed using:
- TF-IDF Vectorization
- Cosine Similarity
- Hybrid TMDB + Local ML recommendations

---

## 🏗️ Tech Stack

- FastAPI
- Streamlit
- Scikit-learn
- Pandas / NumPy
- TMDB API

---

## 📡 API Endpoints

- `/home` → Popular movies
- `/tmdb/search` → Search movies
- `/movie/id/{id}` → Movie details
- `/recommend/tfidf` → Similar movies
- `/recommend/genre` → Genre-based recommendations
- `/movie/search` → Full recommendation bundle

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
uvicorn main:app --reload
streamlit run app.py



## 📸 App Preview

<img width="1919" height="1079" alt="Screenshot 2026-06-10 194517" src="https://github.com/user-attachments/assets/ee92ce65-1ca6-4375-88f0-6cf3fa7037fb" />
