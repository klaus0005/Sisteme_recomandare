#!/usr/bin/env python3
"""
Sistem de Recomandare - Demo UI Backend (FastAPI)

COMENZI PENTRU A RULA:
1. Asigură-te că ai activat venv-ul: source venv/bin/activate
2. Instalează dependențele: pip install -r requirements.txt
3. Rulează serverul: uvicorn src.app:app --reload

DESCRIERE:
Backend minimal FastAPI pentru demo UI.
Citește CSV-urile generate și oferă endpoint-uri pentru recomandări.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Setup paths
ROOT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"
UI_DIR = ROOT_DIR / "ui"

# Aplicație FastAPI
app = FastAPI(
    title="Sistem de Recomandare Muzică - Demo API",
    description="API pentru demo UI - recomandări ALS și Autoencoder",
    version="1.0.0"
)

# CORS pentru localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite toate origin-urile pentru demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (UI)
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")

# Serve index.html at root
@app.get("/")
async def read_root():
    """Serve index.html at root."""
    index_path = UI_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "UI not found. Please ensure ui/index.html exists."}

# Date încărcate în memorie
tracks_meta: Optional[pd.DataFrame] = None
recs_als: Optional[pd.DataFrame] = None
recs_autoencoder: Optional[pd.DataFrame] = None
test_truth: Optional[Dict[int, List[str]]] = None


def load_data():
    """Încarcă datele din CSV-uri în memorie la pornire."""
    global tracks_meta, recs_als, recs_autoencoder, test_truth
    
    print("Încărcare date...")
    
    # tracks_meta.csv
    tracks_meta_path = OUTPUT_DIR / "tracks_meta.csv"
    if tracks_meta_path.exists():
        tracks_meta = pd.read_csv(tracks_meta_path)
        print(f"  ✓ tracks_meta.csv încărcat ({len(tracks_meta)} piese)")
    else:
        print(f"  ✗ tracks_meta.csv nu există!")
        tracks_meta = pd.DataFrame()
    
    # recs_als.csv
    als_path = OUTPUT_DIR / "recs_als.csv"
    if als_path.exists():
        recs_als = pd.read_csv(als_path)
        print(f"  ✓ recs_als.csv încărcat ({len(recs_als)} recomandări)")
    else:
        print(f"  ✗ recs_als.csv nu există!")
        recs_als = pd.DataFrame()
    
    # recs_autoencoder.csv
    ae_path = OUTPUT_DIR / "recs_autoencoder.csv"
    if ae_path.exists():
        recs_autoencoder = pd.read_csv(ae_path)
        print(f"  ✓ recs_autoencoder.csv încărcat ({len(recs_autoencoder)} recomandări)")
    else:
        print(f"  ✗ recs_autoencoder.csv nu există!")
        recs_autoencoder = pd.DataFrame()
    
    # interactions_test.csv (opțional)
    test_path = OUTPUT_DIR / "interactions_test.csv"
    if test_path.exists():
        test_df = pd.read_csv(test_path)
        test_truth = {}
        for _, row in test_df.iterrows():
            user_id = int(row['user_id'])
            track_id = row['track_id']
            if user_id not in test_truth:
                test_truth[user_id] = []
            test_truth[user_id].append(track_id)
        print(f"  ✓ interactions_test.csv încărcat ({len(test_truth)} useri)")
    else:
        print(f"  ⚠  interactions_test.csv nu există (opțional)")
        test_truth = {}
    
    print("Date încărcate!")


@app.on_event("startup")
async def startup_event():
    """Încarcă datele la pornirea serverului."""
    load_data()


@app.get("/health")
async def health():
    """Endpoint pentru health check."""
    return {"status": "ok"}


@app.get("/recommendations")
async def get_recommendations(
    user_id: int = Query(..., description="ID utilizator"),
    k: int = Query(10, ge=1, le=100, description="Număr recomandări (1-100)")
):
    """
    Endpoint pentru recomandări.
    
    Args:
        user_id: ID utilizator
        k: Număr recomandări (default 10, max 100)
    
    Returns:
        JSON cu recomandări ALS, Autoencoder și test_truth
    """
    result = {
        "user_id": user_id,
        "k": k,
        "als": [],
        "autoencoder": [],
        "test_truth": []
    }
    
    # Verifică dacă user_id există în recomandări
    has_als = False
    has_ae = False
    
    if recs_als is not None and len(recs_als) > 0:
        user_als = recs_als[recs_als['user_id'] == user_id]
        if len(user_als) > 0:
            has_als = True
    
    if recs_autoencoder is not None and len(recs_autoencoder) > 0:
        user_ae = recs_autoencoder[recs_autoencoder['user_id'] == user_id]
        if len(user_ae) > 0:
            has_ae = True
    
    if not has_als and not has_ae:
        raise HTTPException(
            status_code=404,
            detail=f"User ID {user_id} nu există în recomandări. "
                   f"Useri disponibili: {sorted(set(recs_als['user_id'].unique()) if recs_als is not None and len(recs_als) > 0 else [])}"
        )
    
    # Obține ground truth pentru user (pentru evidențiere)
    truth_track_ids = set()
    if test_truth is not None and user_id in test_truth:
        truth_track_ids = set(test_truth[user_id])
    
    # Recomandări ALS
    if has_als and tracks_meta is not None:
        user_als = recs_als[recs_als['user_id'] == user_id].sort_values('rank').head(k)
        
        # Join cu tracks_meta pentru nume și artist
        als_with_meta = user_als.merge(
            tracks_meta[['id', 'name', 'artist']],
            left_on='track_id',
            right_on='id',
            how='left'
        )
        
        for _, row in als_with_meta.iterrows():
            track_id = row['track_id']
            is_in_truth = track_id in truth_track_ids
            result["als"].append({
                "rank": int(row['rank']),
                "track_id": track_id,
                "name": row.get('name', 'N/A'),
                "artist": row.get('artist', 'N/A'),
                "score": float(row['score']),
                "in_truth": is_in_truth  # Flag pentru evidențiere în UI
            })
    
    # Recomandări Autoencoder
    if has_ae and tracks_meta is not None:
        user_ae = recs_autoencoder[recs_autoencoder['user_id'] == user_id].sort_values('rank').head(k)
        
        # Join cu tracks_meta pentru nume și artist
        ae_with_meta = user_ae.merge(
            tracks_meta[['id', 'name', 'artist']],
            left_on='track_id',
            right_on='id',
            how='left'
        )
        
        for _, row in ae_with_meta.iterrows():
            track_id = row['track_id']
            is_in_truth = track_id in truth_track_ids
            result["autoencoder"].append({
                "rank": int(row['rank']),
                "track_id": track_id,
                "name": row.get('name', 'N/A'),
                "artist": row.get('artist', 'N/A'),
                "score": float(row['score']),
                "in_truth": is_in_truth  # Flag pentru evidențiere în UI
            })
    
    # Test truth (opțional, max 10)
    if test_truth is not None and user_id in test_truth:
        truth_tracks = test_truth[user_id][:10]  # Max 10
        
        if tracks_meta is not None:
            truth_df = tracks_meta[tracks_meta['id'].isin(truth_tracks)]
            
            for track_id in truth_tracks:
                track_row = truth_df[truth_df['id'] == track_id]
                if len(track_row) > 0:
                    row = track_row.iloc[0]
                    result["test_truth"].append({
                        "track_id": track_id,
                        "name": row.get('name', 'N/A'),
                        "artist": row.get('artist', 'N/A')
                    })
    
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
