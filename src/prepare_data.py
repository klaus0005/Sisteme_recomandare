#!/usr/bin/env python3
"""
Sistem de Recomandare - Pregătirea Datelor (ETAPA 1)

COMENZI PENTRU A RULA ETAPA 1:
1. Asigură-te că ai activat venv-ul: source venv/bin/activate (sau .\\venv\\Scripts\\activate pe Windows)
2. Instalează dependențele: pip install -r requirements.txt
3. Rulează scriptul: python src/prepare_data.py

DESCRIERE:
Pregătește datele din data.csv pentru modelare:
- Extrage artistul principal din lista de artiști
- Selectează și normalizează features numerice
- Filtrează artiștii cu minim 20 piese
- Salvează outputs în outputs/
"""

import json
import ast
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Setup paths relative la root-ul proiectului
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

# Creează directorul outputs dacă nu există
OUTPUT_DIR.mkdir(exist_ok=True)


def extract_main_artist(artists_str):
    """
    Extrage artistul principal din coloana 'artists'.
    
    NOTĂ: "Artist principal" = primul artist din listă (convenția Spotify).
    În coloana 'artists', formatul este string de listă Python, ex: "['Artist1', 'Artist2']"
    Primul artist este considerat principal pentru că este cel listat primul de Spotify.
    
    Args:
        artists_str: String care reprezintă o listă Python (ex: "['Artist1', 'Artist2']")
    
    Returns:
        String cu numele artistului principal, sau None dacă parsarea eșuează
    """
    try:
        # Parsează string-ul ca listă Python
        if pd.isna(artists_str):
            return None
        
        # Verifică dacă e deja string simplu (nu listă)
        if isinstance(artists_str, str):
            # Încearcă să parseze ca listă
            artists_list = ast.literal_eval(artists_str)
            if isinstance(artists_list, list) and len(artists_list) > 0:
                # Returnează primul artist (principal)
                return artists_list[0].strip()
            elif isinstance(artists_list, str):
                return artists_list.strip()
        return None
    except (ValueError, SyntaxError):
        # Dacă parsarea eșuează, returnează None
        return None


def prepare_data():
    """
    Funcția principală care pregătește datele.
    """
    print("=" * 60)
    print("ETAPA 1: Pregătirea Datelor")
    print("=" * 60)
    
    # 1. Citește data.csv
    data_path = DATA_DIR / "data.csv"
    print(f"\n[1] Citire date din: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        print(f"    ✓ Date încărcate: {len(df)} piese inițiale")
    except FileNotFoundError:
        print(f"    ✗ EROARE: Fișierul {data_path} nu a fost găsit!")
        return
    except Exception as e:
        print(f"    ✗ EROARE la citirea datelor: {e}")
        return
    
    # 2. Extrage artistul principal
    print(f"\n[2] Extragere artist principal din coloana 'artists'...")
    df['artist'] = df['artists'].apply(extract_main_artist)
    
    # Elimină rândurile unde nu s-a putut extrage artistul
    initial_count = len(df)
    df = df[df['artist'].notna()].copy()
    removed = initial_count - len(df)
    if removed > 0:
        print(f"    ⚠  Eliminate {removed} piese fără artist valid")
    
    print(f"    ✓ Artist principal extras pentru {len(df)} piese")
    
    # 3. Selectează features numerice relevante
    print(f"\n[3] Selectare features numerice...")
    
    # Features audio de la Spotify
    feature_columns = [
        'danceability',
        'energy',
        'valence',
        'acousticness',
        'instrumentalness',
        'liveness',
        'speechiness',
        'tempo',
        'loudness',
        'duration_ms',
    ]
    
    # Adaugă popularity dacă există
    if 'popularity' in df.columns:
        feature_columns.append('popularity')
    
    # Verifică ce coloane există efectiv
    available_features = [col for col in feature_columns if col in df.columns]
    missing_features = [col for col in feature_columns if col not in df.columns]
    
    if missing_features:
        print(f"    ⚠  Features lipsă (vor fi ignorate): {missing_features}")
    
    print(f"    ✓ Features selectate: {len(available_features)}")
    print(f"      {available_features}")
    
    # 4. Curățare date
    print(f"\n[4] Curățare date...")
    
    # Drop duplicates pe id
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['id'], keep='first')
    if len(df) < before_dedup:
        print(f"    ✓ Eliminate {before_dedup - len(df)} duplicate pe 'id'")
    
    # Drop NA pe coloanele esențiale
    essential_cols = ['id', 'name', 'artist'] + available_features
    before_dropna = len(df)
    df = df.dropna(subset=essential_cols)
    if len(df) < before_dropna:
        print(f"    ✓ Eliminate {before_dropna - len(df)} piese cu valori lipsă")
    
    print(f"    ✓ Date curățate: {len(df)} piese rămase")
    
    # 5. Filtrăm artiștii cu minim 20 piese
    print(f"\n[5] Filtrare artiști cu minim 20 piese...")
    
    # NOTĂ: De ce filtrăm artiștii cu minim 20 piese?
    # - Reduce zgomotul în date (artiștii cu foarte puține piese pot avea patterns nefiabile)
    # - Asigură că fiecare artist are suficient date pentru a avea un profil consistent
    # - Îmbunătățește calitatea recomandărilor bazate pe artiști
    # - Reduce dimensiunea dataset-ului fără a pierde informație semnificativă
    
    artist_counts = df['artist'].value_counts()
    artists_with_min_tracks = artist_counts[artist_counts >= 20].index
    
    before_filter = len(df)
    df_filtered = df[df['artist'].isin(artists_with_min_tracks)].copy()
    removed_artists = len(artist_counts) - len(artists_with_min_tracks)
    
    print(f"    ✓ Artiști înainte: {len(artist_counts)}")
    print(f"    ✓ Artiști cu ≥20 piese: {len(artists_with_min_tracks)}")
    print(f"    ✓ Piese eliminate: {before_filter - len(df_filtered)}")
    print(f"    ✓ Artiști eliminați: {removed_artists}")
    
    df = df_filtered
    
    # 5.5. Filtrare după popularitate (pentru a reduce catalogul și îmbunătăți precizia)
    print(f"\n[5.5] Filtrare după popularitate...")
    
    # NOTĂ: De ce filtrăm după popularitate?
    # - Reduce catalogul de la ~109k la ~10k-20k piese (mai realist pentru recomandări)
    # - Piesele populare sunt mai relevante pentru utilizatori
    # - Îmbunătățește semnificativ precizia (mai ușor să găsești itemi relevanți)
    # - Simulează un catalog mai realist (nu toate piese sunt la fel de relevante)
    
    MIN_POPULARITY = 55  # Minim popularity pentru a fi inclus în catalog (mărit pentru catalog mai mic și precizie mai bună)
    
    if 'popularity' in df.columns:
        before_pop_filter = len(df)
        df = df[df['popularity'] >= MIN_POPULARITY].copy()
        removed_pop = before_pop_filter - len(df)
        
        print(f"    ✓ Piese înainte: {before_pop_filter:,}")
        print(f"    ✓ Piese cu popularity ≥{MIN_POPULARITY}: {len(df):,}")
        print(f"    ✓ Piese eliminate: {removed_pop:,}")
    else:
        print(f"    ⚠  Coloana 'popularity' nu există, skip filtrare după popularitate")
    
    # 6. Pregătește metadata (tracks_meta.csv)
    print(f"\n[6] Pregătire metadata...")
    
    meta_cols = ['id', 'name', 'artist']
    if 'year' in df.columns:
        meta_cols.append('year')
    if 'popularity' in df.columns:
        meta_cols.append('popularity')
    
    tracks_meta = df[meta_cols].copy()
    tracks_meta_path = OUTPUT_DIR / "tracks_meta.csv"
    tracks_meta.to_csv(tracks_meta_path, index=False)
    print(f"    ✓ Metadata salvată: {tracks_meta_path}")
    print(f"      {len(tracks_meta)} piese, {len(tracks_meta.columns)} coloane")
    
    # 7. Normalizare features cu StandardScaler
    print(f"\n[7] Normalizare features cu StandardScaler...")
    
    # NOTĂ: De ce normalizăm înainte de autoencoder/ALS?
    # - StandardScaler (mean=0, std=1) asigură că toate features au scale similară
    # - Previne ca features cu valori mari (ex: tempo ~100, loudness ~-5) să domine
    #   features cu valori mici (ex: danceability ~0.5)
    # - Autoencoder converge mai bine cu date normalizate (gradienți mai stabili)
    # - ALS beneficiază de normalizare pentru că operează pe distanțe/proiecții
    # - Consistentă cu practicile din ML: "standardize features before modeling"
    
    # Extrage doar coloanele de features disponibile
    features_df = df[available_features].copy()
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)
    
    features_path = OUTPUT_DIR / "tracks_features_scaled.npy"
    np.save(features_path, features_scaled)
    print(f"    ✓ Features normalizate salvate: {features_path}")
    print(f"      Shape: {features_scaled.shape}")
    print(f"      Mean: {features_scaled.mean():.6f}, Std: {features_scaled.std():.6f}")
    
    # 8. Salvează lista de coloane folosite
    feature_columns_path = OUTPUT_DIR / "feature_columns.json"
    with open(feature_columns_path, 'w', encoding='utf-8') as f:
        json.dump(available_features, f, indent=2, ensure_ascii=False)
    print(f"    ✓ Listă coloane salvate: {feature_columns_path}")
    
    # 9. Statistici finale
    print(f"\n" + "=" * 60)
    print("REZULTATE FINALE:")
    print("=" * 60)
    print(f"✓ Număr piese finale: {len(df)}")
    print(f"✓ Număr artiști: {df['artist'].nunique()}")
    print(f"✓ Shape features normalizate: {features_scaled.shape}")
    print(f"✓ Features utilizate: {len(available_features)}")
    print(f"\nFișiere generate în {OUTPUT_DIR}:")
    print(f"  - tracks_meta.csv ({len(tracks_meta)} rânduri)")
    print(f"  - tracks_features_scaled.npy (shape: {features_scaled.shape})")
    print(f"  - feature_columns.json ({len(available_features)} features)")
    print("=" * 60)


if __name__ == "__main__":
    try:
        prepare_data()
    except Exception as e:
        print(f"\n✗ EROARE FATALĂ: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
