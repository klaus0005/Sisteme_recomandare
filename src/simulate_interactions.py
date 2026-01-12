#!/usr/bin/env python3
"""
Sistem de Recomandare - Simulare Interacțiuni User-Track

COMENZI PENTRU A RULA:
1. Asigură-te că ai activat venv-ul: source venv/bin/activate
2. Rulează scriptul: python src/simulate_interactions.py

DESCRIERE:
Simulează interacțiuni user-track pentru sistemul de recomandare.
Datele Spotify nu au user_id real, deci generăm utilizatori fictivi cu
interacțiuni bazate pe preferințe pentru artiști (implicit feedback).

METODOLOGIE:
- Fiecare user are un artist favorit (ales ponderat după nr. de piese)
- 85% din interacțiuni sunt cu piese ale artistului favorit
- 15% sunt "noise" random pentru realism (utilizatori exploră și alti artiști)
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Setup paths
ROOT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"

# Parametri configurabili
N_USERS = 1000  # Număr utilizatori fictivi
TRACKS_PER_USER = 60  # Număr piese per utilizator (redus pentru a menține raportul 85/15 corect - majoritatea artiștilor au 20-100 piese)
NOISE_RATIO = 0.15  # 15% noise (piese random, nu din artistul favorit)
FAVORITE_RATIO = 1.0 - NOISE_RATIO  # 85% din artistul favorit
MIN_TRACKS_PER_ARTIST = 20  # Minim piese necesare pentru a fi ales ca artist favorit
RANDOM_SEED = 42

# Setup random generator cu seed fix
rng = np.random.default_rng(RANDOM_SEED)


def simulate_interactions():
    """
    Funcția principală care simulează interacțiunile user-track.
    """
    print("=" * 60)
    print("ETAPA 2: Simulare Interacțiuni User-Track")
    print("=" * 60)
    
    # 1. Citește tracks_meta.csv
    tracks_meta_path = OUTPUT_DIR / "tracks_meta.csv"
    print(f"\n[1] Citire metadata piese din: {tracks_meta_path}")
    
    try:
        tracks_meta = pd.read_csv(tracks_meta_path)
        print(f"    ✓ Date încărcate: {len(tracks_meta)} piese")
    except FileNotFoundError:
        print(f"    ✗ EROARE: Fișierul {tracks_meta_path} nu a fost găsit!")
        print(f"    → Rulează mai întâi: python src/prepare_data.py")
        return
    except Exception as e:
        print(f"    ✗ EROARE la citirea datelor: {e}")
        return
    
    # Validare: verifică că există coloana 'id' și 'artist'
    required_cols = ['id', 'artist']
    missing_cols = [col for col in required_cols if col not in tracks_meta.columns]
    if missing_cols:
        print(f"    ✗ EROARE: Coloane lipsă: {missing_cols}")
        return
    
    # 2. Pregătește datele pentru simulare
    print(f"\n[2] Pregătire date pentru simulare...")
    
    # Filtrează artiștii cu minim MIN_TRACKS_PER_ARTIST piese
    artist_counts = tracks_meta['artist'].value_counts()
    valid_artists = artist_counts[artist_counts >= MIN_TRACKS_PER_ARTIST].index.tolist()
    tracks_valid = tracks_meta[tracks_meta['artist'].isin(valid_artists)].copy()
    
    print(f"    ✓ Artiști valizi (≥{MIN_TRACKS_PER_ARTIST} piese): {len(valid_artists)}")
    print(f"    ✓ Piese valide: {len(tracks_valid)}")
    
    if len(valid_artists) == 0:
        print(f"    ✗ EROARE: Nu există artiști cu ≥{MIN_TRACKS_PER_ARTIST} piese!")
        return
    
    # Creează dicționar artist -> lista de track_ids
    artist_to_tracks = {}
    for artist in valid_artists:
        artist_tracks = tracks_valid[tracks_valid['artist'] == artist]['id'].tolist()
        artist_to_tracks[artist] = artist_tracks
    
    # Pregătește probabilități pentru alegerea artistului favorit
    # Probabilitatea = număr de piese (ponderare)
    artist_probs = np.array([len(artist_to_tracks[artist]) for artist in valid_artists])
    artist_probs = artist_probs / artist_probs.sum()  # Normalizare
    
    print(f"    ✓ Probabilități calculate pentru {len(valid_artists)} artiști")
    
    # 3. Generează utilizatori și interacțiuni
    print(f"\n[3] Generare {N_USERS} utilizatori și interacțiuni...")
    print(f"    Parametri:")
    print(f"      - Tracks per user: {TRACKS_PER_USER}")
    print(f"      - Ratio artist favorit: {FAVORITE_RATIO:.1%}")
    print(f"      - Ratio noise: {NOISE_RATIO:.1%}")
    
    # NOTĂ: De ce adăugăm noise?
    # - Realism: utilizatorii reali explorează și alti artiști, nu doar preferințele principale
    # - Previne overfitting pe preferințe foarte concentrate
    # - Permite modelului să învețe și pattern-uri mai subtile în date
    # - Echilibrează dataset-ul (nu doar clusteruri clare de artiști)
    
    interactions_list = []
    users_profile_list = []
    
    all_track_ids = tracks_valid['id'].tolist()  # Pentru noise
    
    for user_id in range(N_USERS):
        # Alege artist favorit (ponderat după nr. de piese)
        favorite_artist_idx = rng.choice(len(valid_artists), p=artist_probs)
        favorite_artist = valid_artists[favorite_artist_idx]
        favorite_tracks = artist_to_tracks[favorite_artist]
        
        # Calculează numărul de piese din artistul favorit și noise
        n_favorite = int(TRACKS_PER_USER * FAVORITE_RATIO)
        n_noise = TRACKS_PER_USER - n_favorite
        
        # Selectează piese din artistul favorit (fără dubluri)
        if n_favorite > len(favorite_tracks):
            # Dacă nu sunt suficiente piese, iau toate
            selected_favorite = favorite_tracks.copy()
            n_favorite = len(selected_favorite)
            n_noise = TRACKS_PER_USER - n_favorite
        else:
            selected_favorite = rng.choice(favorite_tracks, size=n_favorite, replace=False).tolist()
        
        # Selectează noise (piese random din tot catalogul, excludând cele din artistul favorit)
        # Asigurăm că nu dublăm piese din artistul favorit
        available_noise = [tid for tid in all_track_ids if tid not in selected_favorite]
        
        if n_noise > len(available_noise):
            # Dacă nu sunt suficiente piese pentru noise, iau câte sunt disponibile
            selected_noise = available_noise.copy()
            n_noise = len(selected_noise)
        else:
            selected_noise = rng.choice(available_noise, size=n_noise, replace=False).tolist()
        
        # Combina selecțiile
        selected_tracks = selected_favorite + selected_noise
        
        # Creează interacțiuni (toate au play=1.0 pentru implicit feedback)
        for track_id in selected_tracks:
            interactions_list.append({
                'user_id': user_id,
                'track_id': track_id,
                'play': 1.0
            })
        
        # Salvează profilul user-ului
        users_profile_list.append({
            'user_id': user_id,
            'favorite_artist': favorite_artist,
            'n_tracks': len(selected_tracks),
            'noise_ratio': n_noise / len(selected_tracks) if len(selected_tracks) > 0 else 0.0
        })
    
    # Creează DataFrame-uri
    interactions_df = pd.DataFrame(interactions_list)
    users_profile_df = pd.DataFrame(users_profile_list)
    
    # Validare: verifică că toate track_id-urile există în tracks_meta
    print(f"\n[4] Validare interacțiuni...")
    valid_track_ids = set(tracks_meta['id'].tolist())
    interactions_track_ids = set(interactions_df['track_id'].tolist())
    invalid_tracks = interactions_track_ids - valid_track_ids
    
    if invalid_tracks:
        print(f"    ⚠  Atenție: {len(invalid_tracks)} track_id-uri invalide (vor fi eliminate)")
        interactions_df = interactions_df[interactions_df['track_id'].isin(valid_track_ids)]
    
    # Validare: verifică dubluri (nu ar trebui să existe)
    duplicates = interactions_df.duplicated(subset=['user_id', 'track_id']).sum()
    if duplicates > 0:
        print(f"    ⚠  Atenție: {duplicates} dubluri găsite (vor fi eliminate)")
        interactions_df = interactions_df.drop_duplicates(subset=['user_id', 'track_id'], keep='first')
    
    print(f"    ✓ Validare completă: {len(interactions_df)} interacțiuni valide")
    
    # 5. Calculează statistici
    print(f"\n[5] Statistici interacțiuni...")
    
    n_interactions = len(interactions_df)
    n_unique_tracks = interactions_df['track_id'].nunique()
    n_unique_users = interactions_df['user_id'].nunique()
    
    # Sparsity = 1 - (nr_interacțiuni / (nr_users * nr_items))
    # NOTĂ: Sparsity în matricea user-item
    # - Sparsity = proporția de celule goale (fără interacțiuni) în matricea user×item
    # - Ex: dacă avem 1000 users și 10000 items, matricea are 10M celule
    # - Dacă avem 50000 interacțiuni, sparsity = 1 - 50000/10000000 = 99.5%
    # - Sparsity mare = tipic pentru sisteme de recomandare (utilizatorii interacționează cu puține iteme)
    sparsity = 1.0 - (n_interactions / (n_unique_users * n_unique_tracks))
    
    print(f"    ✓ Număr interacțiuni: {n_interactions:,}")
    print(f"    ✓ Număr utilizatori: {n_unique_users:,}")
    print(f"    ✓ Număr piese unice: {n_unique_tracks:,}")
    print(f"    ✓ Sparsity aproximativă: {sparsity:.2%}")
    
    # Top 5 artiști aleși ca favoriți
    top_artists = users_profile_df['favorite_artist'].value_counts().head(5)
    print(f"\n    Top 5 artiști aleși ca favoriți:")
    for artist, count in top_artists.items():
        print(f"      {count:4d} users → {artist}")
    
    # 6. Salvează fișierele
    print(f"\n[6] Salvare fișiere...")
    
    # Salvează interactions.csv
    interactions_path = OUTPUT_DIR / "interactions.csv"
    interactions_df.to_csv(interactions_path, index=False)
    print(f"    ✓ Interacțiuni salvate: {interactions_path}")
    print(f"      {len(interactions_df):,} interacțiuni, 3 coloane (user_id, track_id, play)")
    
    # Salvează users_profile.csv
    users_profile_path = OUTPUT_DIR / "users_profile.csv"
    users_profile_df.to_csv(users_profile_path, index=False)
    print(f"    ✓ Profiluri utilizatori salvate: {users_profile_path}")
    print(f"      {len(users_profile_df):,} utilizatori, 4 coloane (user_id, favorite_artist, n_tracks, noise_ratio)")
    
    # Rezumat final
    print(f"\n" + "=" * 60)
    print("REZULTATE FINALE:")
    print("=" * 60)
    print(f"✓ Utilizatori generați: {N_USERS}")
    print(f"✓ Interacțiuni totale: {n_interactions:,}")
    print(f"✓ Sparsity: {sparsity:.2%}")
    print(f"✓ Fișiere generate în {OUTPUT_DIR}:")
    print(f"  - interactions.csv")
    print(f"  - users_profile.csv")
    print("=" * 60)


if __name__ == "__main__":
    try:
        simulate_interactions()
    except Exception as e:
        print(f"\n✗ EROARE FATALĂ: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
