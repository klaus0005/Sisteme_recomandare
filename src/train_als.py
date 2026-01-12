#!/usr/bin/env python3
"""
Sistem de Recomandare - Training ALS (Alternating Least Squares)

COMENZI PENTRU A RULA:
1. Asigură-te că ai activat venv-ul: source venv/bin/activate
2. Instalează dependențele: pip install -r requirements.txt
3. Rulează scriptul: python src/train_als.py

PREREQUISIT:
Rulează mai întâi:
- python src/prepare_data.py (ETAPA 1)
- python src/simulate_interactions.py (ETAPA 2)
- python src/split_train_test.py (ETAPA 2)

DESCRIERE:
Antrenează un model ALS (Alternating Least Squares) pentru implicit feedback.
ALS este un algoritm clasic de matrix factorization pentru sisteme de recomandare.
Folosește factorizarea matricei user-item pentru a prezice preferințe.
"""

import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

# Setup paths
ROOT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"

# Parametri configurabili ALS
FACTORS = 128  # Dimensiunea factorilor latenți (mărit pentru mai multă capacitate)
REGULARIZATION = 0.01  # Parametru de regularizare
ITERATIONS = 50  # Număr iterații ALS (mărit pentru convergență mai bună)
K = 500  # Număr recomandări top-K per utilizator (mărit pentru catalog mare - mai multe șanse de matching)
RANDOM_SEED = 42

# Set seed pentru reproducibilitate (implicit nu expune direct seed, dar setăm numpy)
np.random.seed(RANDOM_SEED)


def train_als():
    """
    Funcția principală care antrenează ALS și generează recomandări.
    """
    print("=" * 60)
    print("ETAPA 3: Training ALS (Alternating Least Squares)")
    print("=" * 60)
    
    # 1. Citește interactions_train.csv
    train_interactions_path = OUTPUT_DIR / "interactions_train.csv"
    print(f"\n[1] Citire interacțiuni train din: {train_interactions_path}")
    
    try:
        train_interactions = pd.read_csv(train_interactions_path)
        print(f"    ✓ Date încărcate: {len(train_interactions):,} interacțiuni")
    except FileNotFoundError:
        print(f"    ✗ EROARE: Fișierul {train_interactions_path} nu a fost găsit!")
        print(f"    → Rulează mai întâi: python src/split_train_test.py")
        return
    except Exception as e:
        print(f"    ✗ EROARE la citirea datelor: {e}")
        return
    
    # Validare coloane
    required_cols = ['user_id', 'track_id', 'play']
    missing_cols = [col for col in required_cols if col not in train_interactions.columns]
    if missing_cols:
        print(f"    ✗ EROARE: Coloane lipsă: {missing_cols}")
        return
    
    # 2. Citește tracks_meta.csv pentru validare și mapping
    tracks_meta_path = OUTPUT_DIR / "tracks_meta.csv"
    print(f"\n[2] Citire metadata piese din: {tracks_meta_path}")
    
    try:
        tracks_meta = pd.read_csv(tracks_meta_path)
        print(f"    ✓ Metadata încărcată: {len(tracks_meta)} piese")
    except FileNotFoundError:
        print(f"    ✗ EROARE: Fișierul {tracks_meta_path} nu a fost găsit!")
        print(f"    → Rulează mai întâi: python src/prepare_data.py")
        return
    except Exception as e:
        print(f"    ✗ EROARE la citirea metadata: {e}")
        return
    
    # 3. Validează track_id-urile (elimină track-uri inexistente)
    print(f"\n[3] Validare track_id-uri...")
    
    valid_track_ids = set(tracks_meta['id'].tolist())
    before_validation = len(train_interactions)
    train_interactions = train_interactions[train_interactions['track_id'].isin(valid_track_ids)].copy()
    removed = before_validation - len(train_interactions)
    
    if removed > 0:
        print(f"    ⚠  Eliminate {removed} interacțiuni cu track_id invalide")
    
    print(f"    ✓ Interacțiuni valide: {len(train_interactions):,}")
    
    # 4. Mapează user_id și track_id la indici
    print(f"\n[4] Mapping user_id și track_id la indici...")
    
    # Obține user_id-uri și track_id-uri unice (sortate pentru consistență)
    unique_users = sorted(train_interactions['user_id'].unique())
    unique_tracks = sorted(train_interactions['track_id'].unique())
    
    # Creează mapping-uri: id -> index
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    track_id_to_idx = {track_id: idx for idx, track_id in enumerate(unique_tracks)}
    
    # Mapping invers: index -> id (pentru reconstrucție)
    idx_to_user_id = {idx: user_id for user_id, idx in user_id_to_idx.items()}
    idx_to_track_id = {idx: track_id for track_id, idx in track_id_to_idx.items()}
    
    n_users = len(unique_users)
    n_tracks = len(unique_tracks)
    
    print(f"    ✓ Users: {n_users:,} (indici 0-{n_users-1})")
    print(f"    ✓ Tracks: {n_tracks:,} (indici 0-{n_tracks-1})")
    
    # 5. Construiește matricea user-item sparse (CSR format)
    print(f"\n[5] Construire matrice user-item sparse (CSR)...")
    
    # Convertesc user_id și track_id la indici
    user_indices = train_interactions['user_id'].map(user_id_to_idx).values
    track_indices = train_interactions['track_id'].map(track_id_to_idx).values
    values = train_interactions['play'].values
    
    # Construiesc matrice CSR (Compressed Sparse Row)
    # user_indices = row indices, track_indices = col indices
    user_item_matrix = csr_matrix((values, (user_indices, track_indices)), 
                                   shape=(n_users, n_tracks))
    
    print(f"    ✓ Matrice CSR creată: shape {user_item_matrix.shape}")
    print(f"      Non-zero elements: {user_item_matrix.nnz:,}")
    print(f"      Sparsity: {(1.0 - user_item_matrix.nnz / (n_users * n_tracks)):.2%}")
    
    # 6. Antrenează modelul ALS
    print(f"\n[6] Antrenare model ALS...")
    print(f"    Parametri:")
    print(f"      - Factors (dimensiune latenți): {FACTORS}")
    print(f"      - Regularization: {REGULARIZATION}")
    print(f"      - Iterations: {ITERATIONS}")
    
    start_time = time.time()
    
    try:
        model = AlternatingLeastSquares(
            factors=FACTORS,
            regularization=REGULARIZATION,
            iterations=ITERATIONS
        )
        model.fit(user_item_matrix)
        
        training_time = time.time() - start_time
        print(f"    ✓ Model antrenat în {training_time:.2f} secunde")
    except Exception as e:
        print(f"    ✗ EROARE la antrenare: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 7. Generează recomandări top-K pentru fiecare user
    print(f"\n[7] Generare recomandări top-{K} pentru fiecare user...")
    
    recommendations_list = []
    
    # Pentru fiecare user, generez recomandări
    # model.recommend(user_index, user_items, N=K) excludă automat itemii deja ascultați
    # user_items trebuie să fie CSR matrix cu itemii user-ului (din train)
    
    for user_idx in range(n_users):
        # Obțin itemii user-ului din train (pentru a fi excluși din recomandări)
        user_items = user_item_matrix[user_idx]
        
        try:
            # Generează recomandări (exclude automat itemii din user_items)
            # Returnează (item_indices, scores)
            recommended_items, scores = model.recommend(
                user_idx, 
                user_items, 
                N=K
            )
            
            # Convertesc la track_id și user_id
            user_id = idx_to_user_id[user_idx]
            
            # Adaugă recomandările cu rank
            for rank, (track_idx, score) in enumerate(zip(recommended_items, scores), start=1):
                track_id = idx_to_track_id[track_idx]
                recommendations_list.append({
                    'user_id': user_id,
                    'track_id': track_id,
                    'score': float(score),
                    'rank': rank
                })
        except Exception as e:
            print(f"    ⚠  Eroare la generare recomandări pentru user {user_idx}: {e}")
            continue
    
    # Creează DataFrame cu recomandările
    recommendations_df = pd.DataFrame(recommendations_list)
    
    print(f"    ✓ Recomandări generate: {len(recommendations_df):,} total")
    print(f"      Media recomandări per user: {len(recommendations_df) / n_users:.1f}")
    
    # 8. Salvează recomandările
    print(f"\n[8] Salvare recomandări...")
    
    recs_path = OUTPUT_DIR / "recs_als.csv"
    recommendations_df.to_csv(recs_path, index=False)
    print(f"    ✓ Recomandări salvate: {recs_path}")
    print(f"      {len(recommendations_df):,} recomandări, 4 coloane (user_id, track_id, score, rank)")
    
    # 9. Salvează informații despre model
    model_info = {
        'factors': FACTORS,
        'regularization': REGULARIZATION,
        'iterations': ITERATIONS,
        'K': K,
        'n_users': n_users,
        'n_tracks': n_tracks,
        'n_interactions_train': len(train_interactions),
        'training_time_seconds': round(training_time, 2),
        'random_seed': RANDOM_SEED
    }
    
    model_info_path = OUTPUT_DIR / "als_model_info.json"
    with open(model_info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    print(f"    ✓ Informații model salvate: {model_info_path}")
    
    # 10. Print exemplu recomandări pentru user_id 0 (debug)
    print(f"\n[9] Exemplu: Top 10 recomandări pentru user_id 0...")
    
    if 0 in unique_users:
        user_0_recs = recommendations_df[recommendations_df['user_id'] == 0].head(10)
        
        # Join cu tracks_meta pentru nume piese
        user_0_recs_with_names = user_0_recs.merge(
            tracks_meta[['id', 'name', 'artist']],
            left_on='track_id',
            right_on='id',
            how='left'
        )
        
        print(f"\n    Top 10 recomandări pentru user_id 0:")
        for idx, row in user_0_recs_with_names.iterrows():
            name = row.get('name', 'N/A')
            artist = row.get('artist', 'N/A')
            score = row['score']
            rank = row['rank']
            print(f"      {rank:2d}. {name[:50]:<50} | {artist[:30]:<30} | score: {score:.4f}")
    else:
        print(f"    ⚠  User_id 0 nu există în date")
    
    # Rezumat final
    print(f"\n" + "=" * 60)
    print("REZULTATE FINALE:")
    print("=" * 60)
    print(f"✓ Model ALS antrenat:")
    print(f"  - Factors: {FACTORS}, Regularization: {REGULARIZATION}, Iterations: {ITERATIONS}")
    print(f"  - Users: {n_users:,}, Tracks: {n_tracks:,}")
    print(f"  - Interacțiuni train: {len(train_interactions):,}")
    print(f"  - Timp antrenare: {training_time:.2f} secunde")
    print(f"✓ Recomandări generate:")
    print(f"  - Top-{K} per user: {len(recommendations_df):,} total")
    print(f"✓ Fișiere generate în {OUTPUT_DIR}:")
    print(f"  - recs_als.csv")
    print(f"  - als_model_info.json")
    print("=" * 60)


if __name__ == "__main__":
    try:
        train_als()
    except Exception as e:
        print(f"\n✗ EROARE FATALĂ: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
