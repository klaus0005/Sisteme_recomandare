#!/usr/bin/env python3
"""
Script de diagnostic pentru a identifica probleme care afectează precizia.
"""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"

def diagnostic():
    print("=" * 60)
    print("DIAGNOSTIC PRECIZIE - Identificare Probleme")
    print("=" * 60)
    
    # 1. Verifică datele de bază
    print("\n[1] Verificare date de bază...")
    
    try:
        tracks_meta = pd.read_csv(OUTPUT_DIR / "tracks_meta.csv")
        train = pd.read_csv(OUTPUT_DIR / "interactions_train.csv")
        test = pd.read_csv(OUTPUT_DIR / "interactions_test.csv")
        recs_als = pd.read_csv(OUTPUT_DIR / "recs_als.csv")
        recs_ae = pd.read_csv(OUTPUT_DIR / "recs_autoencoder.csv")
    except FileNotFoundError as e:
        print(f"    ✗ EROARE: Fișier lipsă: {e}")
        return
    
    print(f"    ✓ Catalog (tracks_meta): {len(tracks_meta):,} piese")
    print(f"    ✓ Train: {len(train):,} interacțiuni, {train['user_id'].nunique():,} users")
    print(f"    ✓ Test: {len(test):,} interacțiuni, {test['user_id'].nunique():,} users")
    print(f"    ✓ ALS recs: {len(recs_als):,} recomandări")
    print(f"    ✓ AE recs: {len(recs_ae):,} recomandări")
    
    # 2. Verifică overlap train-test (nu ar trebui să existe pentru același user)
    print("\n[2] Verificare overlap train-test per user...")
    
    train_user_tracks = train.groupby('user_id')['track_id'].apply(set).to_dict()
    test_user_tracks = test.groupby('user_id')['track_id'].apply(set).to_dict()
    
    overlaps = 0
    users_with_overlap = 0
    for user_id in train_user_tracks.keys():
        if user_id in test_user_tracks:
            overlap = train_user_tracks[user_id] & test_user_tracks[user_id]
            if overlap:
                overlaps += len(overlap)
                users_with_overlap += 1
    
    if overlaps > 0:
        print(f"    ⚠  PROBLEMĂ: {overlaps} track-uri duplicate între train și test pentru {users_with_overlap} users")
        print(f"      → Acest lucru poate afecta evaluarea (itemi din train în test)")
    else:
        print(f"    ✓ Nu există overlap între train și test (OK)")
    
    # 3. Verifică dacă track-urile din test sunt în catalog
    print("\n[3] Verificare track-uri test în catalog...")
    
    valid_track_ids = set(tracks_meta['id'].tolist())
    test_track_ids = set(test['track_id'].unique())
    train_track_ids = set(train['track_id'].unique())
    
    test_in_catalog = test_track_ids & valid_track_ids
    test_not_in_catalog = test_track_ids - valid_track_ids
    train_in_catalog = train_track_ids & valid_track_ids
    train_not_in_catalog = train_track_ids - valid_track_ids
    
    print(f"    ✓ Track-uri test în catalog: {len(test_in_catalog):,} / {len(test_track_ids):,}")
    if len(test_not_in_catalog) > 0:
        print(f"    ⚠  PROBLEMĂ: {len(test_not_in_catalog)} track-uri din test NU sunt în catalog")
        print(f"      → Acestea nu pot fi evaluate corect")
    
    print(f"    ✓ Track-uri train în catalog: {len(train_in_catalog):,} / {len(train_track_ids):,}")
    if len(train_not_in_catalog) > 0:
        print(f"    ⚠  PROBLEMĂ: {len(train_not_in_catalog)} track-uri din train NU sunt în catalog")
    
    # 4. Verifică dacă track-urile recomandate sunt în catalog
    print("\n[4] Verificare track-uri recomandate în catalog...")
    
    als_track_ids = set(recs_als['track_id'].unique())
    ae_track_ids = set(recs_ae['track_id'].unique())
    
    als_in_catalog = als_track_ids & valid_track_ids
    als_not_in_catalog = als_track_ids - valid_track_ids
    ae_in_catalog = ae_track_ids & valid_track_ids
    ae_not_in_catalog = ae_track_ids - valid_track_ids
    
    print(f"    ✓ ALS: {len(als_in_catalog):,} / {len(als_track_ids):,} track-uri în catalog")
    if len(als_not_in_catalog) > 0:
        print(f"    ⚠  PROBLEMĂ: {len(als_not_in_catalog)} track-uri ALS NU sunt în catalog")
    
    print(f"    ✓ Autoencoder: {len(ae_in_catalog):,} / {len(ae_track_ids):,} track-uri în catalog")
    if len(ae_not_in_catalog) > 0:
        print(f"    ⚠  PROBLEMĂ: {len(ae_not_in_catalog)} track-uri Autoencoder NU sunt în catalog")
    
    # 5. Verifică dacă ALS recomandă doar track-uri din train
    print("\n[5] Verificare dacă ALS recomandă track-uri din train...")
    
    als_only_train = als_track_ids & train_track_ids
    als_not_in_train = als_track_ids - train_track_ids
    
    print(f"    ✓ ALS recomandă {len(als_only_train):,} track-uri din train")
    if len(als_not_in_train) > 0:
        print(f"    ⚠  ATENȚIE: {len(als_not_in_train)} track-uri ALS NU sunt în train")
        print(f"      → ALS ar trebui să recomande doar track-uri din train")
    else:
        print(f"    ✓ ALS recomandă doar track-uri din train (OK)")
    
    # 6. Analiză catalog vs test
    print("\n[6] Analiză catalog vs test (sparsity)...")
    
    catalog_size = len(tracks_meta)
    test_items_per_user = test.groupby('user_id').size()
    avg_test_items = test_items_per_user.mean()
    
    print(f"    ✓ Catalog: {catalog_size:,} piese")
    print(f"    ✓ Test items per user: min={test_items_per_user.min()}, "
          f"mean={avg_test_items:.1f}, max={test_items_per_user.max()}")
    
    # Probabilitatea de a găsi un item relevant în top-K (random)
    k_values = [5, 10, 20]
    for k in k_values:
        prob_random = (avg_test_items / catalog_size) * k
        print(f"    → Probabilitate random de a găsi item relevant în top-{k}: {prob_random:.4f} ({prob_random*100:.2f}%)")
    
    # 7. Verifică dacă există useri fără recomandări
    print("\n[7] Verificare useri fără recomandări...")
    
    test_users = set(test['user_id'].unique())
    als_users = set(recs_als['user_id'].unique())
    ae_users = set(recs_ae['user_id'].unique())
    
    test_users_no_als = test_users - als_users
    test_users_no_ae = test_users - ae_users
    
    if len(test_users_no_als) > 0:
        print(f"    ⚠  PROBLEMĂ: {len(test_users_no_als)} useri din test NU au recomandări ALS")
    
    if len(test_users_no_ae) > 0:
        print(f"    ⚠  PROBLEMĂ: {len(test_users_no_ae)} useri din test NU au recomandări Autoencoder")
    
    if len(test_users_no_als) == 0 and len(test_users_no_ae) == 0:
        print(f"    ✓ Toți userii din test au recomandări (OK)")
    
    # 8. Verifică numărul de recomandări per user
    print("\n[8] Verificare număr recomandări per user...")
    
    als_recs_per_user = recs_als.groupby('user_id').size()
    ae_recs_per_user = recs_ae.groupby('user_id').size()
    
    print(f"    ✓ ALS: min={als_recs_per_user.min()}, mean={als_recs_per_user.mean():.1f}, max={als_recs_per_user.max()}")
    print(f"    ✓ Autoencoder: min={ae_recs_per_user.min()}, mean={ae_recs_per_user.mean():.1f}, max={ae_recs_per_user.max()}")
    
    # 9. Verifică dacă există track-uri din test care pot fi găsite în recomandări
    print("\n[9] Analiză potențial de matching test-recomandări...")
    
    # Pentru fiecare user, verifică câte track-uri din test sunt în recomandări
    test_truth = test.groupby('user_id')['track_id'].apply(set).to_dict()
    
    als_matches = []
    ae_matches = []
    
    for user_id in test_truth.keys():
        truth_set = test_truth[user_id]
        
        if user_id in als_users:
            als_user_recs = set(recs_als[recs_als['user_id'] == user_id]['track_id'].tolist())
            matches = truth_set & als_user_recs
            als_matches.append(len(matches))
        
        if user_id in ae_users:
            ae_user_recs = set(recs_ae[recs_ae['user_id'] == user_id]['track_id'].tolist())
            matches = truth_set & ae_user_recs
            ae_matches.append(len(matches))
    
    if als_matches:
        print(f"    ✓ ALS: {np.mean(als_matches):.2f} matches per user în medie "
              f"(max={max(als_matches) if als_matches else 0})")
        print(f"      → Users cu ≥1 match: {sum(1 for m in als_matches if m > 0)} / {len(als_matches)}")
    
    if ae_matches:
        print(f"    ✓ Autoencoder: {np.mean(ae_matches):.2f} matches per user în medie "
              f"(max={max(ae_matches) if ae_matches else 0})")
        print(f"      → Users cu ≥1 match: {sum(1 for m in ae_matches if m > 0)} / {len(ae_matches)}")
    
    # 10. Rezumat probleme identificate
    print("\n" + "=" * 60)
    print("REZUMAT PROBLEME IDENTIFICATE:")
    print("=" * 60)
    
    problems = []
    
    if overlaps > 0:
        problems.append(f"⚠  {overlaps} track-uri duplicate între train și test")
    
    if len(test_not_in_catalog) > 0:
        problems.append(f"⚠  {len(test_not_in_catalog)} track-uri din test NU sunt în catalog")
    
    if len(als_not_in_train) > 0:
        problems.append(f"⚠  {len(als_not_in_train)} track-uri ALS NU sunt în train")
    
    if len(test_users_no_als) > 0:
        problems.append(f"⚠  {len(test_users_no_als)} useri din test fără recomandări ALS")
    
    if len(test_users_no_ae) > 0:
        problems.append(f"⚠  {len(test_users_no_ae)} useri din test fără recomandări Autoencoder")
    
    if catalog_size > 20000:
        problems.append(f"⚠  Catalog prea mare ({catalog_size:,} piese) - reduce precizia")
    
    if len(problems) == 0:
        print("✓ Nu s-au identificat probleme majore în logica de recomandare/evaluare")
        print("\n💡 SUGESTIE: Precizia scăzută este probabil cauzată de:")
        print(f"   - Catalog prea mare ({catalog_size:,} piese)")
        print(f"   - Prea puține itemi relevanți în test per user ({avg_test_items:.1f} în medie)")
        print(f"   - Probabilitate random de matching: {prob_random:.4f} pentru top-10")
    else:
        for problem in problems:
            print(problem)
    
    print("\n💡 RECOMANDĂRI:")
    print("   1. Mărește MIN_POPULARITY în prepare_data.py (ex: 50-60)")
    print("   2. Sau filtrează catalogul la track-uri care apar în train")
    print("   3. Sau mărește K (numărul de recomandări) la 500-1000")
    print("=" * 60)

if __name__ == "__main__":
    try:
        diagnostic()
    except Exception as e:
        print(f"\n✗ EROARE: {e}")
        import traceback
        traceback.print_exc()
