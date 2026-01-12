#!/usr/bin/env python3
"""
Sistem de Recomandare - Split Train/Test per User (ETAPA 2)

COMENZI PENTRU A RULA:
1. Asigură-te că ai activat venv-ul: source venv/bin/activate
2. Rulează scriptul: python src/split_train_test.py

PREREQUISIT:
Rulează mai întâi: python src/simulate_interactions.py

DESCRIERE:
Face split train/test pentru interacțiunile user-track.
Split-ul este PER USER (nu global) pentru a evalua corect capacitatea
modelului de a prezice preferințele unui utilizator specific.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Setup paths
ROOT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"

# Parametri configurabili
TRAIN_RATIO = 0.8  # 80% train, 20% test
MIN_TRACKS_FOR_TEST = 2  # Minim interacțiuni necesare pentru a avea cel puțin 1 în test
RANDOM_SEED = 42

# Setup random generator cu seed fix
rng = np.random.default_rng(RANDOM_SEED)


def split_train_test():
    """
    Funcția principală care face split train/test per user.
    """
    print("=" * 60)
    print("ETAPA 2: Split Train/Test per User")
    print("=" * 60)
    
    # 1. Citește interactions.csv
    interactions_path = OUTPUT_DIR / "interactions.csv"
    print(f"\n[1] Citire interacțiuni din: {interactions_path}")
    
    try:
        interactions_df = pd.read_csv(interactions_path)
        print(f"    ✓ Date încărcate: {len(interactions_df):,} interacțiuni")
    except FileNotFoundError:
        print(f"    ✗ EROARE: Fișierul {interactions_path} nu a fost găsit!")
        print(f"    → Rulează mai întâi: python src/simulate_interactions.py")
        return
    except Exception as e:
        print(f"    ✗ EROARE la citirea datelor: {e}")
        return
    
    # Validare coloane
    required_cols = ['user_id', 'track_id', 'play']
    missing_cols = [col for col in required_cols if col not in interactions_df.columns]
    if missing_cols:
        print(f"    ✗ EROARE: Coloane lipsă: {missing_cols}")
        return
    
    # 2. Analizează distribuția interacțiunilor per user
    print(f"\n[2] Analiză distribuție interacțiuni per user...")
    
    user_counts = interactions_df['user_id'].value_counts().sort_index()
    n_users = len(user_counts)
    min_interactions = user_counts.min()
    max_interactions = user_counts.max()
    mean_interactions = user_counts.mean()
    median_interactions = user_counts.median()
    
    print(f"    ✓ Număr utilizatori: {n_users:,}")
    print(f"    ✓ Interacțiuni per user:")
    print(f"      - Min: {min_interactions}")
    print(f"      - Mean: {mean_interactions:.1f}")
    print(f"      - Median: {median_interactions:.0f}")
    print(f"      - Max: {max_interactions}")
    
    # Identifică utilizatorii cu prea puține interacțiuni
    users_too_few = user_counts[user_counts < MIN_TRACKS_FOR_TEST].index.tolist()
    if users_too_few:
        print(f"\n    ⚠  Utilizatori cu <{MIN_TRACKS_FOR_TEST} interacțiuni: {len(users_too_few)}")
        print(f"      → Aceștia vor fi excluși din split (nu pot avea test set valid)")
    
    # 3. Split per user
    print(f"\n[3] Split per user ({TRAIN_RATIO:.0%} train / {1-TRAIN_RATIO:.0%} test)...")
    
    # NOTĂ: De ce split per user (nu global)?
    # - Split-ul per user asigură că fiecare utilizator apare în ambele seturi (train și test)
    # - Permite evaluarea corectă: modelul învață din interacțiunile unui user și prezice
    #   celelalte interacțiuni ale aceluiași user
    # - Split global (random peste toate interacțiunile) ar crea un set de test cu utilizatori
    #   complet noi (cold start), ceea ce e o problemă diferită
    # - Split per user = evaluare a capacității de predicție pentru utilizatori cunoscuți
    # - Practic standard în sisteme de recomandare pentru evaluare
    
    train_interactions = []
    test_interactions = []
    excluded_users = []
    
    for user_id in interactions_df['user_id'].unique():
        user_interactions = interactions_df[interactions_df['user_id'] == user_id].copy()
        n_user_interactions = len(user_interactions)
        
        # Exclude utilizatori cu prea puține interacțiuni
        if n_user_interactions < MIN_TRACKS_FOR_TEST:
            excluded_users.append(user_id)
            continue
        
        # Shuffle interacțiunile user-ului (pentru randomizare)
        user_interactions = user_interactions.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
        
        # Calculează split point
        n_train = max(1, int(n_user_interactions * TRAIN_RATIO))  # Minim 1 în train
        n_test = n_user_interactions - n_train
        
        # Asigură minim 1 item în test (dacă e posibil)
        if n_test == 0 and n_user_interactions > 1:
            n_train = n_user_interactions - 1
            n_test = 1
        
        # Split
        train_user = user_interactions.iloc[:n_train].copy()
        test_user = user_interactions.iloc[n_train:].copy()
        
        train_interactions.append(train_user)
        test_interactions.append(test_user)
    
    # Concatenează rezultatele
    train_df = pd.concat(train_interactions, ignore_index=True)
    test_df = pd.concat(test_interactions, ignore_index=True)
    
    if excluded_users:
        print(f"    ✓ Utilizatori excluși (prea puține interacțiuni): {len(excluded_users)}")
    
    print(f"    ✓ Split complet: {len(train_df):,} train, {len(test_df):,} test")
    
    # 4. Validare și statistici
    print(f"\n[4] Validare și statistici split...")
    
    # Verifică că nu există overlap între train și test pentru același user
    train_user_tracks = train_df.groupby('user_id')['track_id'].apply(set).to_dict()
    test_user_tracks = test_df.groupby('user_id')['track_id'].apply(set).to_dict()
    
    overlaps = 0
    for user_id in train_user_tracks.keys():
        if user_id in test_user_tracks:
            overlap = train_user_tracks[user_id] & test_user_tracks[user_id]
            if overlap:
                overlaps += len(overlap)
    
    if overlaps > 0:
        print(f"    ⚠  Atenție: {overlaps} track-uri duplicate între train și test pentru același user")
        # Elimină duplicatele din test (păstrăm doar în train)
        for user_id in train_user_tracks.keys():
            if user_id in test_user_tracks:
                overlap = train_user_tracks[user_id] & test_user_tracks[user_id]
                if overlap:
                    test_df = test_df[~((test_df['user_id'] == user_id) & (test_df['track_id'].isin(overlap)))]
    
    # Recalculează statistici după curățare
    n_train_users = train_df['user_id'].nunique()
    n_test_users = test_df['user_id'].nunique()
    
    train_user_counts = train_df.groupby('user_id').size()
    test_user_counts = test_df.groupby('user_id').size()
    
    print(f"    ✓ Utilizatori în train: {n_train_users:,}")
    print(f"    ✓ Utilizatori în test: {n_test_users:,}")
    print(f"\n    Distribuție interacțiuni train per user:")
    print(f"      - Min: {train_user_counts.min()}")
    print(f"      - Mean: {train_user_counts.mean():.1f}")
    print(f"      - Max: {train_user_counts.max()}")
    print(f"\n    Distribuție interacțiuni test per user:")
    print(f"      - Min: {test_user_counts.min()}")
    print(f"      - Mean: {test_user_counts.mean():.1f}")
    print(f"      - Max: {test_user_counts.max()}")
    
    # 5. Salvează fișierele
    print(f"\n[5] Salvare fișiere...")
    
    train_path = OUTPUT_DIR / "interactions_train.csv"
    train_df.to_csv(train_path, index=False)
    print(f"    ✓ Train set salvat: {train_path}")
    print(f"      {len(train_df):,} interacțiuni, {n_train_users:,} utilizatori")
    
    test_path = OUTPUT_DIR / "interactions_test.csv"
    test_df.to_csv(test_path, index=False)
    print(f"    ✓ Test set salvat: {test_path}")
    print(f"      {len(test_df):,} interacțiuni, {n_test_users:,} utilizatori")
    
    # Rezumat final
    print(f"\n" + "=" * 60)
    print("REZULTATE FINALE:")
    print("=" * 60)
    print(f"✓ Train set: {len(train_df):,} interacțiuni ({len(train_df)/len(interactions_df)*100:.1f}%)")
    print(f"✓ Test set: {len(test_df):,} interacțiuni ({len(test_df)/len(interactions_df)*100:.1f}%)")
    print(f"✓ Utilizatori excluși: {len(excluded_users)}")
    print(f"✓ Fișiere generate în {OUTPUT_DIR}:")
    print(f"  - interactions_train.csv")
    print(f"  - interactions_test.csv")
    print("=" * 60)


if __name__ == "__main__":
    try:
        split_train_test()
    except Exception as e:
        print(f"\n✗ EROARE FATALĂ: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
