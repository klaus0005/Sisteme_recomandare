#!/usr/bin/env python3
"""
Sistem de Recomandare - Demo CLI

COMENZI PENTRU A RULA:
1. Activare venv-ul: source venv/bin/activate
2. Rulare scriptul: python src/demo_cli.py
   
   Dacă nu specifici user_id, vei fi întrebat de la tastatură.



DESCRIERE:
Afișează recomandări ALS și Autoencoder pentru un user specific,
împreună cu ground truth din test set.
"""

import sys
from pathlib import Path
import pandas as pd

# Setup paths
ROOT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"


def load_recs(recs_path):
    """
    Citește recomandările și returnează dict[user_id] = list(dict).
    
    Args:
        recs_path: Path către fișierul de recomandări
    
    Returns:
        dict: {user_id: [{'rank': int, 'track_id': str, 'score': float}, ...]}
    """
    recs_df = pd.read_csv(recs_path)
    recs_df = recs_df.sort_values(['user_id', 'rank'])
    
    recs = {}
    for _, row in recs_df.iterrows():
        user_id = row['user_id']
        if user_id not in recs:
            recs[user_id] = []
        recs[user_id].append({
            'rank': int(row['rank']),
            'track_id': row['track_id'],
            'score': float(row['score'])
        })
    
    return recs


def load_truth(test_path):
    """
    Citește ground truth din test set.
    
    Args:
        test_path: Path către interactions_test.csv
    
    Returns:
        dict: {user_id: list(track_id)}
    """
    test_df = pd.read_csv(test_path)
    truth = {}
    for _, row in test_df.iterrows():
        user_id = row['user_id']
        track_id = row['track_id']
        if user_id not in truth:
            truth[user_id] = []
        truth[user_id].append(track_id)
    return truth


def demo_cli():
    """
    Funcția principală pentru demo CLI.
    """
    print("=" * 70)
    print("DEMO CLI - Recomandări pentru User")
    print("=" * 70)
    
    # Obține user_id din argument sau de la tastatură
    if len(sys.argv) > 1:
        try:
            user_id = int(sys.argv[1])
        except ValueError:
            print(f"✗ EROARE: '{sys.argv[1]}' nu este un user_id valid (trebuie să fie număr întreg)")
            return
    else:
        try:
            user_id = int(input("\nIntrodu user_id: "))
        except ValueError:
            print("✗ EROARE: Trebuie să introduci un număr întreg")
            return
        except KeyboardInterrupt:
            print("\n\nInteruptat de utilizator.")
            return
    
    # 1. Încarcă fișierele necesare
    print(f"\n[1] Încărcare date...")
    
    tracks_meta_path = OUTPUT_DIR / "tracks_meta.csv"
    als_recs_path = OUTPUT_DIR / "recs_als.csv"
    ae_recs_path = OUTPUT_DIR / "recs_autoencoder.csv"
    test_path = OUTPUT_DIR / "interactions_test.csv"
    
    try:
        tracks_meta = pd.read_csv(tracks_meta_path)
        print(f"    ✓ tracks_meta.csv încărcat ({len(tracks_meta)} piese)")
    except FileNotFoundError:
        print(f"    ✗ EROARE: {tracks_meta_path} nu a fost găsit!")
        print(f"    → Rulează mai întâi: python src/prepare_data.py")
        return
    except Exception as e:
        print(f"    ✗ EROARE la citirea tracks_meta: {e}")
        return
    
    try:
        als_recs = load_recs(als_recs_path)
        print(f"    ✓ recs_als.csv încărcat ({len(als_recs)} useri)")
    except FileNotFoundError:
        print(f"    ✗ EROARE: {als_recs_path} nu a fost găsit!")
        print(f"    → Rulează mai întâi: python src/train_als.py")
        return
    except Exception as e:
        print(f"    ✗ EROARE la citirea recs_als: {e}")
        return
    
    try:
        ae_recs = load_recs(ae_recs_path)
        print(f"    ✓ recs_autoencoder.csv încărcat ({len(ae_recs)} useri)")
    except FileNotFoundError:
        print(f"    ✗ EROARE: {ae_recs_path} nu a fost găsit!")
        print(f"    → Rulează mai întâi: python src/train_autoencoder.py")
        return
    except Exception as e:
        print(f"    ✗ EROARE la citirea recs_autoencoder: {e}")
        return
    
    try:
        truth = load_truth(test_path)
        print(f"    ✓ interactions_test.csv încărcat ({len(truth)} useri)")
    except FileNotFoundError:
        print(f"    ✗ EROARE: {test_path} nu a fost găsit!")
        print(f"    → Rulează mai întâi: python src/split_train_test.py")
        return
    except Exception as e:
        print(f"    ✗ EROARE la citirea interactions_test: {e}")
        return
    
    # 2. Verifică dacă user_id există
    print(f"\n[2] Verificare user_id {user_id}...")
    
    user_found = False
    if user_id in als_recs or user_id in ae_recs or user_id in truth:
        user_found = True
    
    if not user_found:
        # Sugerează user_id valid
        all_users = set(als_recs.keys()) | set(ae_recs.keys()) | set(truth.keys())
        if len(all_users) > 0:
            valid_users = sorted(list(all_users))[:10]
            print(f"    ✗ User_id {user_id} nu există în date!")
            print(f"    ⚠  Useri valizi disponibili (exemplu): {valid_users}")
            print(f"       Interval valid: {min(all_users)} - {max(all_users)}")
        else:
            print(f"    ✗ User_id {user_id} nu există și nu există useri disponibili!")
        return
    
    # 3. Afișează recomandări ALS
    print(f"\n" + "=" * 70)
    print(f"RECOMANDĂRI ALS pentru user_id {user_id}")
    print("=" * 70)
    
    if user_id in als_recs:
        als_user_recs = als_recs[user_id][:10]  # Top 10
        
        als_track_ids = [r['track_id'] for r in als_user_recs]
        als_df = tracks_meta[tracks_meta['id'].isin(als_track_ids)]
        als_df = als_df.set_index('id').loc[als_track_ids].reset_index()
        
        print(f"\n{'Rank':<6} {'Score':<12} {'Nume Piesă':<50} {'Artist':<30}")
        print("-" * 70)
        
        for rec, (_, row) in zip(als_user_recs, als_df.iterrows()):
            rank = rec['rank']
            score = rec['score']
            name = row.get('name', 'N/A')[:48]
            artist = row.get('artist', 'N/A')[:28]
            print(f"{rank:<6} {score:<12.6f} {name:<50} {artist:<30}")
    else:
        print(f"    ⚠  Nu există recomandări ALS pentru user_id {user_id}")
    
    # 4. Afișează recomandări Autoencoder
    print(f"\n" + "=" * 70)
    print(f"RECOMANDĂRI AUTOENCODER pentru user_id {user_id}")
    print("=" * 70)
    
    if user_id in ae_recs:
        ae_user_recs = ae_recs[user_id][:10]  # Top 10
        
        ae_track_ids = [r['track_id'] for r in ae_user_recs]
        ae_df = tracks_meta[tracks_meta['id'].isin(ae_track_ids)]
        ae_df = ae_df.set_index('id').loc[ae_track_ids].reset_index()
        
        print(f"\n{'Rank':<6} {'Score':<12} {'Nume Piesă':<50} {'Artist':<30}")
        print("-" * 70)
        
        for rec, (_, row) in zip(ae_user_recs, ae_df.iterrows()):
            rank = rec['rank']
            score = rec['score']
            name = row.get('name', 'N/A')[:48]
            artist = row.get('artist', 'N/A')[:28]
            print(f"{rank:<6} {score:<12.6f} {name:<50} {artist:<30}")
    else:
        print(f"    ⚠  Nu există recomandări Autoencoder pentru user_id {user_id}")
    
    # 5. Afișează ground truth (test)
    print(f"\n" + "=" * 70)
    print(f"GROUND TRUTH (Test Set) pentru user_id {user_id}")
    print("=" * 70)
    
    if user_id in truth:
        truth_track_ids = truth[user_id][:10]  # Primele 10
        truth_df = tracks_meta[tracks_meta['id'].isin(truth_track_ids)]
        
        print(f"\n{'Nume Piesă':<50} {'Artist':<30}")
        print("-" * 70)
        
        for track_id in truth_track_ids:
            track_row = truth_df[truth_df['id'] == track_id]
            if len(track_row) > 0:
                row = track_row.iloc[0]
                name = row.get('name', 'N/A')[:48]
                artist = row.get('artist', 'N/A')[:28]
                print(f"{name:<50} {artist:<30}")
        
        if len(truth[user_id]) > 10:
            print(f"\n... și {len(truth[user_id]) - 10} alte piese în test set")
    else:
        print(f"    ⚠  Nu există date în test set pentru user_id {user_id}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        demo_cli()
    except KeyboardInterrupt:
        print("\n\nInteruptat de utilizator.")
        exit(0)
    except Exception as e:
        print(f"\n✗ EROARE FATALĂ: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
