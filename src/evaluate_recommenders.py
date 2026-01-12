#!/usr/bin/env python3
"""
Sistem de Recomandare - Evaluare Recomandări (ETAPA 5)

COMENZI PENTRU A RULA:
1. Asigură-te că ai activat venv-ul: source venv/bin/activate
2. Rulează scriptul: python src/evaluate_recommenders.py

PREREQUISIT:
Rulează mai întâi:
- python src/train_als.py (ETAPA 3)
- python src/train_autoencoder.py (ETAPA 4)
- python src/split_train_test.py (ETAPA 2)

DESCRIERE:
Evaluează modelele ALS și Autoencoder folosind metrici top-K:
Precision@K, Recall@K, NDCG@K pentru K = 5, 10, 20.

"""

import math
from pathlib import Path
import pandas as pd
import numpy as np

# Setup paths
ROOT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"

# K values pentru evaluare
K_VALUES = [5, 10, 20]
DEMO_USER_ID = 0  # User ID pentru demo/debug


def load_truth(test_path):
    """
    Citește interactions_test.csv și construiește ground truth.
    
    Args:
        test_path: Path către interactions_test.csv
    
    Returns:
        dict: {user_id: set(track_id)} - ground truth pentru fiecare user
    """
    test_df = pd.read_csv(test_path)
    
    # Construiește dict[user_id] = set(track_id)
    truth = {}
    for _, row in test_df.iterrows():
        user_id = row['user_id']
        track_id = row['track_id']
        
        if user_id not in truth:
            truth[user_id] = set()
        truth[user_id].add(track_id)
    
    return truth


def load_recs(recs_path):
    """
    Citește recomandările (recs_als.csv sau recs_autoencoder.csv).
    
    Args:
        recs_path: Path către fișierul de recomandări
    
    Returns:
        dict: {user_id: list(track_id)} - lista ordonată după rank (1..K)
    """
    recs_df = pd.read_csv(recs_path)
    
    # Sortează pe user_id și rank pentru a asigura ordinea corectă
    recs_df = recs_df.sort_values(['user_id', 'rank'])
    
    # Construiește dict[user_id] = list(track_id) ordonată după rank
    recs = {}
    for _, row in recs_df.iterrows():
        user_id = row['user_id']
        track_id = row['track_id']
        rank = row['rank']
        
        if user_id not in recs:
            recs[user_id] = []
        
        # Adaugă track_id la poziția corectă (rank-1 pentru index 0-based)
        # Verificăm că lista e suficient de mare
        while len(recs[user_id]) < rank:
            recs[user_id].append(None)
        recs[user_id][rank - 1] = track_id
    
    # Elimină None-urile și păstrează doar track_id-urile valide
    for user_id in recs:
        recs[user_id] = [tid for tid in recs[user_id] if tid is not None]
    
    return recs


def precision_recall_ndcg_at_k(recommended_list, truth_set, k):
    """
    Calculează Precision@K, Recall@K și NDCG@K pentru un user.
    
    Args:
        recommended_list: list(track_id) - lista recomandărilor ordonată după rank
        truth_set: set(track_id) - set de track_id-uri relevante (ground truth)
        k: int - valoarea K pentru metrici
    
    Returns:
        tuple: (precision, recall, ndcg) sau (0, 0, 0) dacă truth_set e gol
    """
    if len(truth_set) == 0:
        return 0.0, 0.0, 0.0
    
    # Ia top-K recomandări
    top_k = recommended_list[:k]
    
    # Calculează hits (itemi relevanți în top-K)
    hits = sum(1 for item in top_k if item in truth_set)
    
    # Precision@K = hits / K
    precision = hits / k if k > 0 else 0.0
    
    # Recall@K = hits / |truth|
    recall = hits / len(truth_set) if len(truth_set) > 0 else 0.0
    
    # NDCG@K
    # DCG = sum(relevance_i / log2(i+1)) pentru itemii relevanți
    # Pentru binary relevance: relevance_i = 1 dacă item i e în truth, 0 altfel
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in truth_set:
            # i+1 pentru că log2(1) = 0 (evităm diviziunea cu 0)
            dcg += 1.0 / math.log2(i + 2)
    
    # IDCG = DCG pentru ranking ideal (toți itemii relevanți la început)
    # IDCG = sum(1 / log2(i+1)) pentru i = 0..min(k, |truth|)-1
    idcg = 0.0
    num_relevant = min(k, len(truth_set))
    for i in range(num_relevant):
        idcg += 1.0 / math.log2(i + 2)
    
    # NDCG = DCG / IDCG
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    return precision, recall, ndcg


def evaluate_model(model_name, recs_path, truth, k_values):
    """
    Evaluează un model complet.
    
    Args:
        model_name: str - numele modelului ("ALS" sau "Autoencoder")
        recs_path: Path - path către fișierul de recomandări
        truth: dict - ground truth {user_id: set(track_id)}
        k_values: list - lista de valori K pentru evaluare
    
    Returns:
        dict: {k: {'precision': float, 'recall': float, 'ndcg': float, 'n_users': int}}
    """
    print(f"\n  Evaluare {model_name}...")
    
    # Încarcă recomandări
    recs = load_recs(recs_path)
    
    # Găsește userii eligibili (există în truth și în recs, cu cel puțin 1 item în truth)
    eligible_users = set(truth.keys()) & set(recs.keys())
    eligible_users = [uid for uid in eligible_users if len(truth[uid]) > 0]
    
    print(f"    ✓ Useri eligibili: {len(eligible_users):,}")
    
    if len(eligible_users) == 0:
        print(f"    ⚠  ATENȚIE: Nu există useri eligibili pentru {model_name}!")
        return {}
    
    # Calculează metrici pentru fiecare K
    results = {}
    
    for k in k_values:
        precisions = []
        recalls = []
        ndcgs = []
        
        for user_id in eligible_users:
            recommended_list = recs.get(user_id, [])
            truth_set = truth[user_id]
            
            precision, recall, ndcg = precision_recall_ndcg_at_k(
                recommended_list, truth_set, k
            )
            
            precisions.append(precision)
            recalls.append(recall)
            ndcgs.append(ndcg)
        
        # Agregă metricile (mean)
        results[k] = {
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'ndcg': np.mean(ndcgs),
            'n_users': len(eligible_users)
        }
        
        print(f"    K={k}: P={results[k]['precision']:.4f}, "
              f"R={results[k]['recall']:.4f}, NDCG={results[k]['ndcg']:.4f}")
    
    return results


def evaluate_recommenders():
    """
    Funcția principală care evaluează modelele ALS și Autoencoder.
    """
    print("=" * 60)
    print("ETAPA 5: Evaluare Recomandări (ALS vs Autoencoder)")
    print("=" * 60)
    
    # 1. Încarcă ground truth din test
    test_path = OUTPUT_DIR / "interactions_test.csv"
    print(f"\n[1] Încărcare ground truth din: {test_path}")
    
    try:
        truth = load_truth(test_path)
        print(f"    ✓ Ground truth încărcat: {len(truth)} useri")
        
        # Statistici
        truth_sizes = [len(truth[uid]) for uid in truth]
        print(f"    ✓ Items per user: min={min(truth_sizes)}, "
              f"mean={np.mean(truth_sizes):.1f}, max={max(truth_sizes)}")
    except FileNotFoundError:
        print(f"    ✗ EROARE: Fișierul {test_path} nu a fost găsit!")
        print(f"    → Rulează mai întâi: python src/split_train_test.py")
        return
    except Exception as e:
        print(f"    ✗ EROARE la citirea ground truth: {e}")
        return
    
    # 2. Evaluează modelele
    print(f"\n[2] Evaluare modele (K = {K_VALUES})...")
    
    als_path = OUTPUT_DIR / "recs_als.csv"
    autoencoder_path = OUTPUT_DIR / "recs_autoencoder.csv"
    
    als_results = {}
    autoencoder_results = {}
    
    try:
        als_results = evaluate_model("ALS", als_path, truth, K_VALUES)
    except FileNotFoundError:
        print(f"    ✗ EROARE: Fișierul {als_path} nu a fost găsit!")
        print(f"    → Rulează mai întâi: python src/train_als.py")
    except Exception as e:
        print(f"    ✗ EROARE la evaluarea ALS: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        autoencoder_results = evaluate_model("Autoencoder", autoencoder_path, truth, K_VALUES)
    except FileNotFoundError:
        print(f"    ✗ EROARE: Fișierul {autoencoder_path} nu a fost găsit!")
        print(f"    → Rulează mai întâi: python src/train_autoencoder.py")
    except Exception as e:
        print(f"    ✗ EROARE la evaluarea Autoencoder: {e}")
        import traceback
        traceback.print_exc()
    
    if not als_results or not autoencoder_results:
        print(f"\n    ✗ EROARE: Nu s-au putut evalua ambele modele!")
        return
    
    # 3. Salvează rezultatele în CSV
    print(f"\n[3] Salvare rezultate în CSV...")
    
    results_list = []
    
    for k in K_VALUES:
        # ALS
        if k in als_results:
            results_list.append({
                'model': 'ALS',
                'K': k,
                'precision': als_results[k]['precision'],
                'recall': als_results[k]['recall'],
                'ndcg': als_results[k]['ndcg'],
                'n_users_evaluated': als_results[k]['n_users']
            })
        
        # Autoencoder
        if k in autoencoder_results:
            results_list.append({
                'model': 'Autoencoder',
                'K': k,
                'precision': autoencoder_results[k]['precision'],
                'recall': autoencoder_results[k]['recall'],
                'ndcg': autoencoder_results[k]['ndcg'],
                'n_users_evaluated': autoencoder_results[k]['n_users']
            })
    
    results_df = pd.DataFrame(results_list)
    
    eval_path = OUTPUT_DIR / "eval_results.csv"
    results_df.to_csv(eval_path, index=False)
    print(f"    ✓ Rezultate salvate: {eval_path}")
    
    # 4. Print raport comparativ (leaderboard)
    print(f"\n[4] Raport Comparativ:")
    print("=" * 60)
    
    for k in K_VALUES:
        print(f"\nK = {k}:")
        print(f"  {'Model':<15} {'Precision':<12} {'Recall':<12} {'NDCG':<12} {'Users':<10}")
        print(f"  {'-'*65}")
        
        if k in als_results:
            als_r = als_results[k]
            print(f"  {'ALS':<15} {als_r['precision']:<12.4f} {als_r['recall']:<12.4f} "
                  f"{als_r['ndcg']:<12.4f} {als_r['n_users']:<10}")
        
        if k in autoencoder_results:
            ae_r = autoencoder_results[k]
            print(f"  {'Autoencoder':<15} {ae_r['precision']:<12.4f} {ae_r['recall']:<12.4f} "
                  f"{ae_r['ndcg']:<12.4f} {ae_r['n_users']:<10}")
        
        # Diferență (Autoencoder - ALS)
        if k in als_results and k in autoencoder_results:
            diff_p = autoencoder_results[k]['precision'] - als_results[k]['precision']
            diff_r = autoencoder_results[k]['recall'] - als_results[k]['recall']
            diff_n = autoencoder_results[k]['ndcg'] - als_results[k]['ndcg']
            
            print(f"  {'Difference':<15} {diff_p:<12.4f} {diff_r:<12.4f} {diff_n:<12.4f}")
            print(f"  (Autoencoder - ALS)")
    
    print("\n" + "=" * 60)
    
    # 5. Debug/Demo: recomandări pentru user_id DEMO_USER_ID
    print(f"\n[5] Demo: Recomandări pentru user_id {DEMO_USER_ID}...")
    
    tracks_meta_path = OUTPUT_DIR / "tracks_meta.csv"
    
    try:
        tracks_meta = pd.read_csv(tracks_meta_path)
        
        if DEMO_USER_ID in truth:
            print(f"\n  Ground truth (test) pentru user_id {DEMO_USER_ID}:")
            truth_items = list(truth[DEMO_USER_ID])[:10]  # Primele 10
            
            truth_df = tracks_meta[tracks_meta['id'].isin(truth_items)]
            for idx, row in truth_df.iterrows():
                name = row.get('name', 'N/A')
                artist = row.get('artist', 'N/A')
                print(f"    - {name[:50]:<50} | {artist[:30]}")
            
            if len(truth[DEMO_USER_ID]) > 10:
                print(f"    ... și {len(truth[DEMO_USER_ID]) - 10} alte itemi")
            
            # ALS recomandări
            als_recs = load_recs(als_path)
            if DEMO_USER_ID in als_recs:
                print(f"\n  Top 10 recomandări ALS pentru user_id {DEMO_USER_ID}:")
                als_top10 = als_recs[DEMO_USER_ID][:10]
                
                als_df = tracks_meta[tracks_meta['id'].isin(als_top10)]
                # Păstrează ordinea
                als_df = als_df.set_index('id').loc[als_top10].reset_index()
                
                for idx, row in als_df.iterrows():
                    name = row.get('name', 'N/A')
                    artist = row.get('artist', 'N/A')
                    track_id = row['id']
                    rank = als_top10.index(track_id) + 1
                    print(f"    {rank:2d}. {name[:50]:<50} | {artist[:30]}")
            
            # Autoencoder recomandări
            ae_recs = load_recs(autoencoder_path)
            if DEMO_USER_ID in ae_recs:
                print(f"\n  Top 10 recomandări Autoencoder pentru user_id {DEMO_USER_ID}:")
                ae_top10 = ae_recs[DEMO_USER_ID][:10]
                
                ae_df = tracks_meta[tracks_meta['id'].isin(ae_top10)]
                # Păstrează ordinea
                ae_df = ae_df.set_index('id').loc[ae_top10].reset_index()
                
                for idx, row in ae_df.iterrows():
                    name = row.get('name', 'N/A')
                    artist = row.get('artist', 'N/A')
                    track_id = row['id']
                    rank = ae_top10.index(track_id) + 1
                    print(f"    {rank:2d}. {name[:50]:<50} | {artist[:30]}")
        else:
            print(f"    ⚠  User_id {DEMO_USER_ID} nu există în ground truth")
    
    except Exception as e:
        print(f"    ⚠  Eroare la demo: {e}")
    
    # Rezumat final
    print(f"\n" + "=" * 60)
    print("REZULTATE FINALE:")
    print("=" * 60)
    print(f"✓ Evaluare completă pentru K = {K_VALUES}")
    print(f"✓ Modele evaluate: ALS, Autoencoder")
    print(f"✓ Fișier generat: {eval_path}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        evaluate_recommenders()
    except Exception as e:
        print(f"\n✗ EROARE FATALĂ: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
