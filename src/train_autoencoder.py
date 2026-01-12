#!/usr/bin/env python3
"""
Sistem de Recomandare - Training Autoencoder (ETAPA 4)

COMENZI PENTRU A RULA:
1. Asigură-te că ai activat venv-ul: source venv/bin/activate
2. Instalează dependențele: pip install -r requirements.txt
3. Rulează scriptul: python src/train_autoencoder.py

PREREQUISIT:
Rulează mai întâi:
- python src/prepare_data.py (ETAPA 1)
- python src/simulate_interactions.py (ETAPA 2)
- python src/split_train_test.py (ETAPA 2)

DESCRIERE:
Antrenează un Autoencoder PyTorch pe features audio ale pieselor.
Encoder produce embedding-uri pentru fiecare track.
Recomandările se bazează pe cosine similarity între profilul user-ului
(media embedding-urilor pieselor ascultate) și embedding-urile tuturor pieselor.
"""

import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics.pairwise import cosine_similarity

# Setup paths
ROOT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"

# Parametri configurabili Autoencoder
HIDDEN_DIM = 128  # Dimensiunea stratului hidden (mărit pentru mai multă capacitate)
LATENT_DIM = 32  # Dimensiunea spațiului latent (embedding) (mărit pentru mai multe informații)
EPOCHS = 50  # Număr epoci (mărit pentru convergență mai bună)
LEARNING_RATE = 1e-3  # Learning rate pentru Adam
BATCH_SIZE = 256  # Batch size pentru DataLoader
K = 500  # Număr recomandări top-K per utilizator (mărit pentru catalog mare - mai multe șanse de matching)
RANDOM_SEED = 42

# Quick test mode (pentru debug)
QUICK_TEST = False  # Setează True pentru test rapid (EPOCHS=2)

if QUICK_TEST:
    EPOCHS = 2

# Set seed pentru reproducibilitate
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)


class Autoencoder(nn.Module):
    """
    Autoencoder pentru features audio.
    
    Encoder: Linear -> ReLU -> Linear -> ReLU -> Latent
    Decoder: Linear -> ReLU -> Linear -> Output
    """
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        """
        Forward pass: encode -> decode.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """
        Encodare: returnează embedding-ul (output encoder).
        """
        with torch.no_grad():
            return self.encoder(x)


def train_autoencoder():
    """
    Funcția principală care antrenează Autoencoder și generează recomandări.
    """
    print("=" * 60)
    print("ETAPA 4: Training Autoencoder")
    print("=" * 60)
    
    if QUICK_TEST:
        print("⚠  QUICK TEST MODE: EPOCHS=2 (pentru debug)")
    
    # 1. Citește tracks_features_scaled.npy
    features_path = OUTPUT_DIR / "tracks_features_scaled.npy"
    print(f"\n[1] Citire features din: {features_path}")
    
    try:
        X = np.load(features_path)
        print(f"    ✓ Features încărcate: shape {X.shape}")
        input_dim = X.shape[1]
        print(f"    ✓ Input dimension: {input_dim}")
    except FileNotFoundError:
        print(f"    ✗ EROARE: Fișierul {features_path} nu a fost găsit!")
        print(f"    → Rulează mai întâi: python src/prepare_data.py")
        return
    except Exception as e:
        print(f"    ✗ EROARE la citirea features: {e}")
        return
    
    # 2. Citește tracks_meta.csv
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
    
    # Validare: numărul de piese trebuie să corespundă
    if len(tracks_meta) != X.shape[0]:
        print(f"    ⚠  ATENȚIE: Numărul de piese diferă!")
        print(f"      tracks_meta: {len(tracks_meta)}, features: {X.shape[0]}")
        # Ajustăm la minim
        min_len = min(len(tracks_meta), X.shape[0])
        tracks_meta = tracks_meta.iloc[:min_len]
        X = X[:min_len]
        print(f"      → Ajustat la: {min_len}")
    
    # 3. Citește interactions_train.csv
    interactions_train_path = OUTPUT_DIR / "interactions_train.csv"
    print(f"\n[3] Citire interacțiuni train din: {interactions_train_path}")
    
    try:
        interactions_train = pd.read_csv(interactions_train_path)
        print(f"    ✓ Interacțiuni încărcate: {len(interactions_train):,}")
    except FileNotFoundError:
        print(f"    ✗ EROARE: Fișierul {interactions_train_path} nu a fost găsit!")
        print(f"    → Rulează mai întâi: python src/split_train_test.py")
        return
    except Exception as e:
        print(f"    ✗ EROARE la citirea interacțiunilor: {e}")
        return
    
    # 4. Pregătește datele pentru training
    print(f"\n[4] Pregătire date pentru training...")
    
    # Convertesc la tensor PyTorch
    X_tensor = torch.FloatTensor(X)
    
    # Creează DataLoader pentru mini-batching
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"    ✓ Tensor creat: {X_tensor.shape}")
    print(f"    ✓ DataLoader creat: batch_size={BATCH_SIZE}, batches={len(dataloader)}")
    
    # 5. Creează modelul Autoencoder
    print(f"\n[5] Creare model Autoencoder...")
    print(f"    Arhitectură:")
    print(f"      - Input dim: {input_dim}")
    print(f"      - Hidden dim: {HIDDEN_DIM}")
    print(f"      - Latent dim (embedding): {LATENT_DIM}")
    print(f"      - Encoder: Linear({input_dim}→{HIDDEN_DIM}) → ReLU → Linear({HIDDEN_DIM}→{LATENT_DIM}) → ReLU")
    print(f"      - Decoder: Linear({LATENT_DIM}→{HIDDEN_DIM}) → ReLU → Linear({HIDDEN_DIM}→{input_dim})")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"    ✓ Device: {device}")
    
    model = Autoencoder(input_dim, HIDDEN_DIM, LATENT_DIM).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"    ✓ Model creat și mutat pe {device}")
    print(f"    ✓ Loss: MSELoss")
    print(f"    ✓ Optimizer: Adam(lr={LEARNING_RATE})")
    
    # 6. Antrenează modelul
    print(f"\n[6] Antrenare model ({EPOCHS} epoci)...")
    
    model.train()
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in dataloader:
            batch_data = batch[0].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            reconstructed = model(batch_data)
            loss = criterion(reconstructed, batch_data)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        
        # Print loss la fiecare 5 epoci sau la fiecare epocă dacă QUICK_TEST
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == EPOCHS - 1 or QUICK_TEST:
            print(f"    Epoch {epoch+1:3d}/{EPOCHS}: loss = {avg_loss:.6f}")
    
    training_time = time.time() - start_time
    final_loss = avg_loss
    
    print(f"    ✓ Model antrenat în {training_time:.2f} secunde")
    print(f"    ✓ Final loss: {final_loss:.6f}")
    
    # 7. Calculează embeddings pentru toate piese (Z = encoder(X))
    print(f"\n[7] Calculare embeddings pentru toate piese...")
    
    model.eval()
    X_tensor_device = X_tensor.to(device)
    
    # Calculează embeddings în batch-uri pentru eficiență
    embeddings_list = []
    batch_size_embed = 512
    
    with torch.no_grad():
        for i in range(0, len(X_tensor_device), batch_size_embed):
            batch_X = X_tensor_device[i:i+batch_size_embed]
            batch_embeddings = model.encode(batch_X)
            embeddings_list.append(batch_embeddings.cpu().numpy())
    
    Z = np.vstack(embeddings_list)
    
    print(f"    ✓ Embeddings calculate: shape {Z.shape}")
    print(f"      {Z.shape[0]} piese, {Z.shape[1]} dimensiuni latente")
    
    # 8. Creează mapare track_id -> index
    print(f"\n[8] Creare mapare track_id -> index...")
    
    track_id_to_idx = {track_id: idx for idx, track_id in enumerate(tracks_meta['id'].tolist())}
    idx_to_track_id = {idx: track_id for track_id, idx in track_id_to_idx.items()}
    
    print(f"    ✓ Mapare creată: {len(track_id_to_idx)} piese")
    
    # 9. Generează recomandări pentru fiecare user
    print(f"\n[9] Generare recomandări top-{K} pentru fiecare user...")
    
    # Obține utilizatori unici
    unique_users = sorted(interactions_train['user_id'].unique())
    n_users = len(unique_users)
    
    recommendations_list = []
    
    for user_id in unique_users:
        # Obține track-urile user-ului din train
        user_tracks = interactions_train[interactions_train['user_id'] == user_id]['track_id'].tolist()
        
        # Filtrează track-urile care există în tracks_meta
        user_track_indices = [track_id_to_idx[track_id] 
                             for track_id in user_tracks 
                             if track_id in track_id_to_idx]
        
        if len(user_track_indices) == 0:
            # Dacă user-ul nu are track-uri valide, skip
            continue
        
        # Calculează profil user (media embedding-urilor pieselor ascultate)
        user_embeddings = Z[user_track_indices]
        user_profile = np.mean(user_embeddings, axis=0).reshape(1, -1)  # Shape: (1, latent_dim)
        
        # Normalizează embeddings pentru cosine similarity mai precisă
        # Cosine similarity funcționează mai bine cu embeddings normalizate
        Z_normalized = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)  # Evită împărțirea la 0
        user_profile_normalized = user_profile / (np.linalg.norm(user_profile, axis=1, keepdims=True) + 1e-8)
        
        # Calculează cosine similarity între profil și toate embedding-urile
        # cosine_similarity returnează (1, n_tracks)
        similarities = cosine_similarity(user_profile_normalized, Z_normalized)[0]  # Shape: (n_tracks,)
        
        # Exclude piese deja ascultate (set score = -inf)
        similarities[user_track_indices] = -np.inf
        
        # Obține top-K indici (exclude automat itemii cu -inf prin sortare descrescătoare)
        # argsort cu [::-1] sortează descrescător, deci itemii cu -inf vor fi la final
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Filtrează itemii cu -inf (deja ascultați) și ia top-K valizi
        valid_indices = [idx for idx in sorted_indices if similarities[idx] != -np.inf][:K]
        
        # Adaugă recomandările cu rank
        for rank, track_idx in enumerate(valid_indices, start=1):
            track_id = idx_to_track_id[track_idx]
            score = float(similarities[track_idx])
            
            recommendations_list.append({
                'user_id': user_id,
                'track_id': track_id,
                'score': score,
                'rank': rank
            })
    
    # Creează DataFrame cu recomandările
    recommendations_df = pd.DataFrame(recommendations_list)
    
    print(f"    ✓ Recomandări generate: {len(recommendations_df):,} total")
    print(f"      Media recomandări per user: {len(recommendations_df) / n_users:.1f}")
    
    # 10. Salvează recomandările
    print(f"\n[10] Salvare recomandări...")
    
    recs_path = OUTPUT_DIR / "recs_autoencoder.csv"
    recommendations_df.to_csv(recs_path, index=False)
    print(f"    ✓ Recomandări salvate: {recs_path}")
    print(f"      {len(recommendations_df):,} recomandări, 4 coloane (user_id, track_id, score, rank)")
    
    # 11. Salvează informații despre model
    model_info = {
        'HIDDEN_DIM': HIDDEN_DIM,
        'LATENT_DIM': LATENT_DIM,
        'epochs': EPOCHS,
        'lr': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'K': K,
        'n_tracks': len(tracks_meta),
        'n_users_train': n_users,
        'loss_final': round(final_loss, 6),
        'training_time_seconds': round(training_time, 2),
        'input_dim': input_dim,
        'random_seed': RANDOM_SEED
    }
    
    model_info_path = OUTPUT_DIR / "autoencoder_model_info.json"
    with open(model_info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    print(f"    ✓ Informații model salvate: {model_info_path}")
    
    # 12. Print exemplu recomandări pentru user_id 0 (debug)
    print(f"\n[11] Exemplu: Top 10 recomandări pentru user_id 0...")
    
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
    print(f"✓ Model Autoencoder antrenat:")
    print(f"  - Hidden dim: {HIDDEN_DIM}, Latent dim: {LATENT_DIM}")
    print(f"  - Epochs: {EPOCHS}, LR: {LEARNING_RATE}, Batch size: {BATCH_SIZE}")
    print(f"  - Final loss: {final_loss:.6f}")
    print(f"  - Timp antrenare: {training_time:.2f} secunde")
    print(f"✓ Embeddings:")
    print(f"  - Shape: {Z.shape} ({Z.shape[0]} piese, {Z.shape[1]} dim latente)")
    print(f"✓ Recomandări generate:")
    print(f"  - Top-{K} per user: {len(recommendations_df):,} total")
    print(f"  - Users: {n_users:,}")
    print(f"✓ Fișiere generate în {OUTPUT_DIR}:")
    print(f"  - recs_autoencoder.csv")
    print(f"  - autoencoder_model_info.json")
    print("=" * 60)


if __name__ == "__main__":
    try:
        train_autoencoder()
    except Exception as e:
        print(f"\n✗ EROARE FATALĂ: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
