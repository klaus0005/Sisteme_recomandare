# Sistem de Recomandare pentru Muzică (Spotify Audio Features)

Proiect P2 - Sistem de recomandare bazat pe features audio Spotify, comparând două abordări: ALS (baseline) și Autoencoder (model principal).

## 📋 Descriere

Acest proiect implementează un sistem de recomandare muzicală care utilizează features audio de la Spotify (danceability, energy, valence, etc.) pentru a genera recomandări personalizate. Sistemul compară două metode:

- **ALS (Alternating Least Squares)**: Metodă clasică de matrix factorization pentru implicit feedback
- **Autoencoder**: Model neural care învață embedding-uri din features audio și generează recomandări bazate pe cosine similarity

## 🔧 Cerințe

- **Python**: 3.8+
- **venv**: Mediu virtual Python (recomandat)

## 📦 Instalare

1. **Clonează repository-ul** (sau navighează în directorul proiectului)

2. **Activează venv-ul**:
   ```bash
   source venv/bin/activate
   ```
   Pe Windows:
   ```bash
   venv\Scripts\activate
   ```

3. **Instalează dependențele**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Rulare

### Opțiunea 1: Rulează tot automat (RECOMANDAT)

Pentru a rula toate etapele automat în ordine:

```bash
python src/run_all.py
```

Scriptul va:
- Rula toate etapele în ordine (1-5)
- Sări etapele care au outputs deja generate (skip)
- Afișa progres și timpul pentru fiecare etapă
- Verifica outputs finale

**Notă:** Dacă vrei să regeneri totul, șterge outputs/ înainte sau rulează etapele manual.

### Opțiunea 2: Rulează manual (etape individuale)

Proiectul este organizat în 5 etape, care trebuie rulate în ordine:

### ETAPA 1: Pregătirea Datelor
```bash
python src/prepare_data.py
```
**Outputs generate:**
- `outputs/tracks_meta.csv` - metadata piese (id, name, artist, year, popularity)
- `outputs/tracks_features_scaled.npy` - features audio normalizate
- `outputs/feature_columns.json` - lista coloanelor de features

### ETAPA 2: Simulare Interacțiuni și Split Train/Test
```bash
# Simulează interacțiuni user-track
python src/simulate_interactions.py

# Split train/test per user (80/20)
python src/split_train_test.py
```
**Outputs generate:**
- `outputs/interactions.csv` - toate interacțiunile simulate
- `outputs/interactions_train.csv` - setul de antrenare
- `outputs/interactions_test.csv` - setul de test
- `outputs/users_profile.csv` - profiluri utilizatori (favorite artist, etc.)

### ETAPA 3: Training ALS (Baseline)
```bash
python src/train_als.py
```
**Outputs generate:**
- `outputs/recs_als.csv` - recomandări ALS (user_id, track_id, score, rank)
- `outputs/als_model_info.json` - metadate model ALS

### ETAPA 4: Training Autoencoder (Model Principal)
```bash
python src/train_autoencoder.py
```
**Outputs generate:**
- `outputs/recs_autoencoder.csv` - recomandări Autoencoder (user_id, track_id, score, rank)
- `outputs/autoencoder_model_info.json` - metadate model Autoencoder

### ETAPA 5: Evaluare
```bash
python src/evaluate_recommenders.py
```
**Outputs generate:**
- `outputs/eval_results.csv` - rezultate evaluare (Precision@K, Recall@K, NDCG@K pentru K=5,10,20)

## 🎯 Demo CLI

Pentru a vedea recomandările pentru un user specific în terminal:

```bash
python src/demo_cli.py [user_id]
```

Dacă nu specifici `user_id`, vei fi întrebat de la tastatură.

**Exemplu:**
```bash
python src/demo_cli.py 0
```

Afișează:
- Top 10 recomandări ALS (cu score, nume piesă, artist)
- Top 10 recomandări Autoencoder (cu score, nume piesă, artist)
- Piese din test_truth pentru user (ground truth)

## 🌐 Demo UI

Pentru a folosi interfața web pentru recomandări:

### 1. Pornește Backend (FastAPI)

Într-un terminal, activează venv-ul și pornește serverul:

```bash
source venv/bin/activate  # (sau venv\Scripts\activate pe Windows)
uvicorn src.app:app --reload
```

Serverul va rula pe `http://127.0.0.1:8000`

**Notă:** Backend-ul trebuie să ruleze pentru ca interfața web să funcționeze.

### 2. Deschide Interfața Web

**Opțiunea 1 (Recomandată):** Accesează direct prin FastAPI:
- Deschide în browser: `http://127.0.0.1:8000` sau `http://localhost:8000`
- FastAPI servește automat `ui/index.html` la root

**Opțiunea 2:** Deschide fișierul direct:
- Double-click pe `ui/index.html` sau `open ui/index.html`
- **Notă:** Dacă primești "failed to fetch", asigură-te că backend-ul rulează
- Pentru a evita probleme CORS, folosește Opțiunea 1 (accesează prin FastAPI)

### 3. Folosește Interfața

1. Introdu un **User ID** (ex: 0, 1, 2, etc.)
2. Selectează numărul de recomandări (default: 10)
3. Click pe **"Caută Recomandări"**
4. Vezi recomandările ALS și Autoencoder în două coloane
5. Vezi și ground truth (test set) dacă există pentru user

**Caracteristici:**
- Interfață web modernă și responsive
- Comparație side-by-side ALS vs Autoencoder
- Afișează nume piesă, artist, score și rank
- Ground truth din test set (dacă disponibil)

## 📁 Structura Proiectului

```
.
├── data/                          # Dataset-uri originale
│   ├── data.csv                   # Dataset principal Spotify
│   └── ...
├── outputs/                       # Rezultate generate (CSV, JSON, NPY)
│   ├── tracks_meta.csv
│   ├── tracks_features_scaled.npy
│   ├── interactions_train.csv
│   ├── interactions_test.csv
│   ├── recs_als.csv
│   ├── recs_autoencoder.csv
│   ├── eval_results.csv
│   └── ...
├── src/                           # Scripturi Python
│   ├── prepare_data.py           # ETAPA 1
│   ├── simulate_interactions.py  # ETAPA 2
│   ├── split_train_test.py       # ETAPA 2
│   ├── train_als.py              # ETAPA 3
│   ├── train_autoencoder.py      # ETAPA 4
│   ├── evaluate_recommenders.py  # ETAPA 5
│   └── demo_cli.py               # Demo CLI
├── requirements.txt               # Dependențe Python
├── README.md                      # Acest fișier
├── REPORT.md                      # Raport academic
├── PRESENTATION_OUTLINE.md        # Outline prezentare
└── SPEAKER_NOTES.md               # Script prezentare
```

## 📊 Outputs Principale

### Fișiere CSV
- **tracks_meta.csv**: Metadata piese (id, name, artist, year, popularity)
- **interactions_train.csv / interactions_test.csv**: Interacțiuni user-track (user_id, track_id, play)
- **recs_als.csv / recs_autoencoder.csv**: Recomandări generate (user_id, track_id, score, rank)
- **eval_results.csv**: Rezultate evaluare (model, K, precision, recall, ndcg, n_users_evaluated)

### Fișiere JSON
- **als_model_info.json**: Parametri și statistici model ALS
- **autoencoder_model_info.json**: Parametri și statistici model Autoencoder
- **feature_columns.json**: Lista coloanelor de features folosite

### Fișiere NumPy
- **tracks_features_scaled.npy**: Matrice features normalizate (n_tracks × n_features)

## 📈 Metrici de Evaluare

Sistemul evaluează modelele folosind:
- **Precision@K**: Proporția de itemi relevanți în top-K
- **Recall@K**: Proporția de itemi relevanți găsiți
- **NDCG@K**: Normalized Discounted Cumulative Gain (ordinea contează)

Evaluarea se face pentru K = 5, 10, 20, pe un set de test holdout per user.

## 🔍 Note Tehnice

- **Datele**: Dataset Spotify nu are user_id real → simulăm interacțiuni bazate pe preferințe pentru artiști
- **Split**: Train/test split per user (80/20) pentru evaluare corectă
- **Features**: 11 features audio normalizate (danceability, energy, valence, acousticness, etc.)
- **Baseline**: ALS cu 64 factori, regularizare 0.01, 20 iterații
- **Autoencoder**: Encoder/Decoder cu hidden_dim=64, latent_dim=16, MSE loss, Adam optimizer