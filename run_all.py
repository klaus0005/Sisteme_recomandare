#!/usr/bin/env python3
"""
Sistem de Recomandare - Run All (Rulează toate etapele în ordine)

COMENZI PENTRU A RULA:
1. Asigură-te că ai activat venv-ul: source venv/bin/activate
2. Rulează scriptul: python src/run_all.py
   - Non-interactive: python src/run_all.py --yes
   - Force (regenerare tot): python src/run_all.py --force

DESCRIERE:
Rulează automat toate etapele proiectului în ordine:
1. Pregătire date
2. Simulare interacțiuni
3. Split train/test
4. Training ALS
5. Training Autoencoder
6. Evaluare

Opțional: poți sări etapele care au outputs deja generate.
"""

import subprocess
import sys
from pathlib import Path
import time

# Setup paths
ROOT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"
SRC_DIR = ROOT_DIR / "src"

# Scripturi în ordine
SCRIPTS = [
    {
        'name': 'ETAPA 1: Pregătire Date',
        'script': 'prepare_data.py',
        'outputs': ['tracks_meta.csv', 'tracks_features_scaled.npy', 'feature_columns.json'],
        'skip_if_exists': True
    },
    {
        'name': 'ETAPA 2a: Simulare Interacțiuni',
        'script': 'simulate_interactions.py',
        'outputs': ['interactions.csv', 'users_profile.csv'],
        'skip_if_exists': True
    },
    {
        'name': 'ETAPA 2b: Split Train/Test',
        'script': 'split_train_test.py',
        'outputs': ['interactions_train.csv', 'interactions_test.csv'],
        'skip_if_exists': True
    },
    {
        'name': 'ETAPA 3: Training ALS',
        'script': 'train_als.py',
        'outputs': ['recs_als.csv', 'als_model_info.json'],
        'skip_if_exists': True
    },
    {
        'name': 'ETAPA 4: Training Autoencoder',
        'script': 'train_autoencoder.py',
        'outputs': ['recs_autoencoder.csv', 'autoencoder_model_info.json'],
        'skip_if_exists': True
    },
    {
        'name': 'ETAPA 5: Evaluare',
        'script': 'evaluate_recommenders.py',
        'outputs': ['eval_results.csv'],
        'skip_if_exists': False  # Reevaluează întotdeauna
    }
]


def check_outputs_exist(outputs):
    """Verifică dacă toate output-urile există."""
    all_exist = True
    for output in outputs:
        if not (OUTPUT_DIR / output).exists():
            all_exist = False
            break
    return all_exist


def run_script(script_info):
    """Rulează un script și returnează True dacă a reușit."""
    script_name = script_info['name']
    script_file = script_info['script']
    script_path = SRC_DIR / script_file
    
    print("\n" + "=" * 70)
    print(script_name)
    print("=" * 70)
    
    # Verifică dacă scriptul există
    if not script_path.exists():
        print(f"✗ EROARE: Scriptul {script_file} nu există!")
        return False
    
    # Verifică dacă e force mode (regenerare forțată)
    force_mode = '--force' in sys.argv or '-f' in sys.argv
    
    # Verifică dacă poate sări (dacă outputs există deja și nu e force mode)
    if script_info.get('skip_if_exists', False) and not force_mode:
        if check_outputs_exist(script_info['outputs']):
            print(f"⏭  SKIP: Outputs există deja pentru {script_name}")
            print(f"   Outputs: {', '.join(script_info['outputs'])}")
            print(f"   Pentru a regenera, folosește --force sau rulează manual: python src/{script_file}")
            return True
    
    if force_mode and check_outputs_exist(script_info['outputs']):
        print(f"🔄 FORCE: Regenerare outputs pentru {script_name}")
    
    print(f"▶  Rulează: python src/{script_file}")
    print()
    
    start_time = time.time()
    
    try:
        # Rulează scriptul
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=ROOT_DIR,
            check=False,  # Nu aruncă excepție, verificăm manual
            timeout=1800  # 30 minute timeout
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✓ {script_name} COMPLET în {elapsed_time:.1f} secunde")
            return True
        else:
            print(f"\n✗ {script_name} FAILED (exit code {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n✗ {script_name} TIMEOUT (>30 minute)")
        return False
    except KeyboardInterrupt:
        print(f"\n\n⚠  INTERUPTAT de utilizator la {script_name}")
        return False
    except Exception as e:
        print(f"\n✗ {script_name} EXCEPTION: {e}")
        return False


def main():
    """Funcția principală."""
    print("=" * 70)
    print("SISTEM DE RECOMANDARE - RUN ALL")
    print("=" * 70)
    print(f"Root directory: {ROOT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    print("Acest script va rula toate etapele proiectului în ordine.")
    print("Etapele cu outputs existente vor fi sărite (skip).")
    print()
    
    # Verifică dacă e non-interactive mode (flag --yes sau --force)
    non_interactive = '--yes' in sys.argv or '--force' in sys.argv or '-y' in sys.argv
    force_mode = '--force' in sys.argv or '-f' in sys.argv
    
    if force_mode:
        print("⚠  FORCE MODE: Toate outputs-urile vor fi regenerate!")
        print()
    
    if not non_interactive:
        # Confirmare
        try:
            confirm = input("Continuă? (y/N): ").strip().lower()
            if confirm != 'y':
                print("Anulat.")
                return
        except (KeyboardInterrupt, EOFError):
            print("\nAnulat.")
            return
    else:
        print("Non-interactive mode: rulează automat fără confirmare")
    
    # Rulează toate scripturile
    total_start = time.time()
    results = []
    
    for i, script_info in enumerate(SCRIPTS, 1):
        print(f"\n[{i}/{len(SCRIPTS)}] {script_info['name']}")
        success = run_script(script_info)
        results.append((script_info['name'], success))
        
        if not success:
            print(f"\n⚠  EROARE la {script_info['name']}")
            print("   Procesul s-a oprit. Verifică erorile de mai sus.")
            
            # Întreabă dacă continuă (doar dacă nu e non-interactive)
            if not non_interactive:
                try:
                    continue_choice = input("\nContinuă cu următoarea etapă? (y/N): ").strip().lower()
                    if continue_choice != 'y':
                        break
                except (KeyboardInterrupt, EOFError):
                    break
            else:
                print("  Continuă automat...")
    
    total_time = time.time() - total_start
    
    # Rezumat final
    print("\n" + "=" * 70)
    print("REZUMAT FINAL")
    print("=" * 70)
    
    for name, success in results:
        status = "✓ OK" if success else "✗ FAILED"
        print(f"{status} - {name}")
    
    print(f"\nTimp total: {total_time/60:.1f} minute ({total_time:.1f} secunde)")
    
    # Verifică outputs finale
    print("\nVerificare outputs finale...")
    all_outputs = []
    for script_info in SCRIPTS:
        all_outputs.extend(script_info['outputs'])
    
    missing = []
    for output in set(all_outputs):
        if not (OUTPUT_DIR / output).exists():
            missing.append(output)
    
    if missing:
        print(f"⚠  Outputs lipsă: {', '.join(missing)}")
    else:
        print("✓ Toate outputs-urile principale există!")
    
    print("\n" + "=" * 70)
    print("GATA! Poți folosi acum:")
    print("  - Demo CLI: python src/demo_cli.py 0")
    print("  - Demo UI: uvicorn src.app:app --reload")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInteruptat de utilizator.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ EROARE FATALĂ: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
