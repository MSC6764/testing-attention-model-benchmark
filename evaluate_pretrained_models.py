import os
import subprocess
import re
import json

PROBLEMS = ['tsp', 'cvrp', 'sdvrp']
SIZES = {
    'tsp': [20, 50, 100],
    'cvrp': [20, 50, 100],
    'sdvrp': [20, 50, 100]
}
SEED = 1234

BASE_MODEL_DIR = 'pretrained'
BASE_DATA_DIR = 'data'
RESULTS_PATH = 'results.json'

results = {}

for problem in PROBLEMS:
    results[problem] = {}

    for size in SIZES[problem]:
        model_path = os.path.join(BASE_MODEL_DIR, f"{problem}_{size}", "epoch-99.pt")

        if problem == 'tsp':
            val_data = os.path.join(BASE_DATA_DIR, 'tsp', f"tsp{size}_tsp_{size}_test_seed{SEED}_seed{SEED}.pkl")
        else:
            val_data = os.path.join(BASE_DATA_DIR, 'vrp', f"vrp{size}_vrp_{size}_test_seed{SEED}_seed{SEED}.pkl")

        print(f"\n🚀 Evaluating {problem.upper()}-{size} on seed {SEED}")
        if not os.path.exists(model_path):
            print(f"⚠️ Modell nicht gefunden: {model_path}")
            continue
        if not os.path.exists(val_data):
            print(f"⚠️ Testdaten nicht gefunden: {val_data}")
            continue

        cmd = [
            "python", "run.py",
            "--eval_only",
            "--model", "attention",
            "--problem", problem,
            "--load_path", model_path,
            "--graph_size", str(size),
            "--val_dataset", val_data,
            "--no_tensorboard",
            "--no_progress_bar",
        ]

        try:
            output = subprocess.check_output(cmd, text=True)
            # print("------- RAW OUTPUT -------")
            print(output)
            # print("--------------------------")
        except subprocess.CalledProcessError as e:
            print(f"❌ Fehler beim Ausführen: {e}")
            continue

        # Ausgabe nach Tourlänge parsen
        match = re.search(
            r"Validation overall avg_cost:\s*([\d\.]+)\s*\+\-\s*([\d\.]+)",
            output
        )   
        if match:
            mean, std = map(float, match.groups())
            results[problem][str(size)] = {"mean": round(mean, 4), "std": round(std, 4)}
            print(f"✅ Ergebnis: {mean:.4f} ± {std:.4f}")
        else:
            print("⚠️ Keine Ergebniszeile erkannt!")

# Speichere als JSON
with open(RESULTS_PATH, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n📁 Ergebnisse gespeichert in: {RESULTS_PATH}")

