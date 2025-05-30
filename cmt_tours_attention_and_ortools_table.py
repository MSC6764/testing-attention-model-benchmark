import os
import pickle
import json
import re
from collections import defaultdict

# === 🔢 Benchmarkdaten inkl. Fahrzeuganzahl
CMT_INFO = {
    'cmt01': {'size': 50, 'bks': 524.61, 'vehicles': 5},
    'cmt02': {'size': 75, 'bks': 835.26, 'vehicles': 10},
    'cmt03': {'size': 100, 'bks': 826.14, 'vehicles': 8},
    'cmt04': {'size': 150, 'bks': 1028.42, 'vehicles': 12},
    'cmt05': {'size': 199, 'bks': 1291.29, 'vehicles': 16},
    'cmt11': {'size': 120, 'bks': 1042.11, 'vehicles': 7},
    'cmt12': {'size': 100, 'bks': 819.56, 'vehicles': 10},
}

# === Basisverzeichnisse ===
BASE_DIR = "C:\\Users\\Martin\\Documents\\Studium\\Angewandtes Wissenschaftliches Arbeiten\\attention-learn-to-route"
AM_DIR = os.path.join(BASE_DIR, "outputs", "cmt_runs")
ORTOOLS_DIR = os.path.join(BASE_DIR, "outputs", "cmt_runs_ortools")
OUT_DIR = os.path.join(BASE_DIR, "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# === Datenspeicher initialisieren ===
results = defaultdict(lambda: defaultdict(list))

# === Dateien aus Attention-Modell laden ===
for fname in os.listdir(AM_DIR):
    if not fname.endswith("_tour.pkl"):
        continue

    match = re.match(r"(cmt\d{2})_(\d+)_(cvrp|sdvrp)_run1_tour\.pkl", fname)
    if not match:
        print(f"⚠️ Ungültiger Dateiname: {fname}")
        continue

    instance, train_size, problem = match.groups()
    path = os.path.join(AM_DIR, fname)

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)

        if "scaled_cost" not in data or "gap_to_bks" not in data:
            print(f"⚠️ Datei ohne erforderliche Felder (scaled_cost, gap_to_bks) übersprungen: {fname}")
            continue

        results[instance][f"{problem}_train{train_size}"].append({
            "scaled_cost": data["scaled_cost"],
            "gap_to_bks": data["gap_to_bks"]
        })

    except Exception as e:
        print(f"❌ Fehler beim Laden von {fname}: {e}")

# === Dateien aus OR-Tools laden ===
for fname in os.listdir(ORTOOLS_DIR):
    if not fname.endswith("_ortools_tour.pkl"):
        continue

    match = re.match(r"(cmt\d{2})_ortools_tour\.pkl", fname)
    if not match:
        print(f"⚠️ Ungültiger Dateiname: {fname}")
        continue

    instance = match.group(1)
    path = os.path.join(ORTOOLS_DIR, fname)

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)

        results[instance]["ortools"].append({
          
            "scaled_cost": data["scaled_cost"],
            "gap_to_bks": data["gap_to_bks"]
        })

    except Exception as e:
        print(f"❌ Fehler beim Laden von {fname}: {e}")

# === Sortieren nach Instanznamen und Methoden ===
sorted_results = {}
for instance in sorted(results.keys()):
    sorted_results[instance] = {}
    for method in sorted(results[instance].keys()):
        sorted_results[instance][method] = sorted(results[instance][method])

# === Speichern ===
output_path = os.path.join(OUT_DIR, "cmt_summary.json")
with open(output_path, "w") as f:
    json.dump(sorted_results, f, indent=2)

print(f"✅ Zusammenfassung gespeichert in: {output_path}")