import os
import subprocess
import re
import json
import pickle
import math
import torch
from cmt_utils import validate_route, extract_outputs

BASE_MODEL_DIR = 'pretrained'
BASE_DATA_DIR = 'data/cmt'
RESULTS_PATH = 'results_cmt.json'
PROBLEM = 'sdvrp'  # oder 'cvrp'
TRAINED_SIZES = [20, 50, 100]

# CMT graph_sizes & BKS-Werte (laut Christofides et al.)
CMT_INFO = {
    'cmt01': {'size': 50, 'bks': 524.61},
    'cmt02': {'size': 75, 'bks': 835.26},
    'cmt03': {'size': 100, 'bks': 826.14},
    'cmt04': {'size': 150, 'bks': 1028.42},
    'cmt05': {'size': 199, 'bks': 1291.29},
    'cmt06': {'size': 50, 'bks': 555.43},
    'cmt07': {'size': 75, 'bks': 909.68},
    'cmt08': {'size': 100, 'bks': 865.94},
    'cmt09': {'size': 150, 'bks': 1162.55},
    'cmt10': {'size': 199, 'bks': 1092.50},
    'cmt11': {'size': 120, 'bks': 1042.12},
    'cmt12': {'size': 100, 'bks': 819.56},
    'cmt13': {'size': 120, 'bks': 1150.90},
    'cmt14': {'size': 100, 'bks': 1001.24}
}

results = {}

for filename in os.listdir(BASE_DATA_DIR):
    if not filename.endswith('.pkl'):
        continue

    # Dateiname normalisieren: "cmt01_sdvrp.pkl" → "cmt01"
    instance_key = os.path.splitext(filename)[0].split("_")[0]

    if instance_key not in CMT_INFO:
        print(f"⚠️ Keine graph_size-Info für: {filename}")
        continue

    val_data = os.path.join(BASE_DATA_DIR, filename) 
    

    graph_size = CMT_INFO[instance_key]['size']
    bks = CMT_INFO[instance_key]['bks']
    results[instance_key] = {'bks': bks}

    for train_size in TRAINED_SIZES:
        model_path = os.path.join(BASE_MODEL_DIR, f"{PROBLEM}_{train_size}", "epoch-99.pt")

        print(f"\n🚀 Evaluierung von {filename} mit graph_size={graph_size}, Modell trainiert auf {train_size}")
        if not os.path.exists(model_path):
            print(f"⚠️ Modell nicht gefunden: {model_path}")
            continue
        
        # subprocess: run.py wird mit --dump_outputs ausgeführt
        out_file = f"tmp_output_{instance_key}_{train_size}.pkl"

        cmd = [
            "python", "run.py",
            "--eval_only",
            "--model", "attention",
            "--problem", PROBLEM,
            "--load_path", model_path,
            "--graph_size", str(graph_size),
            "--val_dataset", val_data,
            "--no_tensorboard",
            "--no_progress_bar",
            "--dump_outputs", out_file,
        ]

        
        try:
            print(f"\n Starte subprocess:\n{' '.join(cmd)}")  # NEU: Ausgeführter Befehl
            output = subprocess.check_output(cmd, text=True)
            print("⬅️ Rückgabe von subprocess:\n", output)      # NEU: vollständige Ausgabe
        except subprocess.CalledProcessError as e:
            print(f" Fehler beim Ausführen: {e}")
            print(f" stdout:\n{e.output}")
            continue
#------------- Ursprung ------------------
        # Parsen: "Validation overall avg_cost: 584.12 +- 4.21"
        match_mean = re.search(r"Validation overall avg_cost:\s*([0-9]+\.[0-9]+)", output)
        mean = float(match_mean.group(1)) if match_mean else None

        match_std = re.search(r"\+-\s*(nan|[0-9]+\.[0-9]+)", output)
        try:
            std = float(match_std.group(1)) if match_std else None
        except ValueError:
            std = None
        
        results[instance_key][f"trained_on_{train_size}"] = {
            "mean": round(mean, 4),
            "std": round(std, 4) if std is not None else None,
        }
        print(f" Ergebnis: {mean:.2f} ± {std:.2f}")
       

        # Extrahiere zusätzliche Daten
        # Lade Dump und führe Validierung durch
        try:
            with open(out_file, "rb") as f:
                dump = pickle.load(f)
        except Exception as e:
            print(f" Dump konnte nicht geladen werden: {e}")
            dump = None
# Lade die ursprünglichen Metadaten, um den norm_factor zu erhalten
        meta_json_path = val_data.replace('.pkl', '.json')
        if os.path.exists(meta_json_path):
            with open(meta_json_path, 'r') as f:
                meta = json.load(f)
            norm_factor = meta.get('norm_factor', 1.0)
            capacity = meta.get('capacity', 1.0)
        else:
            norm_factor = 1.0
            capacity = 1.0

        if dump:
            cost = dump['cost'][0]  # Nur 1 Instanz
            pi = torch.tensor(dump['pi'])  # [1, L]
            with open(val_data, "rb") as f:
                loc, demand_list, depot, cap = pickle.load(f)[0]
                demand = torch.tensor(demand_list).unsqueeze(0) * capacity
            valid = validate_route(pi, demand, capacity)
            results[instance_key][f"trained_on_{train_size}"]["valid_tour"] = valid
            print(f" Tour Validierung: {'Valid' if valid else 'Unvalid'}")        
       

        scaled_mean = mean * norm_factor
        scaled_std = std * norm_factor if not math.isnan(std) else None
        scaled_gap = (scaled_mean - bks) / bks * 100
        print(f" Skaliert: {scaled_mean:.2f} → GAP: {scaled_gap:.2f}%")
        results[instance_key][f"trained_on_{train_size}"] = {
            "mean": round(scaled_mean, 4),
            "std": round(scaled_std, 4) if scaled_std else None,
            "gap_to_bks": round(scaled_gap, 2),
            "normalized_mean": round(mean, 4)
        }
# Speichern
with open(RESULTS_PATH, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n📁 Ergebnisse gespeichert in: {RESULTS_PATH}")
