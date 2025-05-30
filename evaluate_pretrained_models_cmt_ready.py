import os
import subprocess
import re
import json
import pickle
import math
import torch
from cmt_utils import validate_route

BASE_MODEL_DIR = 'pretrained'
BASE_DATA_DIR = 'data/cmt'
OUTPUT_DIR = 'outputs/cmt_runs'
JSON_DIR = 'CMT_json'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

PROBLEMS = ['cvrp', 'sdvrp']
TRAINED_SIZES = [20, 50, 100]
NUM_RUNS = 10

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

for PROBLEM in PROBLEMS:
    results = {}
    for filename in os.listdir(BASE_DATA_DIR):
        if not filename.endswith('.pkl'):
            continue

        instance_key = os.path.splitext(filename)[0].split("_")[0]
        if instance_key not in CMT_INFO:
            print(f"⚠️ Keine graph_size-Info für: {filename}")
            continue

        val_data = os.path.join(BASE_DATA_DIR, filename)
        graph_size = CMT_INFO[instance_key]['size']
        bks = CMT_INFO[instance_key]['bks']
        results[instance_key] = {'bks': bks}

        for train_size in TRAINED_SIZES:
            results[instance_key][f"trained_on_{train_size}"] = []
            for run_idx in range(NUM_RUNS):
                model_path = os.path.join(BASE_MODEL_DIR, f"{PROBLEM}_{train_size}", "epoch-99.pt")
                out_file = f"tmp_output_{instance_key}_{train_size}_{PROBLEM}_run{run_idx+1}.pkl"

                print(f"\n🚀 Evaluierung von {filename} mit graph_size={graph_size}, Modell trainiert auf {train_size}, Problem={PROBLEM}, Run={run_idx+1}")
                if not os.path.exists(model_path):
                    print(f"⚠️ Modell nicht gefunden: {model_path}")
                    continue

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
                    "--dump_outputs", out_file
                ]

                try:
                    output = subprocess.check_output(cmd, text=True)
                    print("⬅️ Rückgabe von subprocess:\n", output)
                except subprocess.CalledProcessError as e:
                    print(f"❌ Fehler beim Ausführen: {e}")
                    print(f"stdout:\n{e.output}")
                    continue

                match_mean = re.search(r"Validation overall avg_cost:\s*([0-9]+\.[0-9]+)", output)
                mean = float(match_mean.group(1)) if match_mean else None

                match_std = re.search(r"\+-\s*(nan|[0-9]+\.[0-9]+)", output)
                try:
                    std = float(match_std.group(1)) if match_std else None
                except ValueError:
                    std = None

                # Lade Outputs
                try:
                    with open(out_file, "rb") as f:
                        dump = pickle.load(f)
                except Exception as e:
                    print(f"⚠️ Dump konnte nicht geladen werden: {e}")
                    continue

                # Lade Metadaten
                meta_json_path = val_data.replace('sdvrp.pkl', 'meta.json')
                if os.path.exists(meta_json_path):
                    with open(meta_json_path, 'r') as f:
                        meta = json.load(f)
                    norm_factor = meta.get('norm_factor', 1.0)
                    capacity = meta.get('capacity', 1.0)
                else:
                    print(f"⚠️ Keine Metadaten gefunden für: {meta_json_path}")
                    norm_factor = 1.0
                    capacity = 1.0

                # Rückskalierung und Validierung
                scaled_mean = mean * norm_factor
                scaled_std = std * norm_factor if std and not math.isnan(std) else None
                scaled_gap = (scaled_mean - bks) / bks * 100

                cost = dump['cost'][0]
                pi = torch.tensor(dump['pi'])  # [1, L]
                log_likelihood = dump.get('log_likelihood', [None])[0]

                # Originaldaten wiederherstellen
                with open(val_data, "rb") as f:
                    loc, demand_list, depot, _ = pickle.load(f)[0]
                demand = torch.tensor(demand_list).unsqueeze(0) * capacity
                valid = validate_route(pi, demand, capacity)

                # Update Results
                run_result = {
                    "normalized_mean": round(mean, 4),
                    "std": round(std, 4) if std is not None else None,
                    "mean": round(scaled_mean, 4),
                    "std_scaled": round(scaled_std, 4) if scaled_std else None,
                    "gap_to_bks": round(scaled_gap, 2),
                    "valid_tour": valid,
                    "raw_cost": cost,
                    "log_likelihood": log_likelihood,
                    "tour": pi[0].tolist()
                }
                results[instance_key][f"trained_on_{train_size}"].append(run_result)

                print(f" Ergebnis: {mean:.2f} ± {std:.2f}")
                print(f" Tour Validierung: {'Valid' if valid else 'Unvalid'}")
                print(f" Skaliert: {scaled_mean:.2f} → GAP: {scaled_gap:.2f}%")

                # Optional: Save als einzelne Datei für spätere Visualisierung
                out_data = {
                    "instance": instance_key,
                    "trained_on": train_size,
                    "run": run_idx + 1,
                    "problem": PROBLEM,
                    "loc": loc,
                    "depot": depot,
                    "demand": demand_list,
                    "tour": pi[0].tolist(),
                    "scaled_cost": scaled_mean,
                    "gap_to_bks": scaled_gap
                }
                out_path = os.path.join(OUTPUT_DIR, f"{instance_key}_{train_size}_{PROBLEM}_run{run_idx+1}_tour.pkl")
                with open(out_path, "wb") as f:
                    pickle.dump(out_data, f)

    # JSON speichern
    json_path = os.path.join(JSON_DIR, f"results_cmt_{PROBLEM}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📁 Ergebnisse gespeichert in: {json_path}")