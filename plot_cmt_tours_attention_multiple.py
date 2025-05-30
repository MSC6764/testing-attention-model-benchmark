import os
import pickle
import json
import torch
import matplotlib.pyplot as plt
from plot_cmt_tours_utils import get_bks, plot_tour
import re


# === Basisverzeichnis ===
BASE_DIR = "C:\\Users\\Martin\\Documents\\Studium\\Angewandtes Wissenschaftliches Arbeiten\\attention-learn-to-route"
TOUR_DIR = os.path.join(BASE_DIR, "outputs", "cmt_runs")  # Änderung 9: tour.pkl-Verzeichnis

DATA_DIR = os.path.join(BASE_DIR, "data", "cmt")
OUT_DIR = os.path.join(BASE_DIR, "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

problem = ["cvrp", "sdvrp"]
runs = 1
tour_files = []
# === tour.pkl-Dateien sammeln ===
for prob in problem:
    for run in range(1, runs + 1):
        for f in os.listdir(TOUR_DIR):
            if f.endswith(f"_{prob}_run{run}_tour.pkl"):
                tour_files.append(f)
            if not tour_files:
                print(f"⚠️ Keine Tour-Dateien für {prob} Run {run} gefunden.")
                continue

print(f"🔍 Gefundene Tour-Dateien für {prob} Run {run}: {tour_files}")


# === Iteration & Plotten ===
for tour_name in tour_files:
    match = re.match(r"(cmt\d{2})_(\d+)_(cvrp|sdvrp)_run(\d+)_tour\.pkl", tour_name)
    if not match:
        print(f"⚠️ Name konnte nicht geparst werden: {tour_name}")
        continue

    instance, train_size, problem, run= match.groups()
    dump_pkl = os.path.join(TOUR_DIR, tour_name)
    inst_pkl = os.path.join(DATA_DIR, f"{instance}_sdvrp.pkl")
    meta_json = os.path.join(DATA_DIR, f"{instance}_meta.json")
    save_path = os.path.join(OUT_DIR, f"tour_{instance}_{train_size}_{problem}_{run}.png")

    if not (os.path.exists(inst_pkl) and os.path.exists(meta_json) and os.path.exists(dump_pkl)):
        print(f"⚠️ Fehlende Dateien für {instance}: {inst_pkl}, {meta_json}")
        continue

    if os.path.exists(save_path):
        print(f"📊 Overwriting existing plot for {tour_name} → {save_path}")
    else:
        print(f"📊 Creating new plot for {tour_name} → {save_path}")
    plot_tour(
        inst_pkl=inst_pkl,
        dump_pkl=dump_pkl,
        json_meta=meta_json,
        save_path=save_path,
        show=False
    )
    


print("✅ Alle Tourplots wurden erstellt.")