import os
import pickle
import json
import torch
import matplotlib.pyplot as plt
from plot_cmt_tours_utils import get_bks, plot_tour
import re


# === Basisverzeichnis ===
BASE_DIR = "C:\\Users\\Martin\\Documents\\Studium\\Angewandtes Wissenschaftliches Arbeiten\\attention-learn-to-route"

TOUR_DIR_ORTOOLS = os.path.join(BASE_DIR, "outputs", "cmt_runs_ortools")  # Änderung 10: tour.pkl-Verzeichnis für OR-Tools
DATA_DIR = os.path.join(BASE_DIR, "data", "cmt")
OUT_DIR = os.path.join(BASE_DIR, "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# === tour.pkl-Dateien sammeln ===


tour_files_ortools = [
    f for f in os.listdir(os.path.join(BASE_DIR, "outputs", "cmt_runs_ortools"))
    if f.endswith("_tour.pkl")
]

# === Iteration & Plotten ===

# Schleife für OR-Tools-Touren
for tour_name in tour_files_ortools:
    match = re.match(r"(cmt\d{2})_ortools_tour\.pkl", tour_name)
    if not match:
        print(f"⚠️ Name konnte nicht geparst werden (OR-Tools): {tour_name}")
        continue

    instance = match.group(1)  # Nur das erste Element extrahieren
    dump_pkl = os.path.join(TOUR_DIR_ORTOOLS, tour_name)
    inst_pkl = os.path.join(DATA_DIR, f"{instance}_sdvrp.pkl")
    meta_json = os.path.join(DATA_DIR, f"{instance}_meta.json")
    save_path = os.path.join(OUT_DIR, f"tour_{instance}_ortools.png")  # Eindeutige Kennzeichnung

    # Überprüfen, ob die benötigten Dateien existieren
    if not (os.path.exists(inst_pkl) and os.path.exists(dump_pkl)):
        print(f"⚠️ Fehlende Dateien für {instance} (OR-Tools): {inst_pkl}, {dump_pkl}")
        continue

    try:
        # Plot-Tour aufrufen
        plot_tour(
            inst_pkl=inst_pkl,
            dump_pkl=dump_pkl,
            json_meta=meta_json,
            save_path=save_path,
            show=False
        )
    except Exception as e:
        # Fehler anzeigen, aber die Schleife fortsetzen
        print(f"❌ Fehler beim Plotten der Tour für {tour_name}: {e}")
        continue

    print(f"✅ Plot für OR-Tools-Instanz {instance} gespeichert unter: {save_path}")

print("✅ Alle Tourplots wurden erstellt.")