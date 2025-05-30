import os
import pickle
import json
import torch
import matplotlib.pyplot as plt

def get_bks(instance_key):
    # BKS-Lookup Tabelle
    bks_values = {
        'cmt01': 524.61,
        'cmt02': 835.26,
        'cmt03': 826.14,
        'cmt04': 1028.42,
        'cmt05': 1291.29,
        'cmt06': 555.43,
        'cmt07': 909.68,
        'cmt08': 865.94,
        'cmt09': 1162.55,
        'cmt10': 1092.50,
        'cmt11': 1042.12,
        'cmt12': 819.56,
        'cmt13': 1150.90,
        'cmt14': 1001.24
    }
    return bks_values.get(instance_key.lower(), 1.0)

def plot_tour(inst_pkl, dump_pkl, json_meta=None, save_path=None, show=True):
    # === Lade Instanzdaten ===
    with open(inst_pkl, "rb") as f:
        (loc, demand, depot, capacity), = pickle.load(f)
    loc = torch.tensor(loc)
    demand = torch.tensor(demand)
    depot = torch.tensor(depot)

    # === Lade Vorhersage ===
    with open(dump_pkl, "rb") as f:
        dump = pickle.load(f)
    
    # Änderung 1: Übergeordnete Felder aus tour.pkl extrahieren
    pi = torch.tensor(dump["tour"])              # [T]
    scaled_cost = dump.get("scaled_cost", None)  # Änderung 2
    gap = dump.get("gap_to_bks", None)           # Änderung 3
    train_size = dump.get("trained_on", None)    # Änderung 4
    instance = dump.get("instance", None)        # Änderung 5
    
    
    coords = torch.cat([depot.unsqueeze(0), loc], dim=0)

    if pi[0].item() != 0:
        pi = torch.cat([torch.tensor([0]), pi, torch.tensor([0])], dim=0)

    # === Optional: Normierung rückgängig machen ===
    if json_meta:
        with open(json_meta, 'r') as f:
            meta = json.load(f)
        norm = meta.get("norm_factor", 1.0)
        coords *= norm
        bks = get_bks(instance)  # Änderung 6
    else:
        bks = 1.0  # fallback

    # === Plot Setup ===
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("SDVRP-Tour")

    # Änderung 7: Erweiterter Titel mit Kosten, BKS, GAP usw.
    # Anzahl Fahrzeuge = Anzahl Depotbesuche - 1 (außer bei leerem Prefix)
    vehicle_count = (pi == 0).sum().item() - 1 if pi[0].item() == 0 else (pi == 0).sum().item()
    title = (
    f"{instance.upper()} (Training size {train_size})\n"
    f"costs: {scaled_cost:.2f} | BKS: {bks:.2f} | GAP: {gap:.2f}% | vehicles: {vehicle_count}"
)
    ax.set_title(title)

    # Plot Kundenpunkte
    ax.scatter(coords[1:, 0], coords[1:, 1], c='blue', label='customers', s=demand * 30)
    #for i, (x, y) in enumerate(coords[1:], start=1):  # Änderung 8: Kunden-Label
    #    ax.text(x, y, str(i), fontsize=8, color='blue')
    # Plot Depot
    ax.scatter(coords[0, 0], coords[0, 1], c='red', label='depot', s=80, marker='s')
    ax.text(coords[0, 0], coords[0, 1], "D", fontsize=10, color='red', weight='bold')

    # === Tour zeichnen ===
    tour_coords = coords[pi]
    current_tour = [tour_coords[0].tolist()]
    for i in range(1, len(tour_coords)):
        current_tour.append(tour_coords[i].tolist())
        if pi[i].item() == 0 or i == len(tour_coords) - 1:  # Depot oder Ende
            current_tour = torch.tensor(current_tour)
            ax.plot(current_tour[:, 0], current_tour[:, 1], linestyle='-', linewidth=1.5)
            current_tour = [tour_coords[i].tolist()]

    ax.legend()
    ax.axis("equal")
    ax.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"📍 Tour gespeichert unter: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()

