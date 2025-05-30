import torch
import xml.etree.ElementTree as ET
import pickle
import json
import os
from problems.vrp.problem_vrp import SDVRP  
import xml.etree.ElementTree as ET
from nets.attention_model import AttentionModel
from utils import torch_load_cpu, load_problem
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict



def parse_cmt_xml_to_instance(xml_file):
    import xml.etree.ElementTree as ET
    import torch
    import os

    tree = ET.parse(xml_file)
    root = tree.getroot()

    customer_coords = []
    demands = []
    depot_coord = None
    capacity = None

    for node in root.findall(".//node"):
        x = float(node.find("cx").text)
        y = float(node.find("cy").text)
        node_type = node.attrib.get("type")
        if node_type == "0":
            depot_coord = [x, y]
        elif node_type == "1":
            customer_coords.append([x, y])

    vehicle_node = root.find(".//vehicle_profile")
    if vehicle_node is not None:
        capacity = float(vehicle_node.find("capacity").text)

    for req in root.findall(".//request"):
        qty = float(req.find("quantity").text)
        demands.append(qty)

    assert depot_coord is not None, "Depot fehlt"
    assert capacity is not None, "Kapazität fehlt"

    all_coords = torch.tensor([depot_coord] + customer_coords, dtype=torch.float32)
    norm_factor = all_coords.max().item()
    all_coords = all_coords / norm_factor

    depot_tensor = all_coords[0]  # [2]
    loc_tensor = all_coords[1:]   # [n, 2]
    demand_tensor = torch.tensor(demands, dtype=torch.float32)
    demand_tensor = demand_tensor / capacity  # normalize to [0,1]

    instance = {
        "depot": depot_tensor,
        "loc": loc_tensor,
        "demand": demand_tensor,
        "capacity": 1.0  # demands already normalized
    }

    meta = {
        "filename": os.path.basename(xml_file),
        "num_customers": len(customer_coords),
        "capacity": capacity,
        "norm_factor": norm_factor
    }

    return instance, meta


def save_instance_for_runpy_format(instance_data, meta_data, pkl_path, json_path):
    """
    Speichert ein einzelnes Beispiel im Format: (loc, demand, depot, capacity)
    Kompatibel mit Wouter Kool's run.py und make_instance().
    """
    depot = instance_data["depot"].tolist()           # [2]
    loc = instance_data["loc"].tolist()               # [n, 2]
    demand = instance_data["demand"].tolist()         # [n]
    capacity = instance_data["capacity"]              # 1.0

    assert isinstance(loc, list) and all(len(x) == 2 for x in loc), "❌ loc hat falsches Format"
    assert isinstance(demand, list) and all(isinstance(x, float) for x in demand), "❌ demand ist keine float-Liste"
    assert isinstance(depot, list) and len(depot) == 2, "❌ depot hat falsches Format"

    data_tuple = (loc, demand, depot, capacity)

    with open(pkl_path, "wb") as f:
        pickle.dump([data_tuple], f)

    with open(json_path, "w") as f:
        json.dump(meta_data, f, indent=4)

    print(f"✅ Gespeichert (run.py-kompatibel): {pkl_path} & {json_path}")

def load_meta_json(json_path):
    with open(json_path, "r") as f:
        meta = json.load(f)
    return meta["norm_factor"], meta["capacity"]

def parse_cmt_xml_to_sdvrp(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    customer_coords = []
    demands = []

    depot_coord = None
    capacity = None

    # 📌 Alle Nodes extrahieren
    for node in root.findall(".//node"):
        x = float(node.find("cx").text)
        y = float(node.find("cy").text)
        node_type = node.attrib.get("type")

        if node_type == "0":  # Depot
            depot_coord = [x, y]
        elif node_type == "1":  # Kunde
            customer_coords.append([x, y])

    # 📌 Kapazität extrahieren
    vehicle_node = root.find(".//vehicle_profile")
    if vehicle_node is not None:
        capacity = float(vehicle_node.find("capacity").text)

    assert depot_coord is not None, "Depot konnte nicht gefunden werden"
    assert capacity is not None, "Kapazität konnte nicht gefunden werden"

    # 📌 Demands auslesen
    for req in root.findall(".//request"):
        qty = float(req.find("quantity").text)
        demands.append(qty)

    # 📌 Normierungsfaktor für Koordinaten
    all_coords = torch.tensor([depot_coord] + customer_coords, dtype=torch.float)
    norm_factor = all_coords.max().item()
    all_coords = all_coords / norm_factor

    # 📌 Normierung von demand & capacity
    demand_tensor = torch.tensor([0.0] + demands, dtype=torch.float)
    demand_norm = demand_tensor / capacity  # relative Nachfrage
    norm_capacity = 1.0  # Modell erwartet immer 1.0

    return {
        "loc": all_coords.unsqueeze(0),        # [1, n_nodes, 2]
        "demand": demand_norm.unsqueeze(0),    # [1, n_nodes]
        "depot": torch.tensor([[0.0, 0.0]]),    # [1, 2] (nach Normierung immer [0, 0])
    }, norm_factor, norm_capacity
#%%


def load_model_from_args(model_path, problem_name='cvrp'):
    # === Lade args.json ===
    args_path = os.path.join(os.path.dirname(model_path), "args.json")
    if not os.path.isfile(args_path):
        raise FileNotFoundError(f"args.json nicht gefunden unter: {args_path}")

    with open(args_path, 'r') as f:
        args = json.load(f)

    # === Lade Problem (z. B. CVRP, SDVRP) ===
    problem = load_problem(problem_name)

    # === Initialisiere Modell mit args.json ===
    model = AttentionModel(
        embedding_dim=args["embedding_dim"],
        hidden_dim=args["hidden_dim"],
        problem=problem,
        n_encode_layers=args.get("n_encode_layers", 3),
        mask_inner=args.get("mask_inner", True),
        mask_logits=args.get("mask_logits", True),
        normalization=args.get("normalization", "batch"),
        tanh_clipping=args.get("tanh_clipping", 10.0),
        checkpoint_encoder=args.get("checkpoint_encoder", False),
        shrink_size=args.get("shrink_size", None)
    )

   

    # === Lade Gewichtsdaten ===t
    state_dict = torch.load(model_path, map_location="cpu")
    print(state_dict.keys())
    if "model" in state_dict:
        state_dict = state_dict["model"]  # extrahiere tatsächliche weights
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Modell erfolgreich geladen von: {model_path}")
    return model, args, problem
#%%
# Parameter sind schon OK – aber beachte, dass du capacity jetzt wirklich brauchst!
def evaluate_multiple_runs(model, instance, problem='sdvrp', num_runs=100, method='sampling', norm_factor: float=1.0, capacity: float=1.0, graph_size: int = None):
    """
    Modell wird mehrfach mit derselben Instanz ausgeführt, Kosten werden gemessen.
    
    Args:
        model: Das trainierte VRP-Modell
        instance: Ein einzelnes Instanz-Dict (mit loc, demand, depot)
        problem: z. B. CVRP oder SDVRP
        num_runs: Anzahl der Wiederholungen
        method: 'sampling' oder 'beam_search'
        beam_size: nur für Beam Search

    Returns:
        Dictionary mit statistischen Auswertungen der Tourkosten
    """
    

    model.set_decode_type(method)  # Setze den Decodierungsmodus (Sampling oder Greedy)

    model.eval()
    invalid_runs = 0
    valid_runs = 0
    costs = []
    min_cost = float('inf')
    with torch.no_grad():    
        with tqdm(total=num_runs, desc="Evaluating") as pbar:
            while valid_runs < num_runs:
                if method == 'sampling':
                    # Sampling-Modus (z. B. durch log_probs.sample() oder model.forward)
                    cost, ll, pi = model(instance, return_pi=True, graph_size=graph_size)
                elif method == 'greedy':
                    cost, ll, pi = model(instance, return_pi=True, graph_size=graph_size)
                else:
                    raise ValueError("Unbekannte Methode: {}".format(method))

                is_valid = validate_route(
                pi=pi,
                demand=instance['demand'] * capacity,
                capacity=capacity
                )

                if not is_valid:
                    invalid_runs += 1
                    print(f"Ungültige Tour gefunden! Anzahl: {invalid_runs} \n Tour: {pi.squeeze(0).tolist()}")
                    continue  # Ungültige Tour überspringen
                print(f"Gültige Tour gefunden: {pi.squeeze(0).tolist()}")
                if min_cost > cost.item():
                    best_route = pi.squeeze(0).tolist()
                
                if problem == 'cvrp':
                    from problems.vrp.problem_vrp import CVRP
                    cost, _ = CVRP.get_costs(instance, pi)
                elif problem == 'sdvrp':
                    from problems.vrp.problem_vrp import SDVRP
                    cost, _ = SDVRP.get_costs(instance, pi)
                else:
                    raise ValueError("Unbekanntes Problem: {}".format(problem))
                costs.append(cost.item())
                valid_runs += 1
                pbar.update(1)

    costs = np.array(costs)
    return {
        'mean_cost': np.mean(costs)*norm_factor,
        'std_cost': np.std(costs)*norm_factor,
        'min_cost': np.min(costs)* norm_factor,
        'max_cost': np.max(costs)* norm_factor,
        'median_cost': np.median(costs)* norm_factor,
        'all_costs': costs*norm_factor,  # optional, falls du die Rohdaten brauchst
        'best_route': best_route,  # beste Route
        'graph_size': graph_size  # Größe des Graphen (Anzahl der Knoten)
    }

#%%
def validate_route(pi: torch.Tensor, demand: torch.Tensor, capacity: float, tolerance: float = 1e-3):
    """
    Validiert eine SDVRP-Route.

    Voraussetzungen:
    - Die Tour besteht aus mehreren Depot- und Kundenbesuchen.
    - Ein Kunde kann mehrfach besucht werden (Split Delivery erlaubt).
    - Die Kapazität darf zwischen zwei Depotbesuchen nicht überschritten werden.
    - Alle Kunden müssen vollständig beliefert werden.

    Args:
        pi (Tensor): Tour als Sequenz von Node-IDs, Shape: [1, tour_len]
                     Beispiel: [0, 1, 2, 0, 2, 3, 0]
                     Depot = 0, Kunden = 1 bis N
        demand (Tensor): Kundenbedarf (unnormalisiert), Shape: [1, num_customers]
        capacity (float): Maximale Liefermenge pro Fahrt (unnormalisiert)
        tolerance (float): Erlaubte numerische Abweichung (z.B. durch Floating-Point)

    Returns:
        bool: True → gültige Tour, False → ungültige Tour
    """

    # 🔐 Sicherheitsprüfung: Nur Batchgröße 1 erlaubt
    assert pi.size(0) == 1

    # ➗ Tensoren in Python-Listen umwandeln für einfache Verarbeitung
    pi = pi[0].tolist()            # z. B. [0, 1, 2, 0, 2, 3, 0]
   
    demand = demand[0].tolist()    # z. B. [10.0, 15.0, 20.0]
    
    num_customers = len(demand)
    print(f"Demand: {demand}")
    print(f"Tour: {pi}")
    print(f'Demand-Länge: {len(demand)}')
    print(f'Tour-Länge: {len(pi)}')
    print(f"Kapazität: {capacity}")
    # 🧾 Restnachfrage (pro Kunde), wird während der Route reduziert
    remaining_demand = demand.copy()

    # 🧮 Wie viel wurde an jeden Kunden tatsächlich geliefert (für Debugging)
    visits = [0.0] * (num_customers)  # Index 0 = Depot, 1–n = Kunden

    # 🚚 Aktuelle Fahrzeugladung (beginnt leer)
    load = 0.0

    # 🚀 Hauptschleife über die Tour
    for node in pi:
        if node == 0:
            # ⛽ Bei Depotbesuch wird das Fahrzeug "neu beladen"
            load = 0.0
            continue

        # 🔢 Kunden-Index ermitteln (0-basiert)
        cust_idx = node - 1

        # 📦 Wieviel kann geliefert werden? (so viel wie noch gebraucht wird ODER Kapazitätsrest)
        deliverable = min(remaining_demand[cust_idx], capacity - load)

        # ❌ Fehler: Kann nicht negativ liefern
        if deliverable < -tolerance:
            print(f"❌ Ungültige Tour: Kunde {node} kann nicht beliefert werden!")
            return False

        # ✅ Lieferung ausführen
        remaining_demand[cust_idx] -= deliverable  # Bedarf sinkt
        visits[cust_idx] += deliverable                # Debug: wie viel geliefert
        load += deliverable                        # LKW wird voller

    # 🔍 Kapazitätsprüfung nach letzter Teilroute
    if load - capacity > tolerance:
        print(f"❌ Ungültige Tour: Kapazität überschritten!")
        return False

    # 🧾 Letzte Prüfung: Alle Kunden wurden vollständig beliefert
    for i in range(1, num_customers + 1):
        if abs(remaining_demand[i - 1]) > tolerance:
            print(f"❌ Ungültige Tour: Kunde {i} wurde nicht vollständig beliefert!")
            return False
    print(f'remaining_demand: {remaining_demand}')
    

    # 🧾 Debug-Ausgabe (optional)
    # for i, node in enumerate(pi):
    #     print(f"Node {i}: {node} - Demand: {demand[node]}")
    #     print(f"Node {i}: {node} - Visits: {visits[node]}")
    #     print(f"Node {i}: {node} - RemainingDemand: {remaining_demand[node]}")
    # ✅ Alle Prüfungen bestanden → gültige Tour
    return True


def validate_cmt_pkl_format(data_dir):
    for filename in os.listdir(data_dir):
        if not filename.endswith(".pkl"):
            continue

        full_path = os.path.join(data_dir, filename)
        try:
            with open(full_path, "rb") as f:
                data_list = pickle.load(f)

            assert isinstance(data_list, list), f"{filename}: Muss eine Liste sein"
            for instance in data_list:
                assert isinstance(instance, tuple), f"{filename}: Eintrag ist kein Tuple"
                assert len(instance) >= 4, f"{filename}: Tuple hat weniger als 4 Elemente"

                loc, demand, depot, capacity = instance[:4]
                assert isinstance(loc, list) and all(len(x) == 2 for x in loc), f"{filename}: 'loc' hat falsches Format"
                assert isinstance(demand, list) and all(isinstance(x, float) for x in demand), f"{filename}: 'demand' falsch"
                assert isinstance(depot, list) and len(depot) == 2, f"{filename}: 'depot' hat falsches Format"
                assert isinstance(capacity, (int, float)), f"{filename}: 'capacity' falsch"

        except Exception as e:
            print(f"❌ Fehler in {filename}: {e}")
        else:
            print(f"✅ {filename} ist gültig.")

def extract_outputs(pkl_path):
    """
    Extrahiert Kosten, Log-Likelihoods und Touren aus der von run.py erzeugten Pickle-Datei.
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data['cost'], data['log_likelihood'], data['pi']


