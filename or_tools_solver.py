import os
import json
import pickle
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# === 📁 Robuster Pfad
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(SCRIPT_DIR, "data", "cmt")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs", "cmt_runs_ortools")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "results_cmt_ortools.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 🔢 Benchmarkdaten inkl. Fahrzeuganzahl
CMT_INFO = {
    #'cmt01': {'size': 50, 'bks': 524.61, 'vehicles': 5},
    #'cmt02': {'size': 75, 'bks': 835.26, 'vehicles': 10},
    #'cmt03': {'size': 100, 'bks': 826.14, 'vehicles': 8},
    #'cmt04': {'size': 150, 'bks': 1028.42, 'vehicles': 12},
    'cmt05': {'size': 199, 'bks': 1291.29, 'vehicles': 17},
    #'cmt11': {'size': 120, 'bks': 1042.11, 'vehicles': 7},
    #'cmt12': {'size': 100, 'bks': 819.56, 'vehicles': 10},
    # weitere Instanzen nach Bedarf ergänzen
}

results = {}

def compute_euclidean_matrix(coords):
    """Berechnet Euklidische Distanzmatrix aus 2D-Koordinaten"""
    n = len(coords)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dx, dy = coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]
                mat[i][j] = np.sqrt(dx ** 2 + dy ** 2)
    return mat

for file in os.listdir(BASE_DATA_DIR):
    if not file.endswith(".pkl"):
        continue

    instance_key = os.path.splitext(file)[0].split("_")[0]
    if instance_key not in CMT_INFO:
        continue

    filepath = os.path.join(BASE_DATA_DIR, file)
    with open(filepath, "rb") as f:
        data = pickle.load(f)[0]

    loc, demand, depot = data[:3]
    meta_path = filepath.replace("sdvrp.pkl", "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        capacity = meta.get("capacity")
        norm_factor = meta.get("norm_factor")
    else:
        raise ValueError(f"Meta-Datei nicht gefunden: {meta_path}")

    vehicle_count = CMT_INFO[instance_key]["vehicles"]
    bks = CMT_INFO[instance_key]["bks"]

    # === 🧮 Vorbereitung
    locations = [depot] + loc
    demand = [0] + [round(d * capacity) for d in demand]
    distance_matrix = compute_euclidean_matrix(locations)

    # OR-Tools benötigt int -> Skalierung mit 1000
    def distance_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        return int(distance_matrix[from_node][to_node] * 1000)

    manager = pywrapcp.RoutingIndexManager(len(locations), vehicle_count, [0]*vehicle_count, [0]*vehicle_count)
    routing = pywrapcp.RoutingModel(manager)
    transit_cb = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

    demand_cb = routing.RegisterUnaryTransitCallback(
        lambda idx: int(demand[manager.IndexToNode(idx)])
    )
    routing.AddDimensionWithVehicleCapacity(
        demand_cb, 0, [int(capacity)] * vehicle_count, True, "Capacity"
    )

    # 🚀 Lösung konfigurieren
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.FromSeconds(10)

    print(f"\n🔍 Verarbeite: {instance_key.upper()} | Fahrzeuge: {vehicle_count} | Kunden: {len(loc)}")
    solution = routing.SolveWithParameters(search_params)

    if not solution:
        print(f"❌ Keine Lösung gefunden.")
        continue

    # ✅ Routen auslesen
    total_cost = 0
    tours = []
    tour_details = []

    for v in range(vehicle_count):
        idx = routing.Start(v)
        route = []
        tour_demand = 0
        route_cost = 0

        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            route.append(node)
            tour_demand += demand[node]
            next_idx = solution.Value(routing.NextVar(idx))
            from_node = manager.IndexToNode(idx)
            to_node = manager.IndexToNode(next_idx)
            route_cost += distance_matrix[from_node][to_node] * norm_factor
            idx = next_idx

        print(f"🚚 Fahrzeug {v}: Route={route} | Kosten={route_cost:.2f} | Nachfrage={tour_demand}")

        if len(route) > 1:
            total_cost += route_cost
            tours.append(route)
            tour_details.append({
                "vehicle": v,
                "route": route,
                "length": round(route_cost, 2),
                "total_demand": tour_demand
            })

    gap = (total_cost - bks) / bks * 100

    print(f"📦 Ergebnis: Kosten={total_cost:.2f} | BKS={bks} | GAP={gap:.2f}% | Routen={len(tours)}")
    tour = []
    for route in tours:
        tour.extend(route)
    tour.append(0)
    print(f"📍 Tour: {tour}")
    # 📝 Ergebnisse speichern
    out_data = {
        "instance": instance_key,
        "trained_on": None,  # OR-Tools ist nicht trainiert, daher None
        "loc": loc,
        "depot": depot,
        "demand": [d / capacity for d in demand[1:]],  # Rückskalierung der Nachfrage
        "tour": tour,  # Alle Routen in einer Liste
        "scaled_cost": total_cost,
        "gap_to_bks": gap
    }

    with open(os.path.join(OUTPUT_DIR, f"{instance_key}_ortools_tour.pkl"), "wb") as f:
        pickle.dump(out_data, f)
    
    # 🔚 Gesamtresultat
    with open(RESULTS_PATH, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n✅ OR-Tools Ergebnisse gespeichert in: {RESULTS_PATH}")

