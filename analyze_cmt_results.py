import os
import json
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_FILE = "results_cmt.json"
SAVE_DIR = "analysis"
os.makedirs(SAVE_DIR, exist_ok=True)

# === Laden der Ergebnisse ===
with open(RESULTS_FILE, 'r') as f:
    results = json.load(f)

# === Umwandeln in DataFrame für Analyse ===
records = []
for instance, info in results.items():
    bks = info['bks']
    for key, val in info.items():
        if key.startswith("trained_on_"):
            train_size = int(key.split("_")[-1])
            record = {
                "instance": instance,
                "bks": bks,
                "train_size": train_size,
                "scaled_mean": val.get("mean"),
                "normalized_mean": val.get("normalized_mean"),
                "std": val.get("std"),
                "gap_to_bks": val.get("gap_to_bks"),
                "valid_tour": val.get("valid_tour", True)
            }
            records.append(record)

df = pd.DataFrame(records)
df.to_csv(os.path.join(SAVE_DIR, "cmt_results_summary.csv"), index=False)
print("📄 Tabelle gespeichert unter:", os.path.join(SAVE_DIR, "cmt_results_summary.csv"))

# === Plot 1: Skaliertes Ergebnis vs. BKS ===
plt.figure(figsize=(12, 6))
for train_size in sorted(df['train_size'].unique()):
    subset = df[df['train_size'] == train_size]
    plt.bar(
        [f"{row['instance']}" for _, row in subset.iterrows()],
        subset["scaled_mean"],
        label=f"Trained on {train_size}",
        alpha=0.6
    )
plt.plot(df["instance"].unique(), [results[k]['bks'] for k in df["instance"].unique()], label="BKS", linestyle='--', color='black')
plt.title("Skalierte Kosten vs. BKS")
plt.ylabel("Kosten")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "scaled_vs_bks.png"))
print("📊 Plot gespeichert: scaled_vs_bks.png")
plt.close()

# === Plot 2: GAP zu BKS ===
plt.figure(figsize=(12, 6))
for train_size in sorted(df['train_size'].unique()):
    subset = df[df['train_size'] == train_size]
    plt.bar(
        [f"{row['instance']}" for _, row in subset.iterrows()],
        subset["gap_to_bks"],
        label=f"Trained on {train_size}",
        alpha=0.6
    )
plt.axhline(0, color="black", linestyle="--")
plt.title("Gap zur BKS (%)")
plt.ylabel("Gap (%)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "gap_to_bks.png"))
print("📊 Plot gespeichert: gap_to_bks.png")
plt.close()

# === Plot 3: Histogramm der GAPs ===
plt.figure(figsize=(8, 5))
df['gap_to_bks'].hist(bins=20)
plt.title("Verteilung der Gaps zur BKS")
plt.xlabel("GAP (%)")
plt.ylabel("Häufigkeit")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "gap_histogram.png"))
print("📊 Plot gespeichert: gap_histogram.png")
plt.close()

# === Optional: Ungültige Touren extrahieren ===
invalids = df[df['valid_tour'] == False]
if not invalids.empty:
    invalids.to_csv(os.path.join(SAVE_DIR, "invalid_tours.csv"), index=False)
    print("⚠️ Ungültige Touren gespeichert in: invalid_tours.csv")
else:
    print("✅ Alle Touren gültig.")

print("✅ Analyse abgeschlossen.")