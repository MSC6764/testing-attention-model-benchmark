#!/bin/bash

PROBLEMS=("tsp" "vrp")  # "vrp" deckt cvrp & sdvrp ab
SEED=1234
NUM_SAMPLES=10000

echo "📦 Generiere Testdaten für Attention, Learn to Route..."

# Problem-spezifische Graphgrößen
declare -A SIZES
SIZES[tsp]="20 50 100"
SIZES[vrp]="20 50 100"

for PROBLEM in "${PROBLEMS[@]}"; do
  for SIZE in ${SIZES[$PROBLEM]}; do
    NAME="${PROBLEM}_${SIZE}_test_seed${SEED}"
    echo "🔧 Erzeuge: $NAME"
    python generate_data.py \
      --problem $PROBLEM \
      --name $NAME \
      --seed $SEED \
      --graph_size $SIZE \
      --dataset_size $NUM_SAMPLES
  done
done

echo "✅ Alle Testdatensätze wurden erzeugt und im 'data/' Ordner gespeichert."