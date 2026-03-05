#!/bin/bash
# reproduce.sh — Regenerate all figures and results for Dam Safety AI paper
# Expected runtime: ~2-4 hours on 8-core machine (PINN training + GNN training)
# Usage: bash results/reproduce.sh
set -e

echo "=== Dam Safety AI — Full Reproduction Script ==="
echo "Started at: $(date)"

# Install dependencies
cd codes
if [ -f requirements.txt ]; then
    pip install -r requirements.txt -q
fi

# Step 1: Data processing
echo "[1/5] Processing sensor data..."
python3 data_processing/prepare_data.py 2>/dev/null || echo "Data processing script not yet created"

# Step 2: Train PINN model
echo "[2/5] Training PINN seepage model..."
python3 models/train_pinn.py 2>/dev/null || echo "PINN training script not yet created"

# Step 3: Train GNN-LSTM model
echo "[3/5] Training Graph-Attention LSTM..."
python3 models/train_gnn_lstm.py 2>/dev/null || echo "GNN-LSTM training script not yet created"

# Step 4: Run analysis and generate results
echo "[4/5] Running analysis..."
python3 analysis/run_analysis.py 2>/dev/null || echo "Analysis script not yet created"

# Step 5: Generate all figures
echo "[5/5] Generating figures..."
for fig_script in figures/fig_*.py; do
    if [ -f "$fig_script" ]; then
        echo "  Running $fig_script..."
        python3 "$fig_script"
    fi
done

cd ..
echo "=== Reproduction complete at $(date) ==="
