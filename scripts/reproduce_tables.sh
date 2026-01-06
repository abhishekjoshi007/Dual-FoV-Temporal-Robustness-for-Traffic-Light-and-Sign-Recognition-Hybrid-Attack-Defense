#!/bin/bash

set -e

echo "=========================================="
echo "Reproducing Tables from Paper"
echo "=========================================="
echo ""

CONFIG_DIR="configs/experiments"
CHECKPOINT_DIR="checkpoints"
RESULTS_DIR="results"

mkdir -p $RESULTS_DIR

echo "Step 1: Evaluating Baseline Model (Table 9)"
echo "----------------------------------------"
python scripts/evaluate.py \
    --config $CONFIG_DIR/baseline.yaml \
    --model_path $CHECKPOINT_DIR/baseline/best.pt \
    --split test \
    --eval_mode clean

echo ""
echo "Step 2: Evaluating Det-Natural Model (Table 9)"
echo "----------------------------------------"
python scripts/evaluate.py \
    --config $CONFIG_DIR/det_natural.yaml \
    --model_path $CHECKPOINT_DIR/det_natural/best.pt \
    --split test \
    --eval_mode clean

echo ""
echo "Step 3: Evaluating Unified Defense Stack (Table 9)"
echo "----------------------------------------"
python scripts/evaluate.py \
    --config $CONFIG_DIR/unified_defense.yaml \
    --model_path $CHECKPOINT_DIR/unified_defense/unified_defense.pt \
    --split test \
    --eval_mode clean

echo ""
echo "Step 4: Ablation Study (Table 10)"
echo "----------------------------------------"
python scripts/evaluate.py \
    --config $CONFIG_DIR/unified_defense.yaml \
    --model_path $CHECKPOINT_DIR/unified_defense/unified_defense.pt \
    --split test \
    --eval_mode all

echo ""
echo "Step 5: Per-Class Performance (Table 11)"
echo "----------------------------------------"
python scripts/evaluate.py \
    --config $CONFIG_DIR/unified_defense.yaml \
    --model_path $CHECKPOINT_DIR/unified_defense/unified_defense.pt \
    --split test \
    --eval_mode clean

echo ""
echo "Step 6: ODD-Specific Performance (Table 12)"
echo "----------------------------------------"
python scripts/evaluate.py \
    --config $CONFIG_DIR/unified_defense.yaml \
    --model_path $CHECKPOINT_DIR/unified_defense/unified_defense.pt \
    --split test \
    --eval_mode clean

echo ""
echo "Step 7: Natural Perturbations Evaluation (Figure 3)"
echo "----------------------------------------"
python scripts/evaluate.py \
    --config $CONFIG_DIR/unified_defense.yaml \
    --model_path $CHECKPOINT_DIR/unified_defense/unified_defense.pt \
    --split test \
    --eval_mode perturbations

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "Results saved in: $RESULTS_DIR"
echo "=========================================="