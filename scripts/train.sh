#!/bin/bash
# =============================================================================
# NEMOTRON LoRA SFT TRAINING LAUNCHER
# =============================================================================
# Run this from the project root:
#   cd /scratch2/atang/competitions/nemotron-kaggle
#   bash scripts/train.sh
#
# Or run in background with logging:
#   nohup bash scripts/train.sh > outputs/train.log 2>&1 &
#
# Monitor:
#   tail -f outputs/train.log
#   nvidia-smi -l 5
# =============================================================================

set -e

# Project root
PROJECT_DIR="/scratch2/atang/competitions/nemotron-kaggle"
VENV="${PROJECT_DIR}/.venv/bin"

echo "============================================================"
echo "  NEMOTRON SFT TRAINING LAUNCHER"
echo "  Time: $(date)"
echo "  GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================================"

# Show GPU memory before training
echo ""
echo "GPU Memory (before):"
nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader
echo ""

# Create output directory
mkdir -p "${PROJECT_DIR}/outputs"

# Run training
export CUDA_VISIBLE_DEVICES=0,1,2,3
exec "${VENV}/python" "${PROJECT_DIR}/src/train/sft_lora.py"
