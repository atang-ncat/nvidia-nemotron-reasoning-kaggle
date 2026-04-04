#!/bin/bash
# Package LoRA adapter for Kaggle submission
# Usage: bash scripts/package_submission.sh [adapter_dir]

set -e

ADAPTER_DIR="${1:-/scratch2/atang/competitions/nemotron-kaggle/outputs/sft_adapter/final_adapter}"
OUTPUT_ZIP="/scratch2/atang/competitions/nemotron-kaggle/outputs/submission.zip"

echo "📦 Packaging submission from: $ADAPTER_DIR"

# Verify required files exist
if [ ! -f "$ADAPTER_DIR/adapter_config.json" ]; then
    echo "❌ Error: adapter_config.json not found in $ADAPTER_DIR"
    exit 1
fi

if [ ! -f "$ADAPTER_DIR/adapter_model.safetensors" ]; then
    echo "❌ Error: adapter_model.safetensors not found in $ADAPTER_DIR"
    exit 1
fi

# Check LoRA rank <= 32
RANK=$(python3 -c "import json; print(json.load(open('$ADAPTER_DIR/adapter_config.json'))['r'])")
if [ "$RANK" -gt 32 ]; then
    echo "❌ Error: LoRA rank $RANK exceeds competition limit of 32"
    exit 1
fi
echo "  LoRA rank: $RANK ✓"

# Create zip
rm -f "$OUTPUT_ZIP"
cd "$ADAPTER_DIR"
zip -j "$OUTPUT_ZIP" adapter_config.json adapter_model.safetensors
cd -

# Report size
SIZE=$(du -h "$OUTPUT_ZIP" | cut -f1)
echo "✅ Submission created: $OUTPUT_ZIP ($SIZE)"
echo "  Upload this file to https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/submit"
