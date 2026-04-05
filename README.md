# NVIDIA Nemotron Model Reasoning Challenge

LoRA fine-tuning of Nemotron-3-Nano-30B for the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge) on Kaggle.

## Overview

The competition requires fine-tuning a LoRA adapter (rank <= 32) on top of the Nemotron-3-Nano-30B base model to solve reasoning puzzles across six categories: algebra, bit manipulation, gravity, numeral systems, text encryption, and unit conversion.

Our approach combines deterministic solvers for ground truth verification, solver-guided Chain-of-Thought (CoT) reasoning, LLM-generated CoT (via NVIDIA NIM), and synthetic data augmentation.

## Project Structure

```
src/
  data/
    format_sft.py        # Main SFT data formatter with CoT generators
    nemotron_cot.py       # LLM-based CoT generation via NVIDIA NIM API
    augment_data.py       # Synthetic puzzle generator for weak categories
  solvers/
    algebra_solver.py     # Algebra solver v1 (char-level mapping)
    algebra_solver_v2.py  # Algebra solver v2 (operator inference)
    bit_ops_solver.py     # Per-bit boolean function inference
    cipher_solver.py      # Caesar shift and substitution table solver
  train/
    sft_lora.py           # LoRA SFT training script
  inference/
    eval_local.py         # Local evaluation with vLLM
scripts/
  train.sh                # Training launcher
  package_submission.sh   # Submission packaging
data/
  curated/                # Verified, corrected, and unverified splits
  nemotron_cot/           # LLM-generated CoT traces
  synthetic.jsonl         # Synthetic training puzzles
  sft_train.jsonl         # Final training data
  sft_val.jsonl           # Validation data
outputs/
  sft_v2/                 # v2 adapter (baseline)
  sft_v3/                 # v3 adapter (CoT format changes)
  submission.zip          # Packaged submission
```

## Pipeline

1. **Data verification** -- Deterministic solvers verify and correct answers from the training set. Categories with programmatic solutions (gravity, numeral, unit conversion, bit manipulation, text encryption) are verified against solver output.

2. **CoT generation** -- Each training example is paired with a step-by-step reasoning trace:
   - Solver-guided CoT for categories with working solvers
   - LLM-generated CoT (Nemotron-Super via NVIDIA NIM) for algebra puzzles the solver cannot handle

3. **Synthetic augmentation** -- Additional training puzzles are generated programmatically for weak categories (bit manipulation, algebra) with guaranteed-correct answers.

4. **LoRA SFT** -- Standard supervised fine-tuning with rank-32 LoRA on 4x RTX 6000 Ada (48GB each), bf16 precision.

5. **Evaluation** -- Local evaluation with vLLM (tensor parallelism 4), extracting answers from `\boxed{}` format.

## Training

```bash
# Generate SFT data
python src/data/format_sft.py

# Train
nohup bash scripts/train.sh > outputs/train.log 2>&1 &

# Evaluate
python src/inference/eval_local.py \
  --adapter outputs/sft_v3/final_adapter \
  --data data/sft_val.jsonl

# Package submission
bash scripts/package_submission.sh outputs/sft_v3/final_adapter
```

## Results

| Version | algebra | bit_manip | gravity | numeral | text_enc | unit_conv | Overall | LB Score |
|---------|---------|-----------|---------|---------|----------|-----------|---------|----------|
| v2      | 27.5%   | 39.0%     | 98.7%   | 100%    | 86.8%    | 96.3%     | 76.2%   | 0.66     |
| v3      | 26.2%   | 37.3%     | 100%    | 95.2%   | 84.2%    | 92.7%     | 73.3%   | --       |

## Hardware

- 4x NVIDIA RTX 6000 Ada (48GB each)
- Training time: ~8 hours per run

## Dependencies

- transformers, peft, trl, datasets
- vllm (inference)
- openai (NVIDIA NIM API client)
