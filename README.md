# NVIDIA Nemotron Model Reasoning Challenge

LoRA fine-tuning of **Nemotron-3-Nano-30B** for the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge) on Kaggle.

## Quick Start

### 1. Clone & Setup

```bash
git clone git@github.com:atang-ncat/nvidia-nemotron-reasoning-kaggle.git
cd nvidia-nemotron-reasoning-kaggle

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Base Model

```bash
# Download Nemotron-3-Nano-30B base model from HuggingFace
mkdir -p models
huggingface-cli download nvidia/Nemotron-3-Nano-30B-v1 --local-dir models/nemotron-base
```

### 3. Set API Keys

```bash
# NVIDIA NIM API key (for CoT generation via build.nvidia.com)
export NVIDIA_API_KEY="nvapi-YOUR_KEY_HERE"

# Optional: Google Gemini API key
export GEMINI_API_KEY="YOUR_KEY_HERE"
```

Get your NVIDIA API key from [build.nvidia.com](https://build.nvidia.com) (free tier available).

---

## Pipeline Overview

The full pipeline has 4 stages:

```
1. Data Curation → 2. CoT Generation → 3. SFT Data Formatting → 4. LoRA Training
```

### Stage 1: Data Curation

Deterministic solvers verify/correct answers from the competition training set.

```bash
python src/data/curate.py
```

**Output:** `data/curated/{verified,corrected,unverified}.jsonl`

### Stage 2: CoT (Chain-of-Thought) Generation

Generate step-by-step reasoning traces using LLM APIs. These run against the NVIDIA NIM API and can take several hours.

```bash
# Primary: Nemotron-Super 49B (best balance of speed & quality, ~89% usable)
export NVIDIA_API_KEY="nvapi-..."
nohup python -u src/data/nemotron_cot.py --category algebra > outputs/nemotron_cot.log 2>&1 &

# Secondary: DeepSeek R1 Distill Llama 8B (~86% usable)
nohup python -u src/data/deepseek8b_cot.py --category algebra > outputs/deepseek8b_cot.log 2>&1 &

# Monitor progress
tail -f outputs/nemotron_cot.log
tail -f outputs/deepseek8b_cot.log
```

**Output:** `data/nemotron_cot/algebra_cot.jsonl`, `data/deepseek8b_cot/algebra_cot.jsonl`

> **Note:** Not all LLM endpoints work for these puzzles. The special characters in cipher puzzles trigger content safety filters on larger models (Gemini Pro, GPT-OSS 120B). Nemotron-Super and DeepSeek R1 variants work best.

### Stage 3: Format SFT Training Data

Combines solver-generated CoT, LLM-generated CoT, and synthetic puzzles into the final training format.

```bash
# Generate synthetic puzzles (optional, already committed)
python src/data/augment_data.py

# Format all data into SFT format
python src/data/format_sft.py
```

**Output:** `data/sft_train.jsonl` (~10,800 examples), `data/sft_val.jsonl` (~570 examples)

### Stage 4: LoRA Training

Train a LoRA adapter (rank 32) on 4x GPUs with bf16 precision.

```bash
# Set memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Launch training (~15-20 hours on 4x RTX 6000 Ada 48GB)
nohup python src/train/sft_lora.py > outputs/train_v4.log 2>&1 &

# Monitor
tail -f outputs/train_v4.log
```

**Output:** `outputs/sft_v4/` (adapter checkpoints)

**Training config** (editable in `src/train/sft_lora.py`):
| Parameter | Value | Notes |
|---|---|---|
| LoRA rank | 32 | Max allowed by competition |
| Batch size | 6 | Per-device, adjust based on GPU memory |
| Grad accumulation | 5 | Effective batch = 30 |
| Learning rate | 2e-4 | Cosine schedule with 3% warmup |
| Epochs | 3 | |
| Max seq length | 2048 | |
| Precision | bf16 | |

### Stage 5: Evaluation

```bash
python src/inference/eval_local.py \
  --adapter outputs/sft_v4/final_adapter \
  --data data/sft_val.jsonl
```

### Stage 6: Package Submission

```bash
bash scripts/package_submission.sh outputs/sft_v4/final_adapter
```

**Output:** `outputs/submission.zip` (upload to Kaggle)

---

## Project Structure

```
├── configs/
│   └── ds_zero2.json              # DeepSpeed ZeRO-2 config (experimental)
├── data/
│   ├── curated/                   # Verified, corrected, unverified splits
│   ├── nemotron_cot/              # Nemotron-Super CoT traces (gitignored)
│   ├── deepseek_cot/              # DeepSeek R1 CoT traces (gitignored)
│   ├── deepseek8b_cot/            # DeepSeek 8B CoT traces (gitignored)
│   ├── synthetic.jsonl            # Synthetic training puzzles
│   ├── sft_train.jsonl            # Final training data (gitignored)
│   └── sft_val.jsonl              # Validation data (gitignored)
├── models/
│   └── nemotron-base/             # Base model weights (gitignored)
├── outputs/
│   └── sft_v4/                    # Trained adapter (gitignored)
├── scripts/
│   ├── train.sh                   # Training launcher
│   └── package_submission.sh      # Submission packaging
├── src/
│   ├── data/
│   │   ├── format_sft.py          # SFT data formatter with CoT generators
│   │   ├── nemotron_cot.py        # Nemotron-Super CoT via NVIDIA NIM
│   │   ├── deepseek_cot.py        # DeepSeek R1 32B CoT (endpoint down)
│   │   ├── deepseek8b_cot.py      # DeepSeek R1 Llama 8B CoT
│   │   ├── augment_data.py        # Synthetic puzzle generator
│   │   ├── curate.py              # Data verification & curation
│   │   └── categorizer.py         # Puzzle category classifier
│   ├── solvers/
│   │   ├── algebra_solver.py      # Char-level mapping solver
│   │   ├── algebra_solver_v2.py   # Operator inference solver
│   │   ├── bit_ops_solver.py      # Per-bit boolean function solver
│   │   ├── cipher_solver.py       # Caesar/substitution solver
│   │   ├── gravity_solver.py      # Gravity puzzle solver
│   │   ├── numeral_solver.py      # Numeral system solver
│   │   └── unit_conv_solver.py    # Unit conversion solver
│   ├── inference/
│   │   └── eval_local.py          # Local evaluation with vLLM
│   └── train/
│       └── sft_lora.py            # LoRA SFT training script
├── requirements.txt
└── README.md
```

## Results

| Version | Data | algebra | bit_manip | gravity | numeral | text_enc | unit_conv | Overall | LB Score |
|---------|------|---------|-----------|---------|---------|----------|-----------|---------|----------|
| v2 | Solver CoT only | 27.5% | 39.0% | 98.7% | 100% | 86.8% | 96.3% | 76.2% | 0.66 |
| v3 | + format changes | 26.2% | 37.3% | 100% | 95.2% | 84.2% | 92.7% | 73.3% | -- |
| v4 | + LLM CoT + synthetic | -- | -- | -- | -- | -- | -- | -- | -- |

## Hardware Requirements

- **Training:** 4x NVIDIA GPU with ≥48GB VRAM (tested on RTX 6000 Ada)
  - Adjust `BATCH_SIZE` in `sft_lora.py` if using different GPUs
  - batch=6 → ~43GB peak on heaviest GPU
  - batch=4 → ~39GB peak (safer for 40GB GPUs)
- **Inference:** Same GPUs or single GPU with vLLM tensor parallelism
- **CPU RAM:** ≥64GB (model loading)
- **Disk:** ~120GB (base model + training data + checkpoints)

## CoT Model Comparison

| Model | Endpoint | Usable Rate | Notes |
|---|---|---|---|
| Nemotron-Super 49B | `nvidia/llama-3.1-nemotron-super-49b-v1` | **89%** | Best overall |
| DeepSeek R1 32B | `deepseek-ai/deepseek-r1-distill-qwen-32b` | **94%** | Endpoint currently down (502) |
| DeepSeek R1 8B | `deepseek-ai/deepseek-r1-distill-llama-8b` | **86%** | Good alternative |
| GPT-OSS 120B | `openai/gpt-oss-120b` | 0% | Content filtered |
| Gemini 2.5/3.1 Pro | Google API | 0% | Content filtered |
| Nemotron Ultra 253B | `nvidia/llama-3.1-nemotron-ultra-253b-v1` | 20% | Mostly filtered |

## Competition

- **Kaggle:** [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)
- **Midpoint cutoff:** April 9, 2026
- **Constraints:** LoRA rank ≤ 32, base model = Nemotron-3-Nano-30B
