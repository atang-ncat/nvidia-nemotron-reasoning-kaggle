#!/usr/bin/env python3
"""
Phase 4: LoRA SFT Fine-Tuning — Optimized for 4x RTX 6000 Ada (48GB each)
==========================================================================
Maximizes GPU memory utilization with:
  - bf16 across 4 GPUs (Mamba layers incompatible with bitsandbytes)
  - Batch size 8 per device with gradient accumulation 4 = effective batch 32
  - 3 full training epochs (~1,497 steps, ~15 hours at ~37s/step)
  - Gradient checkpointing to trade compute for memory
  - Max sequence length 2048

Run standalone:
  cd /scratch2/atang/competitions/nemotron-kaggle
  bash scripts/train.sh
"""

import os
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# ---- Configuration ---- #
MODEL_NAME = "/scratch2/atang/competitions/nemotron-kaggle/models/nemotron-base"
DATA_DIR = "/scratch2/atang/competitions/nemotron-kaggle/data"
OUTPUT_DIR = "/scratch2/atang/competitions/nemotron-kaggle/outputs/sft_v1"

# LoRA config (rank must be <= 32 per competition rules)
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training config — MAXIMIZED for 4x48GB GPUs
NUM_EPOCHS = 3
BATCH_SIZE = 8          # per-device batch size (was 4, doubled)
GRADIENT_ACCUMULATION_STEPS = 4   # effective batch = 8 * 4 = 32
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 2048
WARMUP_RATIO = 0.03


def messages_to_text(messages):
    """Convert chat messages to raw text for base model training."""
    parts = []
    for msg in messages:
        if msg["role"] == "user":
            parts.append("### Question:\n" + msg["content"])
        elif msg["role"] == "assistant":
            parts.append("### Answer:\n" + msg["content"])
    return "\n\n".join(parts)


def load_sft_data(split):
    """Load SFT data from JSONL and convert to text format."""
    path = os.path.join(DATA_DIR, f"sft_{split}.jsonl")
    texts = []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            text = messages_to_text(record["messages"])
            texts.append({"text": text})
    return Dataset.from_list(texts)


def main():
    print("=" * 60)
    print("NEMOTRON LoRA SFT — OPTIMIZED (bf16, 4x RTX 6000 Ada)")
    print("=" * 60)

    # 1. Tokenizer
    print("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Model in bf16
    print("[2/6] Loading model (bf16, device_map=auto across 4 GPUs)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Print GPU allocation
    print("  GPU memory allocation:")
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_mem / 1e9
        print(f"    GPU {i}: {alloc:.1f}GB / {total:.1f}GB")

    # 3. LoRA
    print("[3/6] Configuring LoRA adapter (rank=32)...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # 4. Data
    print("[4/6] Loading data...")
    train_dataset = load_sft_data("train")
    val_dataset = load_sft_data("val")
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Val:   {len(val_dataset)} examples")

    # 5. Train
    print(f"[5/6] Training: {NUM_EPOCHS} epochs, batch={BATCH_SIZE}x{GRADIENT_ACCUMULATION_STEPS}={BATCH_SIZE*GRADIENT_ACCUMULATION_STEPS}, lr={LEARNING_RATE}")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_length=MAX_SEQ_LENGTH,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    # 6. Save
    print("[6/6] Saving adapter...")
    adapter_dir = os.path.join(OUTPUT_DIR, "final_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    config_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(config_path) as f:
        adapter_config = json.load(f)
    rank = adapter_config.get("r", "unknown")
    print(f"  Rank: {rank} (must be <= 32)")
    assert int(rank) <= 32, "Rank exceeds limit!"

    print(f"\n✅ Done! Adapter: {adapter_dir}")
    print(f"   Package: bash scripts/package_submission.sh {adapter_dir}")


if __name__ == "__main__":
    main()
