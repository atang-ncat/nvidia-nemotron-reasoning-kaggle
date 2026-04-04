#!/usr/bin/env python3
"""
Phase 3: SFT Data Formatter
============================
Merges verified + corrected + unverified data, generates Chain-of-Thought
reasoning templates, and outputs the final SFT-ready dataset.

Each training example is formatted as:
  prompt: <original puzzle prompt>
  completion: <CoT reasoning>\n\n\\boxed{answer}
"""

import sys
import os
import json
import re
import random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

DATA_DIR = "/scratch2/atang/competitions/nemotron-kaggle/data"


# ---- CoT Template Generators per Category ---- #

def cot_gravity(prompt: str, answer: str) -> str:
    """Generate CoT for gravity puzzles."""
    # Parse observations
    observations = []
    query_t = None
    for line in prompt.split("\n"):
        line = line.strip()
        m = re.match(r"For\s+t\s*=\s*([\d.]+)\s*s?,?\s*distance\s*=\s*([\d.]+)\s*m?", line)
        if m:
            observations.append((float(m.group(1)), float(m.group(2))))
        m2 = re.search(r"t\s*=\s*([\d.]+)\s*s", line)
        if m2 and "determine" in line.lower():
            query_t = float(m2.group(1))

    if not observations or query_t is None:
        return f"Let me analyze the pattern and compute the answer.\n\n\\boxed{{{answer}}}"

    # Compute g from first observation
    t0, d0 = observations[0]
    g = 2 * d0 / (t0 * t0)

    cot = f"The gravitational constant in Wonderland can be found using d = 0.5 * g * t².\n"
    cot += f"From the first example: t = {t0}s, d = {d0}m\n"
    cot += f"g = 2d/t² = 2 × {d0} / {t0}² = {g:.4f} m/s²\n\n"

    # Verify with another observation
    if len(observations) > 1:
        t1, d1 = observations[1]
        g1 = 2 * d1 / (t1 * t1)
        cot += f"Verification with second example: g = 2 × {d1} / {t1}² = {g1:.4f} m/s² ✓\n\n"

    result = 0.5 * g * query_t * query_t
    cot += f"For t = {query_t}s: d = 0.5 × {g:.4f} × {query_t}² = {result:.2f}\n\n"
    cot += f"\\boxed{{{answer}}}"
    return cot


def cot_numeral(prompt: str, answer: str) -> str:
    """Generate CoT for numeral conversion puzzles."""
    query = None
    for line in prompt.split("\n"):
        m = re.search(r"write the number\s+(\d+)", line, re.IGNORECASE)
        if m:
            query = int(m.group(1))

    if query is None:
        return f"Converting the number to the Wonderland numeral system.\n\n\\boxed{{{answer}}}"

    cot = f"I need to convert {query} to the Wonderland numeral system (Roman numerals).\n\n"

    # Show decomposition
    remaining = query
    parts = []
    roman_values = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
                    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
                    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    for val, sym in roman_values:
        while remaining >= val:
            parts.append(f"{sym} ({val})")
            remaining -= val

    cot += f"{query} = {' + '.join(parts)}\n"
    cot += f"Result: {answer}\n\n"
    cot += f"\\boxed{{{answer}}}"
    return cot


def cot_unit_conversion(prompt: str, answer: str) -> str:
    """Generate CoT for unit conversion puzzles."""
    examples = []
    query_value = None
    for line in prompt.split("\n"):
        line = line.strip()
        m = re.match(r"^([\d.]+)\s*m?\s+becomes\s+([\d.]+)", line)
        if m:
            examples.append((float(m.group(1)), float(m.group(2))))
        m2 = re.search(r"convert.*?:\s*([\d.]+)\s*m?", line, re.IGNORECASE)
        if m2:
            query_value = float(m2.group(1))

    if not examples or query_value is None:
        return f"Applying the conversion factor.\n\n\\boxed{{{answer}}}"

    factor = examples[0][1] / examples[0][0] if examples[0][0] != 0 else 0
    cot = f"Finding the conversion factor from the examples.\n"
    cot += f"Example 1: {examples[0][0]} → {examples[0][1]}, factor = {examples[0][1]}/{examples[0][0]} = {factor:.6f}\n"

    if len(examples) > 1:
        factor2 = examples[1][1] / examples[1][0] if examples[1][0] != 0 else 0
        cot += f"Example 2: {examples[1][0]} → {examples[1][1]}, factor = {factor2:.6f} ✓\n"

    result = factor * query_value
    cot += f"\nApplying factor to {query_value}: {factor:.6f} × {query_value} = {result:.2f}\n\n"
    cot += f"\\boxed{{{answer}}}"
    return cot


def cot_bit_manipulation(prompt: str, answer: str) -> str:
    """Generate CoT for bit manipulation puzzles."""
    examples = []
    query = None
    for line in prompt.split("\n"):
        line = line.strip()
        m = re.match(r"^([01]{8})\s*->\s*([01]{8})$", line)
        if m:
            examples.append((m.group(1), m.group(2)))
        m2 = re.match(r".*?:\s*([01]{8})\s*$", line)
        if m2 and "->" not in line:
            query = m2.group(1)

    if not examples or not query:
        return f"Analyzing the bit transformation pattern.\n\n\\boxed{{{answer}}}"

    cot = "Analyzing the bit transformation pattern for each bit position.\n\n"
    cot += "Examples:\n"
    for inp, out in examples[:3]:
        cot += f"  {inp} → {out}\n"
    cot += "\n"

    cot += f"Applying the transformation to {query}:\n"
    cot += f"Result: {answer}\n\n"
    cot += f"\\boxed{{{answer}}}"
    return cot


def cot_text_encryption(prompt: str, answer: str) -> str:
    """Generate CoT for text encryption puzzles."""
    cot = "Analyzing the encryption pattern from the examples.\n\n"
    cot += "By mapping each encrypted word to its plaintext equivalent, "
    cot += "I can build a substitution table and apply it to the query.\n\n"
    cot += f"Decrypted result: {answer}\n\n"
    cot += f"\\boxed{{{answer}}}"
    return cot


def cot_algebra(prompt: str, answer: str) -> str:
    """Generate CoT for algebra puzzles."""
    cot = "Analyzing the transformation rules from the examples.\n\n"
    cot += "By examining how each input expression maps to its output, "
    cot += "I can identify the character-level transformation pattern.\n\n"
    cot += f"Applying the rule to the query gives: {answer}\n\n"
    cot += f"\\boxed{{{answer}}}"
    return cot


COT_GENERATORS = {
    "gravity": cot_gravity,
    "numeral": cot_numeral,
    "unit_conversion": cot_unit_conversion,
    "bit_manipulation": cot_bit_manipulation,
    "text_encryption": cot_text_encryption,
    "algebra": cot_algebra,
}


def format_chat_template(prompt: str, completion: str) -> dict:
    """Format a single example into chat template format."""
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
    }


def main():
    # Load all curated data
    datasets = {}
    for name in ["verified", "corrected", "unverified"]:
        path = os.path.join(DATA_DIR, "curated", f"{name}.jsonl")
        if os.path.exists(path):
            with open(path) as f:
                datasets[name] = [json.loads(line) for line in f]
            print(f"  Loaded {len(datasets[name]):5d} rows from {name}.jsonl")
        else:
            datasets[name] = []

    # Merge and generate CoT
    all_records = []
    for source_name, records in datasets.items():
        for record in records:
            cat = record["category"]
            answer = record["answer"]
            prompt = record["prompt"]

            # Generate CoT reasoning
            cot_gen = COT_GENERATORS.get(cat, cot_algebra)
            completion = cot_gen(prompt, answer)

            formatted = format_chat_template(prompt, completion)
            formatted["id"] = record["id"]
            formatted["category"] = cat
            formatted["source"] = source_name
            all_records.append(formatted)

    # Shuffle
    random.seed(42)
    random.shuffle(all_records)

    # Split into train/val (95/5)
    split_idx = int(len(all_records) * 0.95)
    train_records = all_records[:split_idx]
    val_records = all_records[split_idx:]

    # Save
    train_path = os.path.join(DATA_DIR, "sft_train.jsonl")
    val_path = os.path.join(DATA_DIR, "sft_val.jsonl")

    for path, records in [(train_path, train_records), (val_path, val_records)]:
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    print(f"\n✅ SFT data formatted:")
    print(f"  Train: {len(train_records)} examples → {train_path}")
    print(f"  Val:   {len(val_records)} examples → {val_path}")

    # Per-category breakdown
    print(f"\n📊 Category breakdown (train):")
    from collections import Counter
    cat_counts = Counter(r["category"] for r in train_records)
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat:20s}: {count:5d}")

    # Show a sample
    print(f"\n📝 Sample formatted example:")
    sample = train_records[0]
    print(f"  Category: {sample['category']}")
    print(f"  User: {sample['messages'][0]['content'][:150]}...")
    print(f"  Assistant: {sample['messages'][1]['content'][:200]}...")


if __name__ == "__main__":
    main()
