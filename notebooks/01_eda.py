#!/usr/bin/env python3
"""
Phase 1: Exploratory Data Analysis + Data Audit
================================================
Parses train.csv, categorizes rows, computes per-category statistics,
and flags suspicious/noisy examples.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import re
from collections import Counter
from src.data.categorizer import categorize_prompt


def extract_boxed_answer(answer_str: str) -> str:
    """Extract the content from \\boxed{...} if present."""
    m = re.search(r"\\boxed\{(.+?)\}", answer_str)
    if m:
        return m.group(1)
    return answer_str.strip()


def parse_bit_manipulation(prompt: str):
    """Parse bit manipulation prompt into (examples, query)."""
    lines = prompt.strip().split("\n")
    examples = []
    query = None
    for line in lines:
        line = line.strip()
        m = re.match(r"^([01]{8})\s*->\s*([01]{8})$", line)
        if m:
            examples.append((m.group(1), m.group(2)))
        m2 = re.match(r".*?:\s*([01]{8})\s*$", line)
        if m2 and not re.search(r"->", line):
            query = m2.group(1)
    return examples, query


def parse_gravity(prompt: str):
    """Parse gravity prompt into (observations, query_t)."""
    observations = []
    query_t = None
    for line in prompt.split("\n"):
        line = line.strip()
        # Match: For t = 1.37s, distance = 14.92 m
        m = re.match(
            r"For\s+t\s*=\s*([\d.]+)\s*s?,?\s*distance\s*=\s*([\d.]+)\s*m?", line
        )
        if m:
            t = float(m.group(1))
            d = float(m.group(2))
            observations.append((t, d))
        # Match query: "determine the distance ... t = X s"
        m2 = re.search(r"t\s*=\s*([\d.]+)\s*s", line)
        if m2 and "determine" in line.lower():
            query_t = float(m2.group(1))
    return observations, query_t


def parse_numeral(prompt: str):
    """Parse numeral conversion prompt into (examples, query)."""
    examples = []
    query = None
    for line in prompt.split("\n"):
        line = line.strip()
        m = re.match(r"^(\d+)\s*->\s*(.+)$", line)
        if m:
            examples.append((int(m.group(1)), m.group(2).strip()))
        m2 = re.search(r"write the number\s+(\d+)", line, re.IGNORECASE)
        if m2:
            query = int(m2.group(1))
    return examples, query


def parse_unit_conversion(prompt: str):
    """Parse unit conversion prompt into (examples, query_value)."""
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
    return examples, query_value


def analyze_bit_manipulation(df_cat: pd.DataFrame):
    """Analyze bit_manipulation rows for consistency."""
    results = {"consistent": 0, "inconsistent": 0, "parse_error": 0}
    for _, row in df_cat.iterrows():
        examples, query = parse_bit_manipulation(row["prompt"])
        if not examples or not query:
            results["parse_error"] += 1
            continue

        # For each bit position, try to determine a boolean function
        consistent = True
        for bit_pos in range(8):
            in_bits = [int(e[0][bit_pos]) for e in examples]
            out_bits = [int(e[1][bit_pos]) for e in examples]
            # Check if output bit is a simple function of all input bits
            # We just check for constant, copy, or NOT for now
            unique_out = set(out_bits)
            if len(unique_out) == 1:
                continue  # constant function - always consistent
            # Check if it's a copy or NOT of the corresponding input bit
            if in_bits == out_bits:
                continue  # copy
            if [1 - b for b in in_bits] == out_bits:
                continue  # NOT
            # More complex function - need truth table analysis
            # For now mark as potentially consistent (need full solver)
        results["consistent"] += 1

    return results


def analyze_gravity(df_cat: pd.DataFrame):
    """Analyze gravity rows for consistency of g constant."""
    results = {"consistent": 0, "inconsistent": 0, "parse_error": 0}
    g_values = []
    for _, row in df_cat.iterrows():
        obs, query_t = parse_gravity(row["prompt"])
        if not obs or query_t is None:
            results["parse_error"] += 1
            continue
        # Compute g from each observation: g = 2d / t^2
        gs = [2 * d / (t * t) for t, d in obs]
        if not gs:
            results["parse_error"] += 1
            continue

        mean_g = sum(gs) / len(gs)
        spread = max(gs) - min(gs)
        if spread < 0.5:  # all observations agree on g
            results["consistent"] += 1
            g_values.append(mean_g)
        else:
            results["inconsistent"] += 1

    if g_values:
        print(
            f"    g range: [{min(g_values):.2f}, {max(g_values):.2f}], mean={sum(g_values)/len(g_values):.2f}"
        )
    return results


def main():
    df = pd.read_csv("/scratch2/atang/competitions/nemotron-kaggle/data/raw/train.csv")

    # Categorize
    df["category"] = df["prompt"].apply(categorize_prompt)

    print("=" * 60)
    print("NVIDIA NEMOTRON CHALLENGE — EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # Category distribution
    print("\n📊 CATEGORY DISTRIBUTION")
    print("-" * 40)
    for cat, count in df["category"].value_counts().items():
        pct = 100 * count / len(df)
        print(f"  {cat:20s}: {count:5d} ({pct:.1f}%)")
    print(f"  {'TOTAL':20s}: {len(df):5d}")

    # Check for unknowns
    unknowns = df[df["category"] == "unknown"]
    if len(unknowns) > 0:
        print(f"\n⚠️  {len(unknowns)} rows could not be categorized!")
        for _, row in unknowns.head(3).iterrows():
            print(f"    ID={row['id']}: {row['prompt'][:100]}...")

    # Per-category stats
    print("\n📈 PER-CATEGORY STATISTICS")
    print("-" * 40)
    for cat in sorted(df["category"].unique()):
        subset = df[df["category"] == cat]
        avg_prompt_len = subset["prompt"].str.len().mean()
        avg_answer_len = subset["answer"].str.len().mean()
        print(f"\n  [{cat}] n={len(subset)}")
        print(f"    Avg prompt length: {avg_prompt_len:.0f} chars")
        print(f"    Avg answer length: {avg_answer_len:.1f} chars")
        print(f"    Sample answers: {subset['answer'].head(3).tolist()}")

    # Category-specific analyses
    print("\n🔍 BIT MANIPULATION ANALYSIS")
    print("-" * 40)
    bm = df[df["category"] == "bit_manipulation"]
    bm_results = analyze_bit_manipulation(bm)
    for k, v in bm_results.items():
        print(f"    {k}: {v}")

    print("\n🔍 GRAVITY ANALYSIS")
    print("-" * 40)
    gv = df[df["category"] == "gravity"]
    gv_results = analyze_gravity(gv)
    for k, v in gv_results.items():
        print(f"    {k}: {v}")

    # Prompt length distribution
    print("\n📏 PROMPT LENGTH DISTRIBUTION")
    print("-" * 40)
    for cat in sorted(df["category"].unique()):
        subset = df[df["category"] == cat]
        lengths = subset["prompt"].str.len()
        print(
            f"  {cat:20s}: min={lengths.min():5d}, median={lengths.median():7.0f}, max={lengths.max():5d}"
        )

    # Save categorized data
    out_path = "/scratch2/atang/competitions/nemotron-kaggle/data/raw/train_categorized.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✅ Saved categorized data to {out_path}")

    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
