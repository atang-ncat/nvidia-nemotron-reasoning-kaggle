#!/usr/bin/env python3
"""
Phase 2: Data Curation Pipeline ("The Python Enforcer")
=======================================================
Runs all deterministic solvers against train.csv and outputs:
  - data/curated/verified.jsonl   (solver confirms train.csv answer)
  - data/curated/corrected.jsonl  (solver found a different answer)
  - data/curated/unsolvable.jsonl (solver cannot determine the rule)
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import pandas as pd
from src.data.categorizer import categorize_prompt
from src.solvers.gravity_solver import verify_gravity
from src.solvers.numeral_solver import verify_numeral
from src.solvers.unit_conv_solver import verify_unit_conversion
from src.solvers.bit_ops_solver import verify_bit_manipulation
from src.solvers.cipher_solver import verify_text_encryption
from src.solvers.algebra_solver import verify_algebra


VERIFIERS = {
    "gravity": verify_gravity,
    "numeral": verify_numeral,
    "unit_conversion": verify_unit_conversion,
    "bit_manipulation": verify_bit_manipulation,
    "text_encryption": verify_text_encryption,
    "algebra": verify_algebra,
}


def main():
    data_dir = "/scratch2/atang/competitions/nemotron-kaggle/data"
    df = pd.read_csv(os.path.join(data_dir, "raw", "train.csv"))
    df["category"] = df["prompt"].apply(categorize_prompt)

    # Output buckets
    verified = []
    corrected = []
    unsolvable = []
    unverified = []

    # Per-category stats
    stats = {}

    for _, row in df.iterrows():
        cat = row["category"]
        if cat not in stats:
            stats[cat] = {"verified": 0, "corrected": 0, "unsolvable": 0, "unverified": 0, "total": 0}
        stats[cat]["total"] += 1

        verifier = VERIFIERS.get(cat)
        if verifier is None:
            stats[cat]["unverified"] += 1
            unverified.append({
                "id": row["id"],
                "prompt": row["prompt"],
                "answer": row["answer"],
                "category": cat,
                "status": "unverified",
            })
            continue

        result = verifier(row["prompt"], row["answer"])
        status = result["status"]

        record = {
            "id": row["id"],
            "prompt": row["prompt"],
            "answer": result.get("computed") or row["answer"],
            "original_answer": row["answer"],
            "category": cat,
            "status": status,
        }

        if status == "verified":
            stats[cat]["verified"] += 1
            verified.append(record)
        elif status == "corrected":
            stats[cat]["corrected"] += 1
            corrected.append(record)
        elif status == "unverified":
            stats[cat]["unverified"] += 1
            unverified.append(record)
        else:  # unsolvable
            stats[cat]["unsolvable"] += 1
            unsolvable.append(record)

    # Save outputs
    os.makedirs(os.path.join(data_dir, "curated"), exist_ok=True)

    for name, records in [("verified", verified), ("corrected", corrected),
                          ("unsolvable", unsolvable), ("unverified", unverified)]:
        path = os.path.join(data_dir, "curated", f"{name}.jsonl")
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"  Saved {len(records):5d} rows to {name}.jsonl")

    # Print summary
    print("\n" + "=" * 70)
    print("DATA CURATION SUMMARY")
    print("=" * 70)
    print(f"\n{'Category':<20} {'Total':>6} {'Verified':>9} {'Corrected':>10} {'Unsolvable':>11} {'Unverified':>11}")
    print("-" * 70)
    for cat in sorted(stats.keys()):
        s = stats[cat]
        print(f"  {cat:<18} {s['total']:>6} {s['verified']:>9} {s['corrected']:>10} {s['unsolvable']:>11} {s['unverified']:>11}")
    print("-" * 70)
    total_v = sum(s["verified"] for s in stats.values())
    total_c = sum(s["corrected"] for s in stats.values())
    total_u = sum(s["unsolvable"] for s in stats.values())
    total_uv = sum(s["unverified"] for s in stats.values())
    total = sum(s["total"] for s in stats.values())
    print(f"  {'TOTAL':<18} {total:>6} {total_v:>9} {total_c:>10} {total_u:>11} {total_uv:>11}")

    # Print some corrected examples for inspection
    if corrected:
        print(f"\n📝 SAMPLE CORRECTED ROWS (showing first 5):")
        for r in corrected[:5]:
            print(f"  [{r['category']}] ID={r['id']}")
            print(f"    Expected: {r['original_answer']}")
            print(f"    Computed: {r['answer']}")

    # Print some unsolvable examples
    if unsolvable:
        print(f"\n❌ SAMPLE UNSOLVABLE ROWS (showing first 5):")
        for r in unsolvable[:5]:
            print(f"  [{r['category']}] ID={r['id']}")
            print(f"    Answer: {r['original_answer']}")
            print(f"    Prompt preview: {r['prompt'][:120]}...")

    print(f"\n✅ Curation complete. Usable rows for training: {total_v + total_c + total_uv} / {total}")


if __name__ == "__main__":
    main()
