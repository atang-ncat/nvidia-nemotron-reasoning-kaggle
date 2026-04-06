#!/usr/bin/env python3
"""
Generate Chain-of-Thought reasoning traces using NVIDIA NIM API (Qwen3.5 80B Thinking).

Qwen3-next-80B-A3B-Thinking is a hybrid reasoning MoE model.
It outputs <think>...</think> blocks for internal reasoning, then a clean response.
We capture the full response (thinking + answer) as the CoT trace.

Usage:
  export NVIDIA_API_KEY=nvapi-...
  python src/data/qwen_cot.py [--category algebra] [--limit 100]
"""

import os
import sys
import json
import time
import re
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# NVIDIA NIM API config
NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "qwen/qwen3-next-80b-a3b-thinking"

DATA_DIR = "/scratch2/atang/competitions/nemotron-kaggle/data"
OUTPUT_DIR = "/scratch2/atang/competitions/nemotron-kaggle/data/qwen_cot"

# Rate limiting - conservative for a large model
MAX_RPM = 20
SLEEP_BETWEEN_REQUESTS = 60.0 / MAX_RPM  # ~3 seconds


def get_client():
    """Create OpenAI-compatible client for NVIDIA NIM."""
    from openai import OpenAI
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("Set NVIDIA_API_KEY environment variable")
    return OpenAI(base_url=NVIDIA_API_BASE, api_key=api_key)


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from response, keeping the rest."""
    # Remove think blocks (may be multiline)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def build_cot_prompt(puzzle_prompt: str, answer: str, category: str) -> str:
    """Build the prompt that asks the model to explain the reasoning."""
    return f"""You are an expert puzzle solver. Given the following puzzle and its correct answer, explain step-by-step how to arrive at the answer. Show your reasoning clearly.

PUZZLE:
{puzzle_prompt}

CORRECT ANSWER: {answer}

INSTRUCTIONS:
1. Analyze the examples given in the puzzle carefully
2. Identify the pattern or rule being used
3. Show how the rule applies to each example to verify it
4. Apply the rule to the query to get the answer
5. End your response with \\boxed{{{answer}}}

Think step by step:"""


def extract_and_validate(response_text: str, expected_answer: str) -> bool:
    """Check if the response contains the correct boxed answer."""
    # Find all boxed answers
    matches = re.findall(r"\\boxed\{([^}]+)\}", response_text)
    if not matches:
        return False
    # Check if any match the expected answer
    return any(m.strip() == expected_answer.strip() for m in matches)


def generate_cot_for_puzzle(client, puzzle_prompt: str, answer: str, category: str,
                             max_retries: int = 2) -> dict:
    """Generate CoT for a single puzzle. Returns dict with status and trace."""
    prompt = build_cot_prompt(puzzle_prompt, answer, category)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=NVIDIA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3 if attempt == 0 else 0.6,
                max_tokens=2048,  # larger for thinking model
                top_p=0.9,
            )
            content = response.choices[0].message.content
            if not content:
                continue
            raw_text = content.strip()

            # For thinking models, strip <think> tags to get the clean response
            # but keep the full text as the CoT (the thinking IS the chain-of-thought)
            clean_text = strip_think_tags(raw_text)

            # Use the clean text (post-thinking) for validation
            # but store the full raw text as the CoT trace
            if extract_and_validate(clean_text, answer):
                return {
                    "status": "success",
                    "cot": clean_text,  # clean response with reasoning
                    "attempt": attempt + 1,
                }
            elif extract_and_validate(raw_text, answer):
                # Answer was in the think block
                return {
                    "status": "success",
                    "cot": clean_text if clean_text else raw_text,
                    "attempt": attempt + 1,
                }
            else:
                # Check if the answer appears anywhere in the response
                check_text = clean_text if clean_text else raw_text
                if answer.strip() in check_text:
                    check_text += f"\n\n\\boxed{{{answer}}}"
                    return {
                        "status": "fixed",
                        "cot": check_text,
                        "attempt": attempt + 1,
                    }

        except Exception as e:
            error_msg = str(e)
            if "rate" in error_msg.lower():
                print(f"    Rate limited, sleeping 10s...")
                time.sleep(10)
                continue
            else:
                return {"status": "error", "error": error_msg}

    # Last attempt failed validation
    last_text = ""
    try:
        last_text = clean_text if clean_text else raw_text
    except:
        pass
    return {"status": "invalid", "cot": last_text, "reason": "answer mismatch"}


def load_puzzles(category: str = None):
    """Load puzzles that need CoT generation."""
    from src.solvers.algebra_solver_v2 import solve_algebra

    puzzles = []

    # Load from unverified (algebra) and other sources
    for source in ["unverified", "verified", "corrected"]:
        path = os.path.join(DATA_DIR, "curated", f"{source}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                cat = record["category"]

                if category and cat != category:
                    continue

                # For algebra: skip if our solver can handle it
                if cat == "algebra":
                    result = solve_algebra(record["prompt"])
                    if result and result[0] == record["answer"]:
                        continue  # solver handles this one

                # For other categories: only include if we think CoT will help
                if cat not in ("algebra", "text_encryption", "bit_manipulation"):
                    continue

                puzzles.append(record)

    return puzzles


def main():
    parser = argparse.ArgumentParser(description="Generate CoT using Qwen3.5 80B Thinking")
    parser.add_argument("--category", default="algebra", help="Category to generate for")
    parser.add_argument("--limit", type=int, default=0, help="Max puzzles (0=all)")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, f"{args.category}_cot.jsonl")

    # Load existing results if resuming
    done_ids = set()
    if args.resume and os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                r = json.loads(line)
                done_ids.add(r["id"])
        print(f"📂 Resuming: {len(done_ids)} already done")

    # Load puzzles
    print(f"📦 Loading {args.category} puzzles...")
    puzzles = load_puzzles(args.category)
    puzzles = [p for p in puzzles if p["id"] not in done_ids]
    if args.limit > 0:
        puzzles = puzzles[:args.limit]
    print(f"  {len(puzzles)} puzzles to process")

    if not puzzles:
        print("✅ Nothing to do!")
        return

    # Initialize client
    client = get_client()

    # Test connection
    print(f"🔌 Testing NVIDIA NIM connection to {NVIDIA_MODEL}...")
    try:
        test = client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[{"role": "user", "content": "Say 'hello' in one word."}],
            max_tokens=50,
        )
        raw = test.choices[0].message.content.strip()
        clean = strip_think_tags(raw)
        print(f"  ✅ Connected! Response: {clean[:80]}")
    except Exception as e:
        print(f"  ❌ Connection failed: {e}")
        return

    # Process puzzles
    stats = {"success": 0, "fixed": 0, "invalid": 0, "error": 0}
    start_time = time.time()

    with open(output_file, "a") as f:
        for i, puzzle in enumerate(puzzles):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            eta = (len(puzzles) - i) / rate if rate > 0 else 0

            print(f"  [{i+1}/{len(puzzles)}] {puzzle['id'][:8]}... "
                  f"({rate:.0f}/min, ETA: {eta:.0f}min) ", end="", flush=True)

            result = generate_cot_for_puzzle(
                client, puzzle["prompt"], puzzle["answer"], puzzle["category"]
            )

            status = result["status"]
            stats[status] = stats.get(status, 0) + 1
            print(f"→ {status}")

            # Save result
            output_record = {
                "id": puzzle["id"],
                "category": puzzle["category"],
                "prompt": puzzle["prompt"],
                "answer": puzzle["answer"],
                "cot_status": status,
                "cot": result.get("cot", ""),
            }
            f.write(json.dumps(output_record) + "\n")
            f.flush()

            # Rate limiting
            time.sleep(SLEEP_BETWEEN_REQUESTS)

    elapsed = time.time() - start_time
    print(f"\n✅ Done in {elapsed/60:.1f} minutes!")
    print(f"  Success: {stats.get('success', 0)}")
    print(f"  Fixed:   {stats.get('fixed', 0)}")
    print(f"  Invalid: {stats.get('invalid', 0)}")
    print(f"  Error:   {stats.get('error', 0)}")
    print(f"  Output:  {output_file}")


if __name__ == "__main__":
    main()
