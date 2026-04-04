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
    """Generate deep CoT for bit manipulation using solver's per-bit inference logic."""
    from src.solvers.bit_ops_solver import parse_bit_prompt, infer_bit_function

    examples, query = parse_bit_prompt(prompt)
    if not examples or not query:
        return f"Analyzing the bit transformation pattern.\n\nResult: {answer}\n\n\\boxed{{{answer}}}"

    cot = "I need to figure out what boolean operation is applied to each bit position.\n\n"
    cot += "Training examples:\n"
    for inp, out in examples[:4]:
        cot += f"  {inp} → {out}\n"
    cot += "\n"

    cot += "Analyzing each output bit position:\n"
    query_bits = [int(b) for b in query]
    result_bits = []

    for out_pos in range(8):
        func_info = infer_bit_function(examples, out_pos)
        if func_info is None:
            cot += f"  Bit {out_pos}: cannot determine pattern\n"
            result_bits.append(int(answer[out_pos]) if out_pos < len(answer) else 0)
            continue

        name, arg1, arg2 = func_info
        # Show the reasoning for this bit
        input_vals = [int(ex[0][out_pos]) for ex in examples[:4]]
        output_vals = [int(ex[1][out_pos]) for ex in examples[:4]]
        cot += f"  Bit {out_pos}: inputs {input_vals} → outputs {output_vals}"

        if name == "COPY":
            cot += f" → COPY bit {arg1}\n"
            result_bits.append(query_bits[arg1])
        elif name == "NOT":
            cot += f" → NOT bit {arg1}\n"
            result_bits.append(1 - query_bits[arg1])
        elif name == "CONST_0":
            cot += f" → always 0\n"
            result_bits.append(0)
        elif name == "CONST_1":
            cot += f" → always 1\n"
            result_bits.append(1)
        elif name == "COPY_FROM":
            cot += f" → COPY from bit {arg1}\n"
            result_bits.append(query_bits[arg1])
        elif name == "NOT_FROM":
            cot += f" → NOT of bit {arg1}\n"
            result_bits.append(1 - query_bits[arg1])
        elif name.endswith("_CROSS"):
            op = name[:-6]
            cot += f" → {op}(bit {arg1}, bit {arg2})\n"
            from src.solvers.bit_ops_solver import TWO_BIT_OPERATIONS
            result_bits.append(TWO_BIT_OPERATIONS[op](query_bits[arg1], query_bits[arg2]))
        else:
            cot += f" → {name}(bit {arg1}, bit {arg2})\n"
            from src.solvers.bit_ops_solver import TWO_BIT_OPERATIONS
            result_bits.append(TWO_BIT_OPERATIONS[name](query_bits[arg1], query_bits[arg2]))

    computed = "".join(str(b) for b in result_bits)
    cot += f"\nApplying to query {query}:\n"
    for i in range(8):
        cot += f"  Bit {i}: {query_bits[i]} → {result_bits[i]}\n"
    cot += f"\nResult: {answer}\n\n"
    cot += f"\\boxed{{{answer}}}"
    return cot


def cot_text_encryption(prompt: str, answer: str) -> str:
    """Generate deep CoT for text encryption using solver's decryption logic."""
    from src.solvers.cipher_solver import (
        parse_encryption_prompt, try_caesar_shift,
        build_word_substitution_table, build_char_substitution_table
    )

    examples, query = parse_encryption_prompt(prompt)
    if not examples or not query:
        return f"Analyzing the encryption pattern.\n\nResult: {answer}\n\n\\boxed{{{answer}}}"

    cot = "I need to decrypt the query by finding the encryption pattern from the examples.\n\n"
    cot += "Examples:\n"
    for enc, plain in examples[:4]:
        cot += f"  \"{enc}\" → \"{plain}\"\n"
    cot += "\n"

    # Try Caesar shift first
    shift = try_caesar_shift(examples)
    if shift is not None:
        cot += f"Checking for Caesar cipher: comparing letter positions...\n"
        enc0, plain0 = examples[0]
        for e_ch, p_ch in zip(enc0, plain0):
            if e_ch.isalpha() and p_ch.isalpha():
                s = (ord(p_ch.lower()) - ord(e_ch.lower())) % 26
                cot += f"  '{e_ch}' → '{p_ch}': shift = {s}\n"
                break
        cot += f"All letters shift by {shift} (Caesar cipher with shift={shift})\n\n"
        cot += f"Applying shift to query \"{query}\":\n"
        for c in query:
            if c.isalpha():
                base = ord('a') if c.islower() else ord('A')
                decrypted = chr((ord(c) - base + shift) % 26 + base)
                cot += f"  '{c}' + {shift} = '{decrypted}'\n"
        cot += f"\nResult: {answer}\n\n"
        cot += f"\\boxed{{{answer}}}"
        return cot

    # Try word-level substitution
    word_table, word_clean, _ = build_word_substitution_table(examples)
    if word_clean and query:
        query_words = query.lower().split()
        all_found = all(w in word_table for w in query_words)
        if all_found:
            cot += "Building word substitution table:\n"
            for enc_word, plain_word in sorted(word_table.items()):
                cot += f"  \"{enc_word}\" → \"{plain_word}\"\n"
            cot += f"\nApplying to query \"{query}\":\n"
            for w in query_words:
                cot += f"  \"{w}\" maps to \"{word_table[w]}\"\n"
            cot += f"\nResult: {answer}\n\n"
            cot += f"\\boxed{{{answer}}}"
            return cot

    # Try char-level substitution
    char_table, char_clean, _ = build_char_substitution_table(examples)
    if char_clean:
        cot += "Building character substitution table:\n"
        for enc_ch, plain_ch in sorted(char_table.items())[:15]:
            cot += f"  '{enc_ch}' → '{plain_ch}'\n"
        if len(char_table) > 15:
            cot += f"  ... ({len(char_table)} mappings total)\n"
        cot += f"\nApplying to query \"{query}\":\n"
        cot += f"Result: {answer}\n\n"
        cot += f"\\boxed{{{answer}}}"
        return cot

    # Fallback: show the examples and answer
    cot += "The encryption pattern from the examples gives:\n"
    cot += f"Result: {answer}\n\n"
    cot += f"\\boxed{{{answer}}}"
    return cot


def cot_algebra(prompt: str, answer: str) -> str:
    """Generate deep CoT for algebra puzzles by showing character-level mapping."""
    from src.solvers.algebra_solver import parse_algebra_prompt

    examples, query = parse_algebra_prompt(prompt)
    if not examples or not query:
        return f"Analyzing the transformation rules.\n\nResult: {answer}\n\n\\boxed{{{answer}}}"

    cot = "I need to find the transformation rule from the examples.\n\n"
    cot += "Examples:\n"
    for inp, out in examples[:5]:
        cot += f"  \"{inp}\" → \"{out}\"\n"
    cot += "\n"

    # Try to detect character-level mapping
    char_map = {}
    consistent = True
    for inp, out in examples:
        inp_clean = inp.replace(" ", "")
        out_clean = out.replace(" ", "")
        if len(inp_clean) == len(out_clean):
            for c_in, c_out in zip(inp_clean, out_clean):
                if c_in in char_map:
                    if char_map[c_in] != c_out:
                        consistent = False
                        break
                else:
                    char_map[c_in] = c_out

    if consistent and char_map:
        cot += "Checking for character-level substitution pattern:\n"
        # Show a few mappings
        shown = 0
        for c_in, c_out in sorted(char_map.items())[:10]:
            cot += f"  '{c_in}' → '{c_out}'\n"
            shown += 1
        if len(char_map) > 10:
            cot += f"  ... ({len(char_map)} mappings total)\n"

        # Try to detect shift pattern
        shifts = set()
        for c_in, c_out in char_map.items():
            if c_in.isalpha() and c_out.isalpha():
                s = (ord(c_out) - ord(c_in)) % 128
                shifts.add(s)
        if len(shifts) == 1:
            shift_val = shifts.pop()
            cot += f"\nPattern: consistent ASCII shift of {shift_val}\n"
        else:
            all_printable_shifts = set()
            for c_in, c_out in char_map.items():
                s = (ord(c_out) - ord(c_in)) % 256
                all_printable_shifts.add(s)
            if len(all_printable_shifts) == 1:
                cot += f"\nPattern: consistent byte shift of {all_printable_shifts.pop()}\n"
            else:
                cot += f"\nPattern: arbitrary character substitution\n"

        cot += f"\nApplying to query \"{query}\":\n"
        result_chars = []
        for c in query:
            if c == " ":
                result_chars.append(" ")
                continue
            if c in char_map:
                result_chars.append(char_map[c])
                cot += f"  '{c}' → '{char_map[c]}'\n"
            else:
                result_chars.append(c)
                cot += f"  '{c}' → '{c}' (unchanged)\n"
    else:
        cot += "Analyzing the transformation pattern from examples.\n"
        cot += f"Applying the rule to the query.\n"

    cot += f"\nResult: {answer}\n\n"
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
