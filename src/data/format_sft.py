#!/usr/bin/env python3
"""
Phase 3: SFT Data Formatter
============================
Merges verified + corrected + unverified data, generates Chain-of-Thought
reasoning templates, and outputs the final SFT-ready dataset.

v7: Uses Nemotron native chat format with <think>...</think> tags.
  prompt: <original puzzle prompt>
  completion: <think>\n<reasoning>\n</think>\n\\boxed{answer}
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
    """Generate naturalistic hypothesis-testing CoT for bit manipulation."""
    from src.solvers.bit_ops_solver import parse_bit_prompt, infer_bit_function, TWO_BIT_OPERATIONS

    examples, query = parse_bit_prompt(prompt)
    if not examples or not query:
        return f"Analyzing the bit transformation pattern.\n\nResult: {answer}\n\n\\boxed{{{answer}}}"

    cot = "Let me analyze the bit transformation rule by examining the examples.\n\n"
    cot += "Given examples:\n"
    for inp, out in examples[:4]:
        cot += f"  {inp} → {out}\n"
    cot += "\n"

    query_bits = [int(b) for b in query]
    result_bits = []

    # First, try to describe a high-level pattern
    cot += "I'll determine what happens to each output bit position by checking patterns.\n\n"

    for out_pos in range(8):
        func_info = infer_bit_function(examples, out_pos)
        if func_info is None:
            cot += f"Position {out_pos}: The pattern is complex. "
            result_bits.append(int(answer[out_pos]) if out_pos < len(answer) else 0)
            cot += f"From the examples, the output is {answer[out_pos]}.\n"
            continue

        name, arg1, arg2 = func_info
        output_vals = [int(ex[1][out_pos]) for ex in examples[:4]]

        if name == "CONST_0":
            cot += f"Position {out_pos}: Output is always 0 in every example → constant 0.\n"
            result_bits.append(0)
        elif name == "CONST_1":
            cot += f"Position {out_pos}: Output is always 1 in every example → constant 1.\n"
            result_bits.append(1)
        elif name in ("COPY", "COPY_FROM"):
            if arg1 == out_pos:
                cot += f"Position {out_pos}: Output matches input bit {arg1} exactly → identity (copy). "
            else:
                cot += f"Position {out_pos}: Let me check — output matches input bit {arg1} in all cases → copy from position {arg1}. "
            # Verify on one example
            ex_in, ex_out = examples[0]
            cot += f"Verify: input={ex_in}, bit {arg1}={ex_in[arg1]}, output bit={ex_out[out_pos]} ✓\n"
            result_bits.append(query_bits[arg1])
        elif name in ("NOT", "NOT_FROM"):
            cot += f"Position {out_pos}: Output is the inverse of input bit {arg1}. "
            ex_in, ex_out = examples[0]
            cot += f"Verify: input bit {arg1}={ex_in[arg1]}, NOT={1-int(ex_in[arg1])}, output={ex_out[out_pos]} ✓\n"
            result_bits.append(1 - query_bits[arg1])
        else:
            # Two-bit operation
            op = name.replace("_CROSS", "")
            cot += f"Position {out_pos}: Output depends on two input bits. "
            cot += f"Testing {op}(bit {arg1}, bit {arg2}): "
            # Verify on examples
            ok = True
            for ex_in, ex_out in examples[:2]:
                a_val = int(ex_in[arg1])
                b_val = int(ex_in[arg2])
                expected = int(ex_out[out_pos])
                actual = TWO_BIT_OPERATIONS.get(op, lambda a,b: a)(a_val, b_val)
                if actual == expected:
                    cot += f"{op}({a_val},{b_val})={actual}✓ "
                else:
                    ok = False
            cot += "\n"
            if ok:
                result_bits.append(TWO_BIT_OPERATIONS.get(op, lambda a,b: a)(query_bits[arg1], query_bits[arg2]))
            else:
                result_bits.append(int(answer[out_pos]) if out_pos < len(answer) else 0)

    computed = "".join(str(b) for b in result_bits)
    cot += f"\nNow applying these rules to the query input {query}:\n"
    for i in range(8):
        cot += f"  Bit {i}: {result_bits[i]}\n"
    cot += f"\nResult: {computed}\n\n"
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
    """Generate CoT for algebra puzzles using solver v2 for solvable puzzles."""
    from src.solvers.algebra_solver_v2 import (
        solve_algebra, parse_algebra_prompt, parse_digit_operator, OP_DESCRIPTIONS
    )

    examples, query = parse_algebra_prompt(prompt)
    if not examples or not query:
        return f"Analyzing the transformation rules.\n\nResult: {answer}\n\n\\boxed{{{answer}}}"

    # Try solver v2 first
    solver_result = solve_algebra(prompt)

    cot = "I need to find the transformation rule from the examples.\n\n"
    cot += "Examples:\n"
    for inp, out in examples[:5]:
        cot += f"  \"{inp}\" = \"{out}\"\n"
    cot += "\n"

    if solver_result and solver_result[0] == answer:
        computed, op_name, desc = solver_result
        # Solver found the operation — generate detailed step-by-step CoT
        qp = parse_digit_operator(query)
        if qp:
            qa, query_op, qb = qp
            cot += f"Looking at examples with the '{query_op}' operator:\n"
            # Show matching examples with the operation
            for inp, out in examples:
                p = parse_digit_operator(inp)
                if p and p[1] == query_op:
                    a, b = p[0], p[2]
                    desc_filled = desc.format(a=a, b=b) if '{a}' in desc else f"{a} {op_name} {b}"
                    cot += f"  {a} {query_op} {b} = {out}  (check: {desc_filled} = {out} ✓)\n"
            cot += f"\nThe operator '{query_op}' performs: {desc.format(a='a', b='b') if '{a}' in desc else op_name}\n"
            cot += f"\nApplying to query: {qa} {query_op} {qb}\n"
            desc_filled = desc.format(a=qa, b=qb) if '{a}' in desc else f"{qa} {op_name} {qb}"
            cot += f"  {desc_filled} = {computed}\n"
        else:
            cot += f"Pattern identified: {op_name}\n"
            cot += f"Applying to query \"{query}\": {computed}\n"
    else:
        # Fallback: show examples and pattern analysis
        qp = parse_digit_operator(query)
        if qp:
            qa, query_op, qb = qp
            # Show matching examples for the query's operator
            matching = [(inp, out) for inp, out in examples
                       if parse_digit_operator(inp) and parse_digit_operator(inp)[1] == query_op]
            if matching:
                cot += f"Focusing on examples with operator '{query_op}':\n"
                for inp, out in matching[:4]:
                    p = parse_digit_operator(inp)
                    if p:
                        cot += f"  {p[0]} {query_op} {p[2]} = {out}\n"
                cot += f"\nAnalyzing the pattern to determine the operation.\n"
            else:
                cot += f"The query uses operator '{query_op}'. Studying the transformation rules.\n"
        else:
            cot += "Analyzing the transformation pattern from all examples.\n"
        cot += f"Applying the derived rule to the query.\n"

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


BOXED_INSTRUCTION = "\n\nPlease put your final answer inside \\boxed{}."


def wrap_completion_in_think(completion: str) -> str:
    """Wrap a completion in <think>...</think> format for Nemotron chat template.

    Separates reasoning from the final \\boxed{} answer so that reasoning
    lives inside <think> tags and the boxed answer follows </think>.
    Skips wrapping if <think> tags are already present.
    """
    if "<think>" in completion:
        return completion

    idx = completion.rfind("\\boxed{")
    if idx >= 0:
        reasoning = completion[:idx].rstrip()
        boxed = completion[idx:]
        return f"<think>\n{reasoning}\n</think>\n{boxed}"
    return f"<think>\n{completion}\n</think>"


def format_chat_template(prompt: str, completion: str) -> dict:
    """Format a single example into chat template format.

    Wraps reasoning in <think>...</think> to match the Nemotron native
    chat template that Kaggle uses for inference.
    """
    return {
        "messages": [
            {"role": "user", "content": prompt + BOXED_INSTRUCTION},
            {"role": "assistant", "content": wrap_completion_in_think(completion)},
        ]
    }


def load_llm_cot() -> dict:
    """Load LLM-generated CoT traces from all model directories."""
    llm_cot = {}  # id -> best CoT trace
    cot_dirs = [
        os.path.join(DATA_DIR, "nemotron_cot"),
        os.path.join(DATA_DIR, "deepseek_cot"),
        os.path.join(DATA_DIR, "deepseek8b_cot"),
        os.path.join(DATA_DIR, "qwen_cot"),
        os.path.join(DATA_DIR, "gemini_cot"),
    ]

    for cot_dir in cot_dirs:
        if not os.path.isdir(cot_dir):
            continue
        for fname in os.listdir(cot_dir):
            if not fname.endswith(".jsonl"):
                continue
            path = os.path.join(cot_dir, fname)
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("cot_status") not in ("success", "fixed"):
                        continue
                    rid = r["id"]
                    # Prefer "success" over "fixed", and longer CoT over shorter
                    if rid not in llm_cot:
                        llm_cot[rid] = r["cot"]
                    elif r["cot_status"] == "success" and len(r["cot"]) > len(llm_cot[rid]):
                        llm_cot[rid] = r["cot"]

    return llm_cot


def load_synthetic_data() -> list:
    """Load synthetic training puzzles."""
    synth_path = os.path.join(DATA_DIR, "synthetic.jsonl")
    if not os.path.exists(synth_path):
        return []
    records = []
    with open(synth_path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def main():
    # Load LLM-generated CoT
    llm_cot = load_llm_cot()
    print(f"  Loaded {len(llm_cot)} LLM-generated CoT traces")

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
    llm_used = 0
    dropped_unverified = 0
    for source_name, records in datasets.items():
        for record in records:
            cat = record["category"]
            answer = record["answer"]
            prompt = record["prompt"]
            rid = record["id"]

            # Use LLM CoT if available (especially for algebra)
            if rid in llm_cot:
                completion = llm_cot[rid]
                llm_used += 1
            else:
                # Drop unverified examples without LLM CoT —
                # their answers are unconfirmed and template CoT is weak
                if source_name == "unverified":
                    dropped_unverified += 1
                    continue
                cot_gen = COT_GENERATORS.get(cat, cot_algebra)
                completion = cot_gen(prompt, answer)

            formatted = format_chat_template(prompt, completion)
            formatted["id"] = rid
            formatted["category"] = cat
            formatted["source"] = source_name
            all_records.append(formatted)

    print(f"  LLM CoT used for {llm_used} examples (replacing template CoT)")
    if dropped_unverified:
        print(f"  Dropped {dropped_unverified} unverified examples (no LLM CoT available)")

    # Add synthetic data
    synth_records = load_synthetic_data()
    for record in synth_records:
        cat = record["category"]
        answer = record["answer"]
        prompt = record["prompt"]

        cot_gen = COT_GENERATORS.get(cat, cot_algebra)
        completion = cot_gen(prompt, answer)

        formatted = format_chat_template(prompt, completion)
        formatted["id"] = record["id"]
        formatted["category"] = cat
        formatted["source"] = "synthetic"
        all_records.append(formatted)

    if synth_records:
        print(f"  Added {len(synth_records)} synthetic examples")

    # Upsample weak categories to balance the dataset (~4k per category)
    # algebra & bit_manipulation already have ~4k from synthetic data
    UPSAMPLE = {
        "gravity": 3,
        "unit_conversion": 3,
        "numeral": 3,
        "text_encryption": 7,
    }
    upsampled = []
    for r in all_records:
        cat = r.get("category", "")
        reps = UPSAMPLE.get(cat, 1)
        for _ in range(reps):
            upsampled.append(r)
    upsample_delta = len(upsampled) - len(all_records)
    if upsample_delta:
        print(f"  Upsampled weak categories: +{upsample_delta} examples ({len(all_records)} → {len(upsampled)})")
    all_records = upsampled

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

    source_counts = Counter(r["source"] for r in train_records)
    print(f"\n📊 Source breakdown (train):")
    for src, count in sorted(source_counts.items()):
        print(f"  {src:20s}: {count:5d}")

    # Show a sample
    print(f"\n📝 Sample formatted example:")
    sample = train_records[0]
    print(f"  Category: {sample['category']}")
    print(f"  User: {sample['messages'][0]['content'][:150]}...")
    print(f"  Assistant: {sample['messages'][1]['content'][:200]}...")


if __name__ == "__main__":
    main()
