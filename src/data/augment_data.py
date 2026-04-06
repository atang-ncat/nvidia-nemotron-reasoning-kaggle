#!/usr/bin/env python3
"""
Synthetic Data Augmentation for Weak Categories.

Generates synthetic training puzzles with guaranteed-correct answers
and solver-generated CoT reasoning traces.

Categories:
  - bit_manipulation: random 8-bit boolean functions
  - algebra: known-operator digit puzzles (add, mul, concat, etc.)
"""

import os
import sys
import json
import random
import string

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

DATA_DIR = "/scratch2/atang/competitions/nemotron-kaggle/data"
OUTPUT_PATH = os.path.join(DATA_DIR, "synthetic.jsonl")


# ---- Bit Manipulation Generators ---- #

def random_bit_function():
    """Generate a random per-bit function definition."""
    # For each of 8 output bits, define what it does
    funcs = []
    for out_pos in range(8):
        kind = random.choice(["copy", "not", "const0", "const1", "and", "or", "xor", "nand"])
        if kind == "copy":
            src = random.randint(0, 7)
            funcs.append(("COPY", src, None))
        elif kind == "not":
            src = random.randint(0, 7)
            funcs.append(("NOT", src, None))
        elif kind == "const0":
            funcs.append(("CONST_0", None, None))
        elif kind == "const1":
            funcs.append(("CONST_1", None, None))
        else:
            a = random.randint(0, 7)
            b = random.randint(0, 7)
            while b == a:
                b = random.randint(0, 7)
            funcs.append((kind.upper(), a, b))
    return funcs


def apply_bit_function(funcs, input_bits):
    """Apply per-bit function to input, return output bits."""
    output = []
    for name, arg1, arg2 in funcs:
        if name == "COPY":
            output.append(input_bits[arg1])
        elif name == "NOT":
            output.append(1 - input_bits[arg1])
        elif name == "CONST_0":
            output.append(0)
        elif name == "CONST_1":
            output.append(1)
        elif name == "AND":
            output.append(input_bits[arg1] & input_bits[arg2])
        elif name == "OR":
            output.append(input_bits[arg1] | input_bits[arg2])
        elif name == "XOR":
            output.append(input_bits[arg1] ^ input_bits[arg2])
        elif name == "NAND":
            output.append(1 - (input_bits[arg1] & input_bits[arg2]))
        else:
            output.append(0)
    return output


def generate_bit_puzzle():
    """Generate a single bit_manipulation puzzle with answer."""
    funcs = random_bit_function()

    # Generate 5-8 training examples + 1 query
    n_examples = random.randint(5, 8)
    used_inputs = set()
    examples = []

    for _ in range(n_examples + 1):
        while True:
            val = random.randint(0, 255)
            if val not in used_inputs:
                used_inputs.add(val)
                break
        bits = [int(b) for b in format(val, '08b')]
        out_bits = apply_bit_function(funcs, bits)
        inp_str = "".join(str(b) for b in bits)
        out_str = "".join(str(b) for b in out_bits)
        examples.append((inp_str, out_str))

    train_examples = examples[:-1]
    query_inp, query_out = examples[-1]

    # Build prompt
    prompt = "In Wonderland, there is a secret machine that transforms 8-bit binary strings.\n"
    prompt += "Given examples of input → output transformations:\n\n"
    for inp, out in train_examples:
        prompt += f"{inp} -> {out}\n"
    prompt += f"\nDetermine the result for: {query_inp}"

    return {
        "category": "bit_manipulation",
        "prompt": prompt,
        "answer": query_out,
        "id": f"synth_bit_{random.randint(100000,999999)}",
        "source": "synthetic",
    }


# ---- Algebra Generators ---- #

ALGEBRA_OPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "concat_ab": lambda a, b: int(str(a) + str(b)),
    "concat_ba": lambda a, b: int(str(b) + str(a)),
    "abs_diff": lambda a, b: abs(a - b),
    "add_p1": lambda a, b: a + b + 1,
    "add_m1": lambda a, b: a + b - 1,
    "mul_p1": lambda a, b: a * b + 1,
    "mul_m1": lambda a, b: a * b - 1,
    "xor": lambda a, b: a ^ b,
    "max_val": lambda a, b: max(a, b),
    "min_val": lambda a, b: min(a, b),
    "digit_sum_add": lambda a, b: sum(int(d) for d in str(a)) + sum(int(d) for d in str(b)),
    "first_a_last_b": lambda a, b: int(str(a)[0] + str(b)[-1]),
    "reverse_add": lambda a, b: int(str(a)[::-1]) + int(str(b)[::-1]),
}


def generate_algebra_puzzle():
    """Generate a single algebra puzzle with a known operation."""
    op_symbols = list("@#$%^&*!?<>~|{}[]`'\"\\:;")
    op_char = random.choice(op_symbols)

    op_name = random.choice(list(ALGEBRA_OPS.keys()))
    op_func = ALGEBRA_OPS[op_name]

    n_examples = random.randint(4, 6)
    examples = []
    for _ in range(n_examples + 1):
        a = random.randint(10, 99)
        b = random.randint(10, 99)
        result = op_func(a, b)
        examples.append((a, b, str(result)))

    train_examples = examples[:-1]
    qa, qb, q_answer = examples[-1]

    prompt = "In Wonderland, a secret function transforms pairs of numbers.\n"
    prompt += "Given examples:\n\n"
    for a, b, out in train_examples:
        prompt += f"{a}{op_char}{b} = {out}\n"
    prompt += f"\nDetermine the result for: {qa}{op_char}{qb}"

    return {
        "category": "algebra",
        "prompt": prompt,
        "answer": q_answer,
        "id": f"synth_alg_{random.randint(100000,999999)}",
        "source": "synthetic",
    }


def generate_multi_op_algebra_puzzle():
    """Generate multi-operator algebra puzzle (closer to real test distribution).

    Real algebra puzzles often have 2-3 different operators in one prompt.
    The model must figure out what each operator does from context.
    """
    n_ops = random.randint(2, 3)
    op_symbols = random.sample(list("@#$%^&*!?<>~|"), n_ops)
    op_names = random.sample(list(ALGEBRA_OPS.keys()), n_ops)
    op_funcs = {sym: ALGEBRA_OPS[name] for sym, name in zip(op_symbols, op_names)}

    all_examples = []
    for sym, func in op_funcs.items():
        n = random.randint(2, 4)
        for _ in range(n):
            a = random.randint(10, 99)
            b = random.randint(10, 99)
            result = func(a, b)
            all_examples.append((a, sym, b, str(result)))

    random.shuffle(all_examples)

    query_sym = random.choice(op_symbols)
    qa = random.randint(10, 99)
    qb = random.randint(10, 99)
    q_answer = str(op_funcs[query_sym](qa, qb))

    prompt = "In Wonderland, secret functions transform pairs of numbers.\n"
    prompt += "Given examples:\n\n"
    for a, sym, b, out in all_examples:
        prompt += f"{a}{sym}{b} = {out}\n"
    prompt += f"\nDetermine the result for: {qa}{query_sym}{qb}"

    return {
        "category": "algebra",
        "prompt": prompt,
        "answer": q_answer,
        "id": f"synth_malg_{random.randint(100000,999999)}",
        "source": "synthetic",
    }


def main():
    random.seed(42)

    records = []

    # Generate bit_manipulation puzzles
    print("Generating bit_manipulation puzzles...")
    for i in range(2500):
        records.append(generate_bit_puzzle())
    print(f"  {2500} bit_manipulation puzzles")

    # Generate algebra puzzles (single-operator)
    print("Generating algebra puzzles (single-op)...")
    for i in range(1500):
        records.append(generate_algebra_puzzle())
    print(f"  {1500} single-op algebra puzzles")

    # Generate algebra puzzles (multi-operator, closer to real distribution)
    print("Generating algebra puzzles (multi-op)...")
    for i in range(1000):
        records.append(generate_multi_op_algebra_puzzle())
    print(f"  {1000} multi-op algebra puzzles")

    # Shuffle
    random.shuffle(records)

    # Save
    with open(OUTPUT_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\n✅ Total: {len(records)} synthetic puzzles → {OUTPUT_PATH}")

    # Show samples
    for cat in ["bit_manipulation", "algebra"]:
        sample = [r for r in records if r["category"] == cat][0]
        print(f"\n📝 Sample {cat}:")
        print(f"  Prompt: {sample['prompt'][:200]}...")
        print(f"  Answer: {sample['answer']}")


if __name__ == "__main__":
    main()
