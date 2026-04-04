"""
Bit manipulation solver: infers per-bit boolean functions from examples.
Each output bit is determined by a boolean function of the 8 input bits.
"""

import re
from itertools import product
from typing import Optional


# All boolean functions of a single input bit (position-relative)
SINGLE_BIT_FUNCTIONS = {
    "COPY": lambda bits, pos: bits[pos],
    "NOT": lambda bits, pos: 1 - bits[pos],
    "CONST_0": lambda bits, pos: 0,
    "CONST_1": lambda bits, pos: 1,
}

# Two-input boolean functions between the current bit and each other bit
TWO_BIT_OPERATIONS = {
    "AND": lambda a, b: a & b,
    "OR": lambda a, b: a | b,
    "XOR": lambda a, b: a ^ b,
    "NAND": lambda a, b: 1 - (a & b),
    "NOR": lambda a, b: 1 - (a | b),
    "XNOR": lambda a, b: 1 - (a ^ b),
    "AND_NOT": lambda a, b: a & (1 - b),   # a AND (NOT b)
    "NOT_AND": lambda a, b: (1 - a) & b,   # (NOT a) AND b
    "OR_NOT": lambda a, b: a | (1 - b),    # a OR (NOT b)
    "NOT_OR": lambda a, b: (1 - a) | b,    # (NOT a) OR b
}


def parse_bit_prompt(prompt: str):
    """Parse bit manipulation prompt into (examples: [(in_bits, out_bits), ...], query: str)."""
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

    return examples, query


def infer_bit_function(examples, out_pos):
    """
    Infer the boolean function for a given output bit position.
    Returns (func_name, func) or None if no consistent function found.
    """
    # Extract training data for this output position
    inputs = [[int(b) for b in ex[0]] for ex in examples]
    expected = [int(ex[1][out_pos]) for ex in examples]

    # Try single-bit functions first (cheapest)
    for name, func in SINGLE_BIT_FUNCTIONS.items():
        match = all(func(inp, out_pos) == exp for inp, exp in zip(inputs, expected))
        if match:
            return (name, out_pos, None)

    # Try two-input functions between out_pos and every other position
    for other_pos in range(8):
        for op_name, op_func in TWO_BIT_OPERATIONS.items():
            match = all(
                op_func(inp[out_pos], inp[other_pos]) == exp
                for inp, exp in zip(inputs, expected)
            )
            if match:
                return (op_name, out_pos, other_pos)

    # Try single-bit from a different position (permutation)
    for src_pos in range(8):
        if src_pos == out_pos:
            continue
        # COPY from different position
        match = all(inp[src_pos] == exp for inp, exp in zip(inputs, expected))
        if match:
            return ("COPY_FROM", src_pos, None)
        # NOT from different position
        match = all(1 - inp[src_pos] == exp for inp, exp in zip(inputs, expected))
        if match:
            return ("NOT_FROM", src_pos, None)

    # Try two-input functions between any two positions
    for pos_a in range(8):
        for pos_b in range(pos_a + 1, 8):
            if pos_a == out_pos and pos_b == out_pos:
                continue
            for op_name, op_func in TWO_BIT_OPERATIONS.items():
                match = all(
                    op_func(inp[pos_a], inp[pos_b]) == exp
                    for inp, exp in zip(inputs, expected)
                )
                if match:
                    return (f"{op_name}_CROSS", pos_a, pos_b)

    return None


def apply_bit_function(func_info, input_bits):
    """Apply an inferred bit function to produce one output bit."""
    name, arg1, arg2 = func_info

    if name == "CONST_0":
        return 0
    elif name == "CONST_1":
        return 1
    elif name == "COPY":
        return input_bits[arg1]
    elif name == "NOT":
        return 1 - input_bits[arg1]
    elif name == "COPY_FROM":
        return input_bits[arg1]
    elif name == "NOT_FROM":
        return 1 - input_bits[arg1]
    elif name.endswith("_CROSS"):
        op_name = name[:-6]
        return TWO_BIT_OPERATIONS[op_name](input_bits[arg1], input_bits[arg2])
    else:
        # Two-input operation: arg1 is position of bit a, arg2 is other position
        return TWO_BIT_OPERATIONS[name](input_bits[arg1], input_bits[arg2])


def solve_bit_manipulation(prompt: str) -> Optional[str]:
    """Solve a bit manipulation puzzle by inferring per-bit boolean functions."""
    examples, query = parse_bit_prompt(prompt)

    if not examples or not query:
        return None

    query_bits = [int(b) for b in query]

    # Infer function for each output bit position
    output_bits = []
    for out_pos in range(8):
        func_info = infer_bit_function(examples, out_pos)
        if func_info is None:
            return None  # Cannot determine function for this bit
        output_bits.append(apply_bit_function(func_info, query_bits))

    return "".join(str(b) for b in output_bits)


def verify_bit_manipulation(prompt: str, expected_answer: str) -> dict:
    """Verify a bit manipulation puzzle answer."""
    computed = solve_bit_manipulation(prompt)
    expected = expected_answer.strip()

    if computed is None:
        return {"status": "unsolvable", "computed": None, "expected": expected}

    if computed == expected:
        return {"status": "verified", "computed": computed, "expected": expected}
    else:
        return {"status": "corrected", "computed": computed, "expected": expected}
