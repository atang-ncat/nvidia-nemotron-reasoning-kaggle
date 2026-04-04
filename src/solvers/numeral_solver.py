"""
Numeral system solver: converts integers to Roman numerals.
Parses numeral prompts, identifies the target numeral system, and computes the answer.
"""

import re
from typing import Optional


ROMAN_VALUES = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]


def int_to_roman(num: int) -> str:
    """Convert an integer to a Roman numeral string."""
    result = []
    for value, numeral in ROMAN_VALUES:
        while num >= value:
            result.append(numeral)
            num -= value
    return "".join(result)


def roman_to_int(s: str) -> int:
    """Convert a Roman numeral string to an integer."""
    roman_map = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    prev = 0
    for ch in reversed(s.upper()):
        val = roman_map.get(ch, 0)
        if val < prev:
            total -= val
        else:
            total += val
        prev = val
    return total


def parse_numeral_prompt(prompt: str):
    """Parse numeral prompt into (examples: [(int, str), ...], query: int)."""
    examples = []
    query = None

    for line in prompt.split("\n"):
        line = line.strip()
        # Match: 11 -> XI
        m = re.match(r"^(\d+)\s*->\s*(.+)$", line)
        if m:
            examples.append((int(m.group(1)), m.group(2).strip()))

        # Match query: "write the number 38 in the Wonderland numeral system"
        m2 = re.search(r"write the number\s+(\d+)", line, re.IGNORECASE)
        if m2:
            query = int(m2.group(1))

    return examples, query


def detect_numeral_system(examples):
    """Detect if the numeral system is Roman or something else."""
    # Check if all example outputs look like Roman numerals
    roman_chars = set("IVXLCDM")
    all_roman = all(
        set(out.upper()).issubset(roman_chars) for _, out in examples
    )

    if all_roman:
        # Verify that the conversion is standard Roman
        all_match = all(
            int_to_roman(num) == out for num, out in examples
        )
        if all_match:
            return "roman"

    return "unknown"


def solve_numeral(prompt: str) -> Optional[str]:
    """Solve a numeral conversion puzzle. Returns the answer string or None."""
    examples, query = parse_numeral_prompt(prompt)

    if not examples or query is None:
        return None

    system = detect_numeral_system(examples)

    if system == "roman":
        return int_to_roman(query)

    # For non-standard systems, try to infer the pattern
    # (could be base-N, custom symbols, etc.)
    return None


def verify_numeral(prompt: str, expected_answer: str) -> dict:
    """Verify a numeral puzzle answer."""
    computed = solve_numeral(prompt)
    expected = expected_answer.strip()

    if computed is None:
        return {"status": "unsolvable", "computed": None, "expected": expected}

    if computed == expected:
        return {"status": "verified", "computed": computed, "expected": expected}
    else:
        return {"status": "corrected", "computed": computed, "expected": expected}
