"""
Unit conversion solver: extracts the conversion factor from examples, applies to query.
"""

import re
from typing import Optional


def parse_unit_conversion_prompt(prompt: str):
    """Parse unit conversion prompt into (examples: [(input, output), ...], query_value: float)."""
    examples = []
    query_value = None

    for line in prompt.split("\n"):
        line = line.strip()
        # Match: 10.08 m becomes 6.69
        m = re.match(r"^([\d.]+)\s*m?\s+becomes\s+([\d.]+)", line)
        if m:
            examples.append((float(m.group(1)), float(m.group(2))))

        # Match query: "convert the following measurement: 25.09 m"
        m2 = re.search(r"convert.*?:\s*([\d.]+)\s*m?", line, re.IGNORECASE)
        if m2:
            query_value = float(m2.group(1))

    return examples, query_value


def solve_unit_conversion(prompt: str) -> Optional[str]:
    """Solve a unit conversion puzzle by extracting the scaling factor."""
    examples, query_value = parse_unit_conversion_prompt(prompt)

    if not examples or query_value is None:
        return None

    # Compute the conversion factor from each example
    factors = []
    for inp, out in examples:
        if inp == 0:
            continue
        factors.append(out / inp)

    if not factors:
        return None

    # Check consistency
    mean_factor = sum(factors) / len(factors)
    spread = max(factors) - min(factors)

    if spread > 0.01:
        # Inconsistent — might be a non-linear conversion
        return None

    # Apply the factor to the query
    result = mean_factor * query_value
    return f"{result:.2f}"


def verify_unit_conversion(prompt: str, expected_answer: str) -> dict:
    """Verify a unit conversion puzzle answer."""
    computed = solve_unit_conversion(prompt)
    expected = expected_answer.strip()

    if computed is None:
        return {"status": "unsolvable", "computed": None, "expected": expected}

    try:
        computed_f = float(computed)
        expected_f = float(expected)
        if abs(computed_f - expected_f) < 0.1:
            return {"status": "verified", "computed": computed, "expected": expected}
        else:
            return {"status": "corrected", "computed": computed, "expected": expected}
    except ValueError:
        return {"status": "corrected", "computed": computed, "expected": expected}
