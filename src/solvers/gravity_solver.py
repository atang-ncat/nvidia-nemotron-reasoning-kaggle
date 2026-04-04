"""
Gravity solver: g = 2d / t²
Parses gravity prompts, computes g from examples, and predicts distance for query t.
"""

import re
from typing import Optional


def parse_gravity_prompt(prompt: str):
    """Parse gravity prompt into (examples: [(t, d), ...], query_t: float)."""
    observations = []
    query_t = None

    for line in prompt.split("\n"):
        line = line.strip()
        # Match: For t = 1.37s, distance = 14.92 m
        m = re.match(
            r"For\s+t\s*=\s*([\d.]+)\s*s?,?\s*distance\s*=\s*([\d.]+)\s*m?", line
        )
        if m:
            observations.append((float(m.group(1)), float(m.group(2))))

        # Match query line
        m2 = re.search(r"t\s*=\s*([\d.]+)\s*s", line)
        if m2 and "determine" in line.lower():
            query_t = float(m2.group(1))

    return observations, query_t


def solve_gravity(prompt: str) -> Optional[str]:
    """
    Solve a gravity puzzle.
    Returns the predicted distance as a string (rounded to 2 decimals), or None if unsolvable.
    """
    observations, query_t = parse_gravity_prompt(prompt)

    if not observations or query_t is None:
        return None

    # Compute g from each observation: d = 0.5 * g * t^2 => g = 2d / t^2
    g_values = []
    for t, d in observations:
        if t == 0:
            continue
        g = 2 * d / (t * t)
        g_values.append(g)

    if not g_values:
        return None

    # Check consistency — all g values should be close
    mean_g = sum(g_values) / len(g_values)
    spread = max(g_values) - min(g_values)

    if spread > 1.0:
        # Inconsistent observations — flag as suspicious
        return None

    # Compute distance for query
    distance = 0.5 * mean_g * query_t * query_t
    return f"{distance:.2f}"


def verify_gravity(prompt: str, expected_answer: str) -> dict:
    """
    Verify a gravity puzzle answer.
    Returns {'status': 'verified'|'corrected'|'unsolvable', 'computed': str, 'expected': str}
    """
    computed = solve_gravity(prompt)
    expected = expected_answer.strip()

    if computed is None:
        return {"status": "unsolvable", "computed": None, "expected": expected}

    # Compare with tolerance
    try:
        computed_f = float(computed)
        expected_f = float(expected)
        if abs(computed_f - expected_f) < 0.1:
            return {"status": "verified", "computed": computed, "expected": expected}
        else:
            return {"status": "corrected", "computed": computed, "expected": expected}
    except ValueError:
        return {"status": "corrected", "computed": computed, "expected": expected}
