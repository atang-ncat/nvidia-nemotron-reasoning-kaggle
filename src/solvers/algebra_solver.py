"""
Algebra solver: parses custom operator transformation rules from examples.
These puzzles define secret operators on symbol strings.
"""

import re
from typing import Optional


def parse_algebra_prompt(prompt: str):
    """Parse algebra prompt into (examples: [(input, output), ...], query: str)."""
    examples = []
    query = None

    for line in prompt.split("\n"):
        line = line.strip()
        # Match: expression = result
        m = re.match(r"^(.+?)\s*=\s*(.+)$", line)
        if m and "determine" not in line.lower() and "secret" not in line.lower():
            examples.append((m.group(1).strip(), m.group(2).strip()))

        # Match query: "Now, determine the result for: ..."
        m2 = re.search(r"determine the result for:\s*(.+)", line, re.IGNORECASE)
        if m2:
            query = m2.group(1).strip()

    return examples, query


def solve_algebra(prompt: str) -> Optional[str]:
    """
    Attempt to solve an algebra puzzle.
    These are highly varied and often involve custom symbol transformations.
    For now, we mark most as unsolvable unless we can identify simple patterns.
    """
    examples, query = parse_algebra_prompt(prompt)

    if not examples or not query:
        return None

    # The algebra category contains very diverse puzzle types with custom
    # symbol transformations. Without understanding the specific rule being
    # applied, we cannot reliably solve these programmatically.
    # 
    # Strategy: If we can find a consistent character-level mapping, use it.
    # Otherwise, mark as unsolvable and rely on the model to learn these.

    # Try: check if the operator is simply reversing, or applying a char-level shift
    # For the Wonderland competition, many of these are genuinely hard
    # and require the model's reasoning capabilities.

    return None  # Mark as unsolvable for now


def verify_algebra(prompt: str, expected_answer: str) -> dict:
    """Verify an algebra puzzle answer."""
    computed = solve_algebra(prompt)
    expected = expected_answer.strip()

    if computed is None:
        # For algebra, we trust the provided answer if we can't verify independently
        # But we flag it as "unverified" rather than "unsolvable"
        return {"status": "unverified", "computed": None, "expected": expected}

    if computed == expected:
        return {"status": "verified", "computed": computed, "expected": expected}
    else:
        return {"status": "corrected", "computed": computed, "expected": expected}
