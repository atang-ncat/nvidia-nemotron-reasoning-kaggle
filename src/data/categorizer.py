"""
Categorize each row in train.csv into a puzzle domain.

Categories:
  - bit_manipulation: 8-bit binary transformation rules
  - text_encryption: secret cipher / text encoding puzzles
  - gravity: gravitational constant computation (g = 2d/t²)
  - numeral: numeral system conversion (e.g., Roman numerals)
  - unit_conversion: secret unit conversion factors
  - algebra: equation / symbol transformation rules
"""

import re

CATEGORIES = [
    "bit_manipulation",
    "text_encryption",
    "gravity",
    "numeral",
    "unit_conversion",
    "algebra",
]


def categorize_prompt(prompt: str) -> str:
    """Classify a prompt string into one of the known puzzle categories."""
    first_line = prompt.strip().split("\n")[0].lower()

    if "bit manipulation" in first_line:
        return "bit_manipulation"
    elif "gravitational" in first_line or "gravity" in first_line:
        return "gravity"
    elif "encryption" in first_line or "cipher" in first_line:
        return "text_encryption"
    elif "numeral" in first_line or "roman" in first_line:
        return "numeral"
    elif "unit" in first_line and "conver" in first_line:
        return "unit_conversion"
    elif "equation" in first_line or "transformation rules" in first_line:
        return "algebra"
    else:
        # Fallback: try body-level heuristics
        body = prompt.lower()
        if "bit manipulation" in body:
            return "bit_manipulation"
        elif "gravitational" in body:
            return "gravity"
        elif "encryption" in body:
            return "text_encryption"
        elif "numeral" in body:
            return "numeral"
        elif "unit" in body and "conver" in body:
            return "unit_conversion"
        elif re.search(r"transformation rules", body):
            return "algebra"
        return "unknown"
