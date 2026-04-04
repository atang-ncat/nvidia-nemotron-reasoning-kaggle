"""
Text encryption (cipher) solver: builds substitution table from examples and applies to query.
Handles character-level Caesar-shift ciphers and word-level substitution ciphers.
"""

import re
from typing import Optional


def parse_encryption_prompt(prompt: str):
    """Parse text encryption prompt into (examples: [(encrypted, plain), ...], query: str)."""
    examples = []
    query = None

    lines = prompt.strip().split("\n")
    for i, line in enumerate(lines):
        line = line.strip()
        # Match: encrypted_text -> plain_text
        m = re.match(r"^(.+?)\s*->\s*(.+)$", line)
        if m:
            encrypted = m.group(1).strip()
            plain = m.group(2).strip()
            examples.append((encrypted, plain))

        # Match query: "Now, decrypt..." or "Now, determine..."
        if re.match(r"^now,?\s+(decrypt|determine|apply)", line, re.IGNORECASE):
            # The query text might be on this line or the next
            m2 = re.search(r":\s*(.+)$", line)
            if m2:
                query = m2.group(1).strip()
            elif i + 1 < len(lines):
                query = lines[i + 1].strip()

    return examples, query


def build_char_substitution_table(examples):
    """
    Build a character-level substitution table from encryption examples.
    Returns (table: dict, is_complete: bool, conflicts: list)
    """
    table = {}
    conflicts = []

    for encrypted, plain in examples:
        enc_chars = encrypted.replace(" ", "")
        plain_chars = plain.replace(" ", "")

        if len(enc_chars) != len(plain_chars):
            continue

        for e, p in zip(enc_chars, plain_chars):
            if e in table:
                if table[e] != p:
                    conflicts.append((e, table[e], p))
            else:
                table[e] = p

    return table, len(conflicts) == 0, conflicts


def build_word_substitution_table(examples):
    """Build a word-level substitution table from examples."""
    table = {}
    conflicts = []

    for encrypted, plain in examples:
        enc_words = encrypted.lower().split()
        plain_words = plain.lower().split()

        if len(enc_words) != len(plain_words):
            continue

        for e, p in zip(enc_words, plain_words):
            if e in table:
                if table[e] != p:
                    conflicts.append((e, table[e], p))
            else:
                table[e] = p

    return table, len(conflicts) == 0, conflicts


def try_caesar_shift(examples):
    """Try to detect a simple alphabetic shift (Caesar cipher)."""
    shifts = []
    for encrypted, plain in examples:
        enc_chars = [c for c in encrypted.lower() if c.isalpha()]
        plain_chars = [c for c in plain.lower() if c.isalpha()]

        if len(enc_chars) != len(plain_chars):
            return None

        for e, p in zip(enc_chars, plain_chars):
            shift = (ord(p) - ord(e)) % 26
            shifts.append(shift)

    if not shifts:
        return None

    # All shifts should be the same for a Caesar cipher
    if len(set(shifts)) == 1:
        return shifts[0]

    return None


def solve_text_encryption(prompt: str) -> Optional[str]:
    """Solve a text encryption puzzle."""
    examples, query = parse_encryption_prompt(prompt)

    if not examples or not query:
        return None

    # Strategy 1: Try Caesar shift first (simplest)
    shift = try_caesar_shift(examples)
    if shift is not None:
        result = []
        for c in query:
            if c.isalpha():
                base = ord("a") if c.islower() else ord("A")
                result.append(chr((ord(c) - base + shift) % 26 + base))
            else:
                result.append(c)
        return "".join(result)

    # Strategy 2: Try word-level substitution
    word_table, word_clean, _ = build_word_substitution_table(examples)
    if word_clean:
        query_words = query.lower().split()
        all_found = all(w in word_table for w in query_words)
        if all_found:
            result_words = [word_table[w] for w in query_words]
            return " ".join(result_words)

    # Strategy 3: Try character-level substitution
    char_table, char_clean, _ = build_char_substitution_table(examples)
    if char_clean:
        query_chars = query.replace(" ", "")
        all_found = all(c in char_table for c in query_chars)
        if all_found:
            result = []
            for c in query:
                if c == " ":
                    result.append(" ")
                elif c in char_table:
                    result.append(char_table[c])
                else:
                    return None
            return "".join(result)

    return None


def verify_text_encryption(prompt: str, expected_answer: str) -> dict:
    """Verify a text encryption puzzle answer."""
    computed = solve_text_encryption(prompt)
    expected = expected_answer.strip()

    if computed is None:
        return {"status": "unsolvable", "computed": None, "expected": expected}

    if computed.lower() == expected.lower():
        return {"status": "verified", "computed": computed, "expected": expected}
    else:
        return {"status": "corrected", "computed": computed, "expected": expected}
