"""
Algebra solver v2: Infers custom operators from examples.

Puzzle types:
1. Digit-operator: NN<op>NN = result (single or multi-operator per puzzle)
2. Symbol-only: arbitrary ASCII string transformations

Strategy for digit-operator:
- Filter examples to those using the same operator as the query
- Try all candidate operations on filtered examples
- Apply the matching operation to the query

Strategy for symbol-only:
- Try positional extraction (output = input chars at fixed positions)
- Try per-position character mapping
"""

import re
from typing import Optional, Tuple, List


def parse_algebra_prompt(prompt: str):
    """Parse algebra prompt into (examples: [(input, output), ...], query: str)."""
    examples = []
    query = None

    for line in prompt.split("\n"):
        line = line.strip()
        m = re.match(r"^(.+?)\s*=\s*(.+)$", line)
        if m and "determine" not in line.lower() and "secret" not in line.lower():
            examples.append((m.group(1).strip(), m.group(2).strip()))

        m2 = re.search(r"determine the result for:\s*(.+)", line, re.IGNORECASE)
        if m2:
            query = m2.group(1).strip()

    return examples, query


def parse_digit_operator(expr: str):
    """Parse 'NN<op>NN' into (a, op, b). Returns None if not this format."""
    m = re.match(r'^(\d+)([^\d])(\d+)$', expr)
    if m:
        return int(m.group(1)), m.group(2), int(m.group(3))
    return None


# Named operations: name -> function(a, b) -> str | None
def _build_ops():
    import math
    ops = {}
    ops["add"] = lambda a, b: str(a + b)
    ops["sub_ab"] = lambda a, b: str(a - b)
    ops["sub_ba"] = lambda a, b: str(b - a)
    ops["abs_diff"] = lambda a, b: str(abs(a - b))
    ops["mul"] = lambda a, b: str(a * b)
    ops["mul_p1"] = lambda a, b: str(a * b + 1)
    ops["mul_m1"] = lambda a, b: str(a * b - 1)
    ops["mul_p2"] = lambda a, b: str(a * b + 2)
    ops["mul_m2"] = lambda a, b: str(a * b - 2)
    ops["div_ab"] = lambda a, b: str(a // b) if b != 0 else None
    ops["div_ba"] = lambda a, b: str(b // a) if a != 0 else None
    ops["mod_ab"] = lambda a, b: str(a % b) if b != 0 else None
    ops["mod_ba"] = lambda a, b: str(b % a) if a != 0 else None
    ops["xor"] = lambda a, b: str(a ^ b)
    ops["band"] = lambda a, b: str(a & b)
    ops["bor"] = lambda a, b: str(a | b)
    ops["concat_ab"] = lambda a, b: str(a) + str(b)
    ops["concat_ba"] = lambda a, b: str(b) + str(a)
    ops["add_p1"] = lambda a, b: str(a + b + 1)
    ops["add_m1"] = lambda a, b: str(a + b - 1)
    ops["a_sq_p_b"] = lambda a, b: str(a * a + b)
    ops["a_p_b_sq"] = lambda a, b: str(a + b * b)
    ops["a_sq_m_b"] = lambda a, b: str(a * a - b)
    ops["b_sq_m_a"] = lambda a, b: str(b * b - a)
    ops["a_sq_x_b"] = lambda a, b: str(a * a * b)
    ops["a_x_b_sq"] = lambda a, b: str(a * b * b)
    ops["sum_sq"] = lambda a, b: str(a * a + b * b)
    ops["diff_sq_ab"] = lambda a, b: str(a * a - b * b)
    ops["abs_diff_sq"] = lambda a, b: str(abs(a * a - b * b))
    ops["max_ab"] = lambda a, b: str(max(a, b))
    ops["min_ab"] = lambda a, b: str(min(a, b))
    ops["dsum_add"] = lambda a, b: str(sum(int(d) for d in str(a)) + sum(int(d) for d in str(b)))
    ops["dsum_mul"] = lambda a, b: str(sum(int(d) for d in str(a)) * sum(int(d) for d in str(b)))
    ops["dprod_add"] = lambda a, b: str(_dprod(a) + _dprod(b))
    ops["dprod_mul"] = lambda a, b: str(_dprod(a) * _dprod(b))
    ops["gcd"] = lambda a, b: str(math.gcd(a, b)) if a and b else None
    ops["lcm"] = lambda a, b: str(a * b // math.gcd(a, b)) if math.gcd(a, b) != 0 else None
    ops["rev_a_p_b"] = lambda a, b: str(int(str(a)[::-1]) + b)
    ops["a_p_rev_b"] = lambda a, b: str(a + int(str(b)[::-1]))
    ops["rev_a_x_b"] = lambda a, b: str(int(str(a)[::-1]) * b)
    ops["a_x_rev_b"] = lambda a, b: str(a * int(str(b)[::-1]))
    ops["rev_a_m_b"] = lambda a, b: str(int(str(a)[::-1]) - b)
    ops["a_m_rev_b"] = lambda a, b: str(a - int(str(b)[::-1]))
    ops["neg_sub_ab"] = lambda a, b: str(-(a - b))
    ops["neg_add"] = lambda a, b: str(-(a + b))
    # Digit interleave
    ops["interleave_ab"] = lambda a, b: _interleave(str(a), str(b))
    ops["interleave_ba"] = lambda a, b: _interleave(str(b), str(a))
    return ops

def _dprod(n):
    result = 1
    for d in str(abs(n)):
        result *= int(d)
    return result

def _interleave(s1, s2):
    result = []
    for i in range(max(len(s1), len(s2))):
        if i < len(s1): result.append(s1[i])
        if i < len(s2): result.append(s2[i])
    return "".join(result)

OPERATIONS = _build_ops()

OP_DESCRIPTIONS = {
    "add": "{a} + {b}", "sub_ab": "{a} - {b}", "sub_ba": "{b} - {a}",
    "abs_diff": "|{a} - {b}|", "mul": "{a} × {b}",
    "mul_p1": "{a} × {b} + 1", "mul_m1": "{a} × {b} - 1",
    "div_ab": "{a} ÷ {b}", "mod_ab": "{a} mod {b}",
    "xor": "{a} XOR {b}", "concat_ab": "concat({a}, {b})", "concat_ba": "concat({b}, {a})",
    "a_sq_p_b": "{a}² + {b}", "a_x_b_sq": "{a} × {b}²",
    "sum_sq": "{a}² + {b}²", "abs_diff_sq": "|{a}² - {b}²|",
    "dsum_add": "digitsum({a}) + digitsum({b})",
}


def solve_digit_operator_puzzle(examples, query):
    """Solve by filtering examples to query's operator, then trying all operations."""
    query_parsed = parse_digit_operator(query)
    if query_parsed is None:
        return None
    qa, query_op, qb = query_parsed
    
    # Filter examples to those with the same operator
    matching_examples = []
    for inp, out in examples:
        parsed = parse_digit_operator(inp)
        if parsed and parsed[1] == query_op:
            matching_examples.append((parsed[0], parsed[2], out))
    
    # If no matching examples, try ALL examples (single-op puzzle)
    if not matching_examples:
        for inp, out in examples:
            parsed = parse_digit_operator(inp)
            if parsed:
                matching_examples.append((parsed[0], parsed[2], out))
    
    if not matching_examples:
        return None
    
    # Try each named operation
    for op_name, op_func in OPERATIONS.items():
        all_match = True
        for a, b, expected in matching_examples:
            try:
                result = op_func(a, b)
                if result is None or result != expected:
                    all_match = False
                    break
            except:
                all_match = False
                break
        
        if all_match:
            try:
                answer = op_func(qa, qb)
                if answer is not None:
                    desc = OP_DESCRIPTIONS.get(op_name, op_name)
                    return answer, op_name, desc
            except:
                continue
    
    return None


def solve_symbol_puzzle(examples, query):
    """Try positional extraction and per-position character mapping."""
    if len(examples) < 3 or not query:
        return None
    
    first_in_len = len(examples[0][0])
    if not all(len(e[0]) == first_in_len for e in examples):
        return None
    if len(query) != first_in_len:
        return None
    
    n = first_in_len
    if n > 8:
        return None
    
    # Try positional extraction
    from itertools import combinations
    for r in range(1, n):
        for positions in combinations(range(n), r):
            if all("".join(inp[p] for p in positions) == out for inp, out in examples):
                if all(p < len(query) for p in positions):
                    result = "".join(query[p] for p in positions)
                    return result, "pos_extract", f"Extract positions {list(positions)}"
    
    # Try per-position char mapping with fixed output length
    out_lens = set(len(e[1]) for e in examples)
    if len(out_lens) != 1:
        return None
    out_len = out_lens.pop()
    
    position_maps = []
    for out_pos in range(out_len):
        best_map = None
        for in_pos in range(n):
            char_map = {}
            ok = True
            for inp, out in examples:
                ic, oc = inp[in_pos], out[out_pos]
                if ic in char_map:
                    if char_map[ic] != oc:
                        ok = False
                        break
                else:
                    char_map[ic] = oc
            if ok:
                # Verify query char is in map
                if query[in_pos] in char_map:
                    best_map = (in_pos, char_map)
                    break
        if best_map is None:
            return None
        position_maps.append(best_map)
    
    # Apply maps
    result_chars = []
    for in_pos, char_map in position_maps:
        qc = query[in_pos]
        if qc not in char_map:
            return None
        result_chars.append(char_map[qc])
    
    return "".join(result_chars), "char_map", "Per-position character mapping"


def solve_algebra(prompt: str):
    """Solve an algebra puzzle. Returns (answer, op_name, desc) or None."""
    examples, query = parse_algebra_prompt(prompt)
    if not examples or not query:
        return None
    
    # Try digit-operator format
    query_parsed = parse_digit_operator(query)
    if query_parsed is not None:
        result = solve_digit_operator_puzzle(examples, query)
        if result:
            return result
    
    # Try symbol puzzle
    result = solve_symbol_puzzle(examples, query)
    if result:
        return result
    
    return None


def verify_algebra(prompt: str, expected_answer: str) -> dict:
    """Verify an algebra puzzle answer."""
    result = solve_algebra(prompt)
    expected = expected_answer.strip()
    if result is None:
        return {"status": "unverified", "computed": None, "expected": expected}
    computed, op_name, desc = result
    if computed == expected:
        return {"status": "verified", "computed": computed, "expected": expected,
                "op_name": op_name, "description": desc}
    else:
        return {"status": "mismatch", "computed": computed, "expected": expected,
                "op_name": op_name, "description": desc}


if __name__ == "__main__":
    import json
    
    correct = 0
    incorrect = 0
    unsolved = 0
    total = 0
    op_stats = {}
    
    with open("/scratch2/atang/competitions/nemotron-kaggle/data/sft_train.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r.get("category") != "algebra":
                continue
            total += 1
            prompt = r["messages"][0]["content"]
            ans_text = r["messages"][1]["content"]
            m = re.findall(r"\\boxed\{([^}]+)\}", ans_text)
            expected = m[-1] if m else None
            if expected is None:
                unsolved += 1
                continue
            
            result = solve_algebra(prompt)
            if result is None:
                unsolved += 1
            elif result[0] == expected:
                correct += 1
                op_stats[result[1]] = op_stats.get(result[1], 0) + 1
            else:
                incorrect += 1
                if incorrect <= 3:
                    print(f"  MISMATCH: expected={expected}, got={result[0]} ({result[1]})")
    
    print(f"\nTotal: {total}")
    print(f"Correct: {correct} ({100*correct/total:.1f}%)")
    print(f"Incorrect: {incorrect} ({100*incorrect/total:.1f}%)")
    print(f"Unsolved: {unsolved} ({100*unsolved/total:.1f}%)")
    print(f"\nOperations found:")
    for op, count in sorted(op_stats.items(), key=lambda x: -x[1]):
        print(f"  {op}: {count}")
