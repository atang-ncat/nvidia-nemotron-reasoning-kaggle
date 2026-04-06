#!/usr/bin/env python3
"""
Local Evaluation Script
=======================
Mimics the Kaggle evaluation setup: loads the base model with vLLM,
attaches a LoRA adapter, runs inference, and computes per-category accuracy.
"""

import sys
import os
import json
import re
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def extract_boxed(text: str) -> str:
    """Extract answer from \\boxed{...} format, matching Kaggle's logic.
    
    Uses brace-depth counting to handle nested braces correctly.
    E.g., \\boxed{f(x)} or \\boxed{a{b}c} are parsed properly.
    """
    # Find the LAST \\boxed{ occurrence (model may produce multiple)
    idx = text.rfind("\\boxed{")
    if idx != -1:
        start = idx + len("\\boxed{")
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:
            return text[start:i-1].strip()

    # Try Result: pattern (our CoT template uses this)
    result_match = re.search(r"Result:\s*(.+)$", text, re.MULTILINE)
    if result_match:
        return result_match.group(1).strip()

    # Fallback: last non-empty, non-code, non-comment line
    lines = text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("`") and len(line) < 100:
            return line

    return text.strip()


def grade_answer(predicted: str, expected: str) -> bool:
    """Grade a prediction against expected answer (Kaggle-compatible)."""
    pred = predicted.strip()
    exp = expected.strip()

    # Exact string match
    if pred == exp:
        return True

    # Numeric tolerance (rel tolerance 1e-2)
    try:
        pred_f = float(pred)
        exp_f = float(exp)
        if exp_f == 0:
            return abs(pred_f) < 1e-2
        return abs(pred_f - exp_f) / abs(exp_f) < 1e-2
    except (ValueError, ZeroDivisionError):
        pass

    # Case-insensitive string match
    if pred.lower() == exp.lower():
        return True

    return False


def main():
    parser = argparse.ArgumentParser(description="Local evaluation with vLLM")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to LoRA adapter directory (None for zero-shot)")
    parser.add_argument("--data", type=str,
                        default="/scratch2/atang/competitions/nemotron-kaggle/data/sft_val.jsonl",
                        help="Path to evaluation data (JSONL)")
    parser.add_argument("--model", type=str,
                        default="/scratch2/atang/competitions/nemotron-kaggle/models/nemotron-base",
                        help="Path to base model")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to evaluate")
    args = parser.parse_args()

    print("=" * 60)
    print("LOCAL EVALUATION")
    print("=" * 60)
    print(f"  Model:   {args.model}")
    print(f"  Adapter: {args.adapter or 'None (zero-shot)'}")
    print(f"  Data:    {args.data}")

    # Load evaluation data
    eval_data = []
    with open(args.data) as f:
        for line in f:
            record = json.loads(line)
            eval_data.append(record)

    if args.max_samples:
        eval_data = eval_data[:args.max_samples]

    print(f"  Samples: {len(eval_data)}")

    # Load model with vLLM
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    engine_args = {
        "model": args.model,
        "trust_remote_code": True,
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.7,
        "max_num_seqs": 32,
        "dtype": "bfloat16",
        "tensor_parallel_size": 4,
    }

    if args.adapter:
        engine_args["enable_lora"] = True
        engine_args["max_lora_rank"] = 32

    print("\n🧠 Loading model with vLLM...")
    llm = LLM(**engine_args)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=7680,
    )

    # Prepare prompts — wrap in training format
    prompts = []
    expected_answers = []
    categories = []

    for record in eval_data:
        msgs = record["messages"]
        raw_prompt = msgs[0]["content"]
        # Wrap in the same format we trained with
        formatted_prompt = "### Question:\n" + raw_prompt + "\n\n### Answer:\n"
        expected = extract_boxed(msgs[1]["content"])
        prompts.append(formatted_prompt)
        expected_answers.append(expected)
        categories.append(record.get("category", "unknown"))

    # Run inference
    print("🚀 Running inference...")
    lora_req = LoRARequest("adapter", 1, args.adapter) if args.adapter else None

    if lora_req:
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_req)
    else:
        outputs = llm.generate(prompts, sampling_params)

    # Grade results
    print("\n📊 RESULTS")
    print("-" * 60)

    correct = 0
    total = 0
    per_category = {}

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        predicted = extract_boxed(generated_text)
        expected = expected_answers[i]
        cat = categories[i]

        is_correct = grade_answer(predicted, expected)
        correct += is_correct
        total += 1

        if cat not in per_category:
            per_category[cat] = {"correct": 0, "total": 0}
        per_category[cat]["total"] += 1
        if is_correct:
            per_category[cat]["correct"] += 1

    # Print per-category accuracy
    print(f"\n{'Category':<20} {'Correct':>8} {'Total':>6} {'Accuracy':>9}")
    print("-" * 45)
    for cat in sorted(per_category.keys()):
        c = per_category[cat]["correct"]
        t = per_category[cat]["total"]
        acc = c / t if t > 0 else 0
        print(f"  {cat:<18} {c:>8} {t:>6} {acc:>8.1%}")

    print("-" * 45)
    overall_acc = correct / total if total > 0 else 0
    print(f"  {'OVERALL':<18} {correct:>8} {total:>6} {overall_acc:>8.1%}")

    # Print sample errors
    print("\n❌ SAMPLE ERRORS (first 5):")
    error_count = 0
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        predicted = extract_boxed(generated_text)
        expected = expected_answers[i]
        if not grade_answer(predicted, expected) and error_count < 5:
            print(f"  [{categories[i]}] Expected: {expected}, Got: {predicted}")
            error_count += 1

    # Save per-example predictions for error analysis
    adapter_name = os.path.basename(os.path.dirname(args.adapter)) if args.adapter else "zero_shot"
    predictions_path = os.path.join(
        os.path.dirname(args.adapter) if args.adapter else "outputs",
        "eval_predictions.jsonl"
    )
    with open(predictions_path, "w") as f:
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            predicted = extract_boxed(generated_text)
            expected = expected_answers[i]
            record = {
                "id": eval_data[i].get("id", f"sample_{i}"),
                "category": categories[i],
                "expected": expected,
                "predicted": predicted,
                "correct": grade_answer(predicted, expected),
                "source": eval_data[i].get("source", "unknown"),
                "generated_text": generated_text[:2000],  # truncate to save space
            }
            f.write(json.dumps(record) + "\n")
    print(f"\n💾 Per-example predictions saved to: {predictions_path}")


if __name__ == "__main__":
    main()
