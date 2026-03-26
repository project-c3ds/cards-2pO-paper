"""Generate synthetic training examples for rare categories and hard negatives.

Usage:
    python -m recot.synthetic --target 4_1_1_1
    python -m recot.synthetic --target 4_1_1_1 --n 10
    python -m recot.synthetic --all-rare --threshold 5
    python -m recot.synthetic --hard-negatives --n 20 --batches 10 --output data/hard_negatives.jsonl
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from config import ConfigManager, ModelConfig
from models import ModelClient
from prompts import codebook, synthetic_prompt, hard_negative_prompt, system_instruction, cot_trigger


def get_description(target_code: str) -> str:
    """Get the full description for a category code from taxonomy."""
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(f'{current_dir}/data/taxonomy.csv')
    match = df[df['category_number'] == target_code]
    if match.empty:
        return target_code
    return match.iloc[0]['prompt_label']


def get_existing_examples(target_code: str, data_path: str = "data/training_recot_opus.jsonl") -> str:
    """Find all existing training examples that contain the target code."""
    examples = []
    with open(data_path) as f:
        for line in f:
            row = json.loads(line)
            if target_code in str(row['true_claims']):
                examples.append(f"- \"{row['text'][:300]}\" Labels: {row['true_claims']}")
    if not examples:
        return "No existing examples."
    return "\n".join(examples)


def verify_labels(text: str, claimed_labels: list, client: ModelClient, model_config: ModelConfig, min_overlap: float = 0.6) -> bool:
    """Classify text blind and check if labels match. Returns True if overlap >= min_overlap."""
    result = client.get_model_response(
        messages=[{"role": "system", "content": system_instruction}],
        model_config=model_config,
        prompt=f"### Text:\n{text}\n\n{cot_trigger}",
    )
    # Extract codes from response
    after = result["response"].split('</think>')[-1] if '</think>' in result["response"] else result["response"]
    blind_labels = set(re.findall(r'(\d+_\d+(?:_\d+)*)', after))
    claimed_set = set(claimed_labels)

    if not claimed_set:
        return False

    overlap = len(blind_labels & claimed_set) / len(claimed_set)
    return overlap >= min_overlap


def generate_synthetic(target_code: str, n: int = 10, model_id: int = 29) -> list:
    """Generate n synthetic examples for a target category."""
    description = get_description(target_code)
    examples = get_existing_examples(target_code)

    prompt = synthetic_prompt.format(
        codebook=codebook,
        target_code=target_code,
        target_description=description,
        examples=examples,
        n=n,
    )

    client = ModelClient()
    config = ConfigManager()
    model_config = [m for m in config.get_default_model_configs() if m.id == model_id][0]

    print(f"Target: {target_code} ({description})")
    print(f"Existing examples: {examples.count(chr(10)) + 1}")
    print(f"Generating {n} synthetic examples with {model_config.name}...")

    result = client.get_model_response(
        messages=[{"role": "system", "content": prompt}],
        model_config=model_config,
        prompt="Generate the examples now. Return only the JSON array.",
    )

    # Parse JSON from response
    json_match = re.search(r'\[.*\]', result["response"], re.DOTALL)
    if not json_match:
        print("Failed to parse JSON from response")
        print(result["response"][:500])
        return []

    generated = json.loads(json_match.group())

    # Filter: must contain target code
    has_target = [ex for ex in generated if target_code in ex.get('true_claims', [])]
    print(f"Generated {len(generated)}, contain {target_code}: {len(has_target)}")

    # Verify labels with blind classification
    print(f"Verifying labels...")
    verified = []
    for i, ex in enumerate(has_target):
        try:
            passed = verify_labels(ex['text'], ex['true_claims'], client, model_config)
            status = "PASS" if passed else "FAIL"
            print(f"  [{i+1}/{len(has_target)}] {status} — {ex['true_claims']}")
            if passed:
                verified.append(ex)
        except Exception as e:
            print(f"  [{i+1}/{len(has_target)}] ERROR — {e}")

    print(f"Verified: {len(verified)}/{len(has_target)}")
    return verified


def get_false_positives_by_category(results_path: str, multiplier: int = 5) -> dict:
    """Extract false positive errors grouped by predicted category.
    Returns dict of {code: {count, n_generate, description, error_examples}}."""
    from collections import Counter, defaultdict

    with open(results_path) as f:
        records = [json.loads(line) for line in f]

    # Group false positives by predicted code
    fp_by_code = defaultdict(list)
    for r in records:
        if r['true_claims'] == ['0_0_0'] and r['pred_claims'] != ['0_0_0']:
            for code in r['pred_claims']:
                fp_by_code[code].append(r)

    # Get descriptions from taxonomy
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df_tax = pd.read_csv(f'{current_dir}/data/taxonomy.csv')

    categories = {}
    for code, errors in sorted(fp_by_code.items(), key=lambda x: -len(x[1])):
        match = df_tax[df_tax['category_number'] == code]
        desc = match.iloc[0]['prompt_label'] if not match.empty else code
        n_generate = max(len(errors) * multiplier, 10)  # at least 10

        error_examples = "\n\n".join([
            f'{i+1}. "{r["text"][:300]}"\n   Model predicted: {r["pred_claims"]}. True label: 0_0_0.'
            for i, r in enumerate(errors)
        ])

        categories[code] = {
            "count": len(errors),
            "n_generate": n_generate,
            "description": desc,
            "error_examples": error_examples,
        }

    total_fp = sum(c["count"] for c in categories.values())
    total_gen = sum(c["n_generate"] for c in categories.values())
    print(f"Found {total_fp} false positives across {len(categories)} categories")
    print(f"Will generate {total_gen} hard negatives total ({multiplier}x multiplier)")
    for code, info in categories.items():
        print(f"  {code}: {info['count']} FPs → generate {info['n_generate']}")

    return categories


def generate_hard_negatives(n: int = 50, model_id: int = 29, results_path: str = "data/results/cards_qwen35_recot_full.jsonl", multiplier: int = 5, concurrency: int = 10, output_path: str = "data/hard_negatives.jsonl") -> dict:
    """Generate hard negative 0_0_0 examples per confused category (concurrent).
    Returns dict of {code: [examples]}."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    categories = get_false_positives_by_category(results_path, multiplier)

    client = ModelClient()
    config = ConfigManager()
    model_config = [m for m in config.get_default_model_configs() if m.id == model_id][0]

    def generate_for_category(code, info):
        n_gen = info["n_generate"]
        prompt = hard_negative_prompt.format(
            codebook=codebook,
            confused_categories=f"- {code}: {info['description']} (falsely predicted {info['count']} times)",
            error_examples=info["error_examples"],
            n=n_gen,
        )

        try:
            result = client.get_model_response(
                messages=[{"role": "system", "content": prompt}],
                model_config=model_config,
                prompt="Generate the examples now. Return only the JSON array.",
            )

            json_match = re.search(r'\[.*\]', result["response"], re.DOTALL)
            if not json_match:
                return code, []

            # Try parsing, fix common JSON issues
            raw = json_match.group()
            try:
                generated = json.loads(raw)
            except json.JSONDecodeError:
                # Try fixing truncated JSON by removing last incomplete entry
                raw = raw.rsplit('},', 1)[0] + '}]'
                try:
                    generated = json.loads(raw)
                except json.JSONDecodeError:
                    return code, []

            valid = [ex for ex in generated if ex.get('true_claims') == ['0_0_0']]
            return code, valid
        except Exception as e:
            print(f"  {code}: ERROR — {e}")
            return code, []

    # Check which categories already have results (resume support)
    already_done = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                row = json.loads(line)
                if 'source_category' in row:
                    already_done.add(row['source_category'])
    if already_done:
        print(f"  Resuming — skipping {len(already_done)} already completed categories: {already_done}")

    remaining = {code: info for code, info in categories.items() if code not in already_done}
    if not remaining:
        print("  All categories already generated. Nothing to do.")
        return {}

    all_generated = {}
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(generate_for_category, code, info): code
                   for code, info in remaining.items()}
        for future in as_completed(futures):
            code, valid = future.result()
            all_generated[code] = valid
            # Save progress immediately
            with open(output_path, "a") as f:
                for ex in valid:
                    ex["source_category"] = code
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            print(f"  {code}: generated {len(valid)} hard negatives (saved)")

    total = sum(len(v) for v in all_generated.values())
    print(f"\nTotal generated: {total}")
    return all_generated


def get_rare_categories(data_path: str = "data/training_recot_opus.jsonl", threshold: int = 5) -> dict:
    """Find categories with fewer than threshold examples. Returns {code: count}."""
    from collections import Counter
    all_codes = []
    with open(data_path) as f:
        for line in f:
            row = json.loads(line)
            after = row['response'].split('</think>')[-1] if '</think>' in row['response'] else row['response']
            codes = re.findall(r'(\d+_\d+(?:_\d+)*)', after)
            all_codes.extend(codes)
    dist = Counter(all_codes)
    return {k: v for k, v in sorted(dist.items(), key=lambda x: x[1]) if v < threshold and k != '0_0_0'}


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic examples for rare categories")
    parser.add_argument("--target", help="Target category code (e.g., 4_1_1_1)")
    parser.add_argument("--all-rare", action="store_true", help="Generate for all categories with < threshold examples")
    parser.add_argument("--hard-negatives", action="store_true", help="Generate hard negative 0_0_0 examples from model errors")
    parser.add_argument("--results-path", default="data/results/cards_qwen35_recot_full.jsonl",
                        help="Path to model results JSONL (for --hard-negatives)")
    parser.add_argument("--multiplier", type=int, default=5,
                        help="Generate N x false_positive_count examples per category (default: 5)")
    parser.add_argument("--threshold", type=int, default=5, help="Rare category threshold (default: 5)")
    parser.add_argument("--n", type=int, default=20, help="Number of examples per category (for --target/--all-rare)")
    parser.add_argument("--model", type=int, default=29, help="Model ID from models.json")
    parser.add_argument("--output", default="data/synthetic_rare.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    if args.hard_negatives:
        all_generated = generate_hard_negatives(
            model_id=args.model,
            results_path=args.results_path,
            multiplier=args.multiplier,
            output_path=args.output,
        )
        total = sum(len(v) for v in all_generated.values())
        print(f"\nDone. {total} hard negatives → {args.output}")

    elif args.all_rare:
        rare = get_rare_categories(threshold=args.threshold)
        print(f"Found {len(rare)} rare categories (< {args.threshold} examples):")
        for code, count in rare.items():
            print(f"  {code}: {count}")
        print()

        total = 0
        for i, (code, count) in enumerate(rare.items(), 1):
            print(f"\n[{i}/{len(rare)}] {code} (existing: {count})")
            examples = generate_synthetic(code, args.n, args.model)
            with open(args.output, "a") as f:
                for ex in examples:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            total += len(examples)
            print(f"  → {len(examples)} appended ({total} total so far)")

        print(f"\nDone. {total} total examples across {len(rare)} categories → {args.output}")

    elif args.target:
        examples = generate_synthetic(args.target, args.n, args.model)
        with open(args.output, "a") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Appended {len(examples)} examples to {args.output}")

    else:
        parser.error("Provide --target, --all-rare, or --hard-negatives")


if __name__ == "__main__":
    main()
