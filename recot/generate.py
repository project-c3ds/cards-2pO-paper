"""Generate Reverse Engineered Chain-of-Thought (RECoT) training data.

Usage:
    python -m recot.generate --input data/ft/combined_ft_data.csv --model 1
    python -m recot.generate --input data.csv --model 1 --max 10
    python -m recot.generate --input data.csv --model 1 3 --concurrency 10
"""

import argparse
import ast
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import pandas as pd
from tqdm import tqdm

from config import ConfigManager, ModelConfig
from models import ModelClient
from prompts import system_instruction, cot_trigger, recot_trigger, hard_negative_recot_trigger


def build_messages(text: str, true_labels: list, confused_category: str = None) -> list:
    """Build RECoT messages: same system instruction as inference + text with true labels + recot trigger."""
    if confused_category:
        trigger = hard_negative_recot_trigger.format(confused_category=confused_category)
    else:
        trigger = recot_trigger
    return [
        {"role": "system", "content": [
            {
                "type": "text",
                "text": system_instruction,
                "cache_control": {"type": "ephemeral"},
            }
        ]},
        {"role": "user", "content": f"### Text:\n{text}\n\n### True Labels:\n{true_labels}\n\n{trigger}"},
    ]


def generate_one(client: ModelClient, model_config: ModelConfig, text: str, true_labels: list, confused_category: str = None) -> dict:
    """Generate RECoT reasoning for a single example. Returns raw API response dict."""
    messages = build_messages(text, true_labels, confused_category)
    return client.get_model_response(messages, model_config, cot_trigger)


def parse_labels(val):
    """Parse a stringified list from CSV, handling smart quotes."""
    if not isinstance(val, str):
        return []
    val = val.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    try:
        return ast.literal_eval(val)
    except Exception:
        return [val]


def load_data(path: str, text_col: str = "text", labels_col: str = "true_claims") -> list[dict]:
    """Load CSV or JSONL and return list of dicts with text and parsed labels."""
    if path.endswith('.jsonl'):
        df = pd.read_json(path, lines=True)
    else:
        df = pd.read_csv(path)
        df[labels_col] = df[labels_col].apply(parse_labels)
    return df.to_dict("records")


def run_batch(
    data: list[dict],
    model_configs: list[ModelConfig],
    output_path: str,
    text_col: str = "text",
    labels_col: str = "true_claims",
    concurrency: int = 5,
):
    """Process all examples across all models, writing results to JSONL. Resumes from existing output."""
    client = ModelClient()

    # Check already processed texts for resume support
    already_done = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    already_done.add(row["text"][:200])  # use first 200 chars as key
                except Exception:
                    pass
    if already_done:
        print(f"Resuming — {len(already_done)} already processed, skipping them")

    remaining = [row for row in data if row[text_col][:200] not in already_done]
    total = len(remaining) * len(model_configs)
    if total == 0:
        print("All examples already processed. Nothing to do.")
        return

    print(f"Processing {len(remaining)} remaining examples ({len(data) - len(remaining)} skipped)")

    with open(output_path, "a") as f, ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {}
        for row in remaining:
            for mc in model_configs:
                confused = row.get("source_category")
                fut = pool.submit(generate_one, client, mc, row[text_col], row[labels_col], confused)
                futures[fut] = (row, mc.name)

        for fut in tqdm(as_completed(futures), total=total, desc="RECoT"):
            row, model_name = futures[fut]
            try:
                result = fut.result()
                out = {
                    "text": row[text_col],
                    "true_claims": row[labels_col],
                    "model": model_name,
                    "response": result["response"],
                    "usage": result["usage"],
                }
                if row.get("source_category"):
                    out["source_category"] = row["source_category"]
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                f.flush()
            except Exception as e:
                print(f"Error [{model_name}]: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate RECoT training data")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--models", nargs="+", type=int, default=[1], help="Model IDs from models.json")
    parser.add_argument("--output", help="Output JSONL path (default: data/ft/<date>/recot.jsonl)")
    parser.add_argument("--max", type=int, help="Limit number of examples")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--labels_col", default="true_claims")
    args = parser.parse_args()

    # Load models
    config = ConfigManager()
    all_models = config.get_default_model_configs()
    model_configs = [m for m in all_models if m.id in args.models]
    if not model_configs:
        print(f"No models found with IDs {args.models}. Use --models with valid IDs from models.json.")
        return

    # Load data
    data = load_data(args.input, args.text_col, args.labels_col)
    if args.max:
        data = data[: args.max]

    # Output path
    if args.output:
        output_path = args.output
    else:
        run_dir = os.path.join("data/ft", date.today().isoformat())
        os.makedirs(run_dir, exist_ok=True)
        output_path = os.path.join(run_dir, "recot.jsonl")

    print(f"Processing {len(data)} examples x {len(model_configs)} models -> {output_path}")
    run_batch(data, model_configs, output_path, args.text_col, args.labels_col, args.concurrency)
    print(f"Done. Output: {output_path}")


if __name__ == "__main__":
    main()
