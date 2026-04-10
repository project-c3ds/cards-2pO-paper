"""
Generate metrics_summary.md and metrics_summary.json from all result JSONL files.

Usage:
    python generate_report.py              # default: val split
    python generate_report.py --split test
    python generate_report.py --split val
"""

import argparse
import json
import os
import re
from collections import OrderedDict

import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, hamming_loss, classification_report,
)
from sklearn.preprocessing import MultiLabelBinarizer

# ---------------------------------------------------------------------------
# Config: variant key -> (jsonl filename, display label, training data desc)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--split", default="val", choices=["val", "test"])
args = parser.parse_args()

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results", args.split)
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")

VARIANTS = OrderedDict([
    ("08b_recot", {
        "file": "cards_qwen35_08b_recot_full.jsonl",
        "label": "Qwen3.5-0.8B RECoT",
        "training_data": "1791",
    }),
    ("2b_recot", {
        "file": "cards_qwen35_2b_recot_full.jsonl",
        "label": "Qwen3.5-2B RECoT",
        "training_data": "1791",
    }),
    ("4b_norecot", {
        "file": "cards_qwen35_4b_norecot_full.jsonl",
        "label": "Qwen3.5-4B No RECoT",
        "training_data": "1791 norecot",
    }),
    ("4b_recot", {
        "file": "cards_qwen35_4b_recot_full.jsonl",
        "label": "Qwen3.5-4B RECoT",
        "training_data": "1791",
    }),
    ("4b_recot_hn", {
        "file": ["cards_qwen35_4b_hn_recot_full.jsonl", "cards_qwen35_4b_recot_full_hn.jsonl"],
        "label": "Qwen3.5-4B RECoT+HN",
        "training_data": "1791+542 HN",
    }),
    ("9b_recot", {
        "file": "cards_qwen35_9b_recot_full.jsonl",
        "label": "Qwen3.5-9B RECoT",
        "training_data": "1791",
    }),
    ("27b_recot", {
        "file": "cards_qwen35_27b_recot_full.jsonl",
        "label": "Qwen3.5-27B RECoT",
        "training_data": "1791",
    }),
    ("gpt4o_mini", {
        "file": "gpt-4o-mini.jsonl",
        "label": "GPT-4o-mini",
        "training_data": "—",
    }),
    ("cards_mini_opus", {
        "file": "cards-mini-opus.jsonl",
        "label": "CARDS-mini-opus",
        "training_data": "—",
    }),
    ("cards_mini_sonnet", {
        "file": "cards-mini-sonnet.jsonl",
        "label": "CARDS-mini-sonnet",
        "training_data": "—",
    }),
    ("08b_base", {
        "file": "qwen35-08b.jsonl",
        "label": "Qwen3.5-0.8B Base",
        "training_data": "—",
    }),
    ("2b_base", {
        "file": "qwen35-2b.jsonl",
        "label": "Qwen3.5-2B Base",
        "training_data": "—",
    }),
    ("4b_base", {
        "file": "qwen35-4b.jsonl",
        "label": "Qwen3.5-4B Base",
        "training_data": "—",
    }),
    ("9b_base", {
        "file": "qwen35-9b.jsonl",
        "label": "Qwen3.5-9B Base",
        "training_data": "—",
    }),
    ("27b_base", {
        "file": "qwen35-27b.jsonl",
        "label": "Qwen3.5-27B Base",
        "training_data": "—",
    }),
    ("gemma4_4b_base", {
        "file": "gemma4-4b.jsonl",
        "label": "Gemma4-E4B Base",
        "training_data": "—",
    }),
    ("gemma4_31b_base", {
        "file": "gemma4-31b.jsonl",
        "label": "Gemma4-31B Base",
        "training_data": "—",
    }),
    ("gpt54_nano", {
        "file": "gpt-5-4-nano.jsonl",
        "label": "GPT-5.4-nano",
        "training_data": "—",
    }),
    ("gpt54_mini", {
        "file": "gpt-5-4-mini.jsonl",
        "label": "GPT-5.4-mini",
        "training_data": "—",
    }),
    ("gpt54", {
        "file": "gpt-5-4.jsonl",
        "label": "GPT-5.4",
        "training_data": "—",
    }),
    ("gemini31_pro", {
        "file": "gemini-3-1-pro.jsonl",
        "label": "Gemini 3.1 Pro",
        "training_data": "—",
    }),
    ("gemini25_flash_lite", {
        "file": "gemini-2-5-flash-lite.jsonl",
        "label": "Gemini 2.5 Flash Lite",
        "training_data": "—",
    }),
    ("gemini25_flash", {
        "file": "gemini-2-5-flash.jsonl",
        "label": "Gemini 2.5 Flash",
        "training_data": "—",
    }),
    ("gemini31_flash_lite", {
        "file": "gemini-3-1-flash-lite.jsonl",
        "label": "Gemini 3.1 Flash Lite",
        "training_data": "—",
    }),
    ("claude_sonnet_46", {
        "file": "claude-sonnet-4-6.jsonl",
        "label": "Claude Sonnet 4.6",
        "training_data": "—",
    }),
    ("claude_opus_46", {
        "file": "claude-opus-4-6.jsonl",
        "label": "Claude Opus 4.6",
        "training_data": "—",
    }),
    ("cards_flash_opus", {
        "file": "cards-flash-opus.jsonl",
        "label": "CARDS-flash-opus",
        "training_data": "—",
    }),
    ("cards_flash_lite_opus", {
        "file": "cards-flash-lite-opus.jsonl",
        "label": "CARDS-flash-lite-opus",
        "training_data": "—",
    }),
    ("gemma4_e4b_recot", {
        "file": "cards-gemma4-e4b.jsonl",
        "label": "Gemma4-E4B RECoT",
        "training_data": "1791",
    }),
    ("gemma4_31b_recot", {
        "file": "cards-gemma4-31b.jsonl",
        "label": "Gemma4-31B RECoT",
        "training_data": "1791",
    }),
    # Quantized (Q4_K_M GGUF) variants
    ("4b_quant", {
        "file": "quants/cards_qwen_35_4b.jsonl",
        "label": "Qwen3.5-4B Q4_K_M",
        "training_data": "1791 (quant)",
    }),
    ("9b_quant", {
        "file": "quants/cards_qwen_35_9b.jsonl",
        "label": "Qwen3.5-9B Q4_K_M",
        "training_data": "1791 (quant)",
    }),
    ("27b_quant", {
        "file": "quants/cards_qwen_35_27b.jsonl",
        "label": "Qwen3.5-27B Q4_K_M",
        "training_data": "1791 (quant)",
    }),
])

MIN_SUPPORT_THRESHOLDS = [0, 3]  # 0 = all labels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_response(response):
    if not isinstance(response, str):
        return []
    if '</think>' in response:
        after_think = response.split('</think>')[-1].strip()
    else:
        after_think = response
    cat_match = re.search(r'categories:\s*\n((?:\s*-\s*.+\n?)+)', after_think)
    if cat_match:
        # Handle both <0_0_0> and 0_0_0 formats
        return re.findall(r'-\s*<?(\d[\d_]+\d)>?', cat_match.group(1))
    return []


def get_level_code(code, level=3):
    return '_'.join(code.split('_')[:level])


def compute_metrics(df, level=3, min_support=0):
    y_true = df['true_claims'].apply(
        lambda x: list(set(get_level_code(c, level) for c in x))).tolist()
    y_pred = df['pred_claims'].apply(
        lambda x: list(set(get_level_code(c, level) for c in x))).tolist()

    mlb = MultiLabelBinarizer()
    all_labels = set()
    for labels in y_true + y_pred:
        all_labels.update(labels)
    mlb.fit([sorted(all_labels)])

    y_true_bin = mlb.transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    if min_support > 0:
        label_support = y_true_bin.sum(axis=0)
        mask = label_support >= min_support
        y_true_bin = y_true_bin[:, mask]
        y_pred_bin = y_pred_bin[:, mask]

    return {
        'samples_f1': round(f1_score(y_true_bin, y_pred_bin, average='samples', zero_division=0), 3),
        'macro_f1': round(f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0), 3),
        'micro_f1': round(f1_score(y_true_bin, y_pred_bin, average='micro', zero_division=0), 3),
        'micro_precision': round(precision_score(y_true_bin, y_pred_bin, average='micro', zero_division=0), 3),
        'micro_recall': round(recall_score(y_true_bin, y_pred_bin, average='micro', zero_division=0), 3),
        'accuracy': round(accuracy_score(y_true_bin, y_pred_bin), 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    summary = OrderedDict()

    for key, cfg in VARIANTS.items():
        filenames = cfg["file"] if isinstance(cfg["file"], list) else [cfg["file"]]
        path = None
        for fname in filenames:
            candidate = os.path.join(RESULTS_DIR, fname)
            if os.path.exists(candidate):
                path = candidate
                break
        if path is None:
            print(f"  Skipping {key}: none of {filenames} found")
            continue

        df = pd.read_json(path, lines=True)
        # Always reparse predictions from response
        def extract_preds(row):
            resp = row.get('response', '')
            if isinstance(resp, list):
                # Base models store predictions as list directly
                return [re.match(r'<?(\d[\d_]+\d)>?', str(c).strip()).group(1)
                        for c in resp if re.match(r'<?(\d[\d_]+\d)', str(c).strip())]
            return parse_response(resp)
        df['pred_claims'] = df.apply(extract_preds, axis=1)

        parse_failures = (df['pred_claims'].map(len) == 0).sum()
        print(f"  {cfg['label']}: {len(df)} samples, {parse_failures} parse failures")

        entry = {"label": cfg["label"], "training_data": cfg["training_data"],
                 "n_samples": len(df), "parse_failures": int(parse_failures)}

        for min_sup in MIN_SUPPORT_THRESHOLDS:
            for level in [1, 2, 3]:
                suffix = "all" if min_sup == 0 else f"minsup_{min_sup}"
                metrics = compute_metrics(df, level=level, min_support=min_sup)
                entry[f"level_{level}_{suffix}"] = metrics

        # Per-variant classification reports
        variant_dir = os.path.join(REPORTS_DIR, key)
        os.makedirs(variant_dir, exist_ok=True)
        for min_sup in MIN_SUPPORT_THRESHOLDS:
            for level in [1, 2, 3]:
                suffix = "all" if min_sup == 0 else f"minsup_{min_sup}"
                y_true = df['true_claims'].apply(
                    lambda x: list(set(get_level_code(c, level) for c in x))).tolist()
                y_pred = df['pred_claims'].apply(
                    lambda x: list(set(get_level_code(c, level) for c in x))).tolist()
                mlb = MultiLabelBinarizer()
                all_labels = set()
                for labels in y_true + y_pred:
                    all_labels.update(labels)
                mlb.fit([sorted(all_labels)])
                y_true_bin = mlb.transform(y_true)
                y_pred_bin = mlb.transform(y_pred)
                classes = list(mlb.classes_)
                if min_sup > 0:
                    label_support = y_true_bin.sum(axis=0)
                    mask = label_support >= min_sup
                    y_true_bin = y_true_bin[:, mask]
                    y_pred_bin = y_pred_bin[:, mask]
                    classes = [c for c, k in zip(classes, mask) if k]
                report = classification_report(
                    y_true_bin, y_pred_bin, target_names=classes, zero_division=0)
                fname = f"level_{level}_{suffix}.txt"
                with open(os.path.join(variant_dir, fname), "w") as f:
                    f.write(f"Classification Report — Level {level} (min_support={min_sup})\n")
                    f.write(f"Variant: {cfg['label']}\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(report)

        summary[key] = entry

    # Save JSON
    with open(os.path.join(REPORTS_DIR, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Generate Markdown
    lines = [f"# CARDS Qwen3.5 — Full Model Comparison ({args.split} set)\n"]

    # Overview table
    lines.append("| Variant | Training Data | Parse Failures |")
    lines.append("|---------|--------------|----------------|")
    for entry in summary.values():
        lines.append(f"| {entry['label']} | {entry['training_data']} | "
                     f"{entry['parse_failures']}/{entry['n_samples']} |")

    # Metrics tables
    for min_sup in MIN_SUPPORT_THRESHOLDS:
        suffix = "all" if min_sup == 0 else f"minsup_{min_sup}"
        header = "All Labels" if min_sup == 0 else f"Support >= {min_sup}"
        lines.append(f"\n## {header}\n")
        lines.append("| Level | Variant | Samples F1 | Macro F1 | Micro F1 | Precision | Recall | EM |")
        lines.append("|-------|---------|------------|----------|----------|-----------|--------|----|")
        for level in [1, 2, 3]:
            for entry in summary.values():
                key = f"level_{level}_{suffix}"
                m = entry.get(key, {})
                if not m:
                    continue
                lines.append(
                    f"| {level} | {entry['label']} | {m['samples_f1']} | {m['macro_f1']} | "
                    f"{m['micro_f1']} | {m['micro_precision']} | {m['micro_recall']} | {m['accuracy']} |"
                )

    lines.append("")
    md = "\n".join(lines)

    with open(os.path.join(REPORTS_DIR, "metrics_summary.md"), "w") as f:
        f.write(md)

    print(f"\nSaved: {REPORTS_DIR}/metrics_summary.json")
    print(f"Saved: {REPORTS_DIR}/metrics_summary.md")


if __name__ == "__main__":
    main()
