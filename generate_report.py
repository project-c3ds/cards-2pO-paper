"""
Generate metrics_summary.md and metrics_summary.json from all result JSONL files.

Usage:
    python generate_report.py
"""

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
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "data", "results", "val")
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
        "file": "cards_qwen35_4b_recot_full_hn.jsonl",
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
])

MIN_SUPPORT_THRESHOLDS = [0, 3]  # 0 = all labels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_response(response):
    if '</think>' in response:
        after_think = response.split('</think>')[-1].strip()
    else:
        after_think = response
    cat_match = re.search(r'categories:\s*\n((?:\s*-\s*.+\n?)+)', after_think)
    if cat_match:
        return re.findall(r'-\s*([\d_]+)', cat_match.group(1))
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
        path = os.path.join(RESULTS_DIR, cfg["file"])
        if not os.path.exists(path):
            print(f"  Skipping {key}: {cfg['file']} not found")
            continue

        df = pd.read_json(path, lines=True)
        # Reparse predictions from response if pred_claims missing or empty
        if 'pred_claims' not in df.columns:
            df['pred_claims'] = df['response'].map(parse_response)

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
    lines = ["# CARDS Qwen3.5 — Full Model Comparison\n"]

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
