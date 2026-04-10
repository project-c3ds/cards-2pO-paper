"""
Statistical significance tests between two model result files.

Runs paired bootstrap test (samples F1) and McNemar's test (exact match)
at all hierarchy levels.

Usage:
    python data/results/significance_test.py data/results/file_a.jsonl data/results/file_b.jsonl
    python data/results/significance_test.py file_a.jsonl file_b.jsonl --n-bootstrap 10000
"""

import argparse
import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from statsmodels.stats.contingency_tables import mcnemar


def parse_response(response):
    if not isinstance(response, str):
        return []
    if '</think>' in response:
        after_think = response.split('</think>')[-1].strip()
    else:
        after_think = response
    cat_match = re.search(r'categories:\s*\n((?:\s*-\s*.+\n?)+)', after_think)
    if cat_match:
        return re.findall(r'-\s*<?(\d[\d_]+\d)>?', cat_match.group(1))
    return []


def extract_preds(row):
    resp = row.get('response', '')
    if isinstance(resp, list):
        return [re.match(r'<?(\d[\d_]+\d)>?', str(c).strip()).group(1)
                for c in resp if re.match(r'<?(\d[\d_]+\d)', str(c).strip())]
    return parse_response(resp)


def get_level_code(code, level=3):
    return '_'.join(code.split('_')[:level])


def compute_samples_f1(records, level=3):
    y_true = [list(set(get_level_code(c, level) for c in r['true_claims'])) for r in records]
    y_pred = [list(set(get_level_code(c, level) for c in r['pred_claims'])) for r in records]
    mlb = MultiLabelBinarizer()
    all_labels = set()
    for labels in y_true + y_pred:
        all_labels.update(labels)
    mlb.fit([sorted(all_labels)])
    return f1_score(mlb.transform(y_true), mlb.transform(y_pred), average='samples', zero_division=0)


def per_sample_correct(records, level=3):
    correct = []
    for r in records:
        true_set = set(get_level_code(c, level) for c in r['true_claims'])
        pred_set = set(get_level_code(c, level) for c in r['pred_claims'])
        correct.append(true_set == pred_set)
    return correct


def main():
    parser = argparse.ArgumentParser(description="Statistical significance tests between two model results")
    parser.add_argument("file_a", help="First results JSONL (baseline)")
    parser.add_argument("file_b", help="Second results JSONL (comparison)")
    parser.add_argument("--n-bootstrap", type=int, default=10000, help="Number of bootstrap iterations (default: 10000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load data and reparse predictions from response
    with open(args.file_a) as f:
        records_a = [json.loads(line) for line in f]
    for r in records_a:
        r['pred_claims'] = extract_preds(r)
    with open(args.file_b) as f:
        records_b = [json.loads(line) for line in f]
    for r in records_b:
        r['pred_claims'] = extract_preds(r)

    name_a = os.path.basename(args.file_a).replace('.jsonl', '')
    name_b = os.path.basename(args.file_b).replace('.jsonl', '')

    assert len(records_a) == len(records_b), f"Files have different lengths: {len(records_a)} vs {len(records_b)}"

    n = len(records_a)
    np.random.seed(args.seed)

    print(f"{'='*70}")
    print(f"  Statistical Significance Tests")
    print(f"  A: {name_a} ({n} samples)")
    print(f"  B: {name_b} ({n} samples)")
    print(f"  Bootstrap iterations: {args.n_bootstrap}")
    print(f"{'='*70}")

    # Paired Bootstrap Test
    print(f"\n{'='*70}")
    print(f"  PAIRED BOOTSTRAP TEST (Samples F1)")
    print(f"{'='*70}")

    for level in [1, 2, 3]:
        f1_a = compute_samples_f1(records_a, level)
        f1_b = compute_samples_f1(records_b, level)
        observed_diff = f1_b - f1_a

        diffs = []
        for _ in range(args.n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            sample_a = [records_a[i] for i in idx]
            sample_b = [records_b[i] for i in idx]
            diff = compute_samples_f1(sample_b, level) - compute_samples_f1(sample_a, level)
            diffs.append(diff)

        diffs = np.array(diffs)
        ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
        p_value = np.mean(diffs <= 0)

        sig = "YES" if p_value < 0.05 else "NO"
        print(f"\n  Level {level}:")
        print(f"    A ({name_a}): {f1_a:.3f}")
        print(f"    B ({name_b}): {f1_b:.3f}")
        print(f"    Diff (B-A): {observed_diff:+.3f}")
        print(f"    95% CI: [{ci_low:+.4f}, {ci_high:+.4f}]")
        print(f"    p-value (one-sided): {p_value:.4f}")
        print(f"    Significant (p<0.05): {sig}")

    # McNemar's Test
    print(f"\n{'='*70}")
    print(f"  McNEMAR'S TEST (Exact Match)")
    print(f"{'='*70}")

    for level in [1, 2, 3]:
        correct_a = per_sample_correct(records_a, level)
        correct_b = per_sample_correct(records_b, level)

        a = sum(1 for ca, cb in zip(correct_a, correct_b) if ca and cb)
        b = sum(1 for ca, cb in zip(correct_a, correct_b) if not ca and cb)  # B right, A wrong
        c = sum(1 for ca, cb in zip(correct_a, correct_b) if ca and not cb)  # A right, B wrong
        d = sum(1 for ca, cb in zip(correct_a, correct_b) if not ca and not cb)

        table = np.array([[a, b], [c, d]])
        result = mcnemar(table, exact=True)

        direction = "B better" if b > c else "A better" if c > b else "equal"
        sig = "YES" if result.pvalue < 0.05 else "NO"

        print(f"\n  Level {level}:")
        print(f"    Both correct: {a}")
        print(f"    Only B correct: {b} ({name_b})")
        print(f"    Only A correct: {c} ({name_a})")
        print(f"    Both wrong: {d}")
        print(f"    Direction: {direction} ({b} vs {c})")
        print(f"    McNemar p-value: {result.pvalue:.4f}")
        print(f"    Significant (p<0.05): {sig}")


if __name__ == "__main__":
    main()
