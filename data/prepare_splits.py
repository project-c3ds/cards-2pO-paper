"""
Prepare train/val/test splits for CARDS fine-tuning and evaluation.

Splits:
    cards_train.jsonl           - SFT messages format, RECoT (90% of training data)
    cards_train_eval.jsonl      - SFT messages format, RECoT (10% of training data, stratified)
    cards_train_norecot.jsonl   - SFT messages format, YAML only — no reasoning (same 90% split)
    cards_train_eval_norecot.jsonl - SFT messages format, YAML only (same 10% split)
    cards_val.jsonl             - {id, text, true_claims}, for iteration/prompt tuning (30% of congress_test, stratified)
    cards_test.jsonl            - {id, text, true_claims}, held-out final eval (70% of congress_test, stratified)

Usage:
    python data/prepare_splits.py
"""

import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.model_selection import train_test_split
from prompts import slim_system_instruction, cot_trigger

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Training splits (from training_recot_opus.jsonl)
# ---------------------------------------------------------------------------

def build_sft_record(text, response, use_recot=True):
    """Convert raw RECoT record to SFT chat messages format.

    If use_recot=False, strips <think>...</think> reasoning and keeps only the YAML output,
    and omits the CoT trigger from the user prompt.
    """
    if use_recot:
        user_content = f"### Text:\n{text}\n\n{cot_trigger}"
    else:
        response = strip_reasoning(response)
        user_content = f"### Text:\n{text}"
    return {"messages": [
        {"role": "system", "content": slim_system_instruction},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": response},
    ]}


def strip_reasoning(response):
    """Remove <think>...</think> block, keep only the YAML output."""
    if '</think>' in response:
        return response.split('</think>')[-1].strip()
    return response


def parse_claims_from_response(response):
    """Extract predicted category codes from model response for stratification."""
    after_think = response.split('</think>')[-1] if '</think>' in response else response
    match = re.search(r'categories:\s*\n((?:\s*-\s*.+\n?)+)', after_think)
    if match:
        return sorted(re.findall(r'-\s*([\d_]+)', match.group(1)))
    return ['0_0_0']


def prepare_train_splits(extra_files=None):
    path = os.path.join(DATA_DIR, 'training_recot_opus.jsonl')
    with open(path) as f:
        raw = [json.loads(line) for line in f]
    print(f"Loaded {len(raw)} RECoT training samples from training_recot_opus.jsonl")

    # Load additional training data files (e.g., hard negatives)
    if extra_files:
        for extra_path in extra_files:
            full_path = os.path.join(DATA_DIR, extra_path) if not os.path.isabs(extra_path) else extra_path
            with open(full_path) as f:
                extra = [json.loads(line) for line in f]
            # Normalize key names
            for r in extra:
                if 'true_labels' in r and 'true_claims' not in r:
                    r['true_claims'] = r.pop('true_labels')
            raw.extend(extra)
            print(f"  + {len(extra)} from {extra_path}")
        print(f"  = {len(raw)} total training samples")

    # Extract primary label for stratification
    primary_labels = []
    for r in raw:
        claims = parse_claims_from_response(r['response'])
        primary_labels.append(claims[0] if claims else '0_0_0')

    # Bucket rare labels so stratify works
    label_counts = pd.Series(primary_labels).value_counts()
    rare = set(label_counts[label_counts < 2].index)
    strat_keys = [('_rare_' if l in rare else l) for l in primary_labels]

    # Stratified 90/10 split
    train_idx, eval_idx = train_test_split(
        range(len(raw)), test_size=0.1, random_state=RANDOM_STATE, stratify=strat_keys,
    )

    for name, indices in [('cards_train', train_idx), ('cards_train_eval', eval_idx)]:
        for suffix, use_recot in [('', True), ('_norecot', False)]:
            records = [build_sft_record(raw[i]['text'], raw[i]['response'], use_recot=use_recot) for i in indices]
            fname = f'{name}{suffix}.jsonl'
            out_path = os.path.join(DATA_DIR, fname)
            with open(out_path, 'w') as f:
                for r in records:
                    f.write(json.dumps(r) + '\n')
            print(f"  {fname}: {len(records)} samples")

    return train_idx, eval_idx


# ---------------------------------------------------------------------------
# 2. Val/test splits (from congress_test.csv)
# ---------------------------------------------------------------------------

def parse_true_claims(val):
    """Parse true_claims — handles numpy-style space-separated lists."""
    return sorted(re.findall(r'[\d_]+', str(val)))


def prepare_eval_splits():
    path = os.path.join(DATA_DIR, 'congress_test.csv')
    df = pd.read_csv(path)
    print(f"\nLoaded {len(df)} congress_test samples")

    df['tc'] = df['true_claims'].apply(parse_true_claims)
    df['primary_label'] = df['tc'].apply(lambda x: x[0])

    # Bucket rare labels
    vc = df['primary_label'].value_counts()
    rare = set(vc[vc < 2].index)
    df['strat_key'] = df['primary_label'].apply(lambda x: '_rare_' if x in rare else x)

    # Stratified 30/70 split
    df_val, df_test = train_test_split(
        df, test_size=0.7, random_state=RANDOM_STATE, stratify=df['strat_key'],
    )

    for name, d in [('cards_val', df_val), ('cards_test', df_test)]:
        records = []
        for _, row in d.iterrows():
            records.append({
                'id': row['id'],
                'text': row['text'],
                'true_claims': row['tc'],
            })
        out_path = os.path.join(DATA_DIR, f'{name}.jsonl')
        with open(out_path, 'w') as f:
            for r in records:
                f.write(json.dumps(r) + '\n')

        zero_pct = d['primary_label'].apply(lambda x: x == '0_0_0').mean() * 100
        multi_pct = (d['tc'].map(len) > 1).mean() * 100
        print(f"  {name}.jsonl: {len(d)} samples ({zero_pct:.1f}% zero, {multi_pct:.1f}% multi-label)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Prepare CARDS dataset splits")
    parser.add_argument("--extra", nargs="+", default=[],
                        help="Additional JSONL training data files to combine (e.g., hard_negatives_recot.jsonl)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip regenerating val/test splits (they don't change)")
    args = parser.parse_args()

    print("=" * 60)
    print("Preparing CARDS dataset splits")
    print("=" * 60)

    print("\n--- Training splits (stratified 90/10) ---")
    prepare_train_splits(extra_files=args.extra if args.extra else None)

    if not args.skip_eval:
        print("\n--- Evaluation splits (stratified 30/70) ---")
        prepare_eval_splits()
    else:
        print("\n--- Skipping eval splits (unchanged) ---")

    print("\n" + "=" * 60)
    print("Done. Files written to:", DATA_DIR)
    print("=" * 60)
