"""Metrics calculation and claim processing utilities for CARDS benchmark."""

import json
import os
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, 
    hamming_loss, matthews_corrcoef
)
from sklearn.preprocessing import MultiLabelBinarizer


def process_claims(claim_list: List[str], level: int) -> List[str]:

    """Process hierarchical claims at different levels.
    
    Args:
        claim_list: List of claim strings in format 'X_Y_Z'
        level: Processing level (1=superclaims, 2=subclaims, 3=subsubclaims)
        
    Returns:
        Sorted list of processed claims at the specified level
    """
    # Build the hierarchical claims data structure
    claims = defaultdict(lambda: defaultdict(set))
    for claim in claim_list:
        parts = claim.split('_')
        if len(parts) < 3 or not all(p.isdigit() for p in parts[:3]):
            continue
        else:
            X, Y, Z = parts[:3]
        claims[X][Y].add(Z)

    # Process Superclaims
    superclaims = []
    for X in claims:
        Ys = claims[X]
        # Check if any non-zero Y or Z exists under this superclaim
        non_zero_subclaims = any(
            Y != '0' or any(Z != '0' for Z in Zs)
            for Y, Zs in Ys.items()
        )
        if non_zero_subclaims:
            superclaims.append(X)
        else:
            # Include the superclaim if no non-zero subclaims exist
            superclaims.append(X)

    # Process Subclaims
    subclaims = []
    for X in superclaims:
        Ys = claims[X]
        # Check if any non-zero Y exists
        non_zero_Ys = [Y for Y in Ys if Y != '0']
        if non_zero_Ys:
            for Y in non_zero_Ys:
                subclaims.append(f"{X}_{Y}")
        else:
            # No non-zero Y exists, include all Ys
            for Y in Ys:
                subclaims.append(f"{X}_{Y}")

    # Process Subsubclaims
    subsubclaims = []
    for subclaim in subclaims:
        X, Y = subclaim.split('_')
        Zs = claims[X][Y]
        # Check if any non-zero Z exists
        non_zero_Zs = [Z for Z in Zs if Z != '0']
        if non_zero_Zs:
            for Z in non_zero_Zs:
                subsubclaims.append(f"{X}_{Y}_{Z}")
        else:
            # No non-zero Z exists, include all Zs
            for Z in Zs:
                subsubclaims.append(f"{X}_{Y}_{Z}")

    # Return the claims at the specified level
    if level == 1:
        return sorted(set(superclaims))
    elif level == 2:
        return sorted(set(subclaims))
    elif level == 3:
        return sorted(set(subsubclaims))
    else:
        raise ValueError("Invalid level. Level must be 1, 2, or 3.")


def preprocess_multi_label_data(y_true: List[List[str]], y_pred: List[List[str]]) -> Tuple[Any, Any]:
    """Preprocess multi-label data using MultiLabelBinarizer.
    
    Args:
        y_true: List of true label lists
        y_pred: List of predicted label lists
        
    Returns:
        Tuple of binarized true and predicted labels
    """
    mlb = MultiLabelBinarizer()
    
    # Get all unique labels across true and predicted labels
    all_labels = set()
    for labels in list(y_true) + list(y_pred):
        all_labels.update(labels)
    
    # Fit the binarizer with all unique labels
    mlb.fit([all_labels])
    
    # Transform true and predicted labels
    y_true_bin = mlb.transform(y_true)
    y_pred_bin = mlb.transform(y_pred)
    
    return y_true_bin, y_pred_bin

def calculate_multi_label_metrics(
    y_true: List[List[str]], 
    y_pred: List[List[str]], 
    rounding: int = 3
) -> Dict[str, float]:
    """Calculate comprehensive metrics for multi-label classification.
    
    Args:
        y_true: List of true label lists
        y_pred: List of predicted label lists
        rounding: Number of decimal places for rounding
        
    Returns:
        Dictionary of calculated metrics
    """
    y_true_bin, y_pred_bin = preprocess_multi_label_data(y_true, y_pred)
    
    metrics = {
        'micro_f1': round(f1_score(y_true_bin, y_pred_bin, average='micro', zero_division=0), rounding),
        'macro_f1': round(f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0), rounding),
        'weighted_f1': round(f1_score(y_true_bin, y_pred_bin, average='weighted', zero_division=0), rounding),
        'samples_f1': round(f1_score(y_true_bin, y_pred_bin, average='samples', zero_division=0), rounding),
        'micro_precision': round(precision_score(y_true_bin, y_pred_bin, average='micro', zero_division=0), rounding),
        'macro_precision': round(precision_score(y_true_bin, y_pred_bin, average='macro', zero_division=0), rounding),
        'weighted_precision': round(precision_score(y_true_bin, y_pred_bin, average='weighted', zero_division=0), rounding),
        'samples_precision': round(precision_score(y_true_bin, y_pred_bin, average='samples', zero_division=0), rounding),
        'micro_recall': round(recall_score(y_true_bin, y_pred_bin, average='micro', zero_division=0), rounding),
        'macro_recall': round(recall_score(y_true_bin, y_pred_bin, average='macro', zero_division=0), rounding),
        'weighted_recall': round(recall_score(y_true_bin, y_pred_bin, average='weighted', zero_division=0), rounding),
        'samples_recall': round(recall_score(y_true_bin, y_pred_bin, average='samples', zero_division=0), rounding),
        'accuracy': round(accuracy_score(y_true_bin, y_pred_bin), rounding),
        'hamming_loss': round(hamming_loss(y_true_bin, y_pred_bin), rounding),
        'matthews_corrcoef': round(matthews_corrcoef(y_true_bin.ravel(), y_pred_bin.ravel()), rounding),
    }
    
    return metrics

def compute_metrics_for_groups(df: pd.DataFrame, column_name: str = 'final_claims') -> pd.DataFrame:
    """Compute metrics for each classification type and model in the DataFrame.

    Computes three sets of metrics:
    - all: all samples
    - detection: binary detection (0_0_0 vs non-0_0_0)
    - classification: only non-0_0_0 samples (actual claim classification)

    Args:
        df: DataFrame containing predictions and true labels
        column_name: Name of the column containing true labels

    Returns:
        DataFrame with computed metrics for each model, classification type, and subset
    """

    metrics_list = []

    for (classification_type, model), group in df.groupby(['classification_type', 'model']):
        y_true = group[column_name].tolist()
        y_pred = group['predicted_claims'].tolist()

        # All samples
        metrics = calculate_multi_label_metrics(y_true, y_pred)
        metrics = {**{'model': model, 'classification_type': classification_type, 'subset': 'all'}, **metrics}
        metrics_list.append(metrics)

        # Binary detection: is there a claim or not?
        y_true_bin = [[str(int(t != ['0_0_0']))] for t in y_true]
        y_pred_bin = [[str(int(p != ['0_0_0']))] for p in y_pred]
        metrics_det = calculate_multi_label_metrics(y_true_bin, y_pred_bin)
        metrics_det = {**{'model': model, 'classification_type': classification_type, 'subset': 'detection'}, **metrics_det}
        metrics_list.append(metrics_det)

        # Classification: only non-0_0_0 ground truth samples
        non_zero_mask = [t != ['0_0_0'] for t in y_true]
        if any(non_zero_mask):
            y_true_nz = [t for t, m in zip(y_true, non_zero_mask) if m]
            y_pred_nz = [p for p, m in zip(y_pred, non_zero_mask) if m]
            metrics_cls = calculate_multi_label_metrics(y_true_nz, y_pred_nz)
            metrics_cls = {**{'model': model, 'classification_type': classification_type, 'subset': 'classification'}, **metrics_cls}
            metrics_list.append(metrics_cls)

    # Convert the list of metrics dictionaries to a DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df


def load_claims_mappings(data_dir: str = 'data/mapping') -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load final and true claims mapping dictionaries.
    
    Args:
        data_dir: Directory containing mapping files
        
    Returns:
        Tuple of (final_claims_dict, true_claims_dict)
    """
    with open(f'{data_dir}/final_claims_dict.json', 'r') as f:
        final_claims_dict = json.load(f)
    
    with open(f'{data_dir}/true_claims_dict.json', 'r') as f:
        true_claims_dict = json.load(f)
    
    return final_claims_dict, true_claims_dict


def process_results_dataframe(
    df: pd.DataFrame, 
    final_claims_dict: Dict[str, str],
    true_claims_dict: Dict[str, str], 
    classification_type: str,
    level: int = 3,
    exclude_error_analysis: bool = True
) -> pd.DataFrame:
    """Process results DataFrame with claim mappings and preprocessing.
    
    Args:
        df: Raw results DataFrame
        final_claims_dict: Dictionary mapping text to final claims
        true_claims_dict: Dictionary mapping text to true claims
        classification_type: Type of classification ('fewshot' or 'zeroshot')
        level: Processing level for claims
        exclude_error_analysis: Whether to exclude error analysis texts
        
    Returns:
        Processed DataFrame with claims and metadata
    """
    # Select required columns
    df = df[['id', 'text', 'model', 'response', 'predicted_claims']].copy()
    df['text'] = df['text'].str.strip()
    
    # Map claims
    df['true_claims'] = df['text'].map(true_claims_dict)
    df['final_claims'] = df['text'].map(final_claims_dict)
    
    # Exclude error analysis if requested
    if exclude_error_analysis:
        try:
            df_error_analysis = pd.read_csv('data/recot_error_analysis.csv')
            df_error_analysis['text'] = df_error_analysis['text'].str.strip()
            df = df[~df['text'].isin(df_error_analysis['text'].unique())].reset_index(drop=True)
        except FileNotFoundError:
            print("Warning: recot_error_analysis.csv not found, skipping exclusion")
    
    # Add classification type
    df['classification_type'] = classification_type
    
    # Process claims to lists
    for col in ['true_claims', 'final_claims', 'predicted_claims']:
        first = df[col].iloc[0]
        if isinstance(first, str):
            df[col] = df[col].str.replace(' ', ', ').map(eval)
        else:
            # Convert numpy arrays or other iterables to lists
            df[col] = df[col].map(lambda x: list(x) if hasattr(x, '__iter__') and not isinstance(x, str) else x)
    
    # Process claims at specified level
    df['true_claims'] = df['true_claims'].map(lambda x: process_claims(x, level))
    df['final_claims'] = df['final_claims'].map(lambda x: process_claims(x, level))
    df['predicted_claims'] = df['predicted_claims'].map(lambda x: process_claims(x, level))

    
    return df


def display_metrics_table(metrics_df: pd.DataFrame, title: str) -> None:
    """Display all metrics in a single table with model as index.

    Args:
        metrics_df: DataFrame containing metrics
        title: Title for the table
    """
    from tabulate import tabulate

    display_df = metrics_df.copy()
    display_df = display_df.set_index('model')
    display_df = display_df.drop(columns=['classification_type'], errors='ignore')
    display_df.columns = [c.replace('_', ' ').title() for c in display_df.columns]

    print(f"\n\033[95m\033[1m{'=' * 60}\033[0m")
    print(f"\033[95m\033[1m{title.center(60)}\033[0m")
    print(f"\033[95m\033[1m{'=' * 60}\033[0m")
    print(tabulate(display_df, headers='keys', tablefmt='grid', floatfmt='.3f'))
    print()


def run_full_evaluation() -> None:
    """Run complete evaluation on zeroshot and fewshot results."""
    try:
        # Load mapping dictionaries
        final_claims_dict, true_claims_dict = load_claims_mappings()
        
        # Process fewshot results
        print(f"\n🔄 Processing fewshot results...")
        df_fewshot = pd.read_csv('data/results/fewshot_results.csv')
        df_fewshot = process_results_dataframe(
            df_fewshot, 
            final_claims_dict,
            true_claims_dict, 
            'fewshot'
        )
        
        # Process zeroshot results
        print(f"🔄 Processing zeroshot results...")
        df_zeroshot = pd.read_csv('data/results/zeroshot_results.csv')
        df_zeroshot = process_results_dataframe(
            df_zeroshot,
            final_claims_dict,
            true_claims_dict,
            'zeroshot'
        )

        # Process nocot results if available
        nocot_metrics = None
        try:
            print(f"🔄 Processing nocot results...")
            df_nocot = pd.read_csv('data/results/nocot_results.csv')
            df_nocot = process_results_dataframe(
                df_nocot,
                final_claims_dict,
                true_claims_dict,
                'nocot'
            )
        except FileNotFoundError:
            print("ℹ️  No nocot results found, skipping.")
            df_nocot = None

        # Compute metrics
        print(f"📊 Computing metrics...")
        fewshot_metrics = compute_metrics_for_groups(df_fewshot, column_name='final_claims')
        zeroshot_metrics = compute_metrics_for_groups(df_zeroshot, column_name='final_claims')
        if df_nocot is not None:
            nocot_metrics = compute_metrics_for_groups(df_nocot, column_name='final_claims')

        # Filter out specific models if needed
        models_to_ignore = ['Claude-4-Sonnet', 'CARDS-nano-Sonnet-2025-06-14']
        fewshot_metrics = fewshot_metrics[~fewshot_metrics['model'].isin(models_to_ignore)]
        zeroshot_metrics = zeroshot_metrics[~zeroshot_metrics['model'].isin(models_to_ignore)]
        if nocot_metrics is not None:
            nocot_metrics = nocot_metrics[~nocot_metrics['model'].isin(models_to_ignore)]

        # Combine all metrics
        all_metrics = [fewshot_metrics, zeroshot_metrics]
        if nocot_metrics is not None:
            all_metrics.append(nocot_metrics)
        combined_metrics = pd.concat(all_metrics, ignore_index=True)

        # Save to markdown — zeroshot only, two tables
        key_cols = ['model', 'samples_f1', 'macro_f1', 'micro_f1', 'hamming_loss']

        output_path = 'data/results/metrics.md'
        with open(output_path, 'w') as f:
            # Table 1: Zeroshot key metrics (all samples)
            zs_all = zeroshot_metrics[zeroshot_metrics['subset'] == 'all']
            f.write('## Zeroshot — Key Metrics\n\n')
            f.write(zs_all[key_cols].to_markdown(index=False, floatfmt='.3f'))
            f.write('\n\n')

            # Table 2: Zeroshot binary detection (claim vs no-claim)
            zs_det = zeroshot_metrics[zeroshot_metrics['subset'] == 'detection']
            f.write('## Zeroshot — Claim vs No-Claim (Binary Detection)\n\n')
            f.write(zs_det[key_cols].to_markdown(index=False, floatfmt='.3f'))
            f.write('\n\n')

        print(f"✅ Metrics saved to {output_path}")
        return combined_metrics
        
    except FileNotFoundError as e:
        print(f"\n❌ \033[91mError: Could not find required files. {e}\033[0m")
        print(f"\033[93mMake sure the following files exist:\033[0m")
        print(f"  - data/results/fewshot_results.csv")
        print(f"  - data/results/zeroshot_results.csv")
        print(f"  - data/mapping/final_claims_dict.json")
        print(f"  - data/mapping/true_claims_dict.json")
    except Exception as e:
        print(f"\n❌ \033[91mUnexpected error: {e}\033[0m")


def run_file_evaluation(file_path: str) -> None:
    """Run evaluation on a file path or directory of parquet files."""
    import glob as gl

    try:
        final_claims_dict, true_claims_dict = load_claims_mappings()

        # Collect files
        if os.path.isdir(file_path):
            files = sorted(gl.glob(os.path.join(file_path, '*.parquet')))
            if not files:
                print(f"❌ No parquet files found in {file_path}")
                return
            print(f"Found {len(files)} parquet files")
        else:
            files = [file_path]

        # Load and concat
        dfs = []
        for f in files:
            if f.endswith('.parquet'):
                dfs.append(pd.read_parquet(f))
            else:
                dfs.append(pd.read_csv(f))
        df = pd.concat(dfs, ignore_index=True)
        print(f"Total rows: {len(df)}, Models: {df['model'].unique().tolist()}")

        df = process_results_dataframe(
            df, final_claims_dict, true_claims_dict, 'zeroshot'
        )

        metrics = compute_metrics_for_groups(df, column_name='final_claims')

        key_cols = ['model', 'samples_f1', 'macro_f1', 'micro_f1', 'hamming_loss']
        display_metrics_table(metrics[key_cols + ['subset']], "Key Metrics")

    except Exception as e:
        print(f"\n❌ \033[91mError: {e}\033[0m")


def main():
    """Main function for running metrics calculation as a script."""
    import argparse
    parser = argparse.ArgumentParser(description='CARDS Metrics Evaluation')
    parser.add_argument('--file', type=str, default='data/predictions',
                        help='Path to predictions file or directory')
    parser.add_argument('--full', action='store_true', help='Run full evaluation on all result files')
    args = parser.parse_args()

    if args.full:
        print("\n🚀 Running Full CARDS Metrics Evaluation...")
        run_full_evaluation()
    else:
        print(f"\n🚀 Evaluating {args.file}...")
        run_file_evaluation(args.file)


if __name__ == "__main__":
    main()