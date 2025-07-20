"""Metrics calculation and claim processing utilities for CARDS benchmark."""

import json
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
        X, Y, Z = claim.split('_')[:3]
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
        'samples_f1': round(f1_score(y_true_bin, y_pred_bin, average='samples', zero_division=0), rounding),
        'micro_precision': round(precision_score(y_true_bin, y_pred_bin, average='micro', zero_division=0), rounding),
        'macro_precision': round(precision_score(y_true_bin, y_pred_bin, average='macro', zero_division=0), rounding),
        'samples_precision': round(precision_score(y_true_bin, y_pred_bin, average='samples', zero_division=0), rounding),
        'micro_recall': round(recall_score(y_true_bin, y_pred_bin, average='micro', zero_division=0), rounding),
        'macro_recall': round(recall_score(y_true_bin, y_pred_bin, average='macro', zero_division=0), rounding),
        'samples_recall': round(recall_score(y_true_bin, y_pred_bin, average='samples', zero_division=0), rounding),
        'accuracy': round(accuracy_score(y_true_bin, y_pred_bin), rounding),
        'hamming_loss': round(hamming_loss(y_true_bin, y_pred_bin), rounding),
        'matthews_corrcoef': round(matthews_corrcoef(y_true_bin.ravel(), y_pred_bin.ravel()), rounding),
    }
    
    return metrics

def compute_metrics_for_groups(df: pd.DataFrame, column_name: str = 'final_claims') -> pd.DataFrame:
    """Compute metrics for each classification type and model in the DataFrame.
    
    Args:
        df: DataFrame containing predictions and true labels
        column_name: Name of the column containing true labels
        
    Returns:
        DataFrame with computed metrics for each model and classification type
    """

    metrics_list = []

    for (classification_type, model), group in df.groupby(['classification_type', 'model']):
        y_true = group[column_name].tolist()
        y_pred = group['predicted_claims'].tolist()
        metrics = calculate_multi_label_metrics(y_true, y_pred)
        metrics['classification_type'] = classification_type
        metrics['model'] = model
        # put model and classification type at the beginning of the dictionary
        metrics = {**{'model': model, 'classification_type': classification_type}, **metrics}
        metrics_list.append(metrics)

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
    
    # Process claims strings to lists
    df['true_claims'] = df['true_claims'].str.replace(' ', ', ').map(eval)
    df['final_claims'] = df['final_claims'].str.replace(' ', ', ').map(eval)
    df['predicted_claims'] = df['predicted_claims'].str.replace(' ', ', ').map(eval)
    
    # Process claims at specified level
    df['true_claims'] = df['true_claims'].map(lambda x: process_claims(x, level))
    df['final_claims'] = df['final_claims'].map(lambda x: process_claims(x, level))
    df['predicted_claims'] = df['predicted_claims'].map(lambda x: process_claims(x, level))
    
    return df


def display_metrics_table(metrics_df: pd.DataFrame, title: str) -> None:
    """Display metrics in a colored, formatted table.
    
    Args:
        metrics_df: DataFrame containing metrics
        title: Title for the table
    """
    # ANSI color codes
    COLORS = {
        'header': '\033[95m',  # Magenta
        'model': '\033[94m',   # Blue
        'metric': '\033[92m',  # Green
        'high': '\033[91m',    # Red (for high values)
        'medium': '\033[93m',  # Yellow
        'low': '\033[96m',     # Cyan
        'reset': '\033[0m',    # Reset
        'bold': '\033[1m'      # Bold
    }
    
    print(f"\n{COLORS['header']}{COLORS['bold']}{'='*60}{COLORS['reset']}")
    print(f"{COLORS['header']}{COLORS['bold']}{title.center(60)}{COLORS['reset']}")
    print(f"{COLORS['header']}{COLORS['bold']}{'='*60}{COLORS['reset']}\n")
    
    # Display each model's metrics
    for _, row in metrics_df.iterrows():
        model_name = row['model']
        classification_type = row['classification_type']
        
        print(f"{COLORS['model']}{COLORS['bold']}{model_name} ({classification_type}){COLORS['reset']}")
        print(f"{COLORS['model']}{'-' * (len(model_name) + len(classification_type) + 3)}{COLORS['reset']}")
        
        # Key metrics to highlight
        key_metrics = ['samples_f1', 'samples_precision', 'samples_recall', 'accuracy']
        
        for metric in key_metrics:
            if metric in row:
                value = row[metric]
                # Color code based on value
                if value >= 0.8:
                    color = COLORS['high']
                elif value >= 0.6:
                    color = COLORS['medium']
                else:
                    color = COLORS['low']
                
                metric_display = metric.replace('_', ' ').title()
                print(f"  {COLORS['metric']}{metric_display:15}{COLORS['reset']}: {color}{value:.3f}{COLORS['reset']}")
        
        # Additional metrics
        other_metrics = ['hamming_loss', 'matthews_corrcoef']
        for metric in other_metrics:
            if metric in row:
                value = row[metric]
                metric_display = metric.replace('_', ' ').title()
                print(f"  {COLORS['metric']}{metric_display:15}{COLORS['reset']}: {COLORS['metric']}{value:.3f}{COLORS['reset']}")
        
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
        
        # Compute metrics
        print(f"📊 Computing metrics...")
        fewshot_metrics = compute_metrics_for_groups(df_fewshot, column_name='final_claims')
        zeroshot_metrics = compute_metrics_for_groups(df_zeroshot, column_name='final_claims')
        
        # Filter metrics to key columns
        cols_needed = [
            'model', 'classification_type', 'samples_f1', 'samples_precision', 
            'samples_recall', 'accuracy', 'hamming_loss', 'matthews_corrcoef'
        ]
        
        fewshot_metrics = fewshot_metrics[cols_needed]
        zeroshot_metrics = zeroshot_metrics[cols_needed]
        
        # Filter out specific models if needed
        models_to_ignore = ['Claude-4-Sonnet', 'CARDS-nano-Sonnet-2025-06-14']
        fewshot_metrics = fewshot_metrics[~fewshot_metrics['model'].isin(models_to_ignore)]
        zeroshot_metrics = zeroshot_metrics[~zeroshot_metrics['model'].isin(models_to_ignore)]
        
        # Display results
        display_metrics_table(fewshot_metrics, "FEWSHOT EVALUATION RESULTS")
        display_metrics_table(zeroshot_metrics, "ZEROSHOT EVALUATION RESULTS")
        
        # Combined summary
        combined_metrics = pd.concat([fewshot_metrics, zeroshot_metrics], ignore_index=True)
        
        print(f"\n\033[95m\033[1m{'='*60}\033[0m")
        print(f"\033[95m\033[1m{'SUMMARY STATISTICS'.center(60)}\033[0m")
        print(f"\033[95m\033[1m{'='*60}\033[0m\n")
        
        # Best performing models
        best_fewshot = fewshot_metrics.loc[fewshot_metrics['samples_f1'].idxmax()]
        best_zeroshot = zeroshot_metrics.loc[zeroshot_metrics['samples_f1'].idxmax()]
        
        print(f"🏆 \033[92mBest Fewshot Model:\033[0m {best_fewshot['model']} (F1: {best_fewshot['samples_f1']:.3f})")
        print(f"🏆 \033[92mBest Zeroshot Model:\033[0m {best_zeroshot['model']} (F1: {best_zeroshot['samples_f1']:.3f})")
        
        # Average performance
        avg_fewshot_f1 = fewshot_metrics['samples_f1'].mean()
        avg_zeroshot_f1 = zeroshot_metrics['samples_f1'].mean()
        
        print(f"📊 \033[94mAverage Fewshot F1:\033[0m {avg_fewshot_f1:.3f}")
        print(f"📊 \033[94mAverage Zeroshot F1:\033[0m {avg_zeroshot_f1:.3f}")
        
        print(f"\n✅ \033[92mEvaluation complete!\033[0m")
        
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


def main():
    """Main function for running metrics calculation as a script."""
    print("\n🚀 Running CARDS Metrics Evaluation...")
    run_full_evaluation()


if __name__ == "__main__":
    main()