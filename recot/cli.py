#!/usr/bin/env python3
"""Simple CLI script for generating ReCOT data."""

import argparse
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ConfigManager
from recot.core import ReCOTGenerator


def main():
    """Main function for ReCOT generation."""
    parser = argparse.ArgumentParser(
        description='Generate Reverse Engineered Chain-of-Thought (ReCOT) data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate ReCOT for training data using default models
  python generate_recot.py --input data/congress_test.csv --sample 10
  
  # Use specific models
  python generate_recot.py --input data/congress_test.csv --models 1 3 --sample 5
  
  # Generate for full dataset
  python generate_recot.py --input data/congress_test.csv --labels_column final_claims
        """
    )
    
    parser.add_argument('--input', type=str, required=True, help='Input CSV/parquet file')
    parser.add_argument('--output_dir', type=str, default='results/recot_jsons', help='Output directory for JSON files')
    parser.add_argument('--text_column', type=str, default='text', help='Text column name')
    parser.add_argument('--labels_column', type=str, default='true_claims', help='Labels column name')
    parser.add_argument('--max_workers', type=int, default=5, help='Max workers for parallel processing')
    parser.add_argument('--models', nargs='*', type=int, help='Model IDs to use (default: GPT-4o and Claude-3.5)')
    parser.add_argument('--sample', type=int, help='Sample size for testing')
    parser.add_argument('--list_models', action='store_true', help='List available models')
    
    args = parser.parse_args()
    
    # Initialize config manager
    config_manager = ConfigManager()
    all_models = config_manager.get_default_model_configs()
    
    # Handle list models
    if args.list_models:
        print("Available models:")
        for model in all_models:
            print(f"  {model.id}. {model.name} ({model.provider})")
        return
    
    # Load data
    print(f"📂 Loading data from {args.input}...")
    if args.input.endswith('.csv'):
        df = pd.read_csv(args.input)
    elif args.input.endswith('.parquet'):
        df = pd.read_parquet(args.input)
    else:
        raise ValueError("Unsupported file format. Use CSV or parquet.")
    
    print(f"📊 Loaded {len(df)} rows")
    
    # Check required columns
    if args.text_column not in df.columns:
        raise ValueError(f"Text column '{args.text_column}' not found in data")
    if args.labels_column not in df.columns:
        raise ValueError(f"Labels column '{args.labels_column}' not found in data")
    
    # Apply sampling if requested
    if args.sample:
        df = df.sample(min(args.sample, len(df)), random_state=42).reset_index(drop=True)
        print(f"🎯 Using sample of {len(df)} rows")
    
    # Get model configurations
    if args.models:
        model_configs = [model for model in all_models if model.id in args.models]
        if not model_configs:
            raise ValueError(f"No models found with IDs: {args.models}")
    else:
        # Default to GPT-4o and Claude-3.5 for ReCOT generation
        model_configs = [model for model in all_models if model.name in ['GPT-4o', 'Claude-3.5']]
    
    print(f"🤖 Using models: {[model.name for model in model_configs]}")
    
    # Initialize generator
    generator = ReCOTGenerator(
        openai_api_key=config_manager.openai_api_key,
        anthropic_api_key=config_manager.anthropic_api_key,
        output_dir=args.output_dir
    )
    
    # Generate ReCOT
    results_df = generator.generate_recot_dataset(
        df,
        model_configs,
        args.text_column,
        args.labels_column,
        args.max_workers
    )
    
    # Save combined results
    base_name = args.input.replace('.csv', '').replace('.parquet', '')
    output_file = f"{base_name}_recot.parquet"
    results_df.to_parquet(output_file, index=False)
    
    print(f"✅ Combined results saved to: {output_file}")
    print(f"📁 Individual JSON files saved to: {args.output_dir}")
    print(f"📈 Generated ReCOT for {len(results_df)} examples")


if __name__ == "__main__":
    main()