"""Main script for running LLM evaluations."""

import argparse
import os
import pandas as pd
from typing import List

from config import ConfigManager, ModelConfig, EvaluationConfig
from benchmark import CARDSBenchmark
from prompts import system_instruction, fewshot_instruction, prompt, codebook


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LLM Evaluation Framework')
    parser.add_argument('--input_file', type=str, help='Path to input file', default=None)
    parser.add_argument('--output_suffix', type=str, help='Suffix for output file', default='')
    parser.add_argument('--text_column', type=str, help='Name of the text column', default='text')
    parser.add_argument('--use_fewshot', action='store_true', help='Use few-shot examples')
    parser.add_argument('--max_workers', type=int, help='Maximum number of workers', default=30)
    parser.add_argument('--sample', action='store_true', help='Run on sample data')
    parser.add_argument('--sample_size', type=int, help='Sample size', default=2)
    parser.add_argument('--models', nargs='*', type=int, help='List of model IDs to run (default: all models)')
    parser.add_argument('--list_models', action='store_true', help='List all available models and exit')
    parser.add_argument('--no-cot', action='store_true', help='Disable chain-of-thought trigger prompt')
    return parser.parse_args()




def load_data(args: argparse.Namespace) -> pd.DataFrame:
    """Load data from file or default sources."""
    if args.input_file:
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"File not found: {args.input_file}")
        
        if args.input_file.endswith('.csv'):
            return pd.read_csv(args.input_file)
        elif args.input_file.endswith('.parquet'):
            return pd.read_parquet(args.input_file)
        else:
            raise ValueError(f"Unsupported file format: {args.input_file}")
    
    # Default data loading (if no input file specified)
    try:
        df = pd.read_csv('data/congress_test.csv')
        return df
    except FileNotFoundError:
        raise FileNotFoundError("No input file specified and default data file 'data/congress_test.csv' not found")


def prepare_output_path(args: argparse.Namespace) -> str:
    """Prepare output file path in dedicated predictions directory."""
    # Create predictions directory if it doesn't exist
    predictions_dir = 'data/predictions'
    os.makedirs(predictions_dir, exist_ok=True)
    
    if args.input_file:
        input_filename = os.path.basename(args.input_file)
        input_name, input_ext = os.path.splitext(input_filename)
        if input_ext in ['.csv', '.parquet']:
            output_filename = f'{input_name}_predictions.parquet'
        else:
            output_filename = 'predictions.parquet'
    else:
        # Default case: using congress_test.csv
        output_filename = 'congress_test_predictions.parquet'
    
    if args.output_suffix:
        name, ext = os.path.splitext(output_filename)
        output_filename = f'{name}_{args.output_suffix}{ext}'
    
    return os.path.join(predictions_dir, output_filename)


def main():
    """Main evaluation function."""
    args = parse_arguments()
    
    # Initialize configuration manager
    config_manager = ConfigManager()
    
    # Get all available models
    all_models = config_manager.get_default_model_configs()
    
    # Handle --list_models flag
    if args.list_models:
        print("Available models:")
        for model in all_models:
            print(f"  {model.id}. {model.name}")
        return
    
    # Create evaluation configuration
    eval_config = EvaluationConfig(
        input_file=args.input_file,
        output_suffix=args.output_suffix,
        text_column=args.text_column,
        use_fewshot=args.use_fewshot,
        max_workers=args.max_workers,
        run_sample=args.sample,
        sample_size=args.sample_size,
        fewshot_data_path='data/recot_fewshot.csv'
    )
    
    # Initialize benchmark
    active_prompt = "Classify the text." if args.no_cot else prompt
    benchmark = CARDSBenchmark(
        openai_api_key=config_manager.openai_api_key,
        anthropic_api_key=config_manager.anthropic_api_key,
        system_instruction=system_instruction,
        fewshot_instruction=fewshot_instruction,
        prompt=active_prompt,
        codebook=codebook
    )
    
    # Load few-shot data if needed
    if eval_config.use_fewshot and eval_config.fewshot_data_path:
        benchmark.load_fewshot_data(eval_config.fewshot_data_path)
    
    # Load data
    df = load_data(args)
    print(f"Loaded data shape: {df.shape}")
    
    # Apply sampling if requested
    if eval_config.run_sample:
        df = df.sample(eval_config.sample_size, random_state=42).reset_index(drop=True)
        print(f"Sample data shape: {df.shape}")
    
    # Remove duplicates
    df = df.drop_duplicates(subset=[eval_config.text_column]).reset_index(drop=True)
    print(f"Final data shape: {df.shape}")
    
    # Get model configurations
    if args.models:
        # Filter models by specified IDs
        model_configs = [model for model in all_models if model.id in args.models]
        if not model_configs:
            print(f"Error: No models found with IDs: {args.models}")
            print("Use --list_models to see available models")
            return
        print(f"Running models: {[model.name for model in model_configs]}")
    else:
        # Run all models
        model_configs = all_models
        print(f"Running all {len(model_configs)} models")
    
    # Run benchmark evaluation
    results = benchmark.run_benchmark(
        data=df,
        model_configs=model_configs,
        config=eval_config
    )
    
    # Save results
    output_path = prepare_output_path(args)
    results.to_parquet(output_path, index=False)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()