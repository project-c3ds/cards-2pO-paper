"""Reverse Engineered Chain-of-Thought (ReCOT) generation module."""

import pandas as pd
import json
import os
import sys
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import tenacity

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ModelConfig
from models import ModelClient, ResponseParser
from prompts import cot_instruction, cot_text_prompt, prompt, codebook


class ReCOTGenerator:
    """Generate Reverse Engineered Chain-of-Thought reasoning for training data."""
    
    def __init__(
        self,
        openai_api_key: str = None,
        anthropic_api_key: str = None,
        output_dir: str = "results/recot_jsons"
    ):
        """Initialize ReCOT generator."""
        self.model_client = ModelClient()
        self.response_parser = ResponseParser()
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=60),
        stop=tenacity.stop_after_attempt(5),
        retry=tenacity.retry_if_exception_type(Exception)
    )
    def _generate_recot_single(
        self,
        text: str,
        true_labels: List[str],
        model_config: ModelConfig
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate ReCOT for a single text-label pair."""
        # Build messages for ReCOT generation
        messages = [
            {"role": "system", "content": cot_instruction.format(codebook=codebook)},
            {"role": "system", "content": cot_text_prompt.format(text=text, true_labels=true_labels)}
        ]
        
        # Get model response
        response = self.model_client.get_model_response(messages, model_config, prompt)
        return response["response"], response["usage"]
    
    def generate_recot_for_row(
        self,
        row: pd.Series,
        model_configs: List[ModelConfig],
        text_column: str = 'text',
        labels_column: str = 'true_claims'
    ) -> Dict[str, Any]:
        """Generate ReCOT for a single row using multiple models."""
        text = row[text_column]
        true_labels = row[labels_column]
        
        # Convert labels to list if it's a string
        if isinstance(true_labels, str):
            true_labels = eval(true_labels) if true_labels.startswith('[') else [true_labels]
        
        result = {
            'text': text,
            'true_claims': true_labels,
            'id': row.get('id', row.name)
        }
        
        # Generate ReCOT for each model
        for model_config in model_configs:
            model_key = model_config.name.lower().replace('-', '_')
            try:
                response, usage = self._generate_recot_single(text, true_labels, model_config)
                result[f'{model_key}_response'] = response
                result[f'{model_key}_usage'] = usage
                
                # Parse the response to extract structured data
                try:
                    parsed = self.response_parser.extract_classification_dict(response)
                    result[f'{model_key}_parsed'] = parsed
                    result[f'{model_key}_labels'] = [
                        cat['category'] for cat in parsed.get('categories', [])
                    ]
                except Exception as e:
                    result[f'{model_key}_parse_error'] = str(e)
                    
            except Exception as e:
                result[f'{model_key}_error'] = str(e)
        
        return result
    
    def save_result_to_json(self, result: Dict[str, Any], row_id: str) -> None:
        """Save individual result to JSON file."""
        filename = os.path.join(self.output_dir, f"{row_id}.json")
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
    
    def process_single_row(
        self,
        row: pd.Series,
        model_configs: List[ModelConfig],
        text_column: str = 'text',
        labels_column: str = 'true_claims',
        save_individual: bool = True
    ) -> Dict[str, Any]:
        """Process a single row and optionally save to individual JSON."""
        result = self.generate_recot_for_row(row, model_configs, text_column, labels_column)
        
        if save_individual:
            row_id = str(result['id'])
            self.save_result_to_json(result, row_id)
        
        return result
    
    def generate_recot_dataset(
        self,
        data: pd.DataFrame,
        model_configs: List[ModelConfig],
        text_column: str = 'text',
        labels_column: str = 'true_claims',
        max_workers: int = 5,
        save_individual: bool = True
    ) -> pd.DataFrame:
        """Generate ReCOT for entire dataset using parallel processing."""
        
        print(f"🔄 Generating ReCOT for {len(data)} examples using {len(model_configs)} models...")
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = [
                executor.submit(
                    self.process_single_row,
                    row,
                    model_configs,
                    text_column,
                    labels_column,
                    save_individual
                )
                for _, row in data.iterrows()
            ]
            
            # Collect results with progress bar
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing ReCOT generation"
            ):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
        
        # Create DataFrame from results
        df_results = pd.DataFrame(results)
        
        print(f"✅ ReCOT generation complete! Generated {len(df_results)} examples.")
        if save_individual:
            print(f"📁 Individual JSON files saved to: {self.output_dir}")
        
        return df_results
    
    def load_and_combine_jsons(self) -> pd.DataFrame:
        """Load all individual JSON files and combine into DataFrame."""
        json_files = [f for f in os.listdir(self.output_dir) if f.endswith('.json')]
        
        if not json_files:
            print("No JSON files found in output directory")
            return pd.DataFrame()
        
        results = []
        for json_file in tqdm(json_files, desc="Loading JSON files"):
            try:
                with open(os.path.join(self.output_dir, json_file), 'r') as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
        
        return pd.DataFrame(results)


def main():
    """CLI interface for ReCOT generation."""
    import argparse
    from config import ConfigManager
    
    parser = argparse.ArgumentParser(description='ReCOT Generation CLI')
    parser.add_argument('--input_file', type=str, required=True, help='Input CSV/parquet file')
    parser.add_argument('--output_dir', type=str, default='results/recot_jsons', help='Output directory')
    parser.add_argument('--text_column', type=str, default='text', help='Text column name')
    parser.add_argument('--labels_column', type=str, default='true_claims', help='Labels column name')
    parser.add_argument('--max_workers', type=int, default=5, help='Max workers for parallel processing')
    parser.add_argument('--models', nargs='*', type=int, help='Model IDs to use')
    parser.add_argument('--sample', type=int, help='Sample size for testing')
    
    args = parser.parse_args()
    
    # Load data
    if args.input_file.endswith('.csv'):
        df = pd.read_csv(args.input_file)
    elif args.input_file.endswith('.parquet'):
        df = pd.read_parquet(args.input_file)
    else:
        raise ValueError("Unsupported file format")
    
    # Apply sampling if requested
    if args.sample:
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)
        print(f"Using sample of {len(df)} rows")
    
    # Get model configurations
    config_manager = ConfigManager()
    all_models = config_manager.get_default_model_configs()
    
    if args.models:
        model_configs = [model for model in all_models if model.id in args.models]
    else:
        # Default to Claude and GPT-4o for ReCOT generation
        model_configs = [model for model in all_models if model.id in [1, 3]]  # GPT-4o and Claude-3.5
    
    print(f"Using models: {[model.name for model in model_configs]}")
    
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
    
    # Save results
    output_file = args.input_file.replace('.csv', '_recot.csv').replace('.parquet', '_recot.parquet')
    if output_file.endswith('.csv'):
        results_df.to_csv(output_file, index=False)
    else:
        results_df.to_parquet(output_file, index=False)
    
    print(f"📊 Results saved to: {output_file}")


if __name__ == "__main__":
    main()