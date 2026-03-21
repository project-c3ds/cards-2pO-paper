"""Benchmark framework for LLM model comparison and evaluation."""

import pandas as pd
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

from config import ModelConfig, EvaluationConfig
from models import ModelClient, ResponseParser
from embeddings import EmbeddingManager
from checkpoint import CheckpointStore


class CARDSBenchmark:
    """CARDS benchmark framework for comparing LLM performance on classification tasks."""
    
    def __init__(
        self,
        openai_api_key: str = None,
        anthropic_api_key: str = None,
        system_instruction: str = None,
        slim_system_instruction: str = None,
        fewshot_instruction: str = None,
        prompt: str = None,
        codebook: str = None,
        slim_codebook: str = None,
        db_path: str = "data/results.db"
    ):
        """Initialize the benchmark framework with API clients and prompts."""
        self.model_client = ModelClient()
        self.response_parser = ResponseParser()
        self.checkpoint = CheckpointStore(db_path)
        openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        self.embedding_manager = EmbeddingManager(openai_client)

        # Store prompt templates
        self.system_instruction = system_instruction
        self.slim_system_instruction = slim_system_instruction
        self.fewshot_instruction = fewshot_instruction
        self.prompt = prompt
        self.codebook = codebook
        self.slim_codebook = slim_codebook
        
        # Template for text input
        self.text_template = """
        ### Text:
        {text}
        """
    
    def load_fewshot_data(self, fewshot_data_path: str) -> None:
        """Load few-shot examples for dynamic example selection."""
        self.embedding_manager.load_fewshot_data(fewshot_data_path)
    
    def _build_prompt_messages(self, text: str, use_fewshot: bool = False, model_config: ModelConfig = None) -> List[Dict[str, str]]:
        """Build the complete prompt message sequence for the model."""
        if model_config and model_config.use_slim_codebook:
            system_content = self.slim_system_instruction.format(codebook=self.slim_codebook)
        else:
            system_content = self.system_instruction.format(codebook=self.codebook)

        if use_fewshot:
            fewshot_examples = self.embedding_manager.get_dynamic_fewshot_examples(text)
            system_content += "\n\n" + self.fewshot_instruction.format(fewshot=fewshot_examples)

        system_content += "\n\n" + self.text_template.format(text=text)

        return [{"role": "system", "content": system_content}]
    
    def _evaluate_single_text(
        self,
        row_idx: int,
        text: str,
        model_config: ModelConfig,
        use_fewshot: bool = False
    ) -> Dict[str, Any]:
        """Evaluate a single text with a specific model."""
        messages = self._build_prompt_messages(text, use_fewshot, model_config)
        response = self.model_client.get_model_response(messages, model_config, self.prompt)

        # Handle case where all retries failed
        if response is None:
            result = {
                "model": model_config.name,
                "response": "Error: API call failed after all retries",
                "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
                "text": text
            }
        else:
            response["text"] = text
            result = response

        # Save to checkpoint immediately
        self.checkpoint.save_result(
            model=model_config.name,
            row_idx=row_idx,
            text=text,
            response=result["response"],
            usage=result["usage"]
        )
        return result
    
    def run_benchmark(
        self,
        data: pd.DataFrame,
        model_configs: List[ModelConfig],
        config: EvaluationConfig
    ) -> pd.DataFrame:
        """Run benchmark evaluation on multiple models with the given dataset."""
        tqdm.pandas()
        all_texts = data[config.text_column].tolist()
        all_responses = []

        for model_config in model_configs:
            # Check which rows are already completed
            completed = self.checkpoint.get_completed_indices(model_config.name)

            if len(completed) == len(all_texts):
                print(f"\n✅ {model_config.name} already complete ({len(completed)}/{len(all_texts)}), loading from checkpoint...")
                all_responses.extend(self.checkpoint.get_results(model_config.name))
                continue

            if completed:
                print(f"\n🔄 Resuming {model_config.name} ({len(completed)}/{len(all_texts)} done)...")
                # Load existing results
                all_responses.extend(self.checkpoint.get_results(model_config.name))
            else:
                print(f"\n🚀 Benchmarking {model_config.name}...")

            # Build list of remaining (idx, text) pairs
            remaining = [(i, text) for i, text in enumerate(all_texts) if i not in completed]

            # Adjust max_workers for Anthropic API rate limits
            max_workers = 10 if model_config.provider == 'anthropic' else config.max_workers

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self._evaluate_single_text,
                        idx,
                        text,
                        model_config,
                        config.use_fewshot
                    )
                    for idx, text in remaining
                ]

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Processing {model_config.name} ({max_workers} workers)"
                ):
                    all_responses.append(future.result())

        return self._process_benchmark_results(all_responses, data, config)
    
    def _process_benchmark_results(
        self, 
        all_responses: List[Dict[str, Any]], 
        original_data: pd.DataFrame, 
        config: EvaluationConfig
    ) -> pd.DataFrame:
        """Process and format benchmark results."""
        # Create DataFrame from responses
        df_results = pd.DataFrame(all_responses).reset_index(drop=True)
        
        # Merge with original data
        df_results = df_results.merge(
            original_data, 
            left_on='text', 
            right_on=config.text_column
        ).reset_index(drop=True)
        
        # Extract structured classifications (with checkpoint support)
        print("\n📊 Processing model responses...")
        parsed_cache = {}
        for model_name in df_results['model'].unique():
            parsed_cache[model_name] = self.checkpoint.get_parsed_results(model_name)

        def _parse_with_checkpoint(row):
            model = row['model']
            row_idx = row.get('row_idx', row.name)
            cached = parsed_cache.get(model, {})
            if row_idx in cached:
                return cached[row_idx]
            result = self.response_parser.extract_classification_dict(row['response'])
            self.checkpoint.save_parsed_result(model, row_idx, result)
            return result

        df_results['classification_dict'] = df_results.progress_apply(
            _parse_with_checkpoint, axis=1
        )
        
        # Extract predicted claims
        df_results['predicted_claims'] = df_results['classification_dict'].apply(
            lambda x: [
                (t['category'] if isinstance(t, dict) else t).split('>')[0].replace('<', '')
                for t in (x or {}).get('categories', [])
            ]
        )
        
        # Clean up duplicate text column
        if config.text_column != 'text':
            df_results = df_results.drop("text", axis=1)
        
        # Extract usage metrics
        df_results['usage'] = df_results.progress_apply(
            self.response_parser.extract_usage_metrics, 
            axis=1
        )
        
        # Organize final columns
        initial_columns = original_data.columns.tolist()
        final_columns = initial_columns + ['model', 'usage', 'response', 'predicted_claims']
        df_results = df_results[final_columns]
        
        print(f"✅ Benchmark complete! Results shape: {df_results.shape}")
        return df_results