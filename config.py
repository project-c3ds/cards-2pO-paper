"""Configuration module for the evaluation framework."""

import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ModelConfig:
    """Configuration for a model to be evaluated."""
    id: int  # Numeric identifier for CLI usage
    name: str
    provider: str  # 'openai' or 'anthropic'
    model_id: str
    temperature: float = 0
    max_tokens: int = 4000
    extra_body: Dict[str, Any] = field(default_factory=dict)



@dataclass
class EvaluationConfig:
    """Configuration for evaluation parameters."""
    input_file: str = None
    output_suffix: str = ''
    text_column: str = 'text'
    use_fewshot: bool = False
    max_workers: int = 30
    fewshot_data_path: str = None
    run_sample: bool = False
    sample_size: int = 2


class ConfigManager:
    """Manages configuration for the evaluation framework."""
    
    def __init__(self, models_file: str = "models.json"):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.models_file = models_file
    
    def get_default_model_configs(self) -> List[ModelConfig]:
        """Get model configurations from JSON file."""
        try:
            with open(self.models_file, 'r') as f:
                data = json.load(f)
            
            return [
                ModelConfig(
                    id=model['id'],
                    name=model['name'],
                    provider=model['provider'],
                    model_id=model['model_id'],
                    temperature=model.get('temperature', 0),
                    max_tokens=model.get('max_tokens', 4000),
                    extra_body=model.get('extra_body', {})
                )
                for model in data['models']
            ]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading models from {self.models_file}: {e}")
            return []
    
    def get_evaluation_config(self, **kwargs) -> EvaluationConfig:
        """Get evaluation configuration with optional overrides."""
        config = EvaluationConfig()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config