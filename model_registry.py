"""Model registry for adding new models to the benchmark."""

import json
import argparse
from typing import List, Dict, Any


class ModelRegistry:
    """Registry for managing model configurations in JSON file."""
    
    def __init__(self, models_file: str = "models.json"):
        self.models_file = models_file
    
    def _load_models(self) -> Dict[str, Any]:
        """Load models from JSON file."""
        try:
            with open(self.models_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"models": []}
    
    def _save_models(self, data: Dict[str, Any]) -> None:
        """Save models to JSON file."""
        with open(self.models_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register_model(
        self,
        name: str,
        provider: str,
        model_id: str,
        temperature: float = 0,
        max_tokens: int = 4000,
        category: str = "Custom Models"
    ) -> int:
        """Register a new model and return its assigned ID."""
        data = self._load_models()
        
        # Get the next available ID
        existing_ids = [model['id'] for model in data['models']]
        next_id = max(existing_ids) + 1 if existing_ids else 1
        
        new_model = {
            "id": next_id,
            "name": name,
            "provider": provider,
            "model_id": model_id,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "category": category
        }
        
        data['models'].append(new_model)
        self._save_models(data)
        
        print(f"✅ Registered model '{name}' with ID {next_id}")
        return next_id
    
    def remove_model(self, model_id: int) -> bool:
        """Remove a model by ID."""
        data = self._load_models()
        
        original_count = len(data['models'])
        data['models'] = [m for m in data['models'] if m['id'] != model_id]
        
        if len(data['models']) < original_count:
            self._save_models(data)
            print(f"✅ Removed model with ID {model_id}")
            return True
        else:
            print(f"❌ Model with ID {model_id} not found")
            return False
    
    def list_models(self) -> None:
        """List all registered models."""
        data = self._load_models()
        
        if not data['models']:
            print("No models registered.")
            return
        
        print("Registered Models:")
        current_category = None
        for model in data['models']:
            if model.get('category') != current_category:
                current_category = model.get('category', 'Unknown')
                print(f"\n=== {current_category} ===")
            print(f"  {model['id']}. {model['name']} ({model['provider']})")
    
    def update_model(
        self,
        model_id: int,
        name: str = None,
        provider: str = None,
        model_id_str: str = None,
        temperature: float = None,
        max_tokens: int = None,
        category: str = None
    ) -> bool:
        """Update an existing model."""
        data = self._load_models()
        
        for model in data['models']:
            if model['id'] == model_id:
                if name is not None:
                    model['name'] = name
                if provider is not None:
                    model['provider'] = provider
                if model_id_str is not None:
                    model['model_id'] = model_id_str
                if temperature is not None:
                    model['temperature'] = temperature
                if max_tokens is not None:
                    model['max_tokens'] = max_tokens
                if category is not None:
                    model['category'] = category
                
                self._save_models(data)
                print(f"✅ Updated model with ID {model_id}")
                return True
        
        print(f"❌ Model with ID {model_id} not found")
        return False


def main():
    """CLI interface for model registration."""
    parser = argparse.ArgumentParser(description='Model Registry CLI')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Register command
    register_parser = subparsers.add_parser('register', help='Register a new model')
    register_parser.add_argument('name', help='Model name')
    register_parser.add_argument('provider', choices=['openai', 'anthropic'], help='Model provider')
    register_parser.add_argument('model_id', help='Model ID/identifier')
    register_parser.add_argument('--temperature', type=float, default=0, help='Temperature setting')
    register_parser.add_argument('--max_tokens', type=int, default=4000, help='Max tokens')
    register_parser.add_argument('--category', default='Custom Models', help='Model category')
    
    # List command
    subparsers.add_parser('list', help='List all models')
    
    # Remove command
    remove_parser = subparsers.add_parser('remove', help='Remove a model')
    remove_parser.add_argument('model_id', type=int, help='Model ID to remove')
    
    # Update command
    update_parser = subparsers.add_parser('update', help='Update a model')
    update_parser.add_argument('model_id', type=int, help='Model ID to update')
    update_parser.add_argument('--name', help='New model name')
    update_parser.add_argument('--provider', choices=['openai', 'anthropic'], help='New provider')
    update_parser.add_argument('--model_id_str', help='New model ID/identifier')
    update_parser.add_argument('--temperature', type=float, help='New temperature setting')
    update_parser.add_argument('--max_tokens', type=int, help='New max tokens')
    update_parser.add_argument('--category', help='New model category')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    registry = ModelRegistry()
    
    if args.command == 'register':
        registry.register_model(
            args.name, args.provider, args.model_id,
            args.temperature, args.max_tokens, args.category
        )
    elif args.command == 'list':
        registry.list_models()
    elif args.command == 'remove':
        registry.remove_model(args.model_id)
    elif args.command == 'update':
        registry.update_model(
            args.model_id, args.name, args.provider, args.model_id_str,
            args.temperature, args.max_tokens, args.category
        )


if __name__ == "__main__":
    main()