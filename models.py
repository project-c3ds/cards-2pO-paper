"""Model response handling and structured output parsing."""

import json
import re
from typing import List, Dict, Any
from pydantic import BaseModel
from openai import OpenAI
import anthropic
from config import ModelConfig

import tenacity


class Category(BaseModel):
    """Pydantic model for category classification."""
    category: str


class Categories(BaseModel):
    """Pydantic model for structured classification output."""
    review: str
    categories: List[Category]


class ModelClient:
    """Handles communication with different LLM providers."""
    
    def __init__(self, openai_api_key: str = None, anthropic_api_key: str = None):
        self.openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key) if anthropic_api_key else None
    
    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=60),
        stop=tenacity.stop_after_attempt(5),
        retry=tenacity.retry_if_exception_type(Exception),
        reraise=True
    )
    def get_model_response(
        self,
        messages: List[Dict[str, str]],
        model_config: ModelConfig,
        prompt: str
    ) -> Dict[str, Any]:
        """Get response from either OpenAI or Anthropic model."""
        if model_config.provider == 'openai':
            return self._get_openai_response(messages, model_config, prompt)
        elif model_config.provider == 'anthropic':
            return self._get_anthropic_response(messages, model_config, prompt)
        else:
            raise ValueError(f"Unknown provider: {model_config.provider}")
    
    def _get_openai_response(
        self,
        messages: List[Dict[str, str]],
        model_config: ModelConfig,
        prompt: str
    ) -> Dict[str, Any]:
        """Get response from OpenAI model."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        
        full_messages = messages + [{"role": "user", "content": prompt}]
        
        response = self.openai_client.chat.completions.create(
            model=model_config.model_id,
            messages=full_messages,
            temperature=model_config.temperature,
            max_completion_tokens=model_config.max_tokens
        )
        
        return {
            "model": model_config.name,
            "response": response.choices[0].message.content,
            "usage": response.usage.model_dump()
        }
    
    def _get_anthropic_response(
        self,
        messages: List[Dict[str, str]],
        model_config: ModelConfig,
        prompt: str
    ) -> Dict[str, Any]:
        """Get response from Anthropic model."""
        if not self.anthropic_client:
            raise ValueError("Anthropic client not initialized")
        
        # Convert system messages to Anthropic format
        anthropic_messages = [
            {
                "type": "text",
                "text": msg["content"],
                "cache_control": {"type": "ephemeral"} if i == 0 else None
            }
            for i, msg in enumerate(messages)
        ]
        
        response = self.anthropic_client.messages.create(
            model=model_config.model_id,
            max_tokens=model_config.max_tokens,
            system=anthropic_messages,
            temperature=model_config.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {
            "model": model_config.name,
            "response": response.content[0].text,
            "usage": response.usage.model_dump()
        }


class ResponseParser:
    """Handles parsing of model responses into structured formats."""
    
    def __init__(self, openai_client: OpenAI = None):
        self.openai_client = openai_client
    
    def extract_classification_dict(self, response: str) -> Dict:
        """Extract classification dictionary from model response."""
        try:
            json_text = re.search(r'\{.*\}', response, re.DOTALL).group()
            return json.loads(json_text)
        except Exception:
            try:
                json_text = re.search(r'```json(.*)```', response, re.DOTALL).group(1)
                return json.loads(json_text)
            except Exception:
                return self._convert_to_structured_output(response)
    
    def _convert_to_structured_output(self, text: str) -> Dict:
        """Convert text to structured output using OpenAI's parsing capability."""
        if not self.openai_client:
            raise ValueError("OpenAI client required for structured output conversion")
        
        completion = self.openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"""
                You are an expert in parsing text data structured data. 
                You will be given a text that contains JSON information. 
                Your task is to extract the JSON information from the text in the format given below.

                {text}

                ### Instructions:
                - Be as precise as possible. Do not add or remove any information.
                """},
                {"role": "user", "content": "Let's work this out in a step by step way to be sure we have the right answer"}
            ],
            response_format=Categories
        )
        return completion.choices[0].message.parsed.model_dump()
    
    def extract_usage_metrics(self, row: Dict) -> Dict:
        """Extract usage metrics from response row."""
        model = row['model']
        
        if 'gpt' in model.lower() or 'cards' in model.lower():
            return {
                "completion_tokens": row['usage']['completion_tokens'],
                "prompt_tokens": row['usage']['prompt_tokens'],
                "total_tokens": row['usage']['total_tokens']
            }
        
        if 'claude' in model.lower():
            return {
                "completion_tokens": row['usage']['output_tokens'],
                "prompt_tokens": (row['usage']['input_tokens'] + 
                                row['usage']['cache_read_input_tokens'] + 
                                row['usage']['cache_creation_input_tokens']),
                "total_tokens": (row['usage']['input_tokens'] + 
                               row['usage']['output_tokens'] + 
                               row['usage']['cache_read_input_tokens'] + 
                               row['usage']['cache_creation_input_tokens'])
            }
        
        return {}