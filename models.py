"""Model response handling and structured output parsing."""

import json
import re
from typing import List, Dict, Any
from pydantic import BaseModel
import litellm
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
    """Handles communication with LLM providers via LiteLLM."""

    def __init__(self):
        pass

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
        """Get response from any LLM provider via LiteLLM."""
        full_messages = messages + [{"role": "user", "content": prompt}]

        kwargs = dict(
            model=model_config.model_id,
            messages=full_messages,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
        )
        if model_config.extra_body:
            kwargs["extra_body"] = model_config.extra_body

        response = litellm.completion(**kwargs)

        return {
            "model": model_config.name,
            "response": response.choices[0].message.content,
            "usage": response.usage.model_dump()
        }


class ResponseParser:
    """Handles parsing of model responses into structured formats."""

    def __init__(self):
        pass

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
        """Convert text to structured output using LiteLLM."""
        response = litellm.completion(
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
        result = json.loads(response.choices[0].message.content)
        return result

    def extract_usage_metrics(self, row: Dict) -> Dict:
        """Extract usage metrics from response row."""
        usage = row.get('usage', {})
        return {
            "completion_tokens": usage.get('completion_tokens', 0),
            "prompt_tokens": usage.get('prompt_tokens', 0),
            "total_tokens": usage.get('total_tokens', 0)
        }
