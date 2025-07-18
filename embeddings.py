"""Embedding-based similarity search for few-shot examples."""

import pandas as pd
from typing import List
from scipy.spatial.distance import cosine
from openai import OpenAI


class EmbeddingManager:
    """Manages text embeddings and similarity search."""
    
    def __init__(self, openai_client: OpenAI = None):
        self.openai_client = openai_client
        self.df_fewshot = None
    
    def load_fewshot_data(self, fewshot_data_path: str) -> None:
        """Load few-shot examples from CSV or parquet file."""
        if fewshot_data_path:
            # Load data based on file extension
            if fewshot_data_path.endswith('.csv'):
                self.df_fewshot = pd.read_csv(fewshot_data_path)
            elif fewshot_data_path.endswith('.parquet'):
                self.df_fewshot = pd.read_parquet(fewshot_data_path)
            else:
                raise ValueError(f"Unsupported file format: {fewshot_data_path}")
            
            self.df_fewshot.drop_duplicates(subset=['text'], inplace=True)
    
    def get_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """Get embeddings for text using OpenAI's API."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        
        text = text.replace("\n", " ")
        return self.openai_client.embeddings.create(
            input=[text], 
            model=model
        ).data[0].embedding
    
    def get_dynamic_fewshot_examples(self, text: str, top_k: int = 5) -> str:
        """Get most similar few-shot examples based on embedding similarity."""
        if self.df_fewshot is None:
            return ""
        
        query_embedding = self.get_embedding(text)
        similarities = [
            1 - cosine(query_embedding, embedding) 
            for embedding in self.df_fewshot['embeddings']
        ]
        
        self.df_fewshot['similarity'] = similarities
        sorted_df = self.df_fewshot.sort_values(
            'similarity', 
            ascending=False
        ).head(top_k).reset_index(drop=True)
        
        fewshot = ""
        for index, row in sorted_df.iterrows():
            fewshot += f"Example {index + 1}: {row['text']}\n\n"
            fewshot += f"Answer: {row['true_claims']}\n"
            fewshot += "---\n"
        
        return fewshot