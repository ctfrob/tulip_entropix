from typing import List, Dict, Optional, Union, Any
from functools import partial
from dataclasses import dataclass
import os
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from qdrant_client import QdrantClient
import logging
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

@dataclass
class RetrievalConfig:
    k: int = 3  # Number of documents to retrieve
    min_relevance_score: float = 0.7
    max_context_length: int = 2048  # For prompt context window
    batch_size: int = 32
    collection_name: str = "textbooks"
    context_template: str = "\nContext: {context}\n"  # Template for RAG prompt injection

class RetrievalSystem:
    def __init__(self, config: Optional[RetrievalConfig] = None):
        """Initialize the retrieval system"""
        self.config = config or RetrievalConfig()
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        
        # Initialize model and embeddings
        self.__init_model()

    def __init_model(self):
        """Initialize model and tokenizer"""
        try:
            self.model = AutoModel.from_pretrained(
                "jxm/cde-small-v1",
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
            # Load dataset embeddings
            embeddings_path = Path("vectordb/data/dataset_embeddings.pt")
            if not embeddings_path.exists():
                raise FileNotFoundError(f"Dataset embeddings not found at {embeddings_path}")
                
            self.dataset_embeddings = jnp.array(np.load(embeddings_path))
            logger.info("Model and embeddings initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise

    @partial(jax.jit, static_argnums=(0,))
    def _compute_relevance_scores(self, query_embedding: jnp.ndarray, context_embeddings: jnp.ndarray) -> jnp.ndarray:
        """Compute relevance scores between query and retrieved contexts"""
        return jnp.dot(query_embedding, context_embeddings.T)

    def _filter_by_relevance(self, results: List, scores: jnp.ndarray) -> List:
        """Filter results based on relevance scores"""
        filtered = []
        for result, score in zip(results, scores):
            if score >= self.config.min_relevance_score:
                filtered.append(result)
        return filtered

    def format_for_prompt(self, results: List) -> str:
        """Format retrieved contexts for RAG prompt"""
        formatted_contexts = []
        total_length = 0
        
        for result in results:
            context = result.payload.get('text', '').strip()
            if total_length + len(context) > self.config.max_context_length:
                # Truncate context if needed
                available_length = self.config.max_context_length - total_length
                if available_length > 0:
                    context = context[:available_length]
                else:
                    break
            
            formatted_contexts.append(context)
            total_length += len(context)
        
        combined_context = "\n".join(formatted_contexts)
        return self.config.context_template.format(context=combined_context)

    @partial(jax.jit, static_argnums=(0,))
    def encode_query(self, query: str) -> jnp.ndarray:
        """Encode a query into a JAX array"""
        inputs = self.tokenizer(
            f"search document: {query}",
            return_tensors="jax",
            truncation=True,
            max_length=512
        )
        
        # Generate embeddings
        embeddings = self.model.second_stage_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            dataset_embeddings=self.dataset_embeddings
        )
        
        embeddings = embeddings / jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
        return embeddings[0]

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant context from the vector database"""
        try:
            # Encode query
            query_vector = self.encode_query(query)
            results = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=query_vector.tolist(),
                limit=self.config.k * 2 # Fetch extra to allow for filtering
            )

            return self._process_results(results, query_vector)
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            raise

    def _process_results(self, results: List, query_vector: jnp.ndarray) -> List[Dict]:
        """Process the search results"""
        if not results:
            return []
            
        # Extract vectors and compute relevance scores
        vectors = jnp.array([r.vector for r in results])
        scores = self._compute_relevance_scores(query_vector, vectors)

        processed_results = []
        
        for result, score in zip(results, scores):
            if score < self.config.min_relevance_score:
                continue
        
            metadata = result.payload.get('metadata', {})
            source = metadata.get('source', 'unknown')

            text = result.payload.get('text', '').strip()
            if len(text) > self.config.max_context_length:
                text = text[:self.config.max_context_length] + "..."
            
            processed_results.append({
                'score': float(score), #Convert from jax array to float
                'source': source,
                'text': text,
                'metadata': metadata # Include full metadata for optional use
            })
        
        # Sort by score in descending order
        processed_results.sort(key=lambda x: x['score'], reverse=True)

        # Limit to top k results
        return processed_results[:self.config.k]

    async def batch_retrieve(self, queries: List[str]) -> List[Dict]:
        """Batch retrieval for multiple queries"""
        return [await self.retrieve(query) for query in queries]

    def close(self):
        """Cleanup and close connections"""
        try:
            self.client.close()
            logger.info("Retrieval system closed successfully")
        except Exception as e:
            logger.error(f"Error closing retrieval system: {e}")