from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import os
import numpy as np
from pathlib import Path
from qdrant_client import QdrantClient
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class RetrievalConfig:
    k: int = 3  # Number of documents to retrieve
    min_relevance_score: float = 0.3
    max_context_length: int = 4096  # For prompt context window
    collection_name: str = "textbooks"
    context_template: str = "\nContext: {context}\n"  # Template for RAG prompt injection

class RetrievalSystem:
    def __init__(self, config: Optional[RetrievalConfig] = None):
        """Initialize the retrieval system"""
        self.config = config or RetrievalConfig()
        self.vector_dim = 3072  # text-embedding-3-large dimension
        self.embedding_cache = {}

        print(f"Initializing retrieval system with URL: {os.getenv('QDRANT_URL')}")
        
        try:
            # Initialize OpenAI client
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            # Initialize Qdrant client
            self.client = QdrantClient(
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY")
            )
            collections = self.client.get_collections()
            print(f"Collections: {collections}")
        except Exception as e:
            logger.error(f"Error initializing clients: {e}")
            raise
        
        collection_info = self.client.get_collection('textbooks')
        print(f"Collection info: {collection_info}")

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query using OpenAI embeddings API"""
        try:
            response = self.openai_client.embeddings.create(
                input=query,
                model="text-embedding-3-large"
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Error encoding query: {e}")
            raise

    def _filter_by_relevance(self, results: List, min_score: float) -> List:
        """Filter results based on relevance scores"""
        return [r for r in results if r.score >= min_score]

    def format_for_prompt(self, results: List) -> str:
        """Format retrieved contexts for RAG prompt"""
        formatted_contexts = []
        total_length = 0
        
        for result in results:
            context = result.payload.get('text', '').strip()
            if total_length + len(context) > self.config.max_context_length:
                available_length = self.config.max_context_length - total_length
                if available_length > 0:
                    context = context[:available_length]
                else:
                    break
            
            formatted_contexts.append(context)
            total_length += len(context)
        
        combined_context = "\n".join(formatted_contexts)
        return self.config.context_template.format(context=combined_context)

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant context from the vector database"""
        try:
            # Encode query using OpenAI API
            query_vector = self.encode_query(query)
            print(f"Query vector shape: {query_vector.shape}")
            
            # Search in Qdrant
            results = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=query_vector.tolist(),
                limit=self.config.k * 2
            )
            print(f"Number of results: {len(results)}")

            return self._process_results(results)
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            raise

    def _process_results(self, results: List) -> List[Dict]:
        """Process the search results"""
        if not results:
            return []

        processed_results = []
        if self.config.debug:
            print("\nDebug processing results:")
            print(f"Number of results before filtering: {len(results)}")
        
        for i, result in enumerate(results):
            if self.config.debug:
                print(f"\n=== Result {i+1} Details ===")
                print(f"Score: {result.score}")
                print(f"Payload keys: {result.payload.keys()}")
                print(f"Metadata: {result.payload.get('metadata', {})}")
                print(f"Text length: {len(result.payload.get('text', ''))}")
                print("First 300 chars of text:")
                print(result.payload.get('text', '')[:300])
                print("="*50)
            
            if result.score < self.config.min_relevance_score:
                print(f"Filtered out due to low score ({result.score} < {self.config.min_relevance_score})")
                continue
    
            metadata = result.payload.get('metadata', {})
            source = metadata.get('source', 'unknown')
            text = result.payload.get('text', '').strip()
            
            processed_results.append({
                'score': float(result.score),
                'source': source,
                'text': text,
                'metadata': metadata
            })

        # Sort by score in descending order
        processed_results.sort(key=lambda x: x['score'], reverse=True)

        final_results = processed_results[:self.config.k]
        print(f"\nReturning {len(final_results)} results after processing")
        
        context_str = ""
        for r in final_results:
            context_str += f"[Score: {r['score']:.2f}] From {r['source']}:\n{r['text']}\n\n"
        print("\nFinal formatted context:")
        print(context_str)

        return final_results

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