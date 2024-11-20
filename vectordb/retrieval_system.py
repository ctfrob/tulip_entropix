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
import torch
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class RetrievalConfig:
    k: int = 3  # Number of documents to retrieve
    min_relevance_score: float = 0.3
    max_context_length: int = 4096  # For prompt context window
    batch_size: int = 32
    collection_name: str = "textbooks"
    context_template: str = "\nContext: {context}\n"  # Template for RAG prompt injection
    chunk_size: int = 512  # For tokenization

class RetrievalSystem:
    def __init__(self, config: Optional[RetrievalConfig] = None):
        """Initialize the retrieval system"""
        self.config = config or RetrievalConfig()

        print(f"Initializing retrieval system with URL: {os.getenv('QDRANT_URL')}")
        
        try: 
            # Initialize Qdrant client
            self.client = QdrantClient(
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY")
            )
            collections = self.client.get_collections()
            print(f"Collections: {collections}")
        except Exception as e:
            logger.error(f"Error initializing Qdrant client: {e}")
            raise
        
        collection_info = self.client.get_collection('textbooks')
        print(f"Collection info: {collection_info}")
        
        # Initialize model and embeddings
        self.__init_model()

    def __init_model(self):
        """Initialize model and tokenizer using CDE approach"""
        try:
            # Initialize model and move to appropriate device
            self.model = AutoModel.from_pretrained(
                "jxm/cde-small-v1",
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
            # Move to device if GPU available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise

    def _get_first_stage_embeddings(self, text: str) -> jnp.ndarray:
        """Generate first stage embeddings using CDE approach"""
        document_prefix = "search_document: "
        text_with_prefix = document_prefix + text
        
        # Get corpus size from model config
        corpus_size = self.model.config.transductive_corpus_size
        minicorpus_docs = [text_with_prefix] * corpus_size
        
        with torch.no_grad():
            tokenized = self.tokenizer(
                minicorpus_docs,
                truncation=True,
                padding=True,
                max_length=self.config.chunk_size,
                return_tensors="pt"
            ).to(self.device)
            
            embeddings = self.model.first_stage_model(**tokenized)
            return jnp.array(embeddings.cpu().numpy())

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
                available_length = self.config.max_context_length - total_length
                if available_length > 0:
                    context = context[:available_length]
                else:
                    break
            
            formatted_contexts.append(context)
            total_length += len(context)
        
        combined_context = "\n".join(formatted_contexts)
        return self.config.context_template.format(context=combined_context)

    def encode_query(self, query: str) -> jnp.ndarray:
        """Encode query using two-stage CDE process"""
        document_prefix = "search_document: "
        query_text = document_prefix + query
        
        print("Debug encoding:")

        with torch.no_grad():
            # First stage: get dataset embeddings
            dataset_embeddings = self._get_first_stage_embeddings(query_text)
            print(f"Dataset embeddings shape: {dataset_embeddings.shape}")
            
            # Second stage: generate final embedding
            tokenized_query = self.tokenizer(
                query_text,
                truncation=True,
                padding=True,
                max_length=self.config.chunk_size,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate final query embedding
            query_embedding = self.model.second_stage_model(
                input_ids=tokenized_query["input_ids"],
                attention_mask=tokenized_query["attention_mask"],
                dataset_embeddings=torch.from_numpy(np.array(dataset_embeddings)).to(self.device)
            )

            print(f"Pre-normalized query embedding shape: {query_embedding.shape}")
            
            # Normalize embedding
            query_embedding = query_embedding / torch.norm(query_embedding, p=2, dim=1, keepdim=True)
            
            print(f"Post-normalized query embedding shape: {query_embedding.shape}")
            
            # Convert to JAX array
            return jnp.array(query_embedding.cpu().numpy()[0])

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant context from the vector database"""
        try:
            # Encode query using two-stage process
            query_vector = self.encode_query(query)
            print(f"Query vector shape: {query_vector.shape}")
            
            # Search in Qdrant
            results = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=query_vector.tolist(),
                limit=self.config.k * 2,
                with_vectors=True
            )
            print(f"Number of results: {len(results)}")
            if results:
                print(f"First result vector shape: {len(results[0].vector)}")

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

        print("\nDebug processing results:")
        print(f"Number of results before filtering: {len(results)}")
        print(f"Relevance scores: {scores}")
        print(f"Min relevance threshold: {self.config.min_relevance_score}")

        processed_results = []

        for i, (result, score) in enumerate(zip(results, scores)):
            print(f"\n=== Result {i+1} Details ===")
            print(f"Score: {score}")
            print(f"Payload keys: {result.payload.keys()}")
            print(f"Metadata: {result.payload.get('metadata', {})}")
            print(f"Full payload: {result.payload}")
            print(f"Text length: {len(result.payload.get('text', ''))}")
            print("First 300 chars of text:")
            print(result.payload.get('text', '')[:300])
            print("="*50)
            
            if score < self.config.min_relevance_score:
                print(f"Filtered out due to low score ({float(score)} < {self.config.min_relevance_score})")
                continue
    
            metadata = result.payload.get('metadata', {})
            source = metadata.get('source', 'unknown')
            text = result.payload.get('text', '').strip()
            
            processed_results.append({
                'score': float(score),
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
            context_str += f"[Score: {r ['score']:.2f}] From {r['source']}:\n{r['text']}\n\n"
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