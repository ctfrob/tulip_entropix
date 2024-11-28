from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import os
import numpy as np
import logging
import time
from pathlib import Path
from qdrant_client import QdrantClient
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class ModelSizeConfig:
    k: int
    max_context_length: Optional[int]
    chunk_size: int

@dataclass
class RetrievalConfig:
    k: int = 2
    min_relevance_score: float = 0.25
    max_context_length: Optional[int] = None
    collection_name: str = "textbooks_oaiembed"
    context_template: str = "\nContext: {context}\n"
    chunk_size: int = 512
    debug: bool = False

    # Define MODEL_CONFIGS as a class variable outside the dataclass fields
    MODEL_CONFIGS = {
        "1B": ModelSizeConfig(
            k=1,
            max_context_length=1500,
            chunk_size=384
        ),
        "70B": ModelSizeConfig(
            k=3,
            max_context_length=6000,
            chunk_size=512
        )
    }

    @classmethod
    def for_model(cls, model_size: str) -> 'RetrievalConfig':
        """Create a RetrievalConfig instance optimized for given model size"""
        model_config = cls.MODEL_CONFIGS.get(model_size, cls.MODEL_CONFIGS["1B"])
        return cls(
            k=model_config.k,
            max_context_length=model_config.max_context_length,
            chunk_size=model_config.chunk_size
        )

class RetrievalSystem:
    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        self.vector_dim = 3072  # text-embedding-3-large dimension
        
        try:
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.client = QdrantClient(
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY")
            )
            collections = self.client.get_collections()
            print(f"Connected to collections: {collections}")
        except Exception as e:
            logger.error(f"Error initializing clients: {e}")
            raise

    def _parse_concepts(self, concepts: str) -> Dict[str, str]:
        """Parse structured concepts into a dictionary"""
        concept_dict = {}
        try:
            current_key = None
            current_value = []
            
            for line in concepts.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                if any(key in line for key in ["Primary:", "Risk Factors:", "Assessment Areas:"]):
                    if current_key and current_value:
                        concept_dict[current_key] = ' '.join(current_value)
                    current_key = line.split(':')[0].strip()
                    current_value = [line.split(':', 1)[1].strip()]
                else:
                    if current_key:
                        current_value.append(line)
            
            if current_key and current_value:
                concept_dict[current_key] = ' '.join(current_value)
                
        except Exception as e:
            logger.error(f"Error parsing concepts: {e}")
            
        return concept_dict

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

    def retrieve(self, question: str, concepts: Optional[str] = None) -> List[Dict[str, Any]]:
        print(f"\n[Retrieval] Starting retrieval process...")
        start_time = time.time()
        
        try:
            # Build structured medical query
            structured_query = [
                "Medical search:",
                question.strip()
            ]
            
            # Add concepts if available
            if concepts:
                concept_dict = self._parse_concepts(concepts)
                print(f"[Retrieval Debug] Parsed concepts: {concept_dict}")
                
                if concept_dict.get('Primary'):
                    structured_query.append(f"Key conditions: {concept_dict['Primary']}")
                if concept_dict.get('Risk Factors'):
                    structured_query.append(f"Clinical findings: {concept_dict['Risk Factors']}")
                if concept_dict.get('Assessment Areas'):
                    structured_query.append(f"Medical domains: {concept_dict['Assessment Areas']}")
            
            enhanced_query = '\n'.join(structured_query)
            print(f"[Retrieval Debug] Enhanced query:\n{enhanced_query}")
            
            # Make sure we're in RetrievalSystem
            assert hasattr(self, 'config'), "Must be called from RetrievalSystem"
            
            query_vector = self.encode_query(enhanced_query)
            print(f"[Retrieval Debug] Generated embedding vector")
            
            print(f"[Retrieval Debug] Self type: {type(self)}")
            print(f"[Retrieval Debug] Config type: {type(self.config)}")
            print(f"[Retrieval Debug] Collection name: {self.config.collection_name}")
            
            # Use self.config for RetrievalSystem settings
            results = self.client.search(
                collection_name=self.config.collection_name,  # From RetrievalConfig
                query_vector=query_vector.tolist(),
                limit=self.config.k * 2,  # From RetrievalConfig
                score_threshold=self.config.min_relevance_score  # From RetrievalConfig
            )
            
            if results:
                print(f"[Retrieval Debug] Raw results: {len(results)}")
                for r in results[:2]:
                    print(f"[Retrieval Debug] Score: {r.score:.3f}")
            
            return self._process_results(results)
            
        except Exception as e:
            print(f"[Retrieval Error] {str(e)}")
            logger.error(f"Error in retrieval: {e}")
            return []
        
    def _process_results(self, results: List) -> List[Dict]:
        """Process and filter search results"""
        if not results:
            return []

        processed_results = []
        
        for result in results:
            if result.score < self.config.min_relevance_score:
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

        # Sort by score and take top k
        processed_results.sort(key=lambda x: x['score'], reverse=True)
        final_results = processed_results[:self.config.k]
        
        if self.config.debug:
            print(f"\nReturning {len(final_results)} results after processing")
            for r in final_results:
                print(f"\n[Score: {r['score']:.2f}] From {r['source']}:")
                print(f"Text preview: {r['text'][:200]}...")

        return final_results

    def format_for_prompt(self, results: List[Dict]) -> str:
        """Format retrieved contexts for inclusion in the prompt"""
        if not results:
            return self.config.context_template.format(context="No relevant context found.")
            
        contexts = []
        for result in results:
            context = f"[From {result['source']} (Relevance: {result['score']:.2f})]:\n{result['text']}"
            contexts.append(context)
            
        combined_context = "\n\n".join(contexts)
        return self.config.context_template.format(context=combined_context)

    def close(self):
        """Cleanup and close connections"""
        try:
            self.client.close()
            logger.info("Retrieval system closed successfully")
        except Exception as e:
            logger.error(f"Error closing retrieval system: {e}")