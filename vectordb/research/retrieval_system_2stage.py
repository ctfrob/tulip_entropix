from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import os
import numpy as np
from pathlib import Path
from qdrant_client import QdrantClient
import logging

from openai import OpenAI
from dotenv import load_dotenv

import jax
import jax.numpy as jnp

from entropix.config import ModelParams, MODEL_CONFIGS
from entropix.kvcache import KVCache
from entropix.sampler import DEFAULT_DS_CONFIG
from entropix.tokenizer import Tokenizer
from entropix.dslider import initialize_state

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class RetrievalConfig:
    k: int = 2  # Number of documents to retrieve
    min_relevance_score: float = 0.3
    max_context_length: Optional[int] = None
    batch_size: int = 32
    collection_name: str = "textbooks_oaiembed"
    context_template: str = "\nContext: {context}\n"
    chunk_size: int = 512
    debug: bool = False
    
    def adjust_for_model(self, model_params: ModelParams):
        """Adjust retrieval parameters based on model size"""
        if model_params.n_layers == MODEL_CONFIGS["1B"].n_layers:
            self.k = 1
            self.max_context_length = 2048
        else:
            self.k = 2
            self.max_context_length = 3072
            
        print(f"Adjusted for {'1B' if model_params.n_layers == MODEL_CONFIGS['1B'].n_layers else '70B'} model:")
        print(f"- Retrieved chunks (k): {self.k}")
        print(f"- Max context length: {self.max_context_length}")
        return self

class RetrievalSystem:
    def __init__(self, 
                 config: Optional[RetrievalConfig] = None, 
                 model_params: Optional[ModelParams] = None,
                 xfmr_weights=None,
                 xfmr_fn=None,
                 sample_fn=None):
        """Initialize the retrieval system"""
        self.config = config or RetrievalConfig()
        self.model_params = model_params
        if model_params:
            is_1B = model_params.n_layers == MODEL_CONFIGS["1B"].n_layers
            print(f"Initializing retrieval system for {'1B' if is_1B else '70B'} model")
            self.config = self.config.adjust_for_model(model_params)
        
        # Store model components
        self.xfmr_weights = xfmr_weights
        self.xfmr_fn = xfmr_fn
        self.sample_fn = sample_fn
        self.tokenizer = Tokenizer('entropix/tokenizer.model')
        
        self.vector_dim = 3072  # text-embedding-3-large dimension
        self.embedding_cache = {}

        self.concept_extraction_prompt = """You are a medical expert. Extract only essential medical concepts from this clinical question that would help retrieve relevant medical knowledge. Be extremely concise.
Format as:
Primary: [key diagnoses, conditions, symptoms]
Risk Factors: [relevant patient history, behaviors]
Assessment Areas: [key medical areas to consider]

Question: {question}"""

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
        
        collection_info = self.client.get_collection('textbooks_oaiembed')
        print(f"Collection info: {collection_info}")

    def build_attn_mask(self, seqlen: int, start_pos: int) -> jax.Array:
        """Build attention mask for transformer"""
        mask = jnp.zeros((seqlen, seqlen), dtype=jnp.float32)
        if seqlen > 1:
            mask = jnp.full((seqlen, seqlen), float('-inf'))
            mask = jnp.triu(mask, k=1)
            mask = jnp.hstack([jnp.zeros((seqlen, start_pos)), mask], dtype=jnp.float32)
        return mask

    def precompute_freqs_cis(self, dim: int, end: int, theta: float = 500000.0) -> jax.Array:
        """Precompute frequency cis for rotary embeddings"""
        freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim))
        t = jnp.arange(end, dtype=jnp.float32)
        freqs = jnp.outer(t, freqs)
        return jnp.exp(1j * freqs)

    def _generate_with_model(self, tokens: List[int]) -> str:
        """Generate text using the main model"""
        if not all([self.xfmr_weights, self.xfmr_fn, self.sample_fn, self.model_params]):
            logger.warning("Model components not initialized, skipping concept extraction")
            return ""

        try:
            tokens = jnp.array([tokens], jnp.int32)
            cur_pos = 0
            bsz, seqlen = tokens.shape
            
            # Setup model components
            attn_mask = self.build_attn_mask(seqlen, cur_pos)
            freqs_cis = self.precompute_freqs_cis(
                self.model_params.head_dim, 
                self.model_params.max_seq_len,
                self.model_params.rope_theta
            )
            kvcache = KVCache.new(
                self.model_params.n_layers,
                bsz,
                self.model_params.max_seq_len,
                self.model_params.n_local_kv_heads,
                self.model_params.head_dim
            )

            # Initial forward pass
            logits, kvcache, _, _ = self.xfmr_fn(
                self.xfmr_weights,
                self.model_params,
                tokens,
                cur_pos,
                freqs_cis[:seqlen],
                kvcache,
                attn_mask=attn_mask
            )
            
            # Generate concepts
            state = initialize_state(logits, bsz, DEFAULT_DS_CONFIG)
            generated_text = []
            cur_pos = seqlen
            
            while cur_pos < 8192:
                next_token, state = self.sample_fn(state, logits[:, -1], DEFAULT_DS_CONFIG)
                token_val = next_token.tolist()[0][0]
                
                if token_val in [self.tokenizer.eot_id, self.tokenizer.eom_id]:
                    break
                    
                out_token = self.tokenizer.decode([token_val])
                generated_text.append(out_token)
                
                logits, kvcache, _, _ = self.xfmr_fn(
                    self.xfmr_weights,
                    self.model_params,
                    next_token,
                    cur_pos,
                    freqs_cis[cur_pos:cur_pos+1],
                    kvcache
                )
                cur_pos += 1
            
            return ''.join(generated_text)
            
        except Exception as e:
            logger.error(f"Error in concept generation: {e}")
            return ""

    def extract_medical_concepts(self, query: str) -> str:
        """Extract key medical concepts from the query"""
        try:
            tokens = self.tokenizer.encode(
                self.concept_extraction_prompt.format(question=query),
                bos=False,
                eos=False,
                allowed_special='all'
            )
            
            generated_concepts = self._generate_with_model(tokens)
            concepts = self._clean_concept_output(generated_concepts)
            
            if self.config.debug:
                print(f"Extracted concepts:\n{concepts}")
            
            return concepts
            
        except Exception as e:
            logger.warning(f"Concept extraction failed: {e}")
            return ""

    def _clean_concept_output(self, text: str) -> str:
        """Clean and format concept extraction output"""
        text = text.replace("<|begin_of_text|>", "").replace("<|end_of_text|>", "")
        
        if "Primary:" in text:
            text = text[text.find("Primary:"):]
            
        if "<|" in text:
            text = text[:text.find("<|")]
            
        return text.strip()

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

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant context with concept enhancement"""
        try:
            # Extract medical concepts
            concepts = self.extract_medical_concepts(query) if all([
                self.xfmr_weights, self.xfmr_fn, self.sample_fn, self.model_params
            ]) else ""
            
            # Create enhanced query
            enhanced_query = query
            if concepts:
                enhanced_query = f"""Clinical Question: {query}
Relevant Medical Concepts: {concepts}"""
            
            if self.config.debug:
                print(f"Enhanced query:\n{enhanced_query}")
            
            # Generate embedding and search
            query_vector = self.encode_query(enhanced_query)
            
            results = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=query_vector.tolist(),
                limit=self.config.k * 2
            )
            
            return self._process_results(results)
            
        except Exception as e:
            logger.error(f"Error in enhanced retrieval: {e}")
            raise

    def _filter_by_relevance(self, results: List, min_score: float) -> List:
        """Filter results based on relevance scores"""
        return [r for r in results if r.score >= min_score]

    def format_for_prompt(self, results: List) -> str:
        """Format retrieved contexts for RAG prompt with strict token counting"""
        formatted_contexts = []
        total_tokens = 0
        max_tokens = min(self.config.max_context_length, 3500)

        for result in results:
            context = result.payload.get('text', '').strip()
            context_tokens = self.tokenizer.encode(context)

            if total_tokens + len(context_tokens) > max_tokens:
                available_tokens = max_tokens - total_tokens
                if available_tokens > 100:
                    truncated_text = self.tokenizer.decode(context_tokens[:available_tokens])
                    formatted_contexts.append(truncated_text)
                break
            
            formatted_contexts.append(context)
            total_tokens += len(context_tokens)
            
        combined_context = "\n".join(formatted_contexts)
        formatted = self.config.context_template.format(context=combined_context)
        
        final_tokens = self.tokenizer.encode(formatted)
        if len(final_tokens) > max_tokens:
            print(f"Warning: Final context too long ({len(final_tokens)} tokens), truncating...")
            formatted = self.tokenizer.decode(final_tokens[:max_tokens])
    
        print(f"Final context length in tokens: {len(self.tokenizer.encode(formatted))}")
        return formatted

    def _process_results(self, results: List) -> List[Dict]:
        """Process the search results"""
        if not results:
            return []

        processed_results = []
        if self.config.debug:
            print(f"\nNumber of results before filtering: {len(results)}")
        
        for i, result in enumerate(results):
            if self.config.debug:
                print(f"\n=== Result {i+1} Details ===")
                print(f"Score: {result.score}")
                print(f"Payload keys: {result.payload.keys()}")
                print(f"Metadata: {result.payload.get('metadata', {})}")
            
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

        processed_results.sort(key=lambda x: x['score'], reverse=True)
        final_results = processed_results[:self.config.k]
        
        if self.config.debug:
            print(f"\nReturning {len(final_results)} results after processing")
            for r in final_results:
                print(f"\n[Score: {r['score']:.2f}] From {r['source']}:")
                print(f"Text preview: {r['text'][:200]}...")

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