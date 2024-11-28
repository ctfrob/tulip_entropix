from typing import Optional, Dict, List
import time
from dataclasses import dataclass, replace
import logging
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp

from entropix.config import ModelParams
from entropix.kvcache import KVCache
from entropix.tokenizer import Tokenizer
from entropix.dslider import initialize_state
from entropix.sampler import DEFAULT_DS_CONFIG

logger = logging.getLogger(__name__)

@dataclass
class ConceptExtractorConfig:
    max_seq_length: int = 2048 
    temperature: float = 0.5
    max_generation_steps: int = 150 
    concept_template: str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert medical system and helping support a medical doctor with concept extraction. You are not diagnosing the patient, but extracting relevant medical concepts from the question.
Extract ALL relevant medical concepts from the question:

Primary: List ALL diagnoses, injuries, or conditions
Risk Factors: List ALL symptoms, signs, test results, demographics, and relevant history
Assessment Areas: List ALL relevant medical specialties and anatomical areas

Be thorough and specific. Include ALL relevant findings.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Extract concepts from: {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

class ConceptExtractor:
    def __init__(
        self,
        model_params: ModelParams,
        xfmr_weights,
        xfmr_fn,
        sample_fn,
        config: Optional[ConceptExtractorConfig] = None
    ):
        self.config = config or ConceptExtractorConfig()
        self.model_params = model_params
        self.xfmr_weights = xfmr_weights
        self.xfmr_fn = xfmr_fn
        self.sample_fn = sample_fn
        self.tokenizer = Tokenizer('entropix/tokenizer.model')
    
    @staticmethod
    @partial(jax.jit, static_argnames=("temperature",))
    def _simple_sample(logits: jnp.ndarray, temperature: float = 0.7, key=jax.random.PRNGKey(0)):
        """Simple temperature-based sampling for concept extraction"""
        # Apply temperature
        scaled_logits = logits / jnp.clip(temperature, 0.1, 1.0)
        return jax.random.categorical(key, scaled_logits, axis=-1)

    def extract_concepts(self, question: str) -> str:
        start_time = time.time()
        
        try:
            prompt = self.config.concept_template.format(question=question)
            tokens = self.tokenizer.encode(
                prompt,
                bos=False,
                eos=False,
                allowed_special='all'
            )

            tokens = jnp.array([tokens], jnp.int32)
            initial_pos = 0
            bsz, seqlen = tokens.shape

            attn_mask = self._build_attn_mask(seqlen, initial_pos)
            freqs_cis = self._precompute_freqs_cis(
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
                initial_pos,
                freqs_cis[:seqlen],
                kvcache,
                attn_mask=attn_mask
            )
            
            # Generation loop
            generated_text = []
            generation_pos = 0  # Track number of new tokens generated
            sequence_pos = seqlen  # Track position in full sequence
            key = jax.random.PRNGKey(0)
            
            while generation_pos < self.config.max_generation_steps:
                try:
                    key, subkey = jax.random.split(key)
                    
                    next_token = self._simple_sample(
                        logits[:, -1], 
                        temperature=self.config.temperature,
                        key=subkey
                    )
                    token_val = next_token.item()
                    out_token = self.tokenizer.decode([token_val])
                    generated_text.append(out_token)
                    
                    # Early stopping conditions
                    current_text = ''.join(generated_text)
                    if token_val in [self.tokenizer.eot_id, self.tokenizer.eom_id]:
                        break
                        
                    # Stop if we've generated concept headers and seem to be starting a new section
                    if "Assessment Areas:" in current_text and len(''.join(generated_text)) > 50:
                        break

                    logits, kvcache, _, _ = self.xfmr_fn(
                        self.xfmr_weights,
                        self.model_params,
                        next_token.reshape(1, 1),
                        sequence_pos,
                        freqs_cis[sequence_pos:sequence_pos+1],
                        kvcache
                    )
                    
                    generation_pos += 1
                    sequence_pos += 1

                except Exception as e:
                    print(f"[ConceptExtractor] Generation step failed: {e}")
                    break

            concepts = self._clean_output(''.join(generated_text))
            print(f"[ConceptExtractor] Raw generated text:\n{''.join(generated_text)}")
            print(f"[ConceptExtractor] Parsed concepts:\n{concepts}")
            elapsed = time.time() - start_time
            print(f"[ConceptExtractor] Completed in {elapsed:.2f}s")
            print(f"[ConceptExtractor] Final concepts:\n{concepts}\n")
            return concepts
        
        except Exception as e:
            logger.error(f"Error in concept extraction: {e}")
            return ""
        
    def _clean_output(self, text: str) -> str:
        """Clean and format concept extraction output"""
        try:
            # Initialize default structure
            concept_dict = {
                'Primary': [],
                'Risk Factors': [],
                'Assessment Areas': []
            }

            current_section = 'Primary'  # Default to Primary for initial content
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check for section headers
                if 'Primary:' in line:
                    current_section = 'Primary'
                    content = line.split('Primary:')[1].strip()
                elif 'Risk Factors:' in line:
                    current_section = 'Risk Factors'
                    content = line.split('Risk Factors:')[1].strip()
                elif 'Assessment Areas:' in line:
                    current_section = 'Assessment Areas'
                    content = line.split('Assessment Areas:')[1].strip()
                else:
                    # If no header, add to current section
                    content = line

                if content and current_section:
                    concept_dict[current_section].append(content)

            # Format output
            return "\n".join([
                f"Primary: {', '.join(concept_dict['Primary'])}",
                f"Risk Factors: {', '.join(concept_dict['Risk Factors'])}",
                f"Assessment Areas: {', '.join(concept_dict['Assessment Areas'])}"
            ])

        except Exception as e:
            logger.error(f"Error in clean_output: {e}")
            return ""
    
    def _build_attn_mask(self, seqlen: int, start_pos: int) -> jax.Array:
        """Build attention mask for transformer"""
        mask = jnp.zeros((seqlen, seqlen), dtype=jnp.float32)
        if seqlen > 1:
            mask = jnp.full((seqlen, seqlen), float('-inf'))
            mask = jnp.triu(mask, k=1)
            mask = jnp.hstack([jnp.zeros((seqlen, start_pos)), mask], dtype=jnp.float32)
        return mask

    def _precompute_freqs_cis(self, dim: int, end: int, theta: float = 500000.0) -> jax.Array:
        """Precompute frequency cis for rotary embeddings"""
        freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim))
        t = jnp.arange(end, dtype=jnp.float32)
        freqs = jnp.outer(t, freqs)
        return jnp.exp(1j * freqs)