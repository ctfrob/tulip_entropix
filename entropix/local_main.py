from typing import Tuple, Literal, Optional
from pathlib import Path
import dataclasses

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.experimental import mesh_utils
import tyro

from entropix.config import MODEL_CONFIGS, create_model_params
from entropix.config import LLAMA_1B_PARAMS
from entropix.kvcache import KVCache
from entropix.model import xfmr
from entropix.sampler import SamplerConfig, sample
from entropix.prompts import generate_chat_prompt
from entropix.tokenizer import Tokenizer
from entropix.weights import load_weights
from entropix.dslider import initialize_state, DEFAULT_DS_CONFIG
from MedQAPrompt_Builder import MedicalQAPrompt

DEFAULT_WEIGHTS_PATH = Path(__file__).parent / '../weights'

def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0, use_scaled: bool = False, dtype: jnp.dtype = jnp.float32) -> jax.Array:
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
    t = jnp.arange(end, dtype=dtype)
    freqs = jnp.outer(t, freqs)
    return jnp.exp(1j * freqs)

def build_attn_mask(seqlen: int, start_pos: int) -> jax.Array:
    mask = jnp.zeros((seqlen, seqlen), dtype=jnp.float32)
    if seqlen > 1:
        mask = jnp.full((seqlen, seqlen), float('-inf'))
        mask = jnp.triu(mask, k=1)
        mask = jnp.hstack([jnp.zeros((seqlen, start_pos)), mask], dtype=jnp.float32)
    return mask

def generate(xfmr_weights, model_params, tokens, mesh, tokenizer, xfmr_fn):
    """Original working generation function"""
    gen_tokens = None
    cur_pos = 0
    with mesh:
        tokens = jnp.array([tokens], jnp.int32)
        bsz, seqlen = tokens.shape
        attn_mask = build_attn_mask(seqlen, cur_pos)
        freqs_cis = precompute_freqs_cis(
            model_params.head_dim, 
            model_params.max_seq_len, 
            model_params.rope_theta, 
            model_params.use_scaled_rope
        )
        kvcache = KVCache.new(
            model_params.n_layers, 
            bsz, 
            model_params.max_seq_len, 
            model_params.n_local_kv_heads, 
            model_params.head_dim
        )
        logits, kvcache, scores, stats = xfmr_fn(
            xfmr_weights, 
            model_params, 
            tokens, 
            cur_pos, 
            freqs_cis[:seqlen], 
            kvcache, 
            attn_mask=attn_mask
        )
        
        # Initialize state for sampling
        state = initialize_state(logits, bsz, DEFAULT_DS_CONFIG)
        next_token, state = sample(state, logits[:, -1], DEFAULT_DS_CONFIG)
        print(tokenizer.decode([next_token.item()]), end='', flush=True)
        
        cur_pos = seqlen
        stop = jnp.array([128001, 128008, 128009])
        gen_tokens = [next_token]
        
        while cur_pos < 8192:
            cur_pos += 1
            logits, kvcache, scores, stats = xfmr_fn(
                xfmr_weights, 
                model_params, 
                next_token, 
                cur_pos, 
                freqs_cis[cur_pos:cur_pos+1], 
                kvcache
            )
            next_token, state = sample(state, logits[:, -1], DEFAULT_DS_CONFIG)
            gen_tokens.append(next_token)
            
            try:
                token_val = next_token.tolist()[0][0]
                
                # Check ALL special/stop tokens
                if token_val in [tokenizer.eot_id, tokenizer.eom_id] or token_val in [128001, 128008, 128009]:
                    break
                    
                # Only try to decode if it's not a special token
                out_token = tokenizer.decode([token_val])
                print(out_token, end='', flush=True)
                
            except Exception as e:
                print(f"\nError in token processing: {str(e)}")
                break

@dataclasses.dataclass
class Args:
    """Command line arguments for the script"""
    mode: Literal["hardcoded", "medical"] = "hardcoded"
    max_questions: Optional[int] = None
    dataset: Optional[Path] = None
    weights_path: Path = DEFAULT_WEIGHTS_PATH.joinpath('1B-Instruct')

    def __post_init__(self):
        if self.mode == "medical" and not self.dataset:
            raise ValueError("Dataset path required for medical mode")

def main(args: Args):
    # Initialize model
    model_config = MODEL_CONFIGS["1B"]
    model_params = create_model_params(model_config)
    
    # Get both weights and mesh
    xfmr_weights, mesh = load_weights(args.weights_path.absolute(), model_params)
    tokenizer = Tokenizer('entropix/tokenizer.model')
    xfmr_fn = jax.jit(xfmr, static_argnames=("model_params",))
    sample_fn = jax.jit(sample, static_argnames=("config",))

    if args.mode == "hardcoded":
        # Original hardcoded prompt
        prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Environment: ipython
Cutting Knowledge Date: December 2023
Today Date: 23 July 2024

Think carefully in a step-by-step manner. which number is larger, 9.9 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
        
        print(prompt)
        tokens = tokenizer.encode(prompt, bos=False, eos=False, allowed_special='all')
        generate(xfmr_weights, model_params, tokens, mesh, tokenizer, xfmr_fn)
    
    else:  # medical mode
        prompt_handler = MedicalQAPrompt()
        prompts = prompt_handler.process_jsonl_file(args.dataset, args.max_questions)
        
        for prompt, question in prompts:
            print("\nProcessing question:", question)
            tokens = tokenizer.encode(prompt, bos=False, eos=False, allowed_special='all')
            generate(xfmr_weights, model_params, tokens, mesh, tokenizer, xfmr_fn)
            print("\nPress Enter to continue to next question...")
            input()

# Set environment variables for XLA
import os
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)

if __name__ == '__main__':
    tyro.cli(main)