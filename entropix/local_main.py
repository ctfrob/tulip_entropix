from typing import Tuple, Literal, Optional
from pathlib import Path
import dataclasses

import math
import tyro
import jax
import jax.numpy as jnp

from entropix.config import LLAMA_1B_PARAMS
from entropix.kvcache import KVCache
from entropix.model import xfmr
from entropix.sampler import sample
from entropix.tokenizer import Tokenizer
from entropix.weights import load_weights
from entropix.dslider import initialize_state
from entropix.dslider_config import DEFAULT_DS_CONFIG

from MedQAPrompt_Builder import MedicalQAPrompt

DEFAULT_WEIGHTS_PATH = Path(__file__).parent / '../weights'

def apply_scaling(freqs: jax.Array):
    SCALE_FACTOR = 8
    LOW_FREQ_FACTOR = 1
    HIGH_FREQ_FACTOR = 4
    OLD_CONTEXT_LEN = 8192  # original llama3 length

    low_freq_wavelen = OLD_CONTEXT_LEN / LOW_FREQ_FACTOR
    high_freq_wavelen = OLD_CONTEXT_LEN / HIGH_FREQ_FACTOR

    def scale_freq(freq):
        wavelen = 2 * math.pi / freq

        def scale_mid(_):
            smooth = (OLD_CONTEXT_LEN / wavelen - LOW_FREQ_FACTOR) / (HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR)
            return (1 - smooth) * freq / SCALE_FACTOR + smooth * freq

        return jax.lax.cond(
            wavelen < high_freq_wavelen,
            lambda _: freq,
            lambda _: jax.lax.cond(wavelen > low_freq_wavelen, lambda _: freq / SCALE_FACTOR, scale_mid, None),
            None
        )

    return jax.vmap(scale_freq)(freqs)


def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0, use_scaled: bool = False, dtype: jnp.dtype = jnp.float32) -> jax.Array:
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
    if use_scaled:
        freqs = apply_scaling(freqs)
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

@dataclasses.dataclass
class Args:
    mode: Literal["hardcoded", "medical"] = "hardcoded"
    max_questions: Optional[int] = None
    dataset: Optional[Path] = None
    weights_path: Path = DEFAULT_WEIGHTS_PATH.joinpath('1B-Instruct')

    def __post_init__(self):
        if self.mode == "medical" and not self.dataset:
            raise ValueError("Dataset path required for medical mode")

def run_generation(
    prompt: str,
    model_params,
    xfmr_weights,
    tokenizer,
    xfmr_fn,
    sample_fn,
    freqs_cis
) -> str:
    """Run single generation for a prompt"""
    tokens = tokenizer.encode(prompt, bos=False, eos=False, allowed_special='all')
    
    cur_pos = 0
    tokens = jnp.array([tokens], jnp.int32)
    bsz, seqlen = tokens.shape
    attn_mask = build_attn_mask(seqlen, cur_pos)
    kvcache = KVCache.new(model_params.n_layers, bsz, model_params.max_seq_len, 
                         model_params.n_local_kv_heads, model_params.head_dim)
    
    logits, kvcache, _, _ = xfmr_fn(
        xfmr_weights, model_params, tokens, cur_pos, 
        freqs_cis[:seqlen], kvcache, attn_mask=attn_mask
    )
    
    state = initialize_state(logits, bsz, DEFAULT_DS_CONFIG)
    next_token, state = sample(state, logits[:, -1], DEFAULT_DS_CONFIG)
    output = [tokenizer.decode([next_token.item()])]
    
    cur_pos = seqlen
    gen_tokens = [next_token]
    
    while cur_pos < 8192:
        cur_pos += 1
        logits, kvcache, _, _ = xfmr_fn(
            xfmr_weights, model_params, next_token, cur_pos,
            freqs_cis[cur_pos:cur_pos+1], kvcache
        )
        next_token, state = sample(state, logits[:, -1], DEFAULT_DS_CONFIG)
        gen_tokens.append(next_token)
        
        try:
            token_val = next_token.tolist()[0][0]
            if token_val in [tokenizer.eot_id, tokenizer.eom_id] or token_val in [128001, 128008, 128009]:
                break
            
            out_token = tokenizer.decode([token_val])
            output.append(out_token)
            
        except Exception as e:
            print(f"\nError in token processing: {str(e)}")
            break
    
    return "".join(output)

def main(args: Args):
    # Initialize model components
    model_params = LLAMA_1B_PARAMS
    xfmr_weights, mesh = load_weights(args.weights_path.absolute(), model_params)
    tokenizer = Tokenizer('entropix/tokenizer.model')
    xfmr_fn = jax.jit(xfmr, static_argnames=("model_params",))
    sample_fn = jax.jit(sample)
    
    # Precompute frequency cis
    freqs_cis = precompute_freqs_cis(
        model_params.head_dim, 
        model_params.max_seq_len,
        model_params.rope_theta, 
        model_params.use_scaled_rope
    )

    if args.mode == "hardcoded":
        # Original hardcoded prompt
        prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Environment: ipython
Cutting Knowledge Date: December 2023
Today Date: 12 November 2024

Think carefully in a step-by-step manner. which number is larger, 9.9 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
        
        output = run_generation(
            prompt, model_params, xfmr_weights, tokenizer,
            xfmr_fn, sample_fn, freqs_cis
        )
        print(output)
    
    else:  # medical mode
        prompt_handler = MedicalQAPrompt()
        prompts = prompt_handler.process_jsonl_file(args.dataset, args.max_questions)
        
        for prompt, question in prompts:
            output = run_generation(
                prompt, model_params, xfmr_weights, tokenizer,
                xfmr_fn, sample_fn, freqs_cis
            )
            formatted_output = prompt_handler.format_output(question, output)
            print(formatted_output)
            print("\nPress Enter to continue to next question...")
            input()

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