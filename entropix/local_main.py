from typing import Tuple, Literal, Optional
import time
import os

import math
from pathlib import Path
import dataclasses

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.experimental import mesh_utils
import tyro

from entropix.config import LLAMA_1B_PARAMS, LLAMA_70B_PARAMS
from entropix.kvcache import KVCache
from entropix.model import xfmr
from entropix.sampler import sample # sample wraps DSlider with additional sampling logic
from entropix.tokenizer import Tokenizer
from entropix.weights import load_weights
from entropix.dslider import initialize_state
from entropix.dslider_config import DEFAULT_DS_CONFIG
from entropix.medqaprompt import MedicalQAPrompt
from entropix.resultshandler import ResultsHandler
from vectordb.conceptextractor import OpenAIConceptExtractor
from vectordb.retrieval_system import RetrievalSystem, RetrievalConfig

import logging

logging.basicConfig(filename='medqa_debug.log', level=logging.DEBUG)

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
    if end <= 0:
        raise ValueError(f"Invalid end value for freqs_cis: {end}")
    if dim <= 0:
        raise ValueError(f"Invalid dim value for freqs_cis: {dim}")

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

def main(
        model_size: Literal["1B", "70B"] = "70B",
        weights_path: Optional[Path] = None,
        mode: Literal["sort", "medqa_example", "benchmark"] = "sort",
        max_questions: Optional[int] = None,
        dataset_path: Path = Path("entropix/data/test.jsonl") # Path("entropix/data/US_qbank.jsonl")
):
    """
    Main function to run the model in different modes.
    """
    model_params = LLAMA_1B_PARAMS if model_size == "1B" else LLAMA_70B_PARAMS
    if weights_path is None:
        weights_path = DEFAULT_WEIGHTS_PATH.joinpath(f"{model_size}-Instruct")
    
    print(f"Loading {model_size} model from {weights_path}")
    xfmr_weights, mesh = load_weights(weights_path.absolute(), model_params)
    print(f"Created mesh with devices: {mesh.devices}")

    tokenizer = Tokenizer('entropix/tokenizer.model')
    xfmr_fn = jax.jit(xfmr, static_argnames=("model_params",))
    concept_extractor = OpenAIConceptExtractor()
    retrieval_config = RetrievalConfig(
        k=model_params.retrieval_chunks,
        max_context_length=model_params.retrieval_context_length
    )
    retrieval_system = RetrievalSystem(config=retrieval_config)

    def process_question(prompt, question=None):
        """Process a single question"""
        print("\n[Processing Debug] Starting new question processing")
        print("-" * 80)
        print(f"Input prompt length: {len(prompt)}")
        if question:
            print(f"Question type: {question.meta_info}")
        print("-" * 80)

        # Calculate safe lengths
        SYSTEM_RESERVE = 2048
        GENERATION_RESERVE = 1024  # Reserve tokens for generation
        USER_RESERVE = 1024
        MAX_SAFE_LENGTH = model_params.max_seq_len - (SYSTEM_RESERVE - GENERATION_RESERVE - USER_RESERVE)

        tokens = tokenizer.encode(prompt, bos=False, eos=False, allowed_special='all')
        if len(tokens) > MAX_SAFE_LENGTH:
            print(f"Warning: Question is too long ({len(tokens)} tokens), truncating to {MAX_SAFE_LENGTH} tokens")
            tokens = tokens[:MAX_SAFE_LENGTH]
        
        print(f"\n[Processing Debug] Tokenized input:")
        print(f"Number of tokens: {len(tokens)}")
        print(f"Max sequence length: {model_params.max_seq_len}")

        cur_pos = 0
        tokens = jnp.array([tokens], jnp.int32)
        bsz, seqlen = tokens.shape

        # Setup attention and frequency components
        attn_mask = build_attn_mask(seqlen, cur_pos)

        print(f"model_params.head_dim: {model_params.head_dim}")
        print(f"seqlen: {seqlen}")

        freqs_cis = precompute_freqs_cis(model_params.head_dim, model_params.max_seq_len, model_params.rope_theta, model_params.use_scaled_rope)  

        # Initialize KV Cache
        kvcache = KVCache.new(model_params.n_layers, bsz, model_params.max_seq_len, model_params.n_local_kv_heads, model_params.head_dim)

        key = jax.random.PRNGKey(0)
        generated_text = ""
        
        # Initial FWD pass
        with mesh:
            logits, kvcache, scores, _ = xfmr_fn(xfmr_weights, model_params, tokens, cur_pos, freqs_cis[:seqlen], kvcache, attn_mask=attn_mask)
            # Initialize state for this question
            state = initialize_state(logits, bsz, DEFAULT_DS_CONFIG)
            cur_pos = seqlen

            while cur_pos < model_params.max_seq_len:
                key, subkey = jax.random.split(key)
                next_token, new_state = sample(state, logits[:, -1], DEFAULT_DS_CONFIG, key=subkey)
                state = new_state

                try:
                    if isinstance(next_token, jnp.ndarray):
                        token_val = next_token[0][0]
                    else:
                        raise ValueError(f"Unexpected token type: {type(next_token)}")

                    if token_val in [tokenizer.eot_id, tokenizer.eom_id] or token_val in [128001, 128008, 128009]:
                        break
                    
                    out_token = tokenizer.decode([token_val])
                    generated_text += out_token
                    print(out_token, end='', flush=True)

                    # Update for next iteration
                    next_token = next_token.reshape((bsz, 1))
                    logits, kvcache, scores, _ = xfmr_fn(xfmr_weights, model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos+1], kvcache)
                    cur_pos += 1
                except Exception as e:
                    print(f"\nError processing token {token_val}: {e}")
                    logging.error(f"Error processing token {token_val}: {e}")
                    break

        return generated_text

    if mode == "sort":
        prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> 

Environment: ipython
Cutting Knowledge Date: December 2023
Today Date: 14 November 2024

Think carefully in a step-by-step manner. which number is larger, 9.9 or 9.11? Double check your answer.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        print(prompt)
        process_question(prompt)

    elif mode == "medqa_example":
        print("\nStarting MedQA example...")
        prompt_handler = MedicalQAPrompt()
        example_question= {
            "question": "A 45-year-old man comes to the physician because of severe left knee pain and swelling. He has hypercholesterolemia and hypertension. Current medications include pravastatin and captopril. He eats a low-fat diet that includes fish and leafy green vegetables. He drinks 4–6 cups of coffee daily. He has smoked one pack of cigarettes daily for 26 years and drinks 2–3 beers daily. Vital signs are within normal limits. Examination of the left knee shows swelling, warmth, and severe tenderness to palpation. Arthrocentesis is performed. Gram stain is negative. Analysis of the synovial fluid shows monosodium urate crystals. Which of the following health maintenance recommendations is most appropriate to prevent symptom recurrence?", 
            "answer": "F", 
            "options": {
                "A": "Discontinue captopril", 
                "B": "Start aspirin", 
                "C": "Replace beer with red wine", 
                "D": "Stop smoking", 
                "E": "Reduce coffee intake", 
                "F": "Reduce fish intake", 
                "G": "Discontinue pravastatin", 
                "H": "Start colchicine"
            }, 
            "meta_info": "step2",
            "answer_idx": "F"
        }
        retrieved_context = retrieval_system.retrieve(example_question["question"])
        prompt, question = prompt_handler.get_prompt(example_question)
        prompt = prompt_handler.update_prompt_with_context(prompt, retrieved_context)

        generated_text = process_question(prompt, question)

        with open('debug_output.txt', 'w') as f:
            f.write(f"Generated text length: {len(generated_text)}\n")
            f.write(f"Generated text:\n{generated_text}\n")
            f.write("\nFormatted output:\n")
            f.write(prompt_handler.format_output(question, generated_text))

    else:  # mode == "benchmark"
        prompt_handler = MedicalQAPrompt()
        results_handler = ResultsHandler()
        question_counter = 0
        prompts = prompt_handler.process_jsonl_file(str(dataset_path), max_questions)

        for prompt, question in prompts:
            try:
                print(f"Processing question {question_counter + 1}")
                concepts = concept_extractor.extract_concepts(question.question)

                base_prompt_tokens = tokenizer.encode(prompt, bos=False, eos=False, allowed_special='all')
                SYSTEM_RESERVE = 2048
                USER_RESERVE = 1024
                GENERATION_RESERVE = 1024
                MAX_CONTEXT_LENGTH = model_params.max_seq_len - (len(base_prompt_tokens) + SYSTEM_RESERVE + GENERATION_RESERVE + USER_RESERVE)
                
                retrieval_config.max_context_length = MAX_CONTEXT_LENGTH
                retrieved_context = retrieval_system.retrieve(question.question, concepts)
                augmented_prompt = prompt_handler.update_prompt_with_context(prompt, retrieved_context)
                
                # Truncate the augmented prompt if needed
                augmented_prompt_tokens = tokenizer.encode(augmented_prompt, bos=False, eos=False, allowed_special='all')
                if len(augmented_prompt_tokens) > (model_params.max_seq_len - GENERATION_RESERVE):
                    print(f"Warning: Truncating augmented prompt from {len(augmented_prompt_tokens)} tokens")
                    augmented_prompt = tokenizer.decode(augmented_prompt_tokens[:model_params.max_seq_len - GENERATION_RESERVE])

                generated_text = process_question(augmented_prompt, question=question)
                try:
                    formatted_output = prompt_handler.format_output(question, generated_text)
                    print("\n" + "="*80)
                    print(formatted_output)

                    results_handler.process_result(formatted_output, question_id=f"q_{question_counter}")
                    if (question_counter + 1) % 10 == 0:
                        stats = results_handler.get_stats()
                        print(f"\nProgress: {question_counter + 1}/{len(prompts)} questions processed. "
                            f"Success rate: {stats['success_rate']}")
                except Exception as e:
                    logging.error(f"Error in formatting output for question {question_counter}: {e}")

                question_counter += 1

            except Exception as e:
                logging.error(f"Error in generation for question {question_counter}: {e}")
                question_counter += 1
                continue

        results_handler.finalize()

import os
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)
print("JAX devices:", jax.devices())
print("Number of devices:", jax.device_count())

if __name__ == '__main__':
    tyro.cli(main)