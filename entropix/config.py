"""Configuration definitions for Entropix models."""

from dataclasses import dataclass
from typing import NamedTuple


@dataclass
class ModelConfig:
  """Base configuration for transformer models."""
  dim: int
  n_layers: int
  n_heads: int
  n_kv_heads: int
  vocab_size: int
  ffn_dim_multiplier: float
  multiple_of: int
  norm_eps: float
  rope_theta: float
  use_scaled_rope: bool
  max_seq_len: int
  retrieval_chunks: int = 1  # Number of chunks to retrieve
  retrieval_context_length: int = 1024  # Max context length for retrieval


# Define standard model configurations
MODEL_CONFIGS = {
  "1B": ModelConfig(
    dim=2048,
    n_layers=16,
    n_heads=32,
    n_kv_heads=8,
    vocab_size=128256,
    ffn_dim_multiplier=1.5,
    multiple_of=256,
    norm_eps=1e-05,
    rope_theta=500000.0,
    use_scaled_rope=True,
    max_seq_len=4096,
    retrieval_chunks=1,
    retrieval_context_length=1024,
  ),
  "70B": ModelConfig(
    dim=8192,
    n_layers=80,
    n_heads=64,
    n_kv_heads=8,
    vocab_size=128256,
    ffn_dim_multiplier=1.5,
    multiple_of=256,
    norm_eps=1e-05,
    rope_theta=500000.0,
    use_scaled_rope=True,
    max_seq_len=4096,
    retrieval_chunks=2,
    retrieval_context_length=2048,
  ),
}


class ModelParams(NamedTuple):
  """Runtime parameters for model execution."""

  n_layers: int
  n_local_heads: int
  n_local_kv_heads: int
  head_dim: int
  max_seq_len: int
  rope_theta: float
  use_scaled_rope: bool
  retrieval_chunks: int
  retrieval_context_length: int


def create_model_params(config: ModelConfig) -> ModelParams:
  """Creates ModelParams from a ModelConfig."""
  return ModelParams(
    n_layers=config.n_layers,
    n_local_heads=config.n_heads,
    n_local_kv_heads=config.n_kv_heads,
    head_dim=config.dim // config.n_heads,
    max_seq_len=config.max_seq_len,
    rope_theta=config.rope_theta,
    use_scaled_rope=config.use_scaled_rope,
    retrieval_chunks=config.retrieval_chunks,
    retrieval_context_length=config.retrieval_context_length,
  )

LLAMA_1B_PARAMS = create_model_params(MODEL_CONFIGS["1B"])
LLAMA_70B_PARAMS = create_model_params(MODEL_CONFIGS["70B"])