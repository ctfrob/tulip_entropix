import functools
import math
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as PS

from entropix.config import ModelParams
from entropix.dslider import initialize_state
from entropix.dslider_config import DEFAULT_DS_CONFIG
from entropix.kvcache import KVCache
from entropix.tokenizer import Tokenizer

"""Defines the JetStream API.

These functions are the accelerator functions which an outer sampling loop
could want to call, enabling interleaved (continuous batching) inference.
"""

# The model parameters - their partitioning will be unique for different prefill
# and decode topoologies.
Params = Any
# The result of a prefill operation, often a batch size 1 KVCache.
Prefix = Any

# Accelerator representation of tokens.
DeviceTokens = Any
# Cpus asscociated with the mesh.
CpuDevices = Any


class LayerWeights(NamedTuple):
  wq: jax.Array
  wk: jax.Array
  wv: jax.Array
  wo: jax.Array
  w1: jax.Array
  w2: jax.Array
  w3: jax.Array
  ffn_norm: jax.Array
  attention_norm: jax.Array


class XfmrWeights(NamedTuple):
  tok_embeddings: jax.Array
  norm: jax.Array
  output: jax.Array
  layer_weights: List[LayerWeights]


class Params(NamedTuple):
  n_layers: int
  n_local_heads: int
  n_local_kv_heads: int
  head_dim: int
  max_seq_len: int
  rope_theta: float
  use_scaled_rope: bool


def create_partition_spec(key):
  dp = "dp"
  mp = "mp"
  fsdp = "fsdp"
  if "norm" in key:
    return PS()
  if "rope.freqs" in key:
    return PS()
  elif "tok_embeddings" in key or "output" in key or "w2" in key:
    return PS(fsdp, mp)
  else:
    return PS(mp, fsdp)


class DecodeState(NamedTuple):
  """The inputs into a generation step."""

  prefill_cache: jax.Array
  generate_cache: jax.Array
  generate_cache_index: int
  generate_lengths: jax.Array
  generated_token: jax.Array


class SlotData(NamedTuple):
  """Class to store slot data."""

  tokens: Union[jax.Array, np.ndarray]
  valid: Union[jax.Array, np.ndarray]
  lengths: Union[jax.Array, np.ndarray]


class ResultTokens(NamedTuple):
  """Class to store returned tokens in.

  We store everything in one array, and keep indexes - because copying
  a single array to host is much faster.
  Each tuple represents the indices of the relevant data.
  """

  # Shape: [batch, tokens.shape[1] + validity.shape[1] + lengths.shape[1]]
  data: Union[jax.Array, np.ndarray]
  # The range of indices which contain tokens.
  tokens_idx: Tuple[int, int]
  # The range of indices which contain the validity of
  # the tokens.
  valid_idx: Tuple[int, int]
  # The range of indices which contain the lengths up till now of the lengths
  # of each generated sequence.
  length_idx: Tuple[int, int]
  samples_per_slot: int

  def copy_to_host_async(self: "ResultTokens") -> None:
    """Copy to host asynchronously."""
    # Do nothing for np array
    if isinstance(self.data, np.ndarray):
      return
    self.data.copy_to_host_async()

  def convert_to_numpy(self: "ResultTokens") -> "ResultTokens":
    """Converts to numpy."""
    return ResultTokens(
      np.array(self.data),
      self.tokens_idx,
      self.valid_idx,
      self.length_idx,
      self.samples_per_slot,
    )

  def get_result_at_slot(self, slot: int) -> SlotData:
    """Returns the token at a given slot.

    Args:
      slot: An integer from [0, n) representing an index into the batch.

    Note: implementations of this method must correctly handle
    microbatches, if microbatches are used.
    """
    # Potentially get multiple beams for given slot.
    start_idx = slot * self.samples_per_slot
    end_idx = (slot + 1) * self.samples_per_slot
    # Mask out any non valid tokens.
    return SlotData(
      tokens=self.data[start_idx:end_idx, self.tokens_idx[0] : self.tokens_idx[1]],
      valid=self.data[start_idx:end_idx, self.valid_idx[0] : self.valid_idx[1]],
      lengths=self.data[start_idx:end_idx, self.length_idx[0] : self.length_idx[1]][
        :, 0
      ],
    )


class EntropixEngine:
  """Main engine for running inference with transformer models.

  Handles tokenization, model execution, and result processing.
  """

  def __init__(
    self,
    params: ModelParams,
    xfmr_weights: XfmrWeights,
    mesh: jax.sharding.Mesh,
    tokenizer: Tokenizer,
    xfmr_fn: Callable,
    sample_fn: Callable,
  ):
    """Initialize engine with model parameters and functions.

    Args:
        params: Model architecture parameters
        xfmr_weights: Model weights
        mesh: Device mesh for parallel execution
        tokenizer: Tokenizer instance
        xfmr_fn: Transformer forward function
        sample_fn: Token sampling function
    """
    self.params = params
    self.xfmr_weights = xfmr_weights
    self.mesh = mesh
    self.replicated = jax.NamedSharding(mesh, jax.sharding.PartitionSpec())
    self.tokenizer = tokenizer
    self.freqs_cis = jax.device_put(
      self.precompute_freqs_cis(
        params.head_dim, params.max_seq_len, params.rope_theta, params.use_scaled_rope
      ),
      self.replicated,
    )
    self.xfmr_fn = xfmr_fn
    self.sample_fn = sample_fn

  def get_prefix_destination_sharding(self) -> Any:
    """Returns the shardings necessary to transfer data between engines."""

  def init_decode_state(self, *args, **kwargs) -> DecodeState:
    """Initialises any state which a generation step transforms."""

  def get_tokenizer(
    self,
  ) -> Dict[str, Any]:
    """Returns the info to construct a tokenizer in py/c++."""
    return {}

  def build_tokenizer(
    self,
    metadata: Dict[str, Any],
  ) -> Tokenizer:
    """Builds a new tokenizer object and returns it."""
    return self.tokenizer

  @property
  def max_concurrent_decodes(self) -> int:
    """Maximum number of concurrent decode operations supported."""
    return jax.device_count()

  @property
  def samples_per_slot(self) -> int:
    """Total samples per slot."""
    return 1  # this is actually top_k

  def free_resource(
    self,
    slot: int,  # pylint: disable=unused-argument
  ) -> Any:
    """Free cache and other decode resource for the slot.

    This function is needed for advanced attetnion kenel like PageAttetion.
    After finishing one request, the engine need to free all used page block
    resource and reuse for coming requests.
    """
    return None

  @property
  def max_prefill_length(self) -> int:
    """Maximum prefill length."""
    return 1024

  def mesh(self) -> jax.sharding.Mesh:
    """Mesh which the engine is running on."""
    return self.mesh

  @property
  def colocated_cpus(self) -> Union[list[CpuDevices], None]:
    """CPU devices colocated with the engine's accelerators."""

  def apply_scaling(self, freqs: jax.Array):
    SCALE_FACTOR = 8
    LOW_FREQ_FACTOR = 1
    HIGH_FREQ_FACTOR = 4
    OLD_CONTEXT_LEN = 8192  # original llama3 length

    low_freq_wavelen = OLD_CONTEXT_LEN / LOW_FREQ_FACTOR
    high_freq_wavelen = OLD_CONTEXT_LEN / HIGH_FREQ_FACTOR

    def scale_freq(freq):
      wavelen = 2 * math.pi / freq

      def scale_mid(_):
        smooth = (OLD_CONTEXT_LEN / wavelen - LOW_FREQ_FACTOR) / (
          HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR
        )
        return (1 - smooth) * freq / SCALE_FACTOR + smooth * freq

      return jax.lax.cond(
        wavelen < high_freq_wavelen,
        lambda _: freq,
        lambda _: jax.lax.cond(
          wavelen > low_freq_wavelen, lambda _: freq / SCALE_FACTOR, scale_mid, None
        ),
        None,
      )

    return jax.vmap(scale_freq)(freqs)

  # @functools.partial(jax.jit, static_argnums=(0, 1))
  def precompute_freqs_cis(
    self,
    dim: int,
    end: int,
    theta: float = 500000.0,
    use_scaled: bool = False,
    dtype: jnp.dtype = jnp.float32,
  ) -> jax.Array:
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
    if use_scaled:
      freqs = self.apply_scaling(freqs)
    t = jnp.arange(end, dtype=dtype)
    freqs = jnp.outer(t, freqs)
    return jnp.exp(1j * freqs)

  def build_attn_mask(self, seqlen: int, start_pos: int) -> jax.Array:
    mask = jnp.zeros((seqlen, seqlen), dtype=jnp.float32)
    if seqlen > 1:
      mask = jnp.full((seqlen, seqlen), float("-inf"))
      mask = jnp.triu(mask, k=1)
      mask = jnp.hstack([jnp.zeros((seqlen, start_pos)), mask], dtype=jnp.float32)
    return mask

  @functools.partial(jax.jit, static_argnames=("self", "params"))
  def prefill(
    self,
    *,
    params: Params,
    existing_prefix: Optional[jax.Array] = None,
    padded_tokens: jax.Array,
    true_length: int,
    sampler: Optional[Callable[[Any], Any]] = None,  # pylint: disable=unused-argument
    rng: Optional[jax.random.PRNGKey] = None,
    top_k: int = 6,
  ) -> Tuple[Prefix, ResultTokens]:
    """Computes a kv-cache for a set of tokens conditional on existing cache.

    existing_prefix (if provided) represents a prefix that has already been
    processed by the underlying model. tokens is logically appended
    to the text represented by `existing_prefix`. This method returns a new
    kv_cache (typically) for the resulting text.

    If sampler is passed, then the engine should use it do sample next token.
    """
    cur_pos = 0
    bsz, seqlen = padded_tokens.shape
    attn_mask = self.build_attn_mask(seqlen, cur_pos)
    kvcache = KVCache.new(
      params.n_layers, bsz, params.max_seq_len, params.n_local_kv_heads, params.head_dim
    )
    with self.mesh:
      logits, kvcache, _ = self.xfmr_fn(
        self.xfmr_weights,
        params,
        padded_tokens,
        cur_pos,
        self.freqs_cis[:seqlen],
        kvcache,
        attn_mask=attn_mask,
      )
    # next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32)
    _, next_token = jax.lax.top_k(logits[:, true_length], k=top_k)
    next_token = jnp.array(next_token, dtype=jnp.int32).reshape((top_k, 1))
    # Create arrays for tokens, validity, and lengths
    tokens = next_token
    validity = jnp.ones_like(next_token, dtype=jnp.bool_)
    lengths = jnp.broadcast_to(
      jnp.array([[true_length + 1]], dtype=jnp.int32), (tokens.shape[0], 1)
    )
    data = jnp.concatenate([tokens, validity, lengths], axis=1)
    result = ResultTokens(
      data=data,
      # Tokens are shape [batch, speculations], so when we concatenate
      # tokens, validity and length along their index 1 dimension then they
      # occupy 0:speculations.
      tokens_idx=(0, 1),
      # Validity occupies the same amount of space, but next in line.
      valid_idx=(1, 2),
      # And lengths is rank 1.
      length_idx=(2, 3),
      samples_per_slot=bsz,
    )

    return {
      "logits": logits,
      "cache": kvcache,
      "next_pos": true_length + 1,
      "generated_tokens": jnp.zeros((bsz, 1), dtype=jnp.int32),
      "tokens": next_token,
    }, result

  @functools.partial(jax.jit, static_argnums=(0, 1))
  def generate(
    self,
    params: Params,
    decode_state: DecodeState,
    sampler: Optional[Callable[[Any], Any]] = None,  # pylint: disable=unused-argument
    rng: Optional[jax.random.PRNGKey] = jax.random.PRNGKey(1337),
  ) -> Tuple[DecodeState, ResultTokens]:
    """Generates tokens for each sequence being decoded in parallel.

    Generate takes a batch of pre-computed kv-caches, and computes:
      - the predicted next token for each of the sequences
      - an updated set of kv-caches

    In the case of pipelining, this will handle N cycles (where each cycle
    consists of each microbatch progressing through every stage), in
    non-pipelined code this is a full forward pass. In both cases, this accounts
    for a full embed-layerstack-unembed-sample operation.

    If sampler is passed, then the engine should use it do sample next token.
    """
    cur_pos = decode_state["next_pos"]
    bsz = decode_state["tokens"].shape[0]
    freqs_cis_slice = jax.lax.dynamic_slice(
      self.freqs_cis, (cur_pos, 0), (1, self.freqs_cis.shape[1])
    )
    with self.mesh:
      logits, kvcache, _ = self.xfmr_fn(
        self.xfmr_weights,
        params,
        decode_state["tokens"],
        cur_pos,
        freqs_cis_slice,
        decode_state["cache"],
      )

    # TODO(xjdr): reduce slop tokens by penalizing slop weights
    # logits = logits.at[:, -1, self.slop_tokens].multiply(self.slop_weights[None, :, None])
    new_token, new_state = self.sample_fn(
      decode_state["dslider_state"], logits[:, -1, :], DEFAULT_DS_CONFIG, key=rng
    )
    new_token = new_token.reshape((bsz, 1))

    result = ResultTokens(
      data=jnp.concatenate(
        (
          new_token,
          jnp.ones_like(new_token, dtype=jnp.bool_),
          jnp.full(
            (bsz, 1), decode_state["generated_tokens"][:, -1] + 1, dtype=jnp.int32
          ),
        ),
        axis=1,
      ),
      # Tokens are shape [batch, speculations], so when we concatenate
      # tokens, validity and length along their index 1 dimension then they
      # occupy 0:speculations.
      tokens_idx=(0, 1),
      # Validity occupies the same amount of space, but next in line.
      valid_idx=(1, 2),
      # And lengths is rank 1.
      length_idx=(2, 3),
      samples_per_slot=bsz,
    )

    return {
      "logits": logits,
      "cache": kvcache,
      "next_pos": decode_state["next_pos"] + 1,
      "generated_tokens": decode_state["generated_tokens"] + 1,
      "tokens": new_token,
      "dslider_state": new_state,
    }, result

  @functools.partial(
    jax.jit,
    static_argnums=(0,),
    donate_argnums=(
      1,
      2,
    ),
  )
  def insert(
    self,
    prefix: Prefix,
    decode_state: DecodeState,
    slot: int,
  ) -> DecodeState:
    """Adds `new_request` into `caches` at 'slot'.

    When decoding multiple requests in parallel, when one request finishes, a
    new request must be slotted into the recently vacated spot: `insert`!

    This can occur in between and async to generate calls, and takes a lock over
    that row of the cache.

    The slot may represent a tuple of positions (e.g. microbatch, pipeline stage
    and batch), but at the engine interface level all of these are exposed as
    a [0, n) range of slots and converted internally.
    """
    bsz = prefix["tokens"].shape[0]
    layers, _, max_seq_len, kv_heads, head_dim = prefix["cache"].k.shape
    new_k = jnp.broadcast_to(
      prefix["cache"].k, (layers, bsz, max_seq_len, kv_heads, head_dim)
    )
    new_v = jnp.broadcast_to(
      prefix["cache"].v, (layers, bsz, max_seq_len, kv_heads, head_dim)
    )
    new_cache = KVCache(k=new_k, v=new_v)

    return {
      "logits": prefix["l ogits"],
      "cache": new_cache,
      "next_pos": prefix["next_pos"],
      "generated_tokens": prefix["generated_tokens"],
      "tokens": prefix["tokens"],
      "dslider_state": initialize_state(prefix["logits"], bsz, DEFAULT_DS_CONFIG),
    }
