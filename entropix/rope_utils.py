import jax
import jax.numpy as jnp
import math

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