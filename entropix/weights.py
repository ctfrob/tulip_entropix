from typing import List, NamedTuple, Optional, Tuple
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as PS
from jax.experimental import mesh_utils

from pathlib import Path
import re
import os
import numpy as np
import ml_dtypes

def chunked_device_put(array, sharding, chunk_size=64*1024*1024, transpose_first=False, is_attention=False, is_kv=False):
    """Load large arrays in chunks with specific handling for 70B attention patterns"""
    print(f"Processing array of shape {array.shape} ({array.nbytes / 1024**3:.2f} GB)")
    
    devices = sharding.mesh.devices.flat
    n_devices = len(devices)
    
    def calculate_shards(shape, is_attention, is_kv):
        array_size = np.prod(shape) * 2  # size in bytes (bfloat16 = 2 bytes)
        
        # Base number of shards on array size
        if array_size > 1 * 1024**3:  # > 1GB
            base_shards = 16
        elif array_size > 512 * 1024**2:  # > 512MB
            base_shards = 8
        elif array_size > 256 * 1024**2:  # > 256MB
            base_shards = 4
        else:
            base_shards = 2
            
        # Adjust based on layer type
        if is_attention and not is_kv:
            base_shards *= 2
            
        return max(base_shards, n_devices)
    
    # Helper function to convert mmap array to proper bfloat16
    def convert_to_bfloat16(chunk):
        chunk_uint16 = chunk.view(np.uint16)
        return chunk_uint16.view(ml_dtypes.bfloat16)
    
    # For very small arrays, handle directly
    if array.nbytes <= chunk_size:
        if transpose_first:
            array = np.array(array.T)
        array_bf16 = convert_to_bfloat16(array)
        jax_array = jnp.asarray(array_bf16, dtype=jnp.bfloat16)
        return jax.device_put(jax_array, sharding)

    # Calculate total shards needed
    total_shards = calculate_shards(array.shape, is_attention, is_kv)
    shard_size = array.shape[0] // total_shards
    shards = []
    
    print(f"Splitting into {total_shards} shards {'(attention)' if is_attention else ''}")
    
    for i in range(total_shards):
        start = i * shard_size
        end = start + shard_size if i < total_shards-1 else array.shape[0]
        
        # Process chunk
        if transpose_first:
            chunk = array[start:end].T
        else:
            chunk = array[start:end]
        
        # Convert chunk
        chunk_bf16 = convert_to_bfloat16(chunk)
        
        # Process sub-chunks if chunk is still large
        sub_chunks = []
        if chunk_bf16.nbytes > 256 * 1024 * 1024:  # 256MB
            n_sub_chunks = 4
            sub_size = chunk_bf16.shape[0] // n_sub_chunks
            for j in range(n_sub_chunks):
                sub_start = j * sub_size
                sub_end = sub_start + sub_size if j < n_sub_chunks-1 else chunk_bf16.shape[0]
                sub_chunk = chunk_bf16[sub_start:sub_end]
                sub_chunks.append(jnp.asarray(sub_chunk, dtype=jnp.bfloat16))
            chunk_array = jnp.concatenate(sub_chunks, axis=0)
        else:
            chunk_array = jnp.asarray(chunk_bf16, dtype=jnp.bfloat16)
        
        device_idx = i % n_devices
        device_sharding = jax.sharding.SingleDeviceSharding(devices[device_idx])
        shards.append(jax.device_put(chunk_array, device_sharding))
        
        # Cleanup
        del chunk
        del chunk_bf16
        if sub_chunks:
            del sub_chunks
            del chunk_array
        import gc
        gc.collect()
        
        # Small sleep between large chunks
        if array.nbytes > 512 * 1024 * 1024:
            import time
            time.sleep(0.1)
    
    try:
        # Process device arrays in smaller groups
        device_arrays = []
        for d in range(n_devices):
            device_shards = [s for i, s in enumerate(shards) if i % n_devices == d]
            if device_shards:
                # Combine in smaller groups if there are many shards
                if len(device_shards) > 4:
                    combined = []
                    for i in range(0, len(device_shards), 4):
                        group = device_shards[i:i+4]
                        combined.append(jnp.concatenate(group, axis=0))
                        for s in group:
                            del s
                    device_arrays.append(jnp.concatenate(combined, axis=0))
                    del combined
                else:
                    device_arrays.append(jnp.concatenate(device_shards, axis=0))
                    for s in device_shards:
                        del s
        gc.collect()
        
        result = jax.device_put_sharded(device_arrays, devices)
        return jax.device_put(result, sharding)
    except Exception as e:
        print(f"Failed to combine shards: {e}")
        raise

def load_weight_file(file_path, sharding):
    """Load a weight file with memory mapping and pre-sharding"""
    print(f"Loading file: {file_path}")
    
    # Memory map the file
    array = np.load(file_path, mmap_mode='r').astype(jnp.bfloat16)
    
    # Get devices from sharding
    devices = sharding.mesh.devices.flat
    
    # Calculate number of shards based on available devices
    n_shards = len(devices)
    shard_size = array.shape[0] // n_shards
    
    shards = []
    for i in range(n_shards):
        start = i * shard_size
        end = start + shard_size if i < n_shards-1 else array.shape[0]
        
        # Load this portion of the array
        shard = array[start:end]
        device_sharding = jax.sharding.SingleDeviceSharding(devices[i])
        shards.append(jax.device_put(shard, device_sharding))
    
    return jax.device_put_sharded(shards, sharding)

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

@dataclass
class WeightConfig:
  """Configuration for weight loading and sharding."""

  dp_dim: str = "dp"
  mp_dim: str = "mp"
  fsdp_dim: str = "fsdp"

def create_mesh(device_count: int) -> jax.sharding.Mesh:
    """Creates device mesh for distributed execution."""
    devices = jax.devices()
    mesh_shape = (device_count, 1)
    device_mesh = mesh_utils.create_device_mesh(mesh_shape)
    return jax.sharding.Mesh(device_mesh, ("mp", "fsdp"))

def create_partition_spec(key: str):
    """Create sharding spec with explicit memory alignment"""
    if "norm" in key:
        return PS()
    elif "w2" in key or "wo" in key:
        # Align large matrices on power-of-2 boundaries
        return PS("mp", "fsdp")
    elif "wq" in key or "wk" in key or "wv" in key:
        # Keep attention heads together on same device
        return PS("mp", "fsdp")
    else:
        return PS("fsdp", "mp")

def chunked_device_put(array, sharding, transpose_first=False):
    """Simplified loading with explicit transpose handling"""
    if transpose_first:
        array = array.T
    return jax.device_put(array, sharding)

def load_weights(
    ckpt_dir: Path, model_params, weight_config: Optional[WeightConfig] = None
) -> Tuple[XfmrWeights, jax.sharding.Mesh]:
    """Load weights with explicit memory management"""
    weight_config = weight_config or WeightConfig()
    mesh = create_mesh(jax.device_count())
    print(f"Created mesh with shape: {mesh.devices.shape}")
    
    w = {}
    layer_weights = []
    
    # Pre-sort files to group similar tensors together
    files = sorted(ckpt_dir.glob("*.npy"))
    
    # Load norm layers first (they're small)
    norm_files = [f for f in files if 'norm' in str(f)]
    attn_files = [f for f in files if any(x in str(f) for x in ['wq', 'wk', 'wv', 'wo'])]
    ffn_files = [f for f in files if any(x in str(f) for x in ['w1', 'w2', 'w3'])]
    other_files = [f for f in files if f not in norm_files + attn_files + ffn_files]
    
    ordered_files = norm_files + attn_files + ffn_files + other_files
    
    for file in ordered_files:
        name = ".".join(str(file).split("/")[-1].split(".")[:-1])
        print(f"\nLoading {name}")
        
        weight = jnp.load(file=file, mmap_mode="r", allow_pickle=True)
        partition_spec = create_partition_spec(name)
        sharding = NamedSharding(mesh, partition_spec)
        
        if any(lyr in name for lyr in ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]):
            weight = weight.T
            if "wq" in name or "wk" in name or "wv" in name:
                weight = weight.reshape(
                    -1,
                    model_params.n_local_heads if "wq" in name else model_params.n_local_kv_heads,
                    model_params.head_dim,
                )
        
        # Try to ensure alignment
        if weight.nbytes > 100 * 1024 * 1024:  # 100MB
            # Force memory defrag before large allocations
            import gc
            gc.collect()
            jax.clear_caches()
        
        w[name] = jax.device_put(weight, sharding)
    
    # Create layer weights list
    for i in range(model_params.n_layers):
        layer_weights.append(
            LayerWeights(
                wq=w[f"layers.{i}.attention.wq.weight"],
                wk=w[f"layers.{i}.attention.wk.weight"],
                wv=w[f"layers.{i}.attention.wv.weight"],
                wo=w[f"layers.{i}.attention.wo.weight"],
                w1=w[f"layers.{i}.feed_forward.w1.weight"],
                w2=w[f"layers.{i}.feed_forward.w2.weight"],
                w3=w[f"layers.{i}.feed_forward.w3.weight"],
                ffn_norm=w[f"layers.{i}.ffn_norm.weight"],
                attention_norm=w[f"layers.{i}.attention_norm.weight"],
            )
        )
    
    return XfmrWeights(
        tok_embeddings=w["tok_embeddings.weight"],
        norm=w["norm.weight"],
        output=w["output.weight"],
        layer_weights=layer_weights,
    ), mesh