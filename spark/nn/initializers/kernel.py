#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import jax
import jax.numpy as jnp 
from jax._src import dtypes
from typing import Any, Protocol, runtime_checkable
from spark.core.shape import bShape
from math import prod

#-----------------------------------------------------------------------------------------------------------------------------------------------#

BASE_SCALE = 3E3

# NOTE: Normalization and rescale is useful to prevent quiescent neurons by construction.
# Base scale is roughly the minimum sparse current a LIF neuron needs to fire at an input spike sparsity of 90%. 
# BASE_SCALE mitigates the need to introduce large numbers to make neurons initially responsive and allows for later changes to soma models.

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@runtime_checkable
class KernelInitializer(Protocol):
    """
        Base (abstract) class for all spark delay initializers.
    """

    @staticmethod
    def __call__(key: jax.Array, shape: bShape, input_shape: bShape, dtype: Any = jnp.float16) -> jax.Array:
        raise NotImplementedError

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def uniform_kernel_initializer(scale: Any = 1, dtype: Any = jnp.float16) -> KernelInitializer:
    """
        Builds an initializer that returns real uniformly-distributed random arrays.
    """

    def init(key: jax.Array, shape: bShape, input_shape: bShape, dtype: Any = dtype) -> KernelInitializer:
        dtype = dtypes.canonicalize_dtype(dtype)
        _ax_sum = list(range(len(shape[:-len(input_shape)]), len(shape)))
        _ax_transpose = list(range(len(shape[:-len(input_shape)]), len(shape))) +\
                        list(range(len(shape[:-len(input_shape)])))
        # Random uniform weight distribution
        kernel = jax.random.uniform(key, shape, jnp.float32)
        # Normalize
        norm = jnp.sum(kernel, axis=_ax_sum)
        norm = jnp.where(norm > 0, norm, 1)
        kernel = jnp.transpose(jnp.transpose(kernel, axes=_ax_transpose) / norm, axes=_ax_transpose)
        # Rescale
        kernel = kernel * jnp.array(BASE_SCALE * scale, jnp.float32)
        return kernel.astype(dtype)
    return init

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def sparse_uniform_kernel_initializer(prob: Any = 0.2, scale: Any = 1, dtype: Any = jnp.float16) -> KernelInitializer:
    """
        Builds an initializer that returns a real sparse uniformly-distributed random arrays.
    """

    def init(key: jax.Array, output_shape: bShape, input_shape: bShape, dtype: Any = dtype) -> KernelInitializer:
        dtype = dtypes.canonicalize_dtype(dtype)
        shape = output_shape + input_shape
        _ax_sum = list(range(len(shape[:-len(input_shape)]), len(shape)))
        _ax_transpose = list(range(len(shape[:-len(input_shape)]), len(shape))) +\
                        list(range(len(shape[:-len(input_shape)])))
        key1, key2 = jax.random.split(key, 2)
        # Random masked uniform weight distribution
        values = jnp.repeat(jnp.arange(prod(input_shape))[jnp.newaxis, ...], prod(output_shape), axis=0)
        max_idx = int(prob*prod(input_shape))
        indices = jax.random.permutation(key1, values, axis=1, independent=True)[:, :max_idx].reshape(-1)
        kernel = jnp.zeros((prod(output_shape), prod(input_shape)))
        kernel = kernel.at[jnp.repeat(jnp.arange(prod(output_shape)), max_idx), indices].set(1)
        kernel = kernel.reshape(shape) * jax.random.uniform(key2, shape, jnp.float32)
        # Normalize
        norm = jnp.sum(kernel, axis=_ax_sum)
        norm = jnp.where(norm > 0, norm, 1)
        kernel = jnp.transpose(jnp.transpose(kernel, axes=_ax_transpose) / norm, axes=_ax_transpose)
        # Rescale
        kernel = kernel * jnp.array(BASE_SCALE * scale, jnp.float32)
        return kernel.astype(dtype)
    return init

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################