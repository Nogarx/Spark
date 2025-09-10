#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import jax
import jax.numpy as jnp 
import dataclasses
import typing as tp
from jax._src import dtypes
from jax.typing import DTypeLike
from spark.core.shape import Shape
from math import prod
from spark.core.registry import register_initializer
from spark.core.config import BaseSparkConfig
from spark.core.config_validation import TypeValidator, ZeroOneValidator
from spark.nn.initializers.base import Initializer

#-----------------------------------------------------------------------------------------------------------------------------------------------#

BASE_SCALE = 3E3

# NOTE: Normalization and rescale is useful to prevent quiescent neurons by construction.
# Base scale is roughly the minimum sparse current a LIF neuron needs to fire at an input spike sparsity of 90%. 
# BASE_SCALE mitigates the need to introduce large numbers to make neurons initially responsive and allows for later changes to soma models.

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class KernelInitializerConfig(BaseSparkConfig):
    name: str
    scale: float = dataclasses.field(
        default = 1.0, 
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'description': 'Kernel scale factor.',
        })
    dtype: DTypeLike = dataclasses.field(
        default=jnp.float16, 
        metadata={
            'validators': [
                TypeValidator,
            ], 
            'description': 'Dtype used for JAX dtype promotions.',
        })
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

class UniformKernelInitializerConfig(KernelInitializerConfig):
    name: tp.Literal['uniform_kernel_initializer'] = 'uniform_kernel_initializer'
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_initializer
def uniform_kernel_initializer(config: UniformKernelInitializerConfig) -> Initializer:
    """
        Builds an initializer that returns real uniformly-distributed random arrays.
    """

    def init(key: jax.Array, shape: Shape, input_shape: Shape) -> Initializer:
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
        kernel = kernel * jnp.array(BASE_SCALE * config.scale, jnp.float32)
        return kernel.astype(dtypes.canonicalize_dtype(config.dtype))
    return init

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SparseUniformKernelInitializerConfig(KernelInitializerConfig):
    name: tp.Literal['sparse_uniform_kernel_initializer'] = 'sparse_uniform_kernel_initializer'
    density: float = dataclasses.field(
        default = 0.2, 
        metadata = {
            'validators': [
                TypeValidator,
                ZeroOneValidator,
            ],
            'description': 'Expected ratio of non-zero entries in the kernel.',
        })
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_initializer
def sparse_uniform_kernel_initializer(config: SparseUniformKernelInitializerConfig) -> Initializer:
    """
        Builds an initializer that returns a real sparse uniformly-distributed random arrays.
    """
    def init(key: jax.Array, output_shape: Shape, input_shape: Shape) -> Initializer:
        shape = output_shape + input_shape
        _ax_sum = list(range(len(shape[:-len(input_shape)]), len(shape)))
        _ax_transpose = list(range(len(shape[:-len(input_shape)]), len(shape))) +\
                        list(range(len(shape[:-len(input_shape)])))
        key1, key2 = jax.random.split(key, 2)
        # Random masked uniform weight distribution
        values = jnp.repeat(jnp.arange(prod(input_shape))[jnp.newaxis, ...], prod(output_shape), axis=0)
        max_idx = int(config.density*prod(input_shape))
        indices = jax.random.permutation(key1, values, axis=1, independent=True)[:, :max_idx].reshape(-1)
        kernel = jnp.zeros((prod(output_shape), prod(input_shape)))
        kernel = kernel.at[jnp.repeat(jnp.arange(prod(output_shape)), max_idx), indices].set(1)
        kernel = kernel.reshape(shape) * jax.random.uniform(key2, shape, jnp.float32)
        # Normalize
        norm = jnp.sum(kernel, axis=_ax_sum)
        norm = jnp.where(norm > 0, norm, 1)
        kernel = jnp.transpose(jnp.transpose(kernel, axes=_ax_transpose) / norm, axes=_ax_transpose)
        # Rescale
        kernel = kernel * jnp.array(BASE_SCALE * config.scale, jnp.float32)
        return kernel.astype(dtypes.canonicalize_dtype(config.dtype))
    return init

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################