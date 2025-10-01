#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import jax
import jax.numpy as jnp 
import dataclasses as dc
import typing as tp
from jax._src import dtypes
from jax.typing import DTypeLike
from spark.core.shape import Shape
from math import prod
from spark.core.registry import register_initializer, register_config
from spark.core.config_validation import TypeValidator, ZeroOneValidator, PositiveValidator
from spark.nn.initializers.base import Initializer, InitializerConfig

#-----------------------------------------------------------------------------------------------------------------------------------------------#

BASE_SCALE = 3 #3E3

# NOTE: Normalization and rescale is useful to prevent quiescent neurons by construction.
# Base scale is roughly the minimum sparse current a LIF neuron needs to fire at an input spike sparsity of 90%. 
# BASE_SCALE mitigates the need to introduce large numbers to make neurons initially responsive and allows for later changes to soma models.

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class KernelInitializerConfig(InitializerConfig):
    name: str = dc.field(
        default = 'sparse_uniform_kernel_initializer', 
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'value_options': [
                'uniform_kernel_initializer',
                'sparse_uniform_kernel_initializer',
            ],
            'description': 'Delay initializer protocol.',
        })
    scale: float = dc.field(
        default = 1.0, 
        metadata = {
            'validators': [
                TypeValidator,
                PositiveValidator,
            ], 
            'description': 'Float value to scale the kernel array.',
        })

    dtype: DTypeLike = dc.field(
        default = jnp.float16, 
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'value_options': [
                jnp.float16,
                jnp.float32,
                jnp.float64,
            ],
            'description': 'Dtype used for JAX dtype promotions.',
        })

# NOTE: This is a simple registry needed for the GUI. 
# A more complex implementation would be an overkill for what this is needed.
_KERNEL_CONFIG_REGISTRY: dict[str, type[KernelInitializerConfig]] = {}

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_config
class UniformKernelInitializerConfig(KernelInitializerConfig):
    name: tp.Literal['uniform_kernel_initializer'] = 'uniform_kernel_initializer'
_KERNEL_CONFIG_REGISTRY['uniform_kernel_initializer'] = UniformKernelInitializerConfig

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

@register_config
class SparseUniformKernelInitializerConfig(KernelInitializerConfig):
    name: tp.Literal['sparse_uniform_kernel_initializer'] = 'sparse_uniform_kernel_initializer'
    density: float = dc.field(
        default = 0.2, 
        metadata = {
            'validators': [
                TypeValidator,
                ZeroOneValidator,
            ],
            'description': 'Expected ratio of non-zero entries in the kernel.',
        })
_KERNEL_CONFIG_REGISTRY['sparse_uniform_kernel_initializer'] = SparseUniformKernelInitializerConfig

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