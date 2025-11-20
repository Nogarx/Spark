#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import abc
import jax
import jax.numpy as jnp 
import dataclasses as dc
import typing as tp
import spark.core.utils as utils
from jax._src import dtypes
from jax.typing import DTypeLike, ArrayLike
from spark.core.registry import register_initializer, register_config
from spark.core.config_validation import TypeValidator, ZeroOneValidator, PositiveValidator
from spark.nn.initializers.base import Initializer, InitializerConfig

T = tp.TypeVar('T', bound=ArrayLike)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class ConstantInitializerConfig(InitializerConfig):
    """
        ConstantInitializer configuration class.
    """
    __class_ref__: str = 'ConstantInitializer'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_initializer
class ConstantInitializer(Initializer):
    """
        Initializer that returns real uniformly-distributed random arrays.

        Init:
            scale: numeric, value for the output array (default = 1).

        Input:
            key: jax.Array, key for the random generator (jax.random.key).
            shape: tuple[int],shaoe fir the output array.
    """
    config: ConstantInitializerConfig

    def __call__(self, key: jax.Array, shape: tuple[int]) -> jax.Array:
        array: jax.Array = self.config.scale * jnp.ones(shape, dtype=self.config.dtype)
        return array.astype(self.config.dtype)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_config
class UniformInitializerConfig(InitializerConfig):
    """
        UniformInitializer configuration class.
    """
    __class_ref__: str = 'UniformInitializer'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_initializer
class UniformInitializer(Initializer):
    """
        Initializer that returns real uniformly-distributed random arrays.

        Init:
            scale: numeric, multiplicative factor for the output array (default = 1).
            min_value: numeric, minimum value for the output array (default = None).
            max_value: numeric, maximum value for the output array (default = None).

        Input:
            key: jax.Array, key for the random generator (jax.random.key).
            shape: tuple[int],shaoe fir the output array.
    """
    config: UniformInitializerConfig

    def __call__(self, key: jax.Array, shape: tuple[int]) -> jax.Array:
        array = self.config.scale * jax.random.uniform(key, shape)
        array = jnp.clip(array, min=self.config.min_value, max=self.config.max_value)
        return array.astype(self.config.dtype)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_config
class SparseUniformInitializerConfig(UniformInitializerConfig):
    """
        SparseUniformInitializer configuration class.
    """
    __class_ref__: str = 'SparseUniformInitializer'

    density: float = dc.field(
        default = 0.2, 
        metadata = {
            'validators': [
                TypeValidator,
                ZeroOneValidator,
            ],
            'description': 'Expected ratio of non-zero entries in the output array.',
        })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_initializer
class SparseUniformInitializer(UniformInitializer):
    """
        Initializer that returns a real sparse uniformly-distributed random arrays.
        
        Note that the output will contain zero values even if min_value > 0.

        Init:
            scale: numeric, multiplicative factor for the output array (default = 1).
            min_value: numeric, minimum value for the output array (default = None).
            max_value: numeric, maximum value for the output array (default = None).
            density: float, expected ratio of non-zero entries (default = 0.2).

        Input:
            key: jax.Array, key for the random generator (jax.random.key).
            shape: tuple[int],shaoe fir the output array.
    """
    config: SparseUniformInitializerConfig

    def __call__(self, key: jax.Array, shape: tuple[int]) -> jax.Array:
        key1, key2 = jax.random.split(key, 2)
        # Get uniform array
        array = super().__call__(key1, shape)
        # Zero mask
        mask = jax.random.uniform(key2, shape, dtype=jnp.float16) < self.config.density
        array = jnp.where(mask, array, 0)
        return array.astype(self.config.dtype)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_config
class NormalizedSparseUniformInitializerConfig(SparseUniformInitializerConfig):
    """
        NormalizedSparseUniformInitializer configuration class.
    """
    __class_ref__: str = 'NormalizedSparseUniformInitializer'

    norm_axes: tuple[int] | None = dc.field(
        default = (0,), 
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Axes used to normalize the output array over. Note: This attribute is automatically managed.',
        }
    )

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_initializer
class NormalizedSparseUniformInitializer(SparseUniformInitializer):
    """
        Initializer that returns a real sparse uniformly-distributed random arrays.
        This is a variation of the SparseUniformInitializer that normalizes the array, which may be useful to prevent quiescent neurons.
        Entries in the array are normalized by contracting the array over to the norm_axes and rescaled back to [min_value, max_value].
        
        Normalization example 
        array -> ijk; 
        norm_axes -> (i,k)
        contraction = \'ijk->ik\'
        sum(norm_array[i,:,k]) = 1

        Note that the output will contain zero values even if min_value > 0.

        Init:
            scale: numeric, multiplicative factor for the output array (default = 1).
            min_value: numeric, minimum value for the output array (default = None).
            max_value: numeric, maximum value for the output array (default = None).
            density: float, expected ratio of non-zero entries (default = 0.2).
            norm_axes: tuple[int], axes used for normalization (default = (0,)): 

        Input:
            key: jax.Array, key for the random generator (jax.random.key).
            shape: tuple[int], shape for the output array.

        Output:
            jax.Array[dtype]
    """
    config: NormalizedSparseUniformInitializerConfig

    def __call__(self, key: jax.Array, shape: tuple[int]) -> jax.Array:
        # Get sparse array
        array = super().__call__(key, shape)
        # Normalize
        num_dims = len(shape)
        # Sanity checks  
        if not num_dims > 1:
            raise ValueError(
                f'Normalization is only supported for arrays of dimension 2 or larger but got \"shape\": {shape}.'
            )
        if not utils.is_float(self.config.dtype):
            raise TypeError(
                f'Normalization is only possible for float \"dtype\", but got: \"{self.config.dtype}\".'
            )
        if any([(ax < 0) and (ax >= num_dims) for ax in self.config.norm_axes]):
            raise ValueError(
                f'Expected all indices of \"norm_axes\" to be in the set {{0, ..., {num_dims-1}}}, but got: \"{self.config.norm_axes}\".'
            )
        if len(set(ax for ax in self.config.norm_axes)) != len(self.config.norm_axes):
            raise ValueError(
                f'Expected all indices of \"norm_axes\" to be unique, but got: \"{self.config.norm_axes}\".'
            )
        # Normalize axes labes
        all_labels = utils.get_axes_einsum_labels([i for i in range(len(shape))])
        norm_labels = utils.get_axes_einsum_labels(self.config.norm_axes)
        # Normalize
        norm = jnp.einsum(f'{all_labels}->{norm_labels}', array)
        norm = jnp.where(norm != 0, 1/norm, 1)
        array = jnp.einsum(f'{all_labels},{norm_labels}->{all_labels}', array, norm)
        # Rescale and clip
        array = jnp.where(
            array != 0, 
            jnp.clip(self.config.scale * array, min=self.config.min_value, max=self.config.max_value),
            0
        )
        return array.astype(self.config.dtype)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################