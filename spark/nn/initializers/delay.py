#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import jax
import jax.numpy as jnp 
import typing as tp
import dataclasses
from jax.typing import DTypeLike
from jax._src import dtypes
from spark.core.shape import bShape
from spark.core.registry import register_initializer
from spark.core.config import BaseSparkConfig
from spark.core.config_validation import TypeValidator
from spark.nn.initializers.base import Initializer

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class DelayInitializerConfig(BaseSparkConfig):
    name: str
    dtype: DTypeLike = dataclasses.field(
        default=jnp.uint8, 
        metadata={
            'validators': [
                TypeValidator,
            ], 
            'value_options': [
                jnp.uint8,
                jnp.uint16,
            ],
            'description': 'Dtype used for JAX dtype promotions.',
        })
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ConstantDelayInitializerConfig(DelayInitializerConfig):
    name: tp.Literal['constant_delay_initializer'] = 'constant_delay_initializer'
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_initializer
def constant_delay_initializer(config: ConstantDelayInitializerConfig) -> Initializer:
    """
        Builds an initializer that returns a constant positive integer array.
    """

    def init(key: jax.Array, shape: bShape, buffer_size: int) -> Initializer:
        return buffer_size * jnp.ones(shape, dtype=dtypes.canonicalize_dtype(config.dtype))
    return init

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class UniformDelayInitializerConfig(DelayInitializerConfig):
    name: tp.Literal['uniform_delay_initializer'] = 'uniform_delay_initializer'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_initializer
def uniform_delay_initializer(config: UniformDelayInitializerConfig) -> Initializer:
    """
        Builds an initializer that returns positive integers uniformly-distributed random arrays.
    """

    def init(key: jax.Array, shape: bShape, buffer_size: int) -> Initializer:
        return jax.random.randint(key, shape, 1, buffer_size, dtype=dtypes.canonicalize_dtype(config.dtype))
    return init


#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################