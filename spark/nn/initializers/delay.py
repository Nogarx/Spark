#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import jax
import jax.numpy as jnp 
import typing as tp
import dataclasses as dc
from jax.typing import DTypeLike
from jax._src import dtypes
from spark.core.shape import bShape
from spark.core.registry import register_initializer, register_config
from spark.core.config_validation import TypeValidator
from spark.nn.initializers.base import Initializer, InitializerConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class DelayInitializerConfig(InitializerConfig):
    name: str = dc.field(
        default = 'uniform_delay_initializer', 
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'value_options': [
                'constant_delay_initializer',
                'uniform_delay_initializer',
            ],
            'description': 'Delay initializer protocol.',
        })
    dtype: DTypeLike = dc.field(
        default = jnp.uint8, 
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'value_options': [
                jnp.uint8,
                jnp.uint16,
            ],
            'description': 'Dtype used for JAX dtype promotions.',
        })
    
# NOTE: This is a simple registry needed for the GUI. 
# A more complex implementation would be an overkill for what this is needed.
_DELAY_CONFIG_REGISTRY: dict[str, type[DelayInitializerConfig]] = {}

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_config
class ConstantDelayInitializerConfig(DelayInitializerConfig):
    name: tp.Literal['constant_delay_initializer'] = 'constant_delay_initializer'
_DELAY_CONFIG_REGISTRY['constant_delay_initializer'] = ConstantDelayInitializerConfig

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

@register_config
class UniformDelayInitializerConfig(DelayInitializerConfig):
    name: tp.Literal['uniform_delay_initializer'] = 'uniform_delay_initializer'
_DELAY_CONFIG_REGISTRY['uniform_delay_initializer'] = UniformDelayInitializerConfig

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