#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import jax
import jax.numpy as jnp 
from jax._src import dtypes
from typing import Protocol, runtime_checkable
from spark.core.shape import bShape

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@runtime_checkable
class DelayInitializer(Protocol):
    """
        Base (abstract) class for all spark kernel initializers.
    """

    @staticmethod
    def __call__(key: jax.Array,
                shape: bShape,
                dtype: int = jnp.uint8) -> jax.Array:
        raise NotImplementedError

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def constant_delay_initializer(scale: int = 8, dtype: int = jnp.uint8) -> DelayInitializer:
    """
        Builds an initializer that returns a constant positive integer array.
    """

    def init(key: jax.Array, shape: bShape, dtype: int = dtype) -> DelayInitializer:
        dtype = dtypes.canonicalize_dtype(dtype)
        return scale * jnp.ones(shape, dtype=dtype)
    return init

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def uniform_delay_initializer(scale: int = 8, dtype: int = jnp.uint8) -> DelayInitializer:
    """
        Builds an initializer that returns positive integers uniformly-distributed random arrays.
    """

    def init(key: jax.Array, shape: bShape, dtype: int = dtype) -> DelayInitializer:
        dtype = dtypes.canonicalize_dtype(dtype)
        return jax.random.randint(key, shape, 1, scale-1, dtype=dtype)
    return init


#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################