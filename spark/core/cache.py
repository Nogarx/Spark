#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.payloads import SparkPayload

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from dataclasses import dataclass

import spark.core.validation as validation
from spark.core.variable_containers import SparkVariable


#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@jax.tree_util.register_pytree_node_class
@dataclass(init=False)
class Cache:
    """
        Cache dataclass.
    """
    payload_type: SparkPayload        
    dtype: DTypeLike
    var: SparkVariable  

    def __init__(self, var: SparkVariable, payload_type: SparkPayload, dtype: DTypeLike):
        if not isinstance(var, SparkVariable):
            raise TypeError(f'Expected "value" to be of type "SparkVariable" but got "{type(var).__name__}".')
        if not validation._is_payload_type(payload_type):
            raise TypeError(f'Expected "payload_type" to be of type "SparkPayload" but got "{type(payload_type).__name__}".')
        if not isinstance(jnp.dtype(dtype), jnp.dtype):
            raise TypeError(f'Expected "dtype" to be of type "{DTypeLike}" but got  "{type(dtype).__name__}".')
        self.var = var
        self.payload_type = payload_type
        self.dtype = dtype

    def tree_flatten(self):
        children = (self.payload_type, self.dtype, self.var)
        aux_data = ()
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (payload_type, dtype, var) = children
        return cls(payload_type=payload_type, dtype=dtype, var=var)


#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################