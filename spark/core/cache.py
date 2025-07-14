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
from spark.core.variables import Variable
from spark.core.shape import bShape, Shape, normalize_shape

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@jax.tree_util.register_pytree_node_class
@dataclass(init=False)
class Cache:
    """
        Cache dataclass.
    """
    variable: Variable  
    payload_type: type[SparkPayload]

    def __init__(self, variable: Variable, payload_type: type[SparkPayload]):
        #if not validation.is_shape(shape):
        #    raise TypeError(f'Expected "shape" to be of type "Shape" but got "{type(shape).__name__}".')
        if not validation._is_payload_type(payload_type):
            raise TypeError(f'Expected "payload_type" to be of type "SparkPayload" but got "{type(payload_type).__name__}".')
        #if not validation.is_dtype(dtype):
        #    raise TypeError(f'Expected "dtype" to be of type "{DTypeLike}" but got  "{type(dtype).__name__}".')
        self.variable = variable
        self.payload_type = payload_type

    @property
    def value(self,):
        return self.payload_type(self.variable.value)
    
    @value.setter
    def value(self, new_value):
        self.variable.value = new_value

    @property
    def shape(self,):
        return self.variable.value.shape

    @property
    def dtype(self,):
        return self.variable.value.dtype

    def tree_flatten(self):
        children = (self.variable, self.payload_type)
        aux_data = ()
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (variable, payload_type) = children
        return cls(variable=variable, payload_type=payload_type)


#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################