#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.payloads import ValueSparkPayload

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
import dataclasses as dc
import spark.core.validation as validation
from spark.core.variables import Variable

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@jax.tree_util.register_dataclass
@dc.dataclass(init=False)
class Cache:
    """
        Cache dataclass.
    """
    variable: Variable  
    payload_type: type[ValueSparkPayload]

    def __init__(self, variable: Variable, payload_type: type[ValueSparkPayload]) -> None:
        if not validation._is_payload_type(payload_type):
            raise TypeError(f'Expected "payload_type" to be of type "SparkPayload" but got "{type(payload_type).__name__}".')
        self.variable = variable
        self.payload_type = payload_type

    @property
    def value(self,) -> ValueSparkPayload:
        """
            Current value store in the cache object.
        """
        return self.payload_type(self.variable.value)
    
    @value.setter
    def value(self, new_value) -> None:
        self.variable.value = new_value

    @property
    def shape(self,) -> tuple[int, ...]:
        """
            Shape of the value store in the cache object.
        """
        return self.variable.value.shape

    @property
    def dtype(self,) -> jnp.dtype:
        """
            Dtype of the value store in the cache object.
        """
        return self.variable.value.dtype

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################