#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.payloads import SparkPayload, ValueSparkPayload
    from spark.core.specs import PortSpecs

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
import typing as tp
import dataclasses as dc
import spark.core.validation as validation
from spark.core.variables import Variable

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@jax.tree_util.register_pytree_node_class
@dc.dataclass(init=False)
class Cache:
    """
        Cache dataclass.
    """
    _variable: SparkPayload  

    def __init__(self, _variable: SparkPayload) -> None:
        self._variable = _variable

    @property
    def value(self,) -> SparkPayload:
        """
            Current value store in the cache object.
        """
        return self._variable.value

    def tree_flatten(self) -> tuple[tuple, tuple]:
        return self._variable.tree_flatten()

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> tp.Self:
        obj = cls.__new__(cls)
        obj._variable = aux_data[0].tree_unflatten(aux_data, children)
        return obj

    @property
    def shape(self,) -> tuple[int, ...]:
        """
            Shape of the value store in the cache object.
        """
        return self._variable.shape

    @property
    def dtype(self,) -> jnp.dtype:
        """
            Dtype of the value store in the cache object.
        """
        return self._variable.dtype

    @classmethod
    def _from_spec(cls, spec: PortSpecs) -> tp.Self:
        obj = cls.__new__(cls)
        obj._variable = spec.payload_type._from_spec(spec)
        return obj

    def get(self,) -> SparkPayload:
        return self._variable

    def set(self, payload: SparkPayload) -> None:
        self._variable = payload

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################