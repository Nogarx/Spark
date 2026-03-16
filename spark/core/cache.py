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
from spark.core.utils import TwoKeyDict
from collections import defaultdict
from collections.abc import MutableMapping

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@jax.tree_util.register_pytree_node_class
@dc.dataclass(init=False)
class CacheEntry:
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

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.tree_util.register_pytree_node_class
@dc.dataclass(init=False)
class Cache(TwoKeyDict):

    @tp.overload
    def __setitem__(self, keys: str, value: dict[str, SparkPayload]) -> None: ...
    @tp.overload
    def __setitem__(self, keys: tuple[str, str], value: SparkPayload) -> None: ...
    def __setitem__(self, keys, value) -> None:
        if isinstance(keys, tuple):
            self._data[keys[0]][keys[1]] = CacheEntry(value)
        elif isinstance(value, dict):
            self._data[keys] = {k: CacheEntry(v) for k,v in value.items()}
        else:
            raise ValueError(f'Invalid keys: {keys} or value: {value}.')

    @tp.overload
    def __getitem__(self, keys: tuple[str, str]) -> CacheEntry: ...
    @tp.overload
    def __getitem__(self, keys: str)-> dict[str, CacheEntry]: ...
    def __getitem__(self, keys):
        return super().__getitem__(keys)

    @classmethod
    def from_specs(cls, data: TwoKeyDict[str, str, PortSpecs]) -> tp.Self:
        obj = cls()
        for (key1, key2), spec in data.items():
            # Skip optional
            if spec.shape is not None:
                obj[key1][key2] = CacheEntry._from_spec(spec)
        return obj
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################