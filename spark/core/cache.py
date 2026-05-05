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

@jax.tree_util.register_pytree_with_keys_class
@dc.dataclass(init=False)
class Cache(TwoKeyDict):

    @tp.overload
    def __setitem__(self, keys: str, value: dict[str, SparkPayload]) -> None: ...
    @tp.overload
    def __setitem__(self, keys: tuple[str, str], value: SparkPayload) -> None: ...
    def __setitem__(self, keys, value) -> None:
        if isinstance(keys, tuple):
            self._data[keys[0]][keys[1]] = value
        elif isinstance(value, dict):
            self._data[keys] = {k: v for k,v in value.items()}
        else:
            raise ValueError(f'Invalid keys: {keys} or value: {value}.')

    @tp.overload
    def __getitem__(self, keys: tuple[str, str]) -> SparkPayload: ...
    @tp.overload
    def __getitem__(self, keys: str)-> dict[str, SparkPayload]: ...
    def __getitem__(self, keys):
        return super().__getitem__(keys)

    @classmethod
    def from_specs(cls, data: TwoKeyDict[str, str, PortSpecs]) -> tp.Self:
        obj = cls()
        for (key1, key2), spec in data.items():
            # Skip optional
            if spec.shape is not None:
                obj[key1][key2] = spec.payload_type._from_spec(spec)
        return obj
    
    @classmethod
    def from_payloads(cls, data: TwoKeyDict[str, str, SparkPayload]) -> tp.Self:
        obj = cls()
        for (key1, key2), payload in data.items():
            # Skip optional
            if payload.shape is not None:
                obj[key1][key2] = type(payload)(jnp.zeros_like(payload.value))
        return obj
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################