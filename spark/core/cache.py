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
from spark.core.backend import Variable
from spark.core.utils import TwoKeyDict
from collections import defaultdict
from collections.abc import MutableMapping

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@jax.tree_util.register_pytree_with_keys_class
@dc.dataclass(init=False, eq=False)
class Cache(TwoKeyDict):
    """
       Storage system for a `Brain` controller.

        A two key mapping from (module name, port name) to the payload that port produced. It is
        registered as a pytree, so it is carried through a jit boundary as state.

        A `Brain` reads its inputs from the cache and writes its outputs back once every module
        has run. This system allows module decoupling for one step, which improves model computation 
        speed significantly by allowing jit to schedule more than one module at the same time.

        See Also
        --------
        TwoKeyDict : The mapping this builds on.
    """
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
        """
            Builds a cache of mock payloads from port specifications.

            Parameters
            ----------
            data : TwoKeyDict of (str, str) to PortSpecs
                Specifications by (module name, port name).

            Returns
            -------
            Cache
                One mock payload per specification. Specifications carrying no shape are optional
                ports and are skipped.
        """
        obj = cls()
        for (key1, key2), spec in data.items():
            # Skip optional
            if spec.shape is not None:
                obj[key1, key2] = spec.payload_type._from_spec(spec)
        return obj
    
    @classmethod
    def from_payloads(cls, data: TwoKeyDict[str, str, SparkPayload]) -> tp.Self:
        """
            Builds a cache of zero-filled payloads from existing payloads.

            Parameters
            ----------
            data : TwoKeyDict of (str, str) to SparkPayload
                Payloads by (module name, port name), read for their type and shape.

            Returns
            -------
            Cache
                One zeroed payload per entry, of the same type and shape. Payloads carrying no shape
                are optional ports and are skipped.
        """
        obj = cls()
        for (key1, key2), payload in data.items():
            # Skip optional
            if payload.shape is not None:
                obj[key1, key2] = type(payload)(jnp.zeros_like(payload.value))
        return obj
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################