#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.module import SparkModule
    from spark.core.payloads import SparkPayload
    from spark.core.config import BaseSparkConfig

import jax
import jax.numpy as jnp
import typing as tp
import dataclasses as dc
import spark.core.utils as utils
import spark.core.validation as validation
from jax.typing import DTypeLike
from spark.core.registry import REGISTRY

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@jax.tree_util.register_dataclass
@dc.dataclass(init=False)
class PortSpecs:
    """
        Base specification for a port of an SparkModule.
    """
    payload_type: type[SparkPayload] | None
    shape: tuple[int, ...] | list[tuple[int, ...]] | None
    dtype: DTypeLike | None
    description: str | None = None

    def __init__(
            self, 
            payload_type: type[SparkPayload] | None,
            shape: tuple[int, ...] | list[tuple[int, ...]] | None,
            dtype: DTypeLike | None, 
            description: str | None = None 
        ) -> None:
        if payload_type and not validation._is_payload_type(payload_type):
            raise TypeError(
                f'Expected "payload_type" to be of type "SparkPayload" but got "{type(payload_type).__name__}".'
            )
        if shape and utils.is_shape(shape):
            shape = utils.validate_shape(shape)
        elif shape and utils.is_list_shape(shape):
            shape = utils.validate_list_shape(shape)
        elif shape:
            raise TypeError(
                f'Expected "shape" to be broadcastable to \"tuple[int, ...] | list[tuple[int, ...]]\".'
            )
        if dtype and not isinstance(jnp.dtype(dtype), jnp.dtype):
            raise TypeError(
                f'Expected \"dtype\" to be of type \"{DTypeLike}\" but got "{type(dtype).__name__}".'
            )
        if description and not isinstance(description, str):
            raise TypeError(
                f'Expected \"description\" to be of type \"str\" but got "{type(description).__name__}".'
            )
        self.payload_type = payload_type
        self.shape = shape
        self.dtype = dtype
        self.description = description

    def to_dict(self) -> dict[str, tp.Any]:
        """
            Serialize PortSpecs to dictionary
        """
        reg = REGISTRY.PAYLOADS.get_by_cls(self.payload_type)
        return {
            'payload_type': {
                '__payload_type__': reg.name if reg else None,
            },
            'shape': self.shape,
            'dtype': self.dtype,
            'description': self.description,
        }
    
    @classmethod
    def from_dict(cls, dct: dict) -> tp.Self:
        """
            Deserialize dictionary to  PortSpecs
        """
        return cls(**dct)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.tree_util.register_dataclass
@dc.dataclass(init=False)
class InputSpec(PortSpecs):
    """
        Specification for an input port of an SparkModule.
    """
    is_optional: bool                 

    def __init__(
            self, 
            payload_type: type[SparkPayload] | None, 
            shape: tuple[int, ...] | list[tuple[int, ...]] | None, 
            dtype: DTypeLike | None, 
            is_optional: bool = False, 
            description: str | None = None
        ) -> None:
        super().__init__(payload_type=payload_type, shape=shape, dtype=dtype, description=description)
        if not isinstance(is_optional, bool):
            raise ValueError(
                f'Expected \"is_optional\" to be of type \"bool\" but got \"{type(is_optional).__name__}\".'
            )
        self.is_optional = is_optional

    def to_dict(self) -> dict[str, tp.Any]:
        """
            Serialize InputSpec to dictionary
        """
        reg = REGISTRY.PAYLOADS.get_by_cls(self.payload_type)
        return {
            'payload_type': {
                '__payload_type__': reg.name if reg else None,
            },
            'shape': self.shape,
            'dtype': self.dtype,
            'description': self.description,
            'is_optional': self.is_optional,
        }

    @classmethod
    def from_dict(cls, dct: dict) -> tp.Self:
        """
            Deserialize dictionary to  PortSpecs
        """
        return cls(**dct)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.tree_util.register_dataclass
@dc.dataclass(init=False)
class OutputSpec(PortSpecs):
    """
        Specification for an output port of an SparkModule.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def to_dict(self) -> dict[str, tp.Any]:
        return super().to_dict()

    @classmethod
    def from_dict(cls, dct: dict) -> tp.Self:
        """
            Deserialize dictionary to  PortSpecs
        """
        return cls(**dct)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.tree_util.register_dataclass
@dc.dataclass(init=False)
class PortMap:
    """
        Specification for an output port of an SparkModule.
    """
    origin: str        
    port: str       

    def __init__(self, origin: str, port: str) -> None:
        if not isinstance(origin, str):
            raise TypeError(
                f'Expected "origin" to be of type "str" but got "{type(origin).__name__}".'
            )
        if not isinstance(port, str):
            raise TypeError(
                f'Expected "port" to be of type "str" but got "{type(port).__name__}".'
            )
        self.origin = origin
        self.port = port

    def to_dict(self) -> dict[str, tp.Any]:
        """
            Serialize PortMap to dictionary
        """
        return {
            'origin': self.origin,
            'port': self.port,
        }
    
    @classmethod
    def from_dict(cls, dct: dict) -> tp.Self:
        """
            Deserialize dictionary to PortMap
        """
        return cls(**dct)


#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.tree_util.register_dataclass
@dc.dataclass(init=False)
class ModuleSpecs:
    """
        Specification for SparkModule automatic constructor.
    """

    name: str
    module_cls: type[SparkModule]        
    inputs: dict[str, list[PortMap]]               
    config: BaseSparkConfig

    def __init__(self, name: str, module_cls: type, inputs: dict[str, list[PortMap]], config: BaseSparkConfig) -> None:
        # Validate module_cls
        if not validation._is_module_type(module_cls):
            raise TypeError(
                f'\"module_cls\" must be a valid subclass of \"SparkModule\" but got \"{type(module_cls).__name__}\".'
            )
        if REGISTRY.MODULES.get(module_cls.__name__) is None:  
            raise ValueError(
                f'Class \"{module_cls.__name__}\" does not exists in the registry.'
            )
        # Validate inputs
        if not isinstance(inputs, dict):
            raise TypeError(
                f'\"inputs\" must be of type \"dict\" but got \"{type(inputs).__name__}\".'
            )
        for key in inputs.keys():
            if not isinstance(key, str):
                raise TypeError(
                    f'All keys in \"inputs\" must be strings, but found key \"{key}\" of type \"{type(key).__name__}\".'
                )
            if not isinstance(inputs[key], list):
                raise TypeError(
                    f'All values in \"inputs\" must be a List of PortMap, but found value \"{inputs[key]}\" of type \"{type(inputs[key]).__name__}\".'
                )
            for element in inputs[key]:
                if not isinstance(element, PortMap):
                    raise TypeError(
                        f'Expected PortMap at value {key} of \"inputs\", but found value \"{inputs[key]}\" of type \"{type(inputs[key]).__name__}\".'
                    )
        # Validate model_config
        type_hints = tp.get_type_hints(module_cls)
        if not isinstance(config, type_hints['config']):
            raise TypeError(
                f'\"config\" must be of type \"{type_hints['config'].__name__}\" but got \"{type(config).__name__}\".'
            )
        self.name = name
        self.module_cls = module_cls
        self.inputs = inputs
        self.config = config

    def to_dict(self) -> dict[str, tp.Any]:
        """
            Serialize ModuleSpecs to dictionary
        """
        reg = REGISTRY.MODULES.get_by_cls(self.module_cls)
        return {
            'name': self.name,
            'module_cls': {
                '__module_type__': reg.name if reg else None,
            },
            'inputs': self.inputs,
            'config': self.config.to_dict()
        }
    
    @classmethod
    def from_dict(cls, dct: dict) -> tp.Self:
        """
            Deserialize dictionary to ModuleSpecs
        """
        # Validate dictionary
        name: str | None = dct.get('name', None)
        if not name or not isinstance(name, str):
            raise TypeError(
                f'Expected \"name\" to be of type \"str\", but got \"{name}\".'
            )
        module_cls: type[SparkModule] | None = dct.get('module_cls', None)
        if not module_cls or not issubclass(module_cls, SparkModule):
            raise TypeError(
                f'Expected \"module_cls\" to be of type \"type[SparkModule]\", but got \"{module_cls}\".'
            )
        config: BaseSparkConfig | dict | None = dct.get('config', None)
        if not config or not isinstance(config, (dict, BaseSparkConfig)):
            raise TypeError(
                f'Expected \"config\" to be of type \"type[BaseSparkConfig]\" | dict, but got \"{config}\".'
            )
        config = module_cls.get_config_spec()(**config) if isinstance(config, dict) else config
        inputs: dict | None = dct.get('inputs', None)
        if not inputs or not isinstance(config, dict):
            raise TypeError(
                f'Expected \"inputs\" to be of type \"dict\", but got \"{inputs}\".'
            )
        # Reconstruct spec
        return cls(
            name=name, 
            module_cls=module_cls,
            config=config,
            inputs=inputs,
        )

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################