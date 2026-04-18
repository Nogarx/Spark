#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.module import SparkModule
    from spark.core.payloads import SparkPayload
    from spark.core.config import SparkConfig

import jax
import jax.numpy as jnp
import numpy as np
import typing as tp
import dataclasses as dc
import spark.core.utils as utils
import spark.core.validation as validation
from spark.core.variables import Constant
from spark.core.typing import enforce_annotations
from jax.typing import DTypeLike
from spark.core.registry import REGISTRY
from math import prod

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################


# TODO: We need to reliably distinguish between single and multi input ports.
@jax.tree_util.register_dataclass
@dc.dataclass(init=False)
class PortSpecs:
    """
        Base specification for a port of an SparkModule.
    """
    payload_type: type[SparkPayload] | None
    shape: tuple[int, ...] | list[tuple[int, ...]] | None
    dtype: DTypeLike | None
    description: str | None

    # Auxiliary metadata only used at model build time.
    # These are dynamic metadata variables that are only needed at build time.
    async_spikes: bool | None
    inhibition_mask: bool | None

    def __init__(
            self, 
            payload_type: type[SparkPayload] | None,
            shape: tuple[int, ...] | list[tuple[int, ...]] | None,
            dtype: DTypeLike | None, 
            description: str | None = None,
            async_spikes: bool | None = None,
            inhibition_mask: jax.Array | bool | None = None,
        ) -> None:

        if shape and utils.is_shape(shape):
            shape = utils.validate_shape(shape)
        elif shape and utils.is_list_shape(shape):
            shape = utils.validate_list_shape(shape)

        self.payload_type = payload_type
        self.shape = shape
        self.dtype = dtype
        self.description = description
        self.async_spikes = async_spikes
        self.inhibition_mask = inhibition_mask

    def to_dict(self,) -> dict[str, tp.Any]:
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
    def from_dict(cls, dct: dict[str, tp.Any]) -> tp.Self:
        """
            Deserialize dictionary to  PortSpecs
        """
        return cls(**dct)

    @classmethod
    def from_portspecs_list(cls, portspec_list: list[PortSpecs], validate_async: bool = True) -> tp.Self:
        """
            Merges a list of PortSpecs into a single PortSpecs
        """
        # Return original portspec if list contains a single element
        if len(portspec_list) == 1:
            return portspec_list[0]

        # Payload validation.
        payload_type = set([spec.payload_type for spec in portspec_list])
        if len(payload_type) != 1:
            raise TypeError(
                f'Expect all payload types to be equal  but got {payload_type}.'
                f'In order to merge the PortSpecs into a single PortSpecs all types must be the same.'
            )
        payload_type = list(payload_type)[0]
        # Validate that all  async_spikes has the same value.
        if validate_async:
            async_spikes = set([spec.async_spikes for spec in portspec_list])
            if len(async_spikes) != 1:
                raise TypeError(
                    f'Expect all async_spikes values to be equal but got {async_spikes}.'
                    f'In order to merge the PortSpecs into a single PortSpecs all async_spikes values must be the same.'
                )
            async_spikes = list(async_spikes)[0]
        else:
            async_spikes = None
        # Since we expect everything to be a valid PortSpecs we don't really need to validate anything else.
        # Generic description.
        description = 'Merged PortSpecs'
        # Promote dtypes
        dtype = jnp.result_type(*[spec.dtype for spec in portspec_list])
        # Merge shapes.
        shape = utils.merge_shape_list([spec.shape for spec in portspec_list])
        # Merge inhibition_mask when present.
        from spark.core.payloads import SpikeArray
        if payload_type == SpikeArray:
            inhibition_mask = []
            for spec in portspec_list:
                if isinstance(spec.inhibition_mask, (jax.Array, np.ndarray)):
                    inhibition_mask.append(spec.inhibition_mask.reshape(-1))
                elif isinstance(spec.inhibition_mask, Constant):
                    inhibition_mask.append(spec.inhibition_mask.value.reshape(-1))
                elif isinstance(spec.inhibition_mask, (bool, int, float)):
                    inhibition_mask.append(
                        bool(spec.inhibition_mask) * jnp.ones(spec.shape, dtype=jnp.bool).reshape(-1)
                    )
                else:
                    # Inhibiton mask is ill defined.
                    raise TypeError(
                        f'Expected inhibition mask to be an instance of jax.Array | np.ndarray | bool but got '
                        f'{spec.inhibition_mask}, which is not broadcastable to a mask.'
                    )
            inhibition_mask = jnp.concatenate(inhibition_mask)
            if inhibition_mask.shape != shape:
                raise ValueError(
                    f'Inhibition mask with shape {inhibition_mask.shape} is not compatible with expected shape {shape}.'
                )
        else:
            inhibition_mask = None
        return cls(
            payload_type=payload_type,
            shape=shape,
            dtype=dtype,
            description=description,
            async_spikes=async_spikes,
            inhibition_mask=inhibition_mask,
        )

    def _create_mock_input(self,) -> SparkPayload:
        return self.payload_type._from_spec(self)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.tree_util.register_dataclass
@dc.dataclass(init=False)
class PortMap:
    """
        Specification for an output port of an SparkModule.
    """
    origin: str        
    port: str       
    is_property: bool

    def __init__(self, origin: str, port: str, is_property: bool = False) -> None:
        self.origin = origin
        self.port = port
        self.is_property = is_property

    def to_dict(self,) -> dict[str, tp.Any]:
        """
            Serialize PortMap to dictionary
        """
        return {
            'origin': self.origin,
            'port': self.port,
            'is_property': self.is_property
        }
    
    @classmethod
    def from_dict(cls, dct: dict[str, tp.Any]) -> tp.Self:
        """
            Deserialize dictionary to PortMap
        """
        return cls(**dct)

    def __hash__(self) -> int:
        return hash(self.origin+self.port+str(self.is_property))

    def __eq__(self, other: PortMap) -> bool:
        return self.origin == other.origin and self.port == other.port and self.is_property == other.is_property

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# TODO: Serialization of this class is weak.
@jax.tree_util.register_dataclass
@dc.dataclass(init=False)
class ModuleSpecs:
    """
        Specification for SparkModule automatic constructor.
    """

    name: str
    module_cls: type[SparkModule]        
    inputs: dict[str, tp.Iterable[PortMap]]
    outputs: dict[str, str]
    effects: dict[str, tp.Iterable[PortMap]]
    config: SparkConfig

    def __init__(
            self, 
            name: str, 
            module_cls: type[SparkModule], 
            inputs: dict[str, tp.Iterable[PortMap]] | dict[str, PortMap], 
            config: SparkConfig | None = None,
            outputs: dict[str, str] | None = None,
            effects: dict[str, tp.Iterable[PortMap]] | dict[str, PortMap] | None = None,
        ) -> None:
        # Validate module_cls
        # TODO: In order to add controllers to the registry they need to build the ModuleSpecs,
        # this currently access the REGISTRY to validate the spec, which crashes with the controllers
        # since the registry is not necessarily built
        from spark.core.module import SparkModule
        from spark.nn.controllers.neuron import Neuron
        if REGISTRY.MODULES.__built__ and issubclass(module_cls, SparkModule) and REGISTRY.MODULES.get(module_cls.__name__) is None:  
            raise ValueError(
                f'Module class \"{module_cls.__name__}\" does not exists in the registry.'
            )
        elif REGISTRY.NEURONS.__built__ and issubclass(module_cls, Neuron) and REGISTRY.NEURONS.get(module_cls.__name__) is None:  
            raise ValueError(
                f'Neuron class \"{module_cls.__name__}\" does not exists in the registry.'
            )
        # Validate model_config
        type_hints = tp.get_type_hints(module_cls)
        if config is not None and not isinstance(config, type_hints['config']):
            raise TypeError(
                f'\"config\" must be of type \"{type_hints['config'].__name__}\" but got \"{type(config).__name__}\".'
            )
        # Set values
        self.name = name
        self.module_cls = module_cls
        self.inputs = inputs
        # NOTE: We allow partial configs to simplify controller definitions
        self.config = config if config is not None else module_cls.get_config_spec().partial()
        self.outputs = {} if outputs is None else outputs
        self.effects = {} if effects is None else effects

    def to_dict(self,) -> dict[str, tp.Any]:
        """
            Serialize ModuleSpecs to dictionary
        """
        from spark.core.module import SparkModule
        from spark.nn.controllers.neuron import Neuron
        if issubclass(self.module_cls, SparkModule):
            reg = REGISTRY.MODULES.get_by_cls(self.module_cls)
            subregistry = 'MODULES'
        elif issubclass(self.module_cls, Neuron):
            reg = REGISTRY.NEURONS.get_by_cls(self.module_cls)
            subregistry = 'NEURONS'
        else:
            raise RuntimeError(
                f'Unable to find "{self.module_cls}" registry entry. Confirm that the class is a member of a registry.'
            )
        return {
            'name': self.name,
            'module_cls': {
                '__module_type__': reg.name if reg else None,
                '__subregistry__': subregistry,
            },
            'inputs': self.inputs,
            'config': self.config,
            'outputs': self.outputs,
            'effects': self.effects,
        }
    
    @classmethod
    def from_dict(cls, dct: dict[str, tp.Any],) -> tp.Self:
        """
            Deserialize dictionary to ModuleSpecs
        """
        # Name
        name = dct.get('name', None)
        if name is None:
            raise ValueError(
                'ModuleSpecs name cannot be "None".'
            )
        # Module
        module_cls: SparkModule = dct.get('module_cls', None)
        # Config
        config_cls: SparkConfig = module_cls.get_config_spec()
        config = dct.get('config', None)
        config = config_cls.from_dict(config) if isinstance(config, dict) else config
        # Inputs
        inputs = dct.get('inputs', {})
        for key, port_list in inputs.items():
            inputs[key] = [
                PortMap.from_dict(port) if isinstance(port, dict) else port for port in port_list
            ]
        # Outputs
        outputs = dct.get('outputs', {})
        # Effects
        effects = dct.get('effects', {})
        for key, port_list in effects.items():
            effects[key] = [
                PortMap.from_dict(port) if isinstance(port, dict) else port for port in port_list
            ]
        # Reconstruct spec
        return cls(
            name=name, 
            module_cls=module_cls,
            inputs=inputs,
            config=config,
            outputs=outputs,
            effects=effects,
        )
        
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################