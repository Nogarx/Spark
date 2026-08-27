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
from spark.core.backend import Constant
from jax.typing import DTypeLike
from spark.core.registry import REGISTRY, RegistryNamespace
from math import prod

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@jax.tree_util.register_dataclass
@dc.dataclass(init=False)
class PortSpecs:
    """
        Module port specification.

        Names the payload type, and the shape and dtype once they are known. Shape and dtype are
        None until the module is built, since they are inferred from the values that reach it.

        Parameters
        ----------
        payload_type : type of SparkPayload or None
            Type the port carries. Two ports connect only if this matches.
        shape : tuple of int or list of tuple of int or None
            Shape of the payload. A list when several values arrive on the port.
        dtype : DTypeLike or None
            Dtype of the payload.
        description : str, optional
            Human readable description of the port.
    """
    payload_type: type[SparkPayload] | None
    shape: tuple[int, ...] | list[tuple[int, ...]] | None
    dtype: DTypeLike | None
    description: str | None

    def __init__(
            self, 
            payload_type: type[SparkPayload] | None,
            shape: tuple[int, ...] | list[tuple[int, ...]] | None,
            dtype: DTypeLike | None, 
            description: str | None = None,
        ) -> None:

        if shape and utils.is_shape(shape):
            shape = utils.validate_shape(shape)
        elif shape and utils.is_list_shape(shape):
            shape = utils.validate_list_shape(shape)

        self.payload_type = payload_type
        self.shape = shape
        self.dtype = dtype
        self.description = description

    def to_dict(self,) -> dict[str, tp.Any]:
        """
            Serializes the specification to a dictionary.

            Returns
            -------
            dict
                The fields of the specification, with the payload type as its registered name.
        """
        reg = REGISTRY.Payloads.get_by_cls(self.payload_type)
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
            Builds a specification from a dictionary.

            Parameters
            ----------
            dct : dict
                As produced by `to_dict`.

            Returns
            -------
            PortSpecs
        """
        return cls(**dct)

    @classmethod
    def from_payload(cls, payload: SparkPayload) -> tp.Self:
        """
            Builds a specification describing an existing payload.

            Parameters
            ----------
            payload : SparkPayload
                Payload to read the type, shape and dtype from.

            Returns
            -------
            PortSpecs
        """
        from spark.core.payloads import SpikeArray
        dct = {
            'payload_type': type(payload),
            'shape': payload.shape,
            'dtype': payload.dtype,
            'description': 'Auto-generated payload description',
        }
        return cls(**dct)

    @classmethod
    def from_portspecs_list(cls, portspec_list: list[PortSpecs]) -> tp.Self:
        """
            Merges several specifications into one.

            Used for a port fed by more than one source, whose values are concatenated.

            Parameters
            ----------
            portspec_list : list of PortSpecs
                Specifications to merge. All must carry the same payload type.

            Returns
            -------
            PortSpecs
                The shared payload type, the promoted dtype and the merged shape. The list itself is
                returned unchanged when it holds a single entry.

            Raises
            ------
            TypeError
                If the specifications do not all carry the same payload type.
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
        # Since we expect everything to be a valid PortSpecs we don't really need to validate anything else.
        # Generic description.
        description = 'Merged PortSpecs'
        # Promote dtypes
        dtype = jnp.result_type(*[spec.dtype for spec in portspec_list])
        # Merge shapes.
        shape = utils.merge_shape_list([spec.shape for spec in portspec_list])
        # Merge inhibition_mask when present.
        return cls(
            payload_type=payload_type,
            shape=shape,
            dtype=dtype,
            description=description,
        )

    def _create_mock_payload(self,) -> SparkPayload:
        return self.payload_type._from_spec(self)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.tree_util.register_dataclass
@dc.dataclass(init=False)
class PortMap:
    """
        Module's connections specification.

        A pair of the form (module_name, module_port_name) that specifies a connection within a controller.
        ``'__call__'`` and ``'__self__'`` are speciail module_names used to refer to the controller's inputs
        and the same module defining the mapping.
        

        Parameters
        ----------
        origin : str
            Name of the module the value comes from. ``'__call__'`` for an input of the enclosing
            controller, ``'__self__'`` for a property of the controller itself.
        port : str
            Name of the port on that module.
        is_property : bool, default False
            Read a property of the origin rather than one of its outputs.
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
            Serializes the map to a dictionary.

            Returns
            -------
            dict
                The origin, the port and the property flag.
        """
        return {
            'origin': self.origin,
            'port': self.port,
            'is_property': self.is_property
        }
    
    @classmethod
    def from_dict(cls, dct: dict[str, tp.Any]) -> tp.Self:
        """
            Builds a map from a dictionary.

            Parameters
            ----------
            dct : dict
                As produced by `to_dict`.

            Returns
            -------
            PortMap
        """
        return cls(**dct)

    def __hash__(self) -> int:
        return hash(self.origin+self.port+str(self.is_property))

    def __eq__(self, other: PortMap) -> bool:
        return self.origin == other.origin and self.port == other.port and self.is_property == other.is_property

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.tree_util.register_dataclass
@dc.dataclass(init=False)
class ModuleSpecs:
    """
        Specification of a module within a controller.

        This is what makes a model data rather than code: a controller holds a tuple of these, so
        it can be written to a file, edited, and instantiated again without a Python definition.

        Parameters
        ----------
        name : str
            Name the module answers to inside the controller. Also the attribute it is bound to.
        module_cls : type of SparkModule
            Class to instantiate. Must be registered.
        inputs : dict of str to PortMap or list of PortMap
            For each input port of the module, where its value comes from. Several entries for one
            port are concatenated in order.
        config : SparkConfig, optional
            Configuration of the module. The default configuration is used when omitted.
        outputs : dict of str to str, optional
            Output ports of the module to expose as output ports of the controller, as
            ``{controller port: module port}``.
        effects : dict of str to PortMap or list of PortMap, optional
            Properties of this module to write after the step, as ``{property: source}``. A
            plasticity rule writes weights back onto a synapse this way. The property must have a
            setter.
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
        from spark.nn.components.base import Component
        from spark.nn.interfaces.base import Interface
        from spark.nn.controllers.neuron import Neuron
        if REGISTRY.__built__ and issubclass(module_cls, Component) and REGISTRY.Components.get_by_cls(module_cls) is None:  
            raise ValueError(
                f'Component class \"{module_cls.__name__}\" does not exists in the registry.'
            )
        if REGISTRY.__built__ and issubclass(module_cls, Interface) and REGISTRY.Interfaces.get_by_cls(module_cls) is None:  
            raise ValueError(
                f'Interface class \"{module_cls.__name__}\" does not exists in the registry.'
            )
        elif REGISTRY.__built__ and issubclass(module_cls, Neuron) and REGISTRY.Neurons.get(module_cls.__name__) is None:  
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
            Serializes the specification to a dictionary.

            Returns
            -------
            dict
                The name, the registered name of the module class, the wiring and the configuration.
        """
        reg, subregistry = None, None
        for namespace in (RegistryNamespace.Components, RegistryNamespace.Interfaces, RegistryNamespace.Neurons):
            reg = getattr(REGISTRY, namespace.name).get_by_cls(self.module_cls)
            if reg is not None:
                subregistry = namespace.name
                break
        if reg is None:
            raise RuntimeError(
                f'Unable to find "{self.module_cls}" registry entry. Confirm that the class is a member of a registry.'
            )
        return {
            'name': self.name,
            'module_cls': {
                '__module_type__': reg.name,
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
            Builds a specification from a dictionary.

            Parameters
            ----------
            dct : dict
                As produced by `to_dict`. The module class is looked up in the registry by name.

            Returns
            -------
            ModuleSpecs
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