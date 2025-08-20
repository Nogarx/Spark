#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# NOTE: This is a relaxed version of the specs defined in core, as such this should only be used witin the GraphEditor.

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.specs import OutputSpec, InputSpec

import abc
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Type, Dict, List, Any
from spark.core.shape import bShape, normalize_shape
from spark.core.payloads import SparkPayload

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@dataclass(init=False)
class PortSpecEditor:
    """
        Mutable base specification for a port of an SparkModule.
    """
    payload_type: Type[SparkPayload]        
    shape: bShape | None                  
    dtype: Any | None                               
    description: str | None       

    def __init__(self, 
                 payload_type: Type[SparkPayload],  
                 shape: bShape | None = None, 
                 dtype: Any | None = None,  
                 description: str | None = None):
        
        # Validate attributes
        if not issubclass(payload_type, SparkPayload):
            raise ValueError(f'Expected "payload_type" of type "SparkPayload", got "{type(payload_type).__name__}".')
        try:
            if not shape is None:
                shape = normalize_shape(shape)
        except:
            raise ValueError(f'Expected "shape" of type {bShape.__name__}, got "{type(shape).__name__}".')
        if not isinstance(jnp.dtype(dtype), jnp.dtype):
            raise ValueError(f'Expected "dtype" of type {jnp.dtype.__name__}, got  "{type(dtype).__name__}".')
        if not (isinstance(description, str) or description is None):
            raise ValueError(f'Expected "description" {str.__name__}, got "{type(description).__name__}".')
        
        # Update attributes
        self.payload_type = payload_type
        self.shape = shape
        self.dtype = dtype
        self.description = description


#-----------------------------------------------------------------------------------------------------------------------------------------------#

@dataclass(init=False)
class InputSpecEditor(PortSpecEditor):
    """
        Specification for an input port of an SparkModule.
    """
    is_optional: bool                       
    port_maps: List[PortMap]

    def __init__(self, 
                 payload_type: Type[SparkPayload], 
                 shape: bShape, 
                 dtype: jnp.dtype, 
                 is_optional: bool, 
                 port_maps: List[PortMap] = [],
                 description: str | None = None):
        super().__init__(payload_type=payload_type, shape=shape, dtype=dtype, description=description)
        if not isinstance(is_optional, bool):
            raise TypeError(f'Expected "is_optional" to be of type {bool.__name__} but got "{type(is_optional).__name__}".')
        if not isinstance(port_maps, list):
            raise TypeError(f'Expected "port_maps" to be of type {list.__name__} but got "{type(port_maps).__name__}".')
        for port in port_maps:
            if not isinstance(port, PortMap):
                raise TypeError(f'Expected elements of "port_maps" to be of type {list.__name__} but got "{type(port_maps).__name__}".')
        
        # Update attributes
        self.is_optional = is_optional
        self.port_maps = port_maps

    def add_port_map(self, port_map: PortMap):
        self.port_maps.append(port_map)

    def remove_port_map(self, port_map: PortMap):
        self.port_maps.remove(port_map)

    @classmethod
    def from_input_specs(cls, input_spec: InputSpec, port_maps: List[PortMap] = []) -> InputSpecEditor:
        return cls(payload_type=input_spec.payload_type,  
                   shape=input_spec.shape, 
                   dtype=input_spec.dtype,  
                   is_optional=input_spec.is_optional,
                   port_maps=port_maps,
                   description=input_spec.description)
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@dataclass(init=False)
class OutputSpecEditor(PortSpecEditor):
    """
        Specification for an output port of an SparkModule.
    """
    pass

    @classmethod
    def from_output_specs(cls, output_spec: OutputSpec) -> OutputSpecEditor:
        return cls(payload_type=output_spec.payload_type,  
                   shape=output_spec.shape, 
                   dtype=output_spec.dtype,  
                   description=output_spec.description)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@dataclass(init=False)
class PortMap:
    """
        Specification for a connection between SparkModules within the SparkGraphEditor.
    """
    origin: str        
    port: str       

    def __init__(self, origin: str, port: str):

        # Validate attributes
        if not isinstance(origin, str):
            raise ValueError(f'Expected origin {str.__name__}, got "{type(origin).__name__}".')
        if not isinstance(port, str):
            raise ValueError(f'Expected port {str.__name__}, got "{type(port).__name__}".')
        
        # Update attributes
        self.origin = origin
        self.port = port
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@dataclass(init=False)
class ModuleSpecsEditor:
    """
        Specification for SparkModule automatic constructor within the SparkGraphEditor.
    """

    name: str
    module_cls: str         
    inputs: Dict[str, List[PortMap]]
    init_args: Dict[str, Any]                      

    def __init__(self, name: str, module_cls: str, inputs: Dict[str, List[PortMap]], init_args: Dict[str, Any]):
        
        # Validate module_cls
        if not isinstance(module_cls, str):
            raise TypeError(f'Expected "module_cls" to be of type "str" but got type {type(module_cls).__name__}.')
        #if REGISTRY.MODULES.get(module_cls) is None:  
        #    raise ValueError(f'Module "{module_cls}" does not exists in the registry.')
        # Validate inputs
        if not isinstance(inputs, dict):
            raise TypeError(f'"inputs" must be of type "{dict.__name__}" but got "{type(init_args).__name__}".')
        for key in inputs.keys():
            if not isinstance(key, str):
                raise TypeError(f'All keys in "inputs" must be strings, but found key "{key}" of type {type(key).__name__}.')
            if not isinstance(inputs[key], list):
                raise TypeError(f'All values in "inputs" must be a list of PortMap, but found value "{inputs[key]}" of type {type(inputs[key]).__name__}.')
            for port_map in inputs[key]:
                if not isinstance(port_map, PortMap):
                    raise TypeError(f'All values in "inputs" must be a list of PortMap, but found value "{inputs[key]}" of type {type(inputs[key]).__name__}.')
        # Validate init_args
        if not isinstance(init_args, dict):
            raise TypeError(f'"init_args" must be of type "{dict.__name__}" but got "{type(init_args).__name__}".')
        for key in init_args.keys():
            if not isinstance(key, str):
                raise TypeError(f'All keys in "init_args" must be strings, but found key "{key}" of type {type(key).__name__}.')
        
        # Update attributes
        self.name = name
        self.module_cls = module_cls
        self.inputs = inputs
        self.init_args = init_args

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################