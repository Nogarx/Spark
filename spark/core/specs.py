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
from jax.typing import DTypeLike
from dataclasses import dataclass

from spark.core.shape import Shape
import spark.core.validation as validation

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@jax.tree_util.register_pytree_node_class
@dataclass(init=False, frozen=True)
class PortSpecs:
    """
        Base specification for a port of an SparkModule.
    """
    payload_type: type[SparkPayload] | None
    shape: Shape | list[Shape] | None
    dtype: DTypeLike | None
    description: str | None = None

    def __init__(self, 
                 payload_type: type[SparkPayload] | None,
                 shape: Shape | None, 
                 dtype: DTypeLike | None, 
                 description: str | None = None ):
        if payload_type and not validation._is_payload_type(payload_type):
            raise TypeError(f'Expected "payload_type" to be of type "SparkPayload" but got "{type(payload_type).__name__}".')
        if shape and not (validation.is_shape(shape) or validation.is_list_shape(shape)):
            raise TypeError(f'Expected "shape" to be of type "Shape | list[Shape]" but got "{shape}".')
        if dtype and not isinstance(jnp.dtype(dtype), jnp.dtype):
            raise TypeError(f'Expected "dtype" to be of type "{DTypeLike}" but got "{type(dtype).__name__}".')
        if description and not (isinstance(description, str) or description is None):
            raise TypeError(f'Expected "description" to be of type "str" but got "{type(description).__name__}".')
        object.__setattr__(self, 'payload_type', payload_type)
        object.__setattr__(self, 'shape', shape)
        object.__setattr__(self, 'dtype', dtype)
        object.__setattr__(self, 'description', description)

    def tree_flatten(self):
        children = (self.payload_type, self.shape, self.dtype, self.description)
        aux_data = ()
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (payload_type, shape, dtype, description) = children
        return cls(payload_type=payload_type, shape=shape, dtype=dtype, description=description,)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.tree_util.register_pytree_node_class
@dataclass(init=False, frozen=True)
class InputSpec(PortSpecs):
    """
        Specification for an input port of an SparkModule.
    """
    is_optional: bool                 

    def __init__(self, 
                 payload_type: type[SparkPayload] | None, 
                 shape: Shape | None, 
                 dtype: DTypeLike | None, 
                 is_optional: bool, 
                 description: str | None = None):
        super().__init__(payload_type=payload_type, shape=shape, dtype=dtype, description=description)
        if not isinstance(is_optional, bool):
            raise ValueError(f'Expected "is_optional" to be of type "bool" but got "{type(is_optional).__name__}".')
        object.__setattr__(self, 'is_optional', is_optional)

    def tree_flatten(self):
        children = (self.payload_type, self.shape, self.dtype, self.is_optional, self.description)
        aux_data = ()
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (payload_type, shape, dtype, is_optional, description) = children
        return cls(payload_type=payload_type, shape=shape, dtype=dtype, is_optional=is_optional, description=description,)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class OutputSpec(PortSpecs):
    """
        Specification for an output port of an SparkModule.
    """
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.tree_util.register_pytree_node_class
@dataclass(init=False, frozen=True)
class PortMap:
    """
        Specification for an output port of an SparkModule.
    """
    origin: str        
    port: str       

    def __init__(self, origin: str, port: str):
        if not isinstance(origin, str):
            raise TypeError(f'Expected "origin" to be of type "str" but got "{type(origin).__name__}".')
        if not isinstance(port, str):
            raise TypeError(f'Expected "port" to be of type "str" but got "{type(port).__name__}".')
        object.__setattr__(self, 'origin', origin)
        object.__setattr__(self, 'port', port)

    def tree_flatten(self):
        children = (self.origin, self.port)
        aux_data = ()
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (origin, port) = children
        return cls(origin=origin, port=port)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# TODO: Inspect for missing mandatory fields.
@jax.tree_util.register_pytree_node_class
@dataclass(init=False, frozen=True)
class ModuleSpecs:
    """
        Specification for SparkModule automatic constructor.
    """

    name: str
    module_cls: type[SparkModule]        
    inputs: dict[str, list[PortMap]]               
    config: BaseSparkConfig

    def __init__(self, name: str, module_cls: type, inputs: dict[str, list[PortMap]], config: BaseSparkConfig):
        # TODO: Refactor code to remove lazy imports.
        from spark.core.registry import REGISTRY
        # Validate module_cls
        if not validation._is_module_type(module_cls):
            raise TypeError(f'"module_cls" must be a valid subclass of "SparkModule" but got "{type(module_cls).__name__}".')
        if REGISTRY.MODULES.get(module_cls.__name__) is None:  
            raise ValueError(f'Class "{module_cls.__name__}" does not exists in the registry.')
        # Validate inputs
        if not isinstance(inputs, dict):
            raise TypeError(f'"inputs" must be of type "dict" but got "{type(inputs).__name__}".')
        for key in inputs.keys():
            if not isinstance(key, str):
                raise TypeError(f'All keys in "inputs" must be strings, but found key "{key}" of type {type(key).__name__}.')
            if not isinstance(inputs[key], list):
                raise TypeError(f'All values in "inputs" must be a List of PortMap, but found value "{inputs[key]}" of type {type(inputs[key]).__name__}.')
            for element in inputs[key]:
                if not isinstance(element, PortMap):
                    raise TypeError(f'Expected PortMap at value {key} of "inputs", but found value "{inputs[key]}" of type {type(inputs[key]).__name__}.')
        
        # Validate model_config
        type_hints = tp.get_type_hints(module_cls)
        if not isinstance(config, type_hints['config']):
            raise TypeError(f'"config" must be of type "{type_hints['config'].__name__}" but got "{type(config).__name__}".')

        object.__setattr__(self, 'name', name)
        object.__setattr__(self, 'module_cls', module_cls)
        object.__setattr__(self, 'inputs', inputs)
        object.__setattr__(self, 'config', config)

    def tree_flatten(self):
        children = (self.name, self.module_cls, self.inputs, self.config)
        aux_data = ()
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (name, module_cls, inputs, config) = children
        return cls(name=name, module_cls=module_cls, inputs=inputs, config=config,)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@dataclass(init=False)
class VarSpec:
    
    shape: Shape
    dtype: DTypeLike

    def __init__(self, shape: Shape, dtype: DTypeLike):
        #if shape and not validation.is_shape(shape):
        #    raise TypeError(f'Expected "shape" to be of type "Shape" but got "{shape}".')
        #if dtype and not isinstance(jnp.dtype(dtype), jnp.dtype):
        #    raise TypeError(f'Expected "dtype" to be of type "{DTypeLike}" but got "{type(dtype).__name__}".')
        self.shape = shape
        self.dtype = dtype
        

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################