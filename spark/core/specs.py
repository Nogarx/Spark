#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.module import SparkModule
    from spark.core.payloads import SparkPayload

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Type, Dict, List, Any, Optional
from spark.core.variable_containers import Variable
from spark.core.shape import bShape, normalize_shape
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
    payload_type: Type[SparkPayload]        
    shape: bShape               
    dtype: jnp.dtype                        
    description: Optional[str] = None       

    def __init__(self, 
                 payload_type: Type[SparkPayload], 
                 shape: bShape, 
                 dtype: jnp.dtype, 
                 description: Optional[str] = None):
        if not validation._is_payload_type(payload_type):
            raise ValueError(f'Expected payload_type of type "SparkPayload", got "{type(payload_type).__name__}".')
        try:
            if not shape is None:
                shape = normalize_shape(shape)
        except:
            raise ValueError(f'Expected shape of type {bShape.__name__}, got "{type(shape).__name__}".')
        if not isinstance(jnp.dtype(dtype), jnp.dtype):
            raise ValueError(f'Expected dtype of type {jnp.dtype.__name__}, got  "{type(dtype).__name__}".')
        if not (isinstance(description, str) or description is None):
            raise ValueError(f'Expected description {str.__name__}, got "{type(description).__name__}".')
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
class InputSpecs(PortSpecs):
    """
        Specification for an input port of an SparkModule.
    """
    is_optional: bool                       

    def __init__(self, 
                 payload_type: Type[SparkPayload], 
                 shape: bShape, 
                 dtype: jnp.dtype, 
                 is_optional: bool, 
                 description: Optional[str] = None):
        super().__init__(payload_type=payload_type, shape=shape, dtype=dtype, description=description)
        if not isinstance(is_optional, bool):
            raise ValueError(f'Expected is_optional {bool.__name__}, got "{type(is_optional).__name__}".')
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
class OutputSpecs(PortSpecs):
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
            raise TypeError(f'Expected origin {str.__name__} but got "{type(origin).__name__}".')
        if not isinstance(port, str):
            raise TypeError(f'Expected port {str.__name__} but got "{type(port).__name__}".')
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

@jax.tree_util.register_pytree_node_class
@dataclass(init=False)
class CacheSpec:
    """
        Specification for an output port of an SparkModule.
    """
    payload_type: SparkPayload        
    dtype: jnp.dtype
    var: Variable  

    def tree_flatten(self):
        children = (self.payload_type, self.dtype, self.var)
        aux_data = ()
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (payload_type, dtype, var) = children
        return cls(payload_type=payload_type, dtype=dtype, var=var)

    def __init__(self, var: Variable, payload_type: SparkPayload, dtype: jnp.dtype):
        if not isinstance(var, Variable):
            raise TypeError(f'Expected value {Variable.__name__}, got "{type(var).__name__}".')
        if not validation._is_payload_type(payload_type):
            raise TypeError(f'Expected payload_type "SparkPayload", got "{type(payload_type).__name__}".')
        if not isinstance(jnp.dtype(dtype), jnp.dtype):
            raise TypeError(f'Expected dtype of type {jnp.dtype.__name__}, got  "{type(dtype).__name__}".')
        self.var = var
        self.payload_type = payload_type
        self.dtype = dtype

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# TODO: Inspect for missing mandatory fields.
@jax.tree_util.register_pytree_node_class
@dataclass(init=False, frozen=True)
class ModuleSpecs:
    """
        Specification for SparkModule automatic constructor.
    """

    name: str
    module_cls: Type[SparkModule]        
    inputs: Dict[str, List[PortMap]]               
    init_args: Dict[str, Any]                      

    def __init__(self, name: str, module_cls: Type, inputs: Dict[str, List[PortMap]], init_args: Dict[str, Any]):
        # TODO: Refactor code to remove lazy imports.
        from spark.core.registry import REGISTRY
        # Validate module_cls
        if not validation._is_module_type(module_cls):
            raise TypeError(f'"module_cls" must be a valid subclass of "SparkModule" but got "{type(module_cls).__name__}".')
        if REGISTRY.MODULES.get(module_cls.__name__) is None:  
            raise ValueError(f'Class "{module_cls.__name__}" does not exists in the registry.')
        # Validate inputs
        if not isinstance(inputs, dict):
            raise TypeError(f'"inputs" must be of type "{dict.__name__}" but got "{type(inputs).__name__}".')
        for key in inputs.keys():
            if not isinstance(key, str):
                raise TypeError(f'All keys in "inputs" must be strings, but found key "{key}" of type {type(key).__name__}.')
            if not isinstance(inputs[key], list):
                raise TypeError(f'All values in "inputs" must be a List of PortMap, but found value "{inputs[key]}" of type {type(inputs[key]).__name__}.')
            for element in inputs[key]:
                if not isinstance(element, PortMap):
                    raise TypeError(f'Expected PortMap at value {key} of "inputs", but found value "{inputs[key]}" of type {type(inputs[key]).__name__}.')
        
        # Validate init_args
        if not isinstance(init_args, dict):
            raise TypeError(f'"init_args" must be of type "{dict.__name__}" but got "{type(init_args).__name__}".')
        for key in init_args.keys():
            if not isinstance(key, str):
                raise TypeError(f'All keys in "init_args" must be strings, but found key "{key}" of type {type(key).__name__}.')

        object.__setattr__(self, 'name', name)
        object.__setattr__(self, 'module_cls', module_cls.__name__)
        object.__setattr__(self, 'inputs', inputs)
        object.__setattr__(self, 'init_args', init_args)

    def tree_flatten(self):
        children = (self.name, self.module_cls, self.inputs, self.init_args)
        aux_data = ()
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (name, module_cls, inputs, init_args) = children
        return cls(name=name, module_cls=module_cls, inputs=inputs, init_args=init_args,)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@dataclass
class InputArgSpec:
    
    arg_type: Type
    is_optional: bool

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################