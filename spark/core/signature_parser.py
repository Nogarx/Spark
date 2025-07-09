
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.module import SparkModule

import ast
import typing
import inspect
import textwrap
import jax.numpy as jnp
from typing import List, Tuple, Callable, List, Union, Dict, Any
from spark.core.specs import InputSpecs, OutputSpecs, InputArgSpec
from spark.core.shape import bShape, Shape
import spark.core.validation as validation

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def get_input_specs(module: Union[SparkModule, type[SparkModule]]) -> Dict[str, InputSpecs]:
    """
        Returns a dictionary of the SparkModule's input port specifications.
    """

    # Get input signature.
    annotations = typing.get_type_hints(module.__call__)
    input_specs = {}
    
    # Populate runtime fields if module is an instance of SparkModule.
    if validation._is_module_instance(module):
        input_shapes = module.input_shapes if module.input_shapes != None else [None] * len(annotations)
        dtype = module._dtype
    elif validation._is_module_type(module):
        input_shapes = [None] * len(annotations)
        dtype = jnp.float16
    else:
        raise TypeError(f'Expected module to be of type "SparkModule" or "type[SparkModule]" but got "{type(module).__name__}".')
    
    # Create signatures. 
    for (name, input_annotation), shape in zip(annotations.items(), input_shapes):

        # Skip unimportant parameters.
        if name in ['self', 'cls', 'return']: 
            continue

        # Validate annotation.
        is_optional = False
        base_type = typing.get_origin(input_annotation)
        args = typing.get_args(input_annotation)

        # Handle Optional[T], which is a Union[T, NoneType]
        if base_type is Union and len(args) == 2 and args[1] is type(None):
            is_optional = True
            base_type = typing.get_origin(args[0])

        # If there was no origin the annotation itself is the type
        payload_type = base_type if base_type is not None else input_annotation

        # Check if the payload_type is a valid class and a subclass of SparkPayload
        if not validation._is_payload_type(payload_type):
            # Raise error, payload is not fully compatible with the framework.
            raise TypeError(f'Error: Input parameter "{name}" has type {type(payload_type).__name__}, '
                            f'which is not a valid SparkPayload, Optional[SparkPayload] or sequence of SparkPayload.'
                            f'If you intended to pass a union (e.g. tuple, list, etc.) consider passing each entry '
                            f'in the union directly as SparkPayload type to the __call__ method. '
                            f'Alternatively, consider defining a custom SparkPayload dataclass as a wrapper for your input.')
        
        # Add the spec to collection.
        input_specs[name] = InputSpecs(
            payload_type=payload_type,
            shape=shape,
            is_optional=is_optional,
            dtype=dtype,
            description=f'Input port for {name}',
        )
    return input_specs

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def get_output_specs(module: Union[SparkModule, type[SparkModule]]) -> Dict[str, OutputSpecs]:
    """
        Returns a dictionary of the SparkModule's input port specifications.
    """
    # Get input signature.
    annotations = typing.get_type_hints(module.__call__)
    output_specs = {}

    # Check that the input signature has a return annotation.
    if not 'return' in annotations:
        raise SyntaxError(f'Module "{type(module).__name__}" does not define a return type.')
    annotations = annotations.pop('return')

    # Check if the annotation is a TypedDict
    if not typing.is_typeddict(annotations):
        raise TypeError(f'Module "{type(module).__name__}" does not have a type return annotation of type typing.TypedDict'
                        f'and cannot auto-infer output ports specs. Consider adding a TypedDict annotation to the __call__ method.')
    
    # get_type_hints is required once more since __future__ delays the evaluation of typedict
    annotations = typing.get_type_hints(annotations)

    # Populate runtime fields if module is an instance of SparkModule.
    if validation._is_module_instance(module):
        output_shapes = module.output_shapes if module.output_shapes != None else [None] * len(annotations)
        dtype = module._dtype
    elif validation._is_module_type(module):
        output_shapes = [None] * len(annotations)
        dtype = dtype = jnp.float16
    else:
        raise TypeError(f'Expected module to be of type "SparkModule" or "type[SparkModule]" but got "{type(module).__name__}".')
    
    # Build output specs
    for (name, payload_type), shape in zip(annotations.items(), output_shapes):

        # Check if the payload_type is a valid class and a subclass of SparkPayload
        if not validation._is_payload_type(payload_type):
            # Raise error, payload is not fully compatible with the framework.
            raise TypeError(f'Error: Output parameter "{name}" has type {type(payload_type).__name__}, '
                            f'which is not a valid SparkPayload, Optional[SparkPayload] or sequence of SparkPayload.')

        output_specs[name] = OutputSpecs(
            payload_type=payload_type,
            shape=shape,
            dtype=dtype,
            description=f'Output port for {name}',
    )
    return output_specs

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def get_method_signature(method: Callable) -> Dict[str, InputArgSpec]:
    """
        Returns a dictionary of the method's input signature.
    """

    # Validation
    if not callable(method):
        raise TypeError(f'"method" must be callable but got {method} of type {type(method).__name__}.')

    # Get input signature.
    annotations = typing.get_type_hints(method)
    args_specs = {}
    
    # Create signatures. 
    for name, input_annotation in annotations.items():

        # Skip unimportant parameters.
        if name in ['self', 'cls']: 
            continue

        # Validate annotation.
        is_optional = False
        origin = typing.get_origin(input_annotation)
        args = typing.get_args(input_annotation)

        # Handle Optional[T], which is a Union[T, NoneType]
        if origin is Union and len(args) == 2 and type(None) in args:
            is_optional = True
            input_annotation = next(a for a in args if a is not type(None))

        # Simplify arg_type
        arg_type = normalize_type(input_annotation)

        # Add the spec to collection.
        args_specs[name] = InputArgSpec(arg_type=arg_type, is_optional=is_optional)

    return args_specs


#-----------------------------------------------------------------------------------------------------------------------------------------------#

def normalize_type(tp):
    """
        Normalizes complex typing patterns.
    """
    # Check for shapes
    if tp is bShape:
        return Shape
    #elif Other possible patterns:
    return tp

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################