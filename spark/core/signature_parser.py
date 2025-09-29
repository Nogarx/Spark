
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.module import SparkModule


import inspect
import typing as tp
from spark.core.specs import InputSpec, OutputSpec
from spark.core.shape import bShape, Shape
import spark.core.validation as validation

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def get_input_specs(module: type[SparkModule]) -> dict[str, InputSpec]:
    """
        Returns a dictionary of the SparkModule's input port specifications.
    """

    # Check isinstance of SparkModule.
    if not validation._is_module_type(module):
        raise TypeError(
            f'Expected object class of type "SparkModule" but got "{type(module).__name__}".'
        )

    # Get input signature.
    signature = inspect.signature(module.__call__)
    signature_type_hints = tp.get_type_hints(module.__call__)
    
    
    # Create signatures.
    input_specs = {}
    for parameter in signature.parameters.values():

        # Skip unimportant parameters.
        if parameter.name in ['self', 'cls']: 
            continue

        # Scrap parameter.
        is_optional = parameter.default is not inspect.Parameter.empty
        if tp.get_origin(signature_type_hints[parameter.name]) is list:
            payload_type = tp.get_args(signature_type_hints[parameter.name])[0]
        else:
            payload_type = signature_type_hints[parameter.name]

        # Check if the payload_type is a valid class and a subclass of SparkPayload
        if not validation._is_payload_type(payload_type):
            # Raise error, payload is not fully compatible with the framework.
            raise TypeError(
                f'Error: Input parameter "{parameter.name}" has type {type(payload_type).__name__}, ' 
                f'which is not a valid SparkPayload, None or sequence of SparkPayload.' 
                f'If you intended to pass a union (e.g. tuple, list, etc.) consider passing each entry ' 
                f'in the union directly as SparkPayload type to the __call__ method. ' 
                f'Alternatively, consider defining a custom SparkPayload dataclass as a wrapper for your input.'
            )
        
        # Add the spec to collection.
        input_specs[parameter.name] = InputSpec(
            payload_type=payload_type,
            shape=None,
            dtype=None,
            is_optional=is_optional,
            description=f'Input port for {parameter.name}',
        )
    return input_specs

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def get_output_specs(module: type[SparkModule]) -> dict[str, OutputSpec]:
    """
        Returns a dictionary of the SparkModule's input port specifications.
    """

    # Check isinstance of SparkModule.
    if not validation._is_module_type(module):
        raise TypeError(
            f'Expected object class of type "SparkModule" but got "{type(module).__name__}".'
        )

    # Get output signature.
    signature = inspect.signature(module.__call__)
    signature_type_hints = tp.get_type_hints(module.__call__)

    output_specs = {}

    # Check that the input signature has a return annotation.
    if signature.return_annotation is None:
        raise SyntaxError(
            f'Module "{type(module).__name__}" does not define a return type.'
        )

    # Check if the annotation is a TypedDict
    if not tp.is_typeddict(signature_type_hints['return']):
        raise TypeError(
            f'Module "{type(module).__name__}" does not have a type return annotation of type tp.TypedDict '
            f'and cannot auto-infer output ports specs. Consider adding a TypedDict annotation to the __call__ method.'
        )
    
    # get_type_hints is required once more since __future__ delays the evaluation of typedict
    annotations = tp.get_type_hints(signature_type_hints['return'])

    # Build output specs
    for name, payload_type in annotations.items():

        # Check if the payload_type is a valid class and a subclass of SparkPayload
        if not validation._is_payload_type(payload_type):
            # Raise error, payload is not fully compatible with the framework.
            raise TypeError(
                f'Error: Output parameter "{name}" has type {type(payload_type).__name__}, '
                f'which is not a valid SparkPayload, None or sequence of SparkPayload.'
            )

        output_specs[name] = OutputSpec(
            payload_type=payload_type,
            shape=None,
            dtype=None,
            description=f'Output port for {name}',
    )
    return output_specs

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################