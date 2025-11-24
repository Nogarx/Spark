
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
from collections.abc import Iterable
import spark.core.validation as validation
import itertools
import types

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def _expand(t) -> list[type]:
    """
        Recursively expand a type-hint t into a list of fully concrete non-union types.
    """
    # Unpack typevars
    if type(t) == tp.TypeVar:
        bound = t.__bound__
        if bound is not None:
            t = bound
    # Case 1: union
    origin = tp.get_origin(t)
    if origin is tp.Union or origin is types.UnionType:
        out = []
        for arg in tp.get_args(t):
            out.extend(_expand(arg))
        return out
    # Case 2: parametrized generic
    origin = tp.get_origin(t)
    args = tp.get_args(t)
    if origin is not None:
        # Recursively expand each generic argument
        expanded_args = [ _expand(a) for a in args ]
        # Cartesian product gives all combinations
        combos = itertools.product(*expanded_args)
        results = []
        for combo in combos:
            # combo is a tuple of fully concrete argument types
            if len(combo) == 0:
                results.append(origin)
            elif len(combo) == 1:
                results.append(origin[combo[0]])
            else:
                results.append(origin[combo])
        return results
    # Case 3: plain type
    return [t]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def normalize_typehint(t) -> tuple[type]:
    """
        Produce a tuple of fully expanded, non-union, non-merged type variants.
    """
    expanded = _expand(t)
    # Remove duplicates while preserving order
    seen = set()
    final = []
    for x in expanded:
        if x not in seen:
            seen.add(x)
            final.append(x)
    return tuple(final)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _is_instance(value, annotated_type):
    """
    Validate a value against parameterized generics at runtime.
    Supports list[T], tuple[T], dict[K,V], set[T], and plain classes.
    """
    origin = tp.get_origin(annotated_type)
    args = tp.get_args(annotated_type)
    
    # Case 1: non-parameterized class or builtin
    if origin is None:
        return isinstance(value, annotated_type)

    # Case 2: list[T]
    elif origin is list:
        (elem_type,) = args
        return (
            isinstance(value, list) and
            all(_is_instance(v, elem_type) for v in value)
        )

    # Case 3: tuple[T] or tuple[T1, T2]
    elif origin is tuple:
        # Homogeneous type tuple
        if len(args) == 2 and args[1] is Ellipsis:  
            return (
                isinstance(value, tuple) and
                all(_is_instance(v, args[0]) for v in value)
            )
        # Fixed length tuple
        return (
            isinstance(value, tuple) and
            len(value) == len(args) and
            all(_is_instance(v, t) for v, t in zip(value, args))
        )

    # Case 4: dict[K, V]
    elif origin is dict:
        key_type, val_type = args
        return (
            isinstance(value, dict) and
            all(_is_instance(k, key_type) for k in value.keys()) and
            all(_is_instance(v, val_type) for v in value.values())
        )

    # Case 5: set[T]
    elif origin is set:
        (elem_type,) = args
        return (
            isinstance(value, set) and
            all(_is_instance(v, elem_type) for v in value)
        )

    # Case 6: tp.Union
    elif origin is tp.Union or origin is types.UnionType:
        return any(_is_instance(value, arg) for arg in args)

    # Fallback: treat origin as base
    else:
        return isinstance(value, origin)
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

def is_instance(value, types):
    if isinstance(types, Iterable): 
        for t in types:
            try:
                if _is_instance(value, t):
                    return True
            except:
                pass
    else:
        try:
            if _is_instance(value, types):
                return True
        except:
            pass
    return False

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