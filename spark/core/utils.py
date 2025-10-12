#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import re
import string
import numpy as np
import typing as tp
import collections.abc

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def normalize_str(s: str) -> str:
    """
        Converts any string into a consistent lowercase_snake_case format.

        Args:
            s: str, string to normalize

        Returns:
            str, normalized string
    """
    if not isinstance(s, str) or not s:
        raise TypeError(
            f's must be a non-empty string, got \"{s}\".'
        )
    # Insert underscores between acronyms and other words.
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s)
    # Insert underscores between lowercase letters and uppercase letters.
    s = re.sub(r'([a-z])([A-Z])', r'\1_\2', s)
    # Replace any spaces or hyphens with a single underscore.
    s = re.sub(r'[-\s]+', '_', s)
    # Convert the whole string to lowercase.
    return s.lower()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def get_einsum_labels(num_dims: int, offset: int = 0) -> str:
    """
        Generates labels for a generalized dot product using Einstein notation.

        Args:
            num_dims: int, number of dimensions (labels) to generate
            offset: int, initial dimension (label) offset

        Returns:
            str, a string with num_dims different labels, skipping the first offset characters 
    """
    if (offset + num_dims) > len(string.ascii_letters):
        raise ValueError(
            f'Requested up to {offset + num_dims} symbols but it is only possible to represent up to {len(string.ascii_letters)} '
            f'different symbols. If this was intentional consider defining a custom label map.'
        )
    return string.ascii_letters[offset:offset+num_dims]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def get_axes_einsum_labels(axes: tuple[int, ...], ignore_repeated:bool = False) -> str:
	"""
		Generates labels for a generalized dot product using Einstein notation.

		Args:
			axes: tuple[int, ...], requested dimensions (labels) to generate

		Returns:
			str, a string with num_dims different labels, skipping the first offset characters 
	"""
	
	if any([ax < 0 for ax in axes]):
		raise ValueError(
			f'\"axes\" out of bounds, expected all axis to be positive. '
		)
	
	if any([ax >= len(string.ascii_letters) for ax in axes]):
		raise ValueError(
			f'\"axes\" out of bounds, it is only possible to represent up to {len(string.ascii_letters)-1} symbols. '
			f'If this was intentional consider defining a custom label map.'
		)
	if (not ignore_repeated) and len(set(ax for ax in axes)) != len(axes):
		raise ValueError(
			f'Requested two labels for the same axis. If this was intended use the flag \"ignore_repeated=True\".'
		)
	return ''.join([string.ascii_letters[ax] for ax in axes])

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def validate_shape(obj: tp.Any) -> tuple[int, ...]:
    """
        Verifies that the object is broadcastable to a valid shape (tuple of integers).
        Returns the shape.

        Args:
            obj: tp.Any: the instance to validate

        Returns:
            list[tuple[int, ...]], the shape
    """
    # Sanity checks
    if isinstance(obj, int):
        return tuple([obj])
    elif isinstance(obj, collections.abc.Iterable) and len(obj) > 0:
        for element in obj:
            if not isinstance(element, int):
                raise TypeError(
                    f'Shape elements must be integers, but found type \"{type(element).__name__}\".'
                )
            if element < 0:
                raise ValueError(
                    f'Shape elements cannot be negative, but found value \"{element}\".'
                )
    else:
        raise TypeError(f'Expected obj to be either an iterable or an int, got \"{obj}\".')
    # Cast to tuple
    return tuple(obj)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def validate_list_shape(obj: tp.Any) -> list[tuple[int, ...]]:
    """
        Verifies that the object is broadcastable to a valid list ofshape (a list of tuple of integers).
        Returns the list of shapes.

        Args:
            obj: tp.Any: the instance to validate

        Returns:
            list[tuple[int, ...]], the list of shapes
    """
    # Sanity checks
    if not isinstance(obj, collections.abc.Iterable) or len(obj) == 0:
        raise TypeError(
            f'Expected obj to be an Iterable of Iterables (e.g. list of lists), got \"{obj}\"'
        )
    if is_shape(obj):
        raise TypeError(
            f'Ambiguous input: obj can also be broadcasted to shape, got \"{obj}\". '
            f'To prevent bugs a list of shapes is only broadcastable from an Iterable of Iterables (e.g. list of lists).'
        )
    # Cast to list of shapes
    return [validate_shape(e) for e in obj]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def is_shape(obj: tp.Any) -> bool:
    """
        Checks if the obj is broadcastable to a shape.

        Args:
            obj: tp.Any: the instance to check.

        Returns:
            bool, True if the object is broadcastable to a shape, False otherwise.
    """
    try: 
        validate_shape(obj)
        return True
    except:
        return False

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def is_list_shape(obj: tp.Any) -> bool:
    """
        Checks if the obj is broadcastable to a shape.

        Args:
            obj: tp.Any: the instance to check.

        Returns:
            bool, True if the object is broadcastable to a list of shapes, False otherwise.
    """
    try: 
        validate_list_shape(obj)
        return True
    except:
        return False


#-----------------------------------------------------------------------------------------------------------------------------------------------#

def is_dict_of(obj: tp.Any, value_cls: type[tp.Any], key_cls: type[tp.Any] = str) -> bool:
    """
        Check if an object instance is of 'dict[key_cls, value_cls]'.

        Args:
            obj: tp.Any: the instance to check.
            key_cls: type[tp.Any], the class to compare keys against.
            value_cls: type[tp.Any], the class to compare values against.

        Returns:
            bool, True if the object is an instance of 'dict[key_cls, value_cls]', False otherwise.
    """
    if not isinstance(key_cls, type):
        raise TypeError(
            f'Expected \"key_cls\" to be of a type but got \"{key_cls}\".'
        )
    if not isinstance(value_cls, type):
        raise TypeError(
            f'Expected \"value_cls\" to be of a type but got \"{key_cls}\".'
        )
    if isinstance(obj, dict):
        if all(isinstance(k, key_cls) and isinstance(v, value_cls) for k, v in obj.items()):    
            return True
    return False

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def is_list_of(obj: tp.Any, cls: type[tp.Any]) -> bool:
    """
        Check if an object instance is of 'list[cls]'.

        Args:
            obj: tp.Any, the instance to check.
            cls: type[tp.Any], the class to compare values against.

        Returns:
            bool, True if the object is an instance of 'list[cls]', False otherwise.
    """
    if not isinstance(cls, type):
        raise TypeError(
            f'Expected \"cls\" to be of a type but got \"{cls}\".'
        )
    if isinstance(obj, list):
        if all(isinstance(x, cls) for x in obj):    
            return True
    return False

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def is_dtype(obj: tp.Any) -> bool:
	"""
		Check if an object is a 'DTypeLike'.

		Args:
			obj (tp.Any): The instance to check.
		Returns:
			bool, True if the object is a 'DTypeLike', False otherwise.
	"""
	try:
		if np.isdtype(obj, ('numeric', 'bool')):
			return True
	except: 
		pass
	return False

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def is_float(obj: tp.Any) -> bool:
	"""
		Check if an object is a 'DTypeLike'.

		Args:
			obj (tp.Any): The instance to check.
		Returns:
			bool, True if the object is a 'DTypeLike', False otherwise.
	"""
	try:
		if np.isdtype(obj, ('real floating',)):
			return True
	except: 
		pass
	return False

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
