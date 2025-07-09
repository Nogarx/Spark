
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

from typing import Union, List, Tuple

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# Canonical shape type.
Shape = Tuple[int, ...]

# Broadcastable shape type.
bShape = Union[int, Shape, List[Union[int, Shape]]]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def normalize_shape(x : bShape):
    """
        Validates and normalizes a variable x, which can be an integer, a tuple
        of ints, or a list containing a mixture of integers and tuples of ints.

        Normalization rules:
            - An integer is converted to a tuple (e.g., 5 -> (5,)).
            - A tuple of integers remains unchanged.
            - A list of shapes is converted to a list of tuples.

        Args:
            x: The variable to validate and normalize.

        Returns:
            The normalized shape as a tuple or a list of tuples.
            Returns None if the input is invalid.
    """
    # Case 1: x is an integer
    if isinstance(x, int):
        return (x,)

    # Case 2: x is a tuple
    if isinstance(x, tuple):
        # Check is not empty
        if len(x) == 0:
            raise ValueError(f'Invalid tuple: empty tuple is not a valid bShape.')
        # Check if all elements are integers
        if all(isinstance(i, int) for i in x):
            return x
        else:
            # Find the first invalid element to include in the error message
            invalid_element = next(item for item in x if not isinstance(item, int))
            raise ValueError(f'Invalid tuple: all elements must be integers, but found '
                             f'element "{invalid_element}" of type {type(invalid_element).__name__}.')

    # Case 3: x is a list
    if isinstance(x, list):
        normalized_list = []
        # Check is not empty
        if len(x) == 0:
            raise ValueError(f'Invalid list: empty list is not a valid bShape.')
        for item in x:
            if isinstance(item, int):
                normalized_list.append((item,))
            elif isinstance(item, tuple) and all(isinstance(i, int) for i in item):
                normalized_list.append(item)
            else:
                # Invalid item found in the list
                raise ValueError(f'Invalid item in list: "{item}". All items in the list must be an int or a tuple of ints.')
        return normalized_list

    # If x is not an int, tuple, or list, it's an invalid type
    raise TypeError(f'Input must be an int, tuple, or list, but received type "{type(x).__name__}."')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################