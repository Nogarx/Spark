
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

from typing import Union, List, Tuple

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# Canonical shape type.
Shape = tuple[int, ...]

# Broadcastable shape type.
bShape = int | Shape | list[int]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def normalize_shape(x : bShape) -> Shape:
    """
        Validates and normalizes a variable x, which can be an integer,
        a tuple of ints, or a list of ints.

        Normalization rules:
            - An integer is converted to a tuple (e.g., 5 -> (5,)).
            - A tuple of integers remains unchanged.
            - A list of ints is converted to a tuple of ints.

        Input:
            x: bShape.

        Output:
            The normalized shape as a tuple or a list of tuples.
    """
    # x is an integer
    if isinstance(x, int):
        return (x,)

    # x is a tuple
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

    # x is a list
    if isinstance(x, list):
        # Check is not empty
        if len(x) == 0:
            raise ValueError(f'Invalid list: empty list is not a valid bShape.')
        # Check if all elements are integers
        if all(isinstance(i, int) for i in x):
            return tuple(x)
        else:
            # Find the first invalid element to include in the error message
            invalid_element = next(item for item in x if not isinstance(item, int))
            raise ValueError(f'Invalid tuple: all elements must be integers, but found '
                             f'element "{invalid_element}" of type {type(invalid_element).__name__}.')

    # If x is not an int, tuple, or list, it's an invalid type
    raise TypeError(f'Input must be an int, tuple, or list, but received type "{type(x).__name__}."')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def normalize_list_shape(x : list[bShape]) -> list[Shape]:
    """
        Validates and normalizes a variable x, which can be a list of bShapes, 
        every item inside of the list is broadcasted to a Shape.

        Normalization rules:
            - A list of bShapes is converted to a list of Shapes.

        Input:
            x: list[bShape].

        Output:
            A list of normalized shapes.
    """
    if isinstance(x, list):
        # Check is not empty
        if len(x) == 0:
            raise ValueError(f'Invalid list: empty list is not a valid bShape.')
        return [normalize_shape(v) for v in x]

    # If x is not an int, tuple, or list, it's an invalid type
    raise TypeError(f'Input must be a list of bShape, but received type "{type(x).__name__}."')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################