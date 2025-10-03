
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import pytest
from typing import List, Any
from spark.core.shape import bShape, Shape, Shape

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# Tests for valid inputs that resolve to a SINGLE canonical shape
@pytest.mark.parametrize(
    'input_shape, expected_shape',
    [
        (5, (5,)),                          # Single integer
        ((2, 4), (2, 4)),                   # Tuple of integers
        ([5, 3, 1], [(5,), (3,), (1,)]),    # List of integers
        ([7], [(7,)]),                      # Single element list of int
    ],
    ids=[
        'single_int',
        'tuple_of_ints',
        'list_of_ints',
        'single_element_list'
    ]
)

def test_normalize_single_shapes(input_shape: bShape, expected_shape: Shape):
    """
        Tests various inputs that should normalize to a single tuple shape.
    """
    assert Shape(input_shape) == expected_shape

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# Tests for valid inputs that resolve to a list of Shape's
@pytest.mark.parametrize(
    'input_list, expected_list',
    [
        ([5, (2, 4)], [(5,), (2, 4)]),                              # Mixed list of int and tuple
        ([(1,), (2, 3), (4, 5, 6)], [(1,), (2, 3), (4, 5, 6)]),     # List of only tuples
        ([(5,)], [(5,)]),                                           # List with a single tuple
    ],
    ids=[
        'mixed_list',
        'list_of_tuples',
        'list_with_single_tuple'
    ]
)

def test_normalize_list_of_shapes(input_list: List, expected_list: List[Shape]):
    """
        Tests various inputs that should normalize to a list of tuple shapes.
    """
    assert Shape(input_list) == expected_list

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# Tests for TypeErrors on fundamentally invalid types
@pytest.mark.parametrize(
    'invalid_input',
    [              
        3.14,
        'hello',
        {'shape': (2, 2)},
        {1, 2, 3},
        None
    ],
    ids=[
        'float',
        'string',
        'dict',
        'set',
        'NoneType'
    ]
)

def test_raises_type_error_for_invalid_types(invalid_input: Any):
    """
        Asserts that a TypeError is raised for inputs of an invalid type.
    """
    with pytest.raises(TypeError, match='Input must be an int, tuple, or list'):
        Shape(invalid_input)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# Tests for ValueErrors on inputs with invalid contents 
@pytest.mark.parametrize(
    'invalid_input, expected_error_message',
    [
        ((), 'Invalid tuple: empty tuple is not a valid bShape.'), 
        ([], 'Invalid list: empty list is not a valid bShape.'),       
        ((1, 2.0), f'Invalid tuple: all elements must be integers, but found element "{2.0}" of type {float.__name__}.'),
        ([1, 'a'], f'Invalid item in list: "a". All items in the list must be an int or a tuple of ints.'),
        ([1, (2, 3.0)], f'Invalid item in list: "\(2, 3.0\)". All items in the list must be an int or a tuple of ints.'),
        ([(1, 2), None], f'Invalid item in list: "{None}". All items in the list must be an int or a tuple of ints.'),
    ],
    ids=[
        'empty_tuple',
        'empty_list',
        'tuple_with_float',
        'list_with_string',
        'list_with_invalid_tuple',
        'list_with_none'
    ]
)

def test_raises_value_error_for_invalid_contents(invalid_input: Any, expected_error_message: str):
    """
        Asserts that a ValueError is raised for inputs with invalid contents.
    """
    with pytest.raises(ValueError, match=expected_error_message):
        Shape(invalid_input)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################