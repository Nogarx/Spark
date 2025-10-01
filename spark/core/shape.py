
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import collections.abc

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class Shape(tuple):
    """
		Custom tuple subclass that represents the shape of an array or tensor.

		This class has a similar behavior to tuples. Useful for type checking and extra validation.
    """
    def __new__(cls, *args):
        elements = args[0] if len(args) == 1 and isinstance(args[0], collections.abc.Iterable) else args
        # Sanity checks
        for element in elements:
            if not isinstance(element, int):
                raise TypeError(
                    f'Shape elements must be integers, but found type \"{type(element).__name__}\".'
				)
            if element < 0:
                raise ValueError(
                    f'Shape elements cannot be negative, but found value \"{element}\".'
				)
        return super().__new__(cls, elements)

    def __repr__(self):
        return f'Shape{super().__repr__()}'
    
    def to_dict(self,):
        return {
            '__type__': 'shape',
            'data': self,
        }

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ShapeCollection(tuple):
    """
        Custom list subclass that represents a collection of shapes.

        This class has a similar behavior to tuples. Useful for type checking and extra validation.
    """
    def __new__(cls, *args):
        elements = args[0] if len(args) == 1 and isinstance(args[0], collections.abc.Iterable) else args
        return super().__new__(cls, [Shape(e) for e in elements])

    def __repr__(self):
        return f'ShapeCollection{super().__repr__()}'
    
    def to_dict(self,):
        return {
            '__type__': 'shape_collection',
            'data': self,
        }

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# Broadcastable shape type.
bShape = int | list[int] | Shape

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################