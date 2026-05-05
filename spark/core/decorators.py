#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp
import inspect

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class spark_property:
    """
		Custom property descriptor to expose Spark module properties to the rest of the framework in a safe way.
        Properties must be properly wrapper in payloads to be valid.
        
        Behaves identically to the default property descriptor.
    """
    
    def __init__(self, fget=None, fset=None, fdel=None, doc=None) -> None:
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        if doc is None and fget is not None:
            doc = fget.__doc__
        self.__doc__ = doc

    def __set_name__(self, owner, name) -> None:
        self.__name__ = name

    def __get__(self, obj, objtype=None) -> tp.Self | tp.Any:
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError(f'{self.__name__} does not define a get method.')
        return self.fget(obj)

    def __set__(self, obj, value) -> None:
        if self.fset is None:
            raise AttributeError(f'{self.__name__} does not define a set method.')
        self.fset(obj, value)

    def __delete__(self, obj) -> None:
        if self.fdel is None:
            raise AttributeError(f'{self.__name__} does not define a delete method.')
        self.fdel(obj)

    def getter(self, fget) -> tp.Self:
        return type(self)(fget, self.fset, self.fdel, self.__doc__)

    def setter(self, fset) -> tp.Self:
        return type(self)(self.fget, fset, self.fdel, self.__doc__)

    def deleter(self, fdel) -> tp.Self:
        return type(self)(self.fget, self.fset, fdel, self.__doc__)
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

def limit_recursion(limit) -> tp.Callable[..., tp.Callable[..., tp.Any]]:
    """
		Decorator to limit recursion depth, used in some post config validation.
	"""
    def decorator(func):
        func.current_depth = 0 
        def wrapper(*args, **kwargs):
            if wrapper.current_depth >= limit:
                return args[0] 
            wrapper.current_depth += 1
            try:
                # Standard recursion
                result = func(*args, **kwargs)
            finally:
                # Decrease stack counter
                wrapper.current_depth -= 1
            return result
        # Initialize the depth tracker on the wrapper
        wrapper.current_depth = 0
        return wrapper
    return decorator

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################