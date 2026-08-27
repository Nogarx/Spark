#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp
import inspect
import threading
from functools import wraps

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class spark_property:
    """
        Declares a property port on a module.

        Behaves like the built-in property, and additionally marks the attribute as a port the
        framework can wire. The getter must be annotated with the `SparkPayload` it returns, which
        is what the port carries.

        A property with no setter is read only: other modules may read it, but it cannot be the
        target of an effect.

        Examples
        --------
        >>> class Synapses(Component):
        ...     @spark_property
        ...     def kernel(self) -> FloatArray:
        ...         return FloatArray(self._kernel.value)
        ...
        ...     @kernel.setter
        ...     def kernel(self, new_kernel: FloatArray) -> None:
        ...         self._kernel.value = new_kernel.value
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
        Decorator bounding how deep a function may re-enter itself.

        Used by the configuration hooks that hand values down to nested configurations, where a
        nested configuration would otherwise call back into the one above it.

        Parameters
        ----------
        limit : int
            Depth at which a call returns its first argument instead of running.

        Returns
        -------
        callable
            The decorator.
    """
    def decorator(func):
        state = threading.local()

        @wraps(func)
        def wrapper(*args, **kwargs):
            depth = getattr(state, 'depth', 0)
            if depth >= limit:
                return args[0] if args else None
            state.depth = depth + 1
            try:
                # Standard recursion
                result = func(*args, **kwargs)
            finally:
                # Decrease stack counter
                state.depth = depth
            return result
        return wrapper
    return decorator

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################