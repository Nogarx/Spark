#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import jax
import flax.nnx as nnx
import typing as tp
import numpy as np
import jax.numpy as jnp
import typing_extensions as tpe
from jax._src.pjit import JitWrapped
from flax.nnx.graph import GraphDef, GraphState
from flax.nnx.variablelib import VariableState
from collections.abc import Iterable
A = tp.TypeVar('A')

# NOTE: Currently this code is just a shortcut of all the basic Flax's LAX methods. 
# Its only purpose is to reduce imports for the final user.

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def data(
        value: A, /
    ) -> A:
    return nnx.data(value)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def grad(*args, **kwargs) -> (tp.Callable[..., tp.Any] | tp.Callable[[tp.Callable[..., tp.Any]], tp.Callable[..., tp.Any]] ):
    """
        Wrapper around flax.nnx.grad to simply imports.
    """
    return nnx.grad(*args, **kwargs)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def jit(*args, **kwargs) -> JitWrapped | tp.Callable[[tp.Callable[..., tp.Any]], JitWrapped]:
    """
        Wrapper around flax.nnx.jit to simply imports.
    """
    return nnx.jit(*args, **kwargs)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def eval_shape(*args, **kwargs) -> A:
    """
        Wrapper around flax.nnx.eval_shape to simply imports.
    """
    return nnx.eval_shape(*args, **kwargs)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def split(*args, **kwargs) -> tuple[GraphDef[A], GraphState | VariableState, tpe.Unpack[tuple[GraphState | VariableState, ...]],]:
    """
        Wrapper around flax.nnx.split to simply imports.
    """
    return nnx.split(*args, **kwargs)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def merge(*args, **kwargs) -> A:
    """
        Wrapper around flax.nnx.merge to simply imports.
    """
    return nnx.merge(*args, **kwargs)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class Module(nnx.Module):
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ModuleMeta(nnx.module.ModuleMeta):
    pass

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# TODO: Currently we constraint Constant/Variable to cast everything to arrays. 
# Initially, the plan was to simplify  the use of the class by removing the .value element, 
# however it may be useful to allow for the full flexibility of the original Variable. 

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _as_array(value: tp.Any, dtype: tp.Any = None) -> jax.Array:
    """
        Casts value to a jax array.

        Args:
            value: tp.Any, value to cast
            dtype: tp.Any, dtype to cast to

        Returns:
            jax.Array, the value as jax.Array
    """
    if isinstance(value, (Variable, Constant)):
        value = value.value
    if isinstance(value, jax.Array):
        array = value.astype(dtype) if dtype else value
    elif isinstance(value, np.ndarray):
        array = jnp.asarray(value, dtype=dtype if dtype else value.dtype)
    elif isinstance(value, (bool, int, float, complex)):
        array = jnp.asarray(value, dtype=dtype if dtype else type(value))
    elif isinstance(value, str):
        # NOTE: A string is iterable, and letting it reach the branch below fails inside jax with a message
        # about converting characters to floats.
        raise TypeError(f'Expected data of type Array, Iterable or Scalar, got "{type(value)}".')
    elif isinstance(value, Iterable):
        array = jnp.asarray(value, dtype=dtype) if dtype else jnp.asarray(value)
    else:
        raise TypeError(f'Expected data of type Array, Iterable or Scalar, got "{type(value)}".')
    # NOTE: A scalar is kept as a one element vector, so that everything downstream can assume a shape.
    return array.reshape(-1) if array.ndim == 0 else array

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Variable(nnx.Variable):
    """
        The base class for all ``Variable`` types.
        Note that this is just a convinience wrapper around Flax's Variable to simplify imports.
    """
    # Type hint
    value: jax.Array

    def __init__(self, value: tp.Any, dtype: tp.Any = None, **metadata) -> None:
        super().__init__(_as_array(value, dtype), **metadata)

    @property
    def value(self) -> jax.Array:
        return self.get_value()

    @value.setter
    def value(self, value: tp.Any) -> None:
        self.set_value(_as_array(value))

    def __jax_array__(self) -> jax.Array: 
        return self.value
    
    def __array__(self, dtype=None) -> np.ndarray: 
        return np.array(self.value).astype(dtype if dtype else self.value.dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.tree_util.register_static
class Constant:
    """
        Jax.Array wrapper for constant arrays.
    """

    def __init__(self, data: tp.Any, dtype: tp.Any = None) -> None:
        self._value = _as_array(data, dtype)

    @property
    def value(self) -> jax.Array:
        return self._value

    @value.setter
    def value(self, value: tp.Any) -> None:
        raise AttributeError(
            f'A "{type(self).__name__}" cannot be updated. Wrap the value in a "Variable" if it changes.'
        )

    def __jax_array__(self) -> jax.Array: 
        return self.value
    
    def __array__(self, dtype=None) -> np.ndarray: 
        return np.array(self.value).astype(dtype if dtype else self.value.dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    @property
    def dtype(self) -> tp.Any:
        return self.value.dtype
        
    @property
    def ndim(self) -> int:
        return self.value.ndim
        
    @property
    def size(self) -> int:
        return self.value.size

    @property
    def T(self) -> jax.Array:
        return self.value.T


#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
