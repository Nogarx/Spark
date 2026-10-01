#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import jax
import flax.nnx as nnx
import typing as tp
import numpy as np
import jax.numpy as jnp
from collections.abc import Iterable

# TODO: Currently we constraint Constant/Variable to cast everything to arrays. 
# Initially, the plan was to simplify  the use of the class by removing the .value element, 
# however it may be useful to allow for the full flexibility of the original Variable. 

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def _as_array(value: tp.Any, dtype: tp.Any = None) -> jax.Array:
    """
        Casts a value to a jax array.

        Parameters
        ----------
        value : Any
            Value to cast.
        dtype : DTypeLike, optional
            Dtype of the result. The dtype jax infers is kept when omitted.

        Returns
        -------
        jax.Array
            The value as an array.
    """
    if isinstance(value, (Variable, Constant)):
        value = value.value
    if isinstance(value, jax.Array):
        array = value.astype(dtype) if dtype else value
    elif isinstance(value, np.ndarray):
        array = jnp.asarray(value, dtype=dtype if dtype else value.dtype)
    elif isinstance(value, (bool, int, float, complex)):
        # NOTE: Asked for no dtype, a python scalar is left weakly typed, which is what keeps it from
        # deciding the dtype of everything it touches. Naming its python type instead ("dtype=float") makes
        # it a strong float32, and one such scalar promotes every array it multiplies: a float16 kernel of
        # N by N becomes float32 for the rest of the expression, and is converted back afterwards.
        array = jnp.asarray(value, dtype=dtype) if dtype else jnp.asarray(value)
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
        Representation of a variable array/object.

        Wrapper around the Flax variable, to simplify imports. The dtype given at construction is
        applied once, to the initial value; a later assignment to ``value`` is converted to an
        array but keeps its own dtype.
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
        Representation of a constant array/object.

        Holds a quantity fixed at build time, such as a decay constant or a delay kernel.
        Assigning to ``value`` raises `AttributeError`; a quantity that changes belongs in a
        `Variable`.

        Registered as a static pytree node, so it travels in the treedef rather than as a traced
        leaf.
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

