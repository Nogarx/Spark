#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.shape import Shape
    
import jax
import numpy as np
import jax.numpy as jnp
import flax.nnx as nnx
from typing import Any
from collections.abc import Iterable

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@jax.tree_util.register_static
class Constant:
    """
        Jax.Array wrapper for constant arrays.
    """

    def __init__(self, data: Any, dtype: Any = None):
        if isinstance(data, jax.Array):
            # Input is an array.
            self.value = data.astype(dtype=dtype if dtype else data.dtype)
        elif isinstance(data, (np.ndarray, Constant)):
            # Input is an array.
            self.value = jnp.array(data, dtype=dtype if dtype else data.dtype)
        elif isinstance(data, Iterable):
            # Input is an iterable
            self.value = jnp.array(data, dtype=dtype if dtype else jnp.float32)
        elif isinstance(data, (int, float, complex, bool)):
            # Input is an scalar
            self.value = jnp.array(data, dtype=dtype if dtype else type(data))
        elif isinstance(data, Variable):
            # Input is an scalar
            self.value = jnp.array(data.value, dtype=dtype if dtype else data.value.dtype)
        else:
            raise TypeError(f'Expected data of type Array, Iterable or Scalar, got "{type(data)}".')
        if len(self.value.shape) == 0:
            self.value = self.value.reshape(-1)

    def tree_flatten(self):
        children = (self.value, self.value.dtype)
        aux_data = ()
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (value, dtype) = children
        return cls(value=value, dtype=dtype)

    def __jax_array__(self) -> jax.Array: 
        return self.value
    
    def __array__(self, dtype=None) -> jax.Array: 
        return np.array(self.value).astype(dtype if dtype else self.value.dtype)

    @property
    def shape(self) -> Shape:
        return self.value.shape

    @property
    def dtype(self) -> Any:
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


    # Unary Operators
    def __neg__(self) -> jax.Array: 
        return -self.value
    
    def __pos__(self) -> jax.Array: 
        return +self.value

    def __abs__(self) -> jax.Array: 
        return jnp.abs(self.value)

    def __invert__(self) -> jax.Array: 
        return ~self.value

    # Binary Arithmetic Operators
    def __add__(self, other) -> jax.Array: 
        return jnp.add(self.value, other)

    def __sub__(self, other) -> jax.Array: 
        return jnp.subtract(self.value, other)

    def __mul__(self, other) -> jax.Array: 
        return jnp.multiply(self.value, other)

    def __truediv__(self, other) -> jax.Array: 
        return jnp.divide(self.value, other)

    def __floordiv__(self, other) -> jax.Array: 
        return jnp.floor_divide(self.value, other)

    def __mod__(self, other) -> jax.Array: 
        return jnp.mod(self.value, other)

    def __matmul__(self, other) -> jax.Array: 
        return jnp.matmul(self.value, other)

    def __pow__(self, other) -> jax.Array: 
        return jnp.pow(self.value, other)

    # Reflected Arithmetic Operators
    def __radd__(self, other) -> jax.Array: 
        return jnp.add(other, self.value)

    def __rsub__(self, other) -> jax.Array: 
        return jnp.subtract(other, self.value)

    def __rmul__(self, other) -> jax.Array: 
        return jnp.multiply(other, self.value)

    def __rtruediv__(self, other) -> jax.Array: 
        return jnp.divide(other, self.value)

    def __rfloordiv__(self, other) -> jax.Array: 
        return jnp.floor_divide(other, self.value)

    def __rmod__(self, other) -> jax.Array: 
        return jnp.mod(other, self.value)

    def __rmatmul__(self, other) -> jax.Array: 
        return jnp.matmul(other, self.value)

    def __rpow__(self, other) -> jax.Array: 
        return jnp.pow(other, self.value)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class Variable(nnx.Variable):
    """
        The base class for all ``Variable`` types.
        Note that this is just a convinience wrapper around Flax's nnx.Variable to simplify imports.
    """
    
    def __init__(self, value: Any, dtype: Any = None, **metadata):

        if isinstance(value, jax.Array):
            # Input is an array.
            value = value.astype(dtype=dtype if dtype else value.dtype)
        elif isinstance(value, np.ndarray):
            # Input is an array.
            value = jnp.array(value, dtype=dtype if dtype else value.dtype)
        elif isinstance(value, Iterable):
            # Input is an iterable
            value = jnp.array(value, dtype=dtype if dtype else jnp.float32)
        elif isinstance(value, (int, float, complex, bool)):
            # Input is an scalar
            value = jnp.array(value, dtype=dtype if dtype else type(value))
        #else:
        #   Variable is a custom object, pass it directly to nnx.Variable
        if isinstance(value, jax.Array) and len(value.shape) == 0:
            value = value.reshape(-1)
        super().__init__(value=value, metadata=metadata)

    def __jax_array__(self) -> jax.Array: 
        return self.value
    
    def __array__(self, dtype=None) -> jax.Array: 
        return np.array(self.value).astype(dtype if dtype else self.value.dtype)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################