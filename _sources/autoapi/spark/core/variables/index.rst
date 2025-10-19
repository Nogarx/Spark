spark.core.variables
====================

.. py:module:: spark.core.variables


Classes
-------

.. autoapisummary::

   spark.core.variables.Constant
   spark.core.variables.Variable


Module Contents
---------------

.. py:class:: Constant(data, dtype = None)

   Jax.Array wrapper for constant arrays.


   .. py:attribute:: value
      :type:  jax.Array


   .. py:method:: __jax_array__()


   .. py:method:: __array__(dtype=None)


   .. py:property:: shape
      :type: tuple[int, Ellipsis]



   .. py:property:: dtype
      :type: Any



   .. py:property:: ndim
      :type: int



   .. py:property:: size
      :type: int



   .. py:property:: T
      :type: jax.Array



   .. py:method:: __neg__()


   .. py:method:: __pos__()


   .. py:method:: __abs__()


   .. py:method:: __invert__()


   .. py:method:: __add__(other)


   .. py:method:: __sub__(other)


   .. py:method:: __mul__(other)


   .. py:method:: __truediv__(other)


   .. py:method:: __floordiv__(other)


   .. py:method:: __mod__(other)


   .. py:method:: __matmul__(other)


   .. py:method:: __pow__(other)


   .. py:method:: __radd__(other)


   .. py:method:: __rsub__(other)


   .. py:method:: __rmul__(other)


   .. py:method:: __rtruediv__(other)


   .. py:method:: __rfloordiv__(other)


   .. py:method:: __rmod__(other)


   .. py:method:: __rmatmul__(other)


   .. py:method:: __rpow__(other)


.. py:class:: Variable(value, dtype = None, **metadata)

   Bases: :py:obj:`flax.nnx.Variable`


   The base class for all ``Variable`` types.
   Note that this is just a convinience wrapper around Flax's nnx.Variable to simplify imports.


   .. py:attribute:: value
      :type:  jax.Array


   .. py:method:: __jax_array__()


   .. py:method:: __array__(dtype=None)


   .. py:property:: shape
      :type: tuple[int, Ellipsis]



