spark.core.backend.variables
============================

.. py:module:: spark.core.backend.variables


Classes
-------

.. autoapisummary::

   spark.core.backend.variables.Variable
   spark.core.backend.variables.Constant


Module Contents
---------------

.. py:class:: Variable(value, dtype = None, **metadata)

   Bases: :py:obj:`flax.nnx.Variable`


   Representation of a variable array/object.

   Wrapper around the Flax variable, to simplify imports. The dtype given at construction is
   applied once, to the initial value; a later assignment to ``value`` is converted to an
   array but keeps its own dtype.


   .. py:property:: value
      :type: jax.Array



   .. py:method:: __jax_array__()


   .. py:method:: __array__(dtype=None)


   .. py:property:: shape
      :type: tuple[int, ...]



.. py:class:: Constant(data, dtype = None)

   Representation of a constant array/object.

   Holds a quantity fixed at build time, such as a decay constant or a delay kernel.
   Assigning to ``value`` raises `AttributeError`; a quantity that changes belongs in a
   `Variable`.

   Registered as a static pytree node, so it travels in the treedef rather than as a traced
   leaf.


   .. py:property:: value
      :type: jax.Array



   .. py:method:: __jax_array__()


   .. py:method:: __array__(dtype=None)


   .. py:property:: shape
      :type: tuple[int, ...]



   .. py:property:: dtype
      :type: Any



   .. py:property:: ndim
      :type: int



   .. py:property:: size
      :type: int



   .. py:property:: T
      :type: jax.Array



