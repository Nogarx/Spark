spark.core.backend
==================

.. py:module:: spark.core.backend


Attributes
----------

.. autoapisummary::

   spark.core.backend.A


Classes
-------

.. autoapisummary::

   spark.core.backend.Module
   spark.core.backend.ModuleMeta
   spark.core.backend.Variable
   spark.core.backend.Constant


Functions
---------

.. autoapisummary::

   spark.core.backend.data
   spark.core.backend.grad
   spark.core.backend.jit
   spark.core.backend.eval_shape
   spark.core.backend.split
   spark.core.backend.merge


Module Contents
---------------

.. py:data:: A

.. py:function:: data(value, /)

.. py:function:: grad(*args, **kwargs)

   Wrapper around flax.nnx.grad, to simplify imports.


.. py:function:: jit(*args, **kwargs)

   Wrapper around flax.nnx.jit, to simplify imports.


.. py:function:: eval_shape(*args, **kwargs)

   Wrapper around flax.nnx.eval_shape, to simplify imports.


.. py:function:: split(*args, **kwargs)

   Wrapper around flax.nnx.split, to simplify imports.


.. py:function:: merge(*args, **kwargs)

   Wrapper around flax.nnx.merge, to simplify imports.


.. py:class:: Module

   Bases: :py:obj:`flax.nnx.Module`


   Base class of the module hierarchy.

   Alias of the Flax module, to simplify imports and to give the framework one place to
   change if the backend does.


.. py:class:: ModuleMeta

   Bases: :py:obj:`flax.nnx.module.ModuleMeta`


   Metaclass of `Module`.

   Alias of the Flax module metaclass, to simplify imports.


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



