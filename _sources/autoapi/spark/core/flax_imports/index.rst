spark.core.flax_imports
=======================

.. py:module:: spark.core.flax_imports


Attributes
----------

.. autoapisummary::

   spark.core.flax_imports.A


Functions
---------

.. autoapisummary::

   spark.core.flax_imports.data
   spark.core.flax_imports.grad
   spark.core.flax_imports.jit
   spark.core.flax_imports.eval_shape
   spark.core.flax_imports.split
   spark.core.flax_imports.merge


Module Contents
---------------

.. py:data:: A

.. py:function:: data(value, /)

.. py:function:: grad(f = MISSING, *, argnums = 0, has_aux = False, holomorphic = False, allow_int = False, reduce_axes = ())

   Wrapper around flax.nnx.grad to simply imports.


.. py:function:: jit(fun = Missing, *, in_shardings = None, out_shardings = None, static_argnums = None, static_argnames = None, donate_argnums = None, donate_argnames = None, keep_unused = False, device = None, backend = None, inline = False, abstracted_axes = None)

   Wrapper around flax.nnx.jit to simply imports.


.. py:function:: eval_shape(f, *args, **kwargs)

   Wrapper around flax.nnx.eval_shape to simply imports.


.. py:function:: split(node, *filters)

   Wrapper around flax.nnx.split to simply imports.


.. py:function:: merge(graphdef, state, /, *states)

   Wrapper around flax.nnx.merge to simply imports.


