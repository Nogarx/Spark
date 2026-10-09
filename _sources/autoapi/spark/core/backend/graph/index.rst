spark.core.backend.graph
========================

.. py:module:: spark.core.backend.graph


Attributes
----------

.. autoapisummary::

   spark.core.backend.graph.A


Classes
-------

.. autoapisummary::

   spark.core.backend.graph.Module
   spark.core.backend.graph.ModuleMeta


Functions
---------

.. autoapisummary::

   spark.core.backend.graph.data
   spark.core.backend.graph.split
   spark.core.backend.graph.merge


Module Contents
---------------

.. py:data:: A

.. py:function:: data(value, /)

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


