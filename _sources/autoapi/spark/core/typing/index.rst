spark.core.typing
=================

.. py:module:: spark.core.typing


Functions
---------

.. autoapisummary::

   spark.core.typing.is_dtype_like
   spark.core.typing.is_array_like
   spark.core.typing.is_object_of_type
   spark.core.typing.enforce_annotations


Module Contents
---------------

.. py:function:: is_dtype_like(obj)

   Check if an object is a 'DTypeLike'.

   :param obj: The instance to check.
   :type obj: tp.Any

   :returns: bool, True if the object is a 'DTypeLike', False otherwise.


.. py:function:: is_array_like(obj)

   Check if an object is a 'ArrayLike'.

   :param obj: The instance to check.
   :type obj: tp.Any

   :returns: bool, True if the object is a 'ArrayLike', False otherwise.


.. py:function:: is_object_of_type(obj, _type)

.. py:function:: enforce_annotations(fn, strict = True, validate_keys = True)

