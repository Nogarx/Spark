spark.core.shape
================

.. py:module:: spark.core.shape


Attributes
----------

.. autoapisummary::

   spark.core.shape.bShape


Classes
-------

.. autoapisummary::

   spark.core.shape.Shape
   spark.core.shape.ShapeCollection


Module Contents
---------------

.. py:class:: Shape

   Bases: :py:obj:`tuple`


   Custom tuple subclass that represents the shape of an array or tensor.

   This class has a similar behavior to tuples. Useful for type checking and extra validation.

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:method:: __repr__()

      Return repr(self).



   .. py:method:: to_dict()


   .. py:method:: tree_flatten()


   .. py:method:: tree_unflatten(aux_data, children)
      :classmethod:



.. py:class:: ShapeCollection

   Bases: :py:obj:`tuple`


   Custom list subclass that represents a collection of shapes.

   This class has a similar behavior to tuples. Useful for type checking and extra validation.

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:method:: __repr__()

      Return repr(self).



   .. py:method:: to_dict()


   .. py:method:: tree_flatten()


   .. py:method:: tree_unflatten(aux_data, children)
      :classmethod:



.. py:data:: bShape

