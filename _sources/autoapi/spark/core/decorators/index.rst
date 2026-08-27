spark.core.decorators
=====================

.. py:module:: spark.core.decorators


Classes
-------

.. autoapisummary::

   spark.core.decorators.spark_property


Functions
---------

.. autoapisummary::

   spark.core.decorators.limit_recursion


Module Contents
---------------

.. py:class:: spark_property(fget=None, fset=None, fdel=None, doc=None)

   Declares a property port on a module.

   Behaves like the built-in property, and additionally marks the attribute as a port the
   framework can wire. The getter must be annotated with the `SparkPayload` it returns, which
   is what the port carries.

   A property with no setter is read only: other modules may read it, but it cannot be the
   target of an effect.

   .. rubric:: Examples

   >>> class Synapses(Component):
   ...     @spark_property
   ...     def kernel(self) -> FloatArray:
   ...         return FloatArray(self._kernel.value)
   ...
   ...     @kernel.setter
   ...     def kernel(self, new_kernel: FloatArray) -> None:
   ...         self._kernel.value = new_kernel.value


   .. py:attribute:: fget
      :value: None



   .. py:attribute:: fset
      :value: None



   .. py:attribute:: fdel
      :value: None



   .. py:attribute:: __doc__
      :value: None



   .. py:method:: __set_name__(owner, name)


   .. py:method:: __get__(obj, objtype=None)


   .. py:method:: __set__(obj, value)


   .. py:method:: __delete__(obj)


   .. py:method:: getter(fget)


   .. py:method:: setter(fset)


   .. py:method:: deleter(fdel)


.. py:function:: limit_recursion(limit)

   Decorator bounding how deep a function may re-enter itself.

   Used by the configuration hooks that hand values down to nested configurations, where a
   nested configuration would otherwise call back into the one above it.

   :param limit: Depth at which a call returns its first argument instead of running.
   :type limit: int

   :returns: The decorator.
   :rtype: callable


