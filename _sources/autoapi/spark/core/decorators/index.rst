spark.core.decorators
=====================

.. py:module:: spark.core.decorators


Classes
-------

.. autoapisummary::

   spark.core.decorators.spark_property


Module Contents
---------------

.. py:class:: spark_property(fget=None, fset=None, fdel=None, doc=None)

           Custom property descriptor to expose Spark module properties to the rest of the framework in a safe way.
   Properties must be properly wrapper in payloads to be valid.

   Behaves identically to the default property descriptor.


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


