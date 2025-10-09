spark.core.cache
================

.. py:module:: spark.core.cache


Classes
-------

.. autoapisummary::

   spark.core.cache.Cache


Module Contents
---------------

.. py:class:: Cache(variable, payload_type)

   Cache dataclass.


   .. py:attribute:: variable
      :type:  spark.core.variables.Variable


   .. py:attribute:: payload_type
      :type:  type[spark.core.payloads.ValueSparkPayload]


   .. py:property:: value
      :type: spark.core.payloads.ValueSparkPayload


      Current value store in the cache object.


   .. py:property:: shape
      :type: tuple[int, Ellipsis]


      Shape of the value store in the cache object.


   .. py:property:: dtype
      :type: jax.numpy.dtype


      Dtype of the value store in the cache object.


