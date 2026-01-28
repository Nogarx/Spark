spark.core.cache
================

.. py:module:: spark.core.cache


Classes
-------

.. autoapisummary::

   spark.core.cache.Cache


Module Contents
---------------

.. py:class:: Cache(_variable)

   Cache dataclass.


   .. py:property:: value
      :type: spark.core.payloads.SparkPayload


      Current value store in the cache object.


   .. py:method:: tree_flatten()


   .. py:method:: tree_unflatten(aux_data, children)
      :classmethod:



   .. py:property:: shape
      :type: tuple[int, Ellipsis]


      Shape of the value store in the cache object.


   .. py:property:: dtype
      :type: jax.numpy.dtype


      Dtype of the value store in the cache object.


   .. py:method:: get()


   .. py:method:: set(payload)


