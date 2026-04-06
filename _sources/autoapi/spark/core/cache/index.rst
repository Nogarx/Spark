spark.core.cache
================

.. py:module:: spark.core.cache


Classes
-------

.. autoapisummary::

   spark.core.cache.Cache


Module Contents
---------------

.. py:class:: Cache(data = None)

   Bases: :py:obj:`spark.core.utils.TwoKeyDict`


   A MutableMapping is a generic container for associating
   key/value pairs.

   This class provides concrete generic implementations of all
   methods except for __getitem__, __setitem__, __delitem__,
   __iter__, and __len__.


   .. py:method:: __setitem__(keys: str, value: dict[str, spark.core.payloads.SparkPayload]) -> None
                  __setitem__(keys: tuple[str, str], value: spark.core.payloads.SparkPayload) -> None


   .. py:method:: __getitem__(keys: tuple[str, str]) -> spark.core.payloads.SparkPayload
                  __getitem__(keys: str) -> dict[str, spark.core.payloads.SparkPayload]


   .. py:method:: from_specs(data)
      :classmethod:



