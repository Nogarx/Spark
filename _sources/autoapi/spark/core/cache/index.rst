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


   Storage system for a `Brain` controller.

    A two key mapping from (module name, port name) to the payload that port produced. It is
    registered as a pytree, so it is carried through a jit boundary as state.

    A `Brain` reads its inputs from the cache and writes its outputs back once every module
    has run. This system allows module decoupling for one step, which improves model computation
    speed significantly by allowing jit to schedule more than one module at the same time.

    See Also
    --------
    TwoKeyDict : The mapping this builds on.


   .. py:method:: __setitem__(keys: str, value: dict[str, spark.core.payloads.SparkPayload]) -> None
                  __setitem__(keys: tuple[str, str], value: spark.core.payloads.SparkPayload) -> None


   .. py:method:: __getitem__(keys: tuple[str, str]) -> spark.core.payloads.SparkPayload
                  __getitem__(keys: str) -> dict[str, spark.core.payloads.SparkPayload]


   .. py:method:: from_specs(data)
      :classmethod:


      Builds a cache of mock payloads from port specifications.

      :param data: Specifications by (module name, port name).
      :type data: TwoKeyDict of (str, str) to PortSpecs

      :returns: One mock payload per specification. Specifications carrying no shape are optional
                ports and are skipped.
      :rtype: Cache



   .. py:method:: from_payloads(data)
      :classmethod:


      Builds a cache of zero-filled payloads from existing payloads.

      :param data: Payloads by (module name, port name), read for their type and shape.
      :type data: TwoKeyDict of (str, str) to SparkPayload

      :returns: One zeroed payload per entry, of the same type and shape. Payloads carrying no shape
                are optional ports and are skipped.
      :rtype: Cache



