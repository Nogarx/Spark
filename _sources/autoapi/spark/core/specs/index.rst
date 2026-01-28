spark.core.specs
================

.. py:module:: spark.core.specs


Classes
-------

.. autoapisummary::

   spark.core.specs.PortSpecs
   spark.core.specs.PortMap
   spark.core.specs.ModuleSpecs


Module Contents
---------------

.. py:class:: PortSpecs(payload_type, shape, dtype, description = None, async_spikes = None, inhibition_mask = None)

   Base specification for a port of an SparkModule.


   .. py:attribute:: payload_type
      :type:  type[spark.core.payloads.SparkPayload] | None


   .. py:attribute:: shape
      :type:  tuple[int, Ellipsis] | list[tuple[int, Ellipsis]] | None


   .. py:attribute:: dtype
      :type:  jax.typing.DTypeLike | None


   .. py:attribute:: description
      :type:  str | None
      :value: None



   .. py:attribute:: async_spikes
      :type:  bool | None
      :value: (None,)



   .. py:attribute:: inhibition_mask
      :type:  bool | None
      :value: (None,)



   .. py:method:: to_dict(is_partial = False)

      Serialize PortSpecs to dictionary



   .. py:method:: from_dict(dct, is_partial = False)
      :classmethod:


      Deserialize dictionary to  PortSpecs



   .. py:method:: from_portspecs_list(portspec_list, validate_async = True)
      :classmethod:


      Merges a list of PortSpecs into a single PortSpecs



.. py:class:: PortMap(origin, port)

   Specification for an output port of an SparkModule.


   .. py:attribute:: origin
      :type:  str


   .. py:attribute:: port
      :type:  str


   .. py:method:: to_dict(is_partial = False)

      Serialize PortMap to dictionary



   .. py:method:: from_dict(dct, is_partial = False)
      :classmethod:


      Deserialize dictionary to PortMap



.. py:class:: ModuleSpecs(name, module_cls, inputs, config)

   Specification for SparkModule automatic constructor.


   .. py:attribute:: name
      :type:  str


   .. py:attribute:: module_cls
      :type:  type[spark.core.module.SparkModule]


   .. py:attribute:: inputs
      :type:  dict[str, list[PortMap]]


   .. py:attribute:: config
      :type:  spark.core.config.BaseSparkConfig


   .. py:method:: to_dict(is_partial = False)

      Serialize ModuleSpecs to dictionary



   .. py:method:: from_dict(dct, is_partial = False)
      :classmethod:


      Deserialize dictionary to ModuleSpecs



