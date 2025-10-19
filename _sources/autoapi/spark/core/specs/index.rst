spark.core.specs
================

.. py:module:: spark.core.specs


Classes
-------

.. autoapisummary::

   spark.core.specs.PortSpecs
   spark.core.specs.InputSpec
   spark.core.specs.OutputSpec
   spark.core.specs.PortMap
   spark.core.specs.ModuleSpecs


Module Contents
---------------

.. py:class:: PortSpecs(payload_type, shape, dtype, description = None)

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



   .. py:method:: to_dict()

      Serialize PortSpecs to dictionary



   .. py:method:: from_dict(dct)
      :classmethod:


      Deserialize dictionary to  PortSpecs



.. py:class:: InputSpec(payload_type, shape, dtype, description = None)

   Bases: :py:obj:`PortSpecs`


   Specification for an input port of an SparkModule.


   .. py:method:: to_dict()

      Serialize InputSpec to dictionary



   .. py:method:: from_dict(dct)
      :classmethod:


      Deserialize dictionary to  PortSpecs



.. py:class:: OutputSpec(**kwargs)

   Bases: :py:obj:`PortSpecs`


   Specification for an output port of an SparkModule.


   .. py:method:: to_dict()

      Serialize PortSpecs to dictionary



   .. py:method:: from_dict(dct)
      :classmethod:


      Deserialize dictionary to  PortSpecs



.. py:class:: PortMap(origin, port)

   Specification for an output port of an SparkModule.


   .. py:attribute:: origin
      :type:  str


   .. py:attribute:: port
      :type:  str


   .. py:method:: to_dict()

      Serialize PortMap to dictionary



   .. py:method:: from_dict(dct)
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


   .. py:method:: to_dict()

      Serialize ModuleSpecs to dictionary



   .. py:method:: from_dict(dct)
      :classmethod:


      Deserialize dictionary to ModuleSpecs



