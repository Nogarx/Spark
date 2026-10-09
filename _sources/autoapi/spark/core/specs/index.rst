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

.. py:class:: PortSpecs(payload_type, shape, dtype, description = None)

   Module port specification.

   Names the payload type, and the shape and dtype once they are known. Shape and dtype are
   None until the module is built, since they are inferred from the values that reach it.

   :param payload_type: Type the port carries. Two ports connect only if this matches.
   :type payload_type: type of SparkPayload or None
   :param shape: Shape of the payload. A list when several values arrive on the port.
   :type shape: tuple of int or list of tuple of int or None
   :param dtype: Dtype of the payload.
   :type dtype: DTypeLike or None
   :param description: Human readable description of the port.
   :type description: str, optional


   .. py:attribute:: payload_type
      :type:  type[spark.core.payloads.SparkPayload] | None


   .. py:attribute:: shape
      :type:  tuple[int, ...] | list[tuple[int, ...]] | None


   .. py:attribute:: dtype
      :type:  jax.typing.DTypeLike | None


   .. py:attribute:: description
      :type:  str | None


   .. py:method:: to_dict()

      Serializes the specification to a dictionary.

      :returns: The fields of the specification, with the payload type as its registered name.
      :rtype: dict



   .. py:method:: from_dict(dct)
      :classmethod:


      Builds a specification from a dictionary.

      :param dct: As produced by `to_dict`.
      :type dct: dict

      :rtype: PortSpecs



   .. py:method:: from_payload(payload)
      :classmethod:


      Builds a specification describing an existing payload.

      :param payload: Payload to read the type, shape and dtype from.
      :type payload: SparkPayload

      :rtype: PortSpecs



   .. py:method:: from_portspecs_list(portspec_list)
      :classmethod:


      Merges several specifications into one.

      Used for a port fed by more than one source, whose values are concatenated.

      :param portspec_list: Specifications to merge. All must carry the same payload type.
      :type portspec_list: list of PortSpecs

      :returns: The shared payload type, the promoted dtype and the merged shape. The list itself is
                returned unchanged when it holds a single entry.
      :rtype: PortSpecs

      :raises TypeError: If the specifications do not all carry the same payload type.



.. py:class:: PortMap(origin, port, is_property = False)

   Module's connections specification.

   A pair of the form (module_name, module_port_name) that specifies a connection within a controller.
   ``'__call__'`` and ``'__self__'`` are speciail module_names used to refer to the controller's inputs
   and the same module defining the mapping.


   :param origin: Name of the module the value comes from. ``'__call__'`` for an input of the enclosing
                  controller, ``'__self__'`` for a property of the controller itself.
   :type origin: str
   :param port: Name of the port on that module.
   :type port: str
   :param is_property: Read a property of the origin rather than one of its outputs.
   :type is_property: bool, default False


   .. py:attribute:: origin
      :type:  str


   .. py:attribute:: port
      :type:  str


   .. py:attribute:: is_property
      :type:  bool


   .. py:method:: to_dict()

      Serializes the map to a dictionary.

      :returns: The origin, the port and the property flag.
      :rtype: dict



   .. py:method:: from_dict(dct)
      :classmethod:


      Builds a map from a dictionary.

      :param dct: As produced by `to_dict`.
      :type dct: dict

      :rtype: PortMap



   .. py:method:: __hash__()


   .. py:method:: __eq__(other)


.. py:class:: ModuleSpecs(name, module_cls, inputs, config = None, outputs = None, effects = None)

   Specification of a module within a controller.

   This is what makes a model data rather than code: a controller holds a tuple of these, so
   it can be written to a file, edited, and instantiated again without a Python definition.

   :param name: Name the module answers to inside the controller. Also the attribute it is bound to.
   :type name: str
   :param module_cls: Class to instantiate. Must be registered.
   :type module_cls: type of SparkModule
   :param inputs: For each input port of the module, where its value comes from. Several entries for one
                  port are concatenated in order.
   :type inputs: dict of str to PortMap or list of PortMap
   :param config: Configuration of the module. The default configuration is used when omitted.
   :type config: SparkConfig, optional
   :param outputs: Output ports of the module to expose as output ports of the controller, as
                   ``{controller port: module port}``.
   :type outputs: dict of str to str, optional
   :param effects: Properties of this module to write after the step, as ``{property: source}``. A
                   plasticity rule writes weights back onto a synapse this way. The property must have a
                   setter.
   :type effects: dict of str to PortMap or list of PortMap, optional


   .. py:attribute:: name
      :type:  str


   .. py:attribute:: module_cls
      :type:  type[spark.core.module.SparkModule]


   .. py:attribute:: inputs
      :type:  dict[str, Iterable[PortMap]]


   .. py:attribute:: outputs
      :type:  dict[str, str]


   .. py:attribute:: effects
      :type:  dict[str, Iterable[PortMap]]


   .. py:attribute:: config
      :type:  spark.core.config.SparkConfig


   .. py:method:: to_dict()

      Serializes the specification to a dictionary.

      :returns: The name, the registered name of the module class, the wiring and the configuration.
      :rtype: dict



   .. py:method:: from_dict(dct)
      :classmethod:


      Builds a specification from a dictionary.

      :param dct: As produced by `to_dict`. The module class is looked up in the registry by name.
      :type dct: dict

      :rtype: ModuleSpecs



