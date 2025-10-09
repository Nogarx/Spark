spark.graph_editor.specs
========================

.. py:module:: spark.graph_editor.specs


Classes
-------

.. autoapisummary::

   spark.graph_editor.specs.BaseSpecEditor
   spark.graph_editor.specs.PortSpecEditor
   spark.graph_editor.specs.InputSpecEditor
   spark.graph_editor.specs.OutputSpecEditor
   spark.graph_editor.specs.PortMap
   spark.graph_editor.specs.ModuleSpecsEditor


Module Contents
---------------

.. py:class:: BaseSpecEditor

   Bases: :py:obj:`abc.ABC`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: to_dict()

      Returns a JSON serializable version of the spec.



   .. py:method:: from_dict(**kwargs)
      :classmethod:


      Instantiates the spec from a valid JSON serializable of the spec.



.. py:class:: PortSpecEditor(payload_type, shape = None, dtype = None, description = None)

   Bases: :py:obj:`BaseSpecEditor`


   Mutable base specification for a port of an SparkModule.


   .. py:attribute:: payload_type
      :type:  type[spark.core.payloads.SparkPayload]


   .. py:attribute:: shape
      :type:  spark.core.shape.bShape | None


   .. py:attribute:: dtype
      :type:  Any | None


   .. py:attribute:: description
      :type:  str | None


.. py:class:: InputSpecEditor(payload_type, shape, dtype, is_optional, port_maps = [], description = None)

   Bases: :py:obj:`PortSpecEditor`


   Specification for an input port of an SparkModule.


   .. py:attribute:: is_optional
      :type:  bool


   .. py:attribute:: port_maps
      :type:  list[PortMap]


   .. py:method:: add_port_map(port_map)


   .. py:method:: remove_port_map(port_map)


   .. py:method:: from_input_specs(input_spec, port_maps = [])
      :classmethod:



.. py:class:: OutputSpecEditor(payload_type, shape = None, dtype = None, description = None)

   Bases: :py:obj:`PortSpecEditor`


   Specification for an output port of an SparkModule.


   .. py:method:: from_output_specs(output_spec)
      :classmethod:



.. py:class:: PortMap(origin, port)

   Specification for a connection between SparkModules within the SparkGraphEditor.


   .. py:attribute:: origin
      :type:  str


   .. py:attribute:: port
      :type:  str


.. py:class:: ModuleSpecsEditor(name, module_cls, inputs, config)

   Bases: :py:obj:`BaseSpecEditor`


   Specification for SparkModule automatic constructor within the SparkGraphEditor.


   .. py:attribute:: name
      :type:  str


   .. py:attribute:: module_cls
      :type:  str


   .. py:attribute:: inputs
      :type:  dict[str, list[PortMap]]


   .. py:attribute:: config
      :type:  spark.core.config.BaseSparkConfig


