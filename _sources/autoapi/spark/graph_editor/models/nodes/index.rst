spark.graph_editor.models.nodes
===============================

.. py:module:: spark.graph_editor.models.nodes


Classes
-------

.. autoapisummary::

   spark.graph_editor.models.nodes.AbstractNode
   spark.graph_editor.models.nodes.SourceNode
   spark.graph_editor.models.nodes.SinkNode
   spark.graph_editor.models.nodes.SparkModuleNode


Functions
---------

.. autoapisummary::

   spark.graph_editor.models.nodes.module_to_nodegraph


Module Contents
---------------

.. py:class:: AbstractNode

   Bases: :py:obj:`NodeGraphQt.BaseNode`, :py:obj:`abc.ABC`


   Abstract node model use to represent different components of a Spark model.


   .. py:attribute:: __identifier__
      :value: 'spark'



   .. py:attribute:: NODE_NAME
      :value: 'Abstract Node'



   .. py:attribute:: input_specs
      :type:  dict[str, spark.core.specs.PortSpecs]


   .. py:attribute:: output_specs
      :type:  dict[str, spark.core.specs.PortSpecs]


   .. py:attribute:: graph
      :type:  spark.graph_editor.models.graph.SparkNodeGraph


   .. py:method:: update_io_spec(spec, port_name, payload_type = None, shape = None, dtype = None, description = None)

      Method to update node IO specs. Valid updates are broadcasted.

      :param spec: str, target spec {input, output}
      :param port_name: str, the name of the port to update
      :param payload_type: type[SparkPayload] | None, the new payload_type
      :param shape: tuple[int, ...] | None,  the new shape
      :param dtype: DTypeLike | None,  the new dtype
      :param description: str | None,  the new description



   .. py:property:: metadata
      :type: dict

      :abstractmethod:



.. py:class:: SourceNode

   Bases: :py:obj:`AbstractNode`


   Node representing the input to the system.


   .. py:attribute:: NODE_NAME
      :value: 'Source Node'



   .. py:attribute:: output_specs


   .. py:property:: metadata
      :type: dict



.. py:class:: SinkNode

   Bases: :py:obj:`AbstractNode`


   Node representing the output of the system.


   .. py:attribute:: NODE_NAME
      :value: 'Sink Node'



   .. py:attribute:: input_specs


   .. py:property:: metadata
      :type: dict



.. py:class:: SparkModuleNode

   Bases: :py:obj:`AbstractNode`, :py:obj:`abc.ABC`


   Abstract node representing a SparkModule.


   .. py:attribute:: NODE_NAME
      :value: 'SparkModule'



   .. py:attribute:: module_cls
      :type:  type[spark.core.module.SparkModule]


   .. py:attribute:: node_config
      :type:  spark.core.config.BaseSparkConfig


   .. py:attribute:: input_specs


   .. py:attribute:: output_specs


   .. py:property:: metadata
      :type: dict



.. py:function:: module_to_nodegraph(entry)

   Factory function that creates a new NodeGraphQt node class from an Spark module class.


