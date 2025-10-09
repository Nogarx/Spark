spark.graph_editor.nodes
========================

.. py:module:: spark.graph_editor.nodes


Classes
-------

.. autoapisummary::

   spark.graph_editor.nodes.AbstractNode
   spark.graph_editor.nodes.SourceNode
   spark.graph_editor.nodes.SinkNode
   spark.graph_editor.nodes.SparkModuleNode


Functions
---------

.. autoapisummary::

   spark.graph_editor.nodes.module_to_nodegraph


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
      :type:  dict[str, spark.graph_editor.specs.InputSpecEditor]


   .. py:attribute:: output_specs
      :type:  dict[str, spark.graph_editor.specs.OutputSpecEditor]


   .. py:method:: update_input_shape(port_name, value)

      Updates the shape of an input port and broadcast the update.

      :param port_name: The name of the port.
      :type port_name: str
      :param value: The new value for the attribute.
      :type value: Any



   .. py:method:: update_output_shape(port_name, value)

      Updates the shape of an input port and broadcast the update.

      :param port_name: The name of the port.
      :type port_name: str
      :param value: The new value for the attribute.
      :type value: Any



.. py:class:: SourceNode

   Bases: :py:obj:`AbstractNode`


   Node representing the input to the system.


   .. py:attribute:: NODE_NAME
      :value: 'Source Node'



   .. py:attribute:: output_specs


.. py:class:: SinkNode

   Bases: :py:obj:`AbstractNode`


   Node representing the output of the system.


   .. py:attribute:: NODE_NAME
      :value: 'Sink Node'



   .. py:attribute:: input_specs


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


.. py:function:: module_to_nodegraph(entry)

   Factory function that creates a new NodeGraphQt node class from an Spark module class.


