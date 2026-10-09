spark.graph_editor.models.graph_export
======================================

.. py:module:: spark.graph_editor.models.graph_export


Attributes
----------

.. autoapisummary::

   spark.graph_editor.models.graph_export.logger


Classes
-------

.. autoapisummary::

   spark.graph_editor.models.graph_export.ExportedGraph


Functions
---------

.. autoapisummary::

   spark.graph_editor.models.graph_export.is_module_node
   spark.graph_editor.models.graph_export.build_module_specs
   spark.graph_editor.models.graph_export.build_controller_config


Module Contents
---------------

.. py:data:: logger

.. py:class:: ExportedGraph

   Result of translating a graph back into a controller configuration.


   .. py:attribute:: config
      :type:  Any
      :value: None



   .. py:attribute:: specs
      :type:  list[spark.core.specs.ModuleSpecs]
      :value: []



   .. py:attribute:: layout
      :type:  dict[str, tuple[float, float]]


   .. py:attribute:: problems
      :type:  list[str]
      :value: []



   .. py:property:: is_complete
      :type: bool


      True if the graph describes a model the framework can instantiate.


.. py:function:: is_module_node(node)

   True if a node becomes a module of the controller.

   Sources, sinks and controller properties are not modules: they stand for the controller itself.


.. py:function:: build_module_specs(graph_model)

   Builds the ModuleSpecs of every module on the canvas.

   :param graph_model: Graph to read.
   :type graph_model: GraphModel

   :returns: * **specs** (*list of ModuleSpecs*) -- One entry per module node.
             * **problems** (*list of str*) -- What would stop the model from being instantiated.


.. py:function:: build_controller_config(graph_model, strict = False)

   Translates a graph into the configuration of its controller.

   :param graph_model: Graph to translate.
   :type graph_model: GraphModel
   :param strict: Also report everything that would stop the framework from instantiating the model.
   :type strict: bool, default False

   :returns: The configuration, the specs, the node layout and the problems found.
   :rtype: ExportedGraph


