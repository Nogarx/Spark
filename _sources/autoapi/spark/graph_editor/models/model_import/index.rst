spark.graph_editor.models.model_import
======================================

.. py:module:: spark.graph_editor.models.model_import


Attributes
----------

.. autoapisummary::

   spark.graph_editor.models.model_import.logger


Classes
-------

.. autoapisummary::

   spark.graph_editor.models.model_import.ImportedGraph


Functions
---------

.. autoapisummary::

   spark.graph_editor.models.model_import.expand_controller_config


Module Contents
---------------

.. py:data:: logger

.. py:class:: ImportedGraph

   Result of expanding a controller configuration.


   .. py:attribute:: nodes
      :type:  list[spark.graph_editor.models.node_model.NodeModel]
      :value: []



   .. py:attribute:: edges
      :type:  list[spark.graph_editor.models.edge_model.EdgeModel]
      :value: []



   .. py:attribute:: warnings
      :type:  list[str]
      :value: []



.. py:function:: expand_controller_config(config, graph_model, profile = None, layout = None)

   Expands a controller configuration into the nodes and edges of its modules.

   :param config: Configuration of the controller to expand.
   :type config: SparkConfig
   :param graph_model: Graph the result is destined to. Only read, never modified.
   :type graph_model: GraphModel
   :param profile: Active profile. Used to type the controller properties.
   :type profile: ControllerProfile, optional
   :param layout: Positions by node name. A configuration cannot carry the layout by itself (see session_io).
   :type layout: dict of str to tuple of float, optional

   :returns: The nodes and edges to add, already positioned.
   :rtype: ImportedGraph


