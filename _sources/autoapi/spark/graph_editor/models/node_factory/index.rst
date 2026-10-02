spark.graph_editor.models.node_factory
======================================

.. py:module:: spark.graph_editor.models.node_factory


Attributes
----------

.. autoapisummary::

   spark.graph_editor.models.node_factory.logger
   spark.graph_editor.models.node_factory.NODE_REGISTRY


Classes
-------

.. autoapisummary::

   spark.graph_editor.models.node_factory.NodeFactory
   spark.graph_editor.models.node_factory.NodeRegistry


Module Contents
---------------

.. py:data:: logger

.. py:class:: NodeFactory

   Builds NodeModel classes from registry entries.


   .. py:method:: create_node_from_registry(entry, base_node_cls)
      :staticmethod:


      Creates a NodeModel class from a registry entry.



.. py:class:: NodeRegistry

   Graph Editor Registry for node models.


   .. py:attribute:: NAMESPACE_BASE_MODEL


   .. py:method:: get(node_cls)


   .. py:method:: get_namespace(node_cls)

      Returns the registry namespace of a node.



.. py:data:: NODE_REGISTRY

