spark.graph_editor.inspector
============================

.. py:module:: spark.graph_editor.inspector


Classes
-------

.. autoapisummary::

   spark.graph_editor.inspector.NodeInspectorWidget


Module Contents
---------------

.. py:class:: NodeInspectorWidget(**kwargs)

   Bases: :py:obj:`Qt.QtWidgets.QWidget`


   A widget to inspect and edit the properties of a node in a graph.


   .. py:attribute:: onWidgetUpdate


   .. py:method:: set_node(node)

      Sets the node to be inspected. If the node is None, it clears the inspector.

      :param node: The node to inspect, or None to clear.



