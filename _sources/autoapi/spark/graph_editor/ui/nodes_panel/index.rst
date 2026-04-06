spark.graph_editor.ui.nodes_panel
=================================

.. py:module:: spark.graph_editor.ui.nodes_panel


Classes
-------

.. autoapisummary::

   spark.graph_editor.ui.nodes_panel.NodesPanel
   spark.graph_editor.ui.nodes_panel.QNodeList
   spark.graph_editor.ui.nodes_panel.ListEntry


Module Contents
---------------

.. py:class:: NodesPanel(node_graph, name = 'Nodes', **kwargs)

   Bases: :py:obj:`spark.graph_editor.widgets.dock_panel.QDockPanel`


   Dockable panel that shows a summary of the nodes in the graph.


   .. py:attribute:: node_list


.. py:class:: QNodeList(node_graph, **kwargs)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   Constructs the nodes list associated with the SparkNodeGraph.


   .. py:attribute:: content


   .. py:method:: addWidget(widget)

      Add a widget to the central content widget's layout.



   .. py:method:: on_nodes_deletion(nodes_ids)


   .. py:method:: on_session_changed()


   .. py:method:: on_node_created(node)


   .. py:method:: on_entry_clicked(node_id)


   .. py:method:: on_selection_update(new_selection, prev_selection)


   .. py:method:: on_node_name_update(node_id, name)


.. py:class:: ListEntry(node_id, node_name)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   QWidget associated with each entry in the list.


   .. py:attribute:: clicked


   .. py:attribute:: label


   .. py:method:: on_name_update(name)


   .. py:method:: mousePressEvent(event)


   .. py:method:: mouseReleaseEvent(event)


   .. py:method:: enterEvent(event)


   .. py:method:: leaveEvent(event)


   .. py:method:: set_selection(value)


