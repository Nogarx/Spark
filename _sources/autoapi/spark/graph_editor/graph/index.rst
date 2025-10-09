spark.graph_editor.graph
========================

.. py:module:: spark.graph_editor.graph


Classes
-------

.. autoapisummary::

   spark.graph_editor.graph.SparkNodeViewer
   spark.graph_editor.graph.SparkNodeGraph


Module Contents
---------------

.. py:class:: SparkNodeViewer(parent=None)

   Bases: :py:obj:`NodeGraphQt.widgets.viewer.NodeViewer`


   Custom reimplementation of NodeViwer use to solve certain visual bugs.


   .. py:method:: mousePressEvent(event)


.. py:class:: SparkNodeGraph(parent=None, **kwargs)

   Bases: :py:obj:`NodeGraphQt.NodeGraph`


   .. py:attribute:: stateChanged


   .. py:attribute:: context_menu_prompt


   .. py:method:: ignore_next_click_event(state = True)


   .. py:method:: name_exist(name)


   .. py:method:: update_node_name(id, name)


   .. py:method:: create_node(node_type, name=None, selected=True, color=None, text_color=None, pos=None, push_undo=True)


   .. py:method:: delete_node(node, push_undo=True)


   .. py:method:: build_raw_graph()


   .. py:method:: get_nodes_by_map()


