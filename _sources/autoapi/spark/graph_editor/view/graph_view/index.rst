spark.graph_editor.view.graph_view
==================================

.. py:module:: spark.graph_editor.view.graph_view


Attributes
----------

.. autoapisummary::

   spark.graph_editor.view.graph_view.logger


Classes
-------

.. autoapisummary::

   spark.graph_editor.view.graph_view.TempPipeItem
   spark.graph_editor.view.graph_view.GraphScene
   spark.graph_editor.view.graph_view.GraphClipboard
   spark.graph_editor.view.graph_view.GraphView


Module Contents
---------------

.. py:data:: logger

.. py:class:: TempPipeItem(start_pos, is_disconnecting=False, port_type=None)

   Bases: :py:obj:`PySide6.QtWidgets.QGraphicsPathItem`


   Visual feedback for dragging a new connection or a disconnected one.


   .. py:attribute:: start_pos


   .. py:method:: update_path(end_pos)


.. py:class:: GraphScene(model=None, parent=None)

   Bases: :py:obj:`PySide6.QtWidgets.QGraphicsScene`


   .. py:attribute:: model


   .. py:attribute:: active_port
      :value: None



   .. py:attribute:: drag_source_port
      :value: None



   .. py:attribute:: temp_pipe
      :value: None



   .. py:attribute:: is_disconnecting
      :value: False



   .. py:attribute:: original_pipe_model
      :value: None



   .. py:method:: on_graph_cleared()


   .. py:method:: find_port_item(port_id)


   .. py:method:: on_node_added(node_model)


   .. py:method:: on_node_removed(node_model)


   .. py:method:: on_edge_added(edge_model)


   .. py:method:: on_edge_removed(edge_model)


   .. py:method:: mousePressEvent(event)


   .. py:method:: update_all_pipes()


   .. py:method:: mouseMoveEvent(event)


   .. py:method:: mouseReleaseEvent(event)


   .. py:method:: drawBackground(painter, rect)


.. py:class:: GraphClipboard

   .. py:method:: add_node(node, pos)


   .. py:method:: add_pipe(pipe_dict)


   .. py:method:: get_nodes_data()


   .. py:method:: get_pipes_data()


   .. py:method:: shift_pos(shift)


   .. py:method:: __len__()


   .. py:method:: clear()


.. py:class:: GraphView(scene, parent=None)

   Bases: :py:obj:`PySide6.QtWidgets.QGraphicsView`


   .. py:method:: scene()


   .. py:method:: keyPressEvent(event)


   .. py:method:: contextMenuEvent(event)


   .. py:method:: add_node_for(module_cls, label = 'Add Node')

      Places a node for a module class at the centre of the view.

      Menu driven counterpart of dropping a node from the context menu, which knows where the pointer was.



   .. py:method:: import_model(entry)

      Expands a registered model into the graph.



   .. py:method:: import_config(config, label = 'Model', layout = None)

      Expands a controller configuration into nodes and edges, as a single undoable step.

      :param config: The configuration to expand.
      :type config: SparkConfig
      :param label: Name the undo entry carries.
      :type label: str, default 'Model'
      :param layout: Node positions by name. Supplied when a session is reopened, since a configuration cannot
                     carry the layout by itself.
      :type layout: dict of str to tuple of float, optional



   .. py:method:: delete_selected()


   .. py:method:: copy_selected()


   .. py:method:: paste_selected()


   .. py:method:: mousePressEvent(event)


   .. py:method:: mouseMoveEvent(event)


   .. py:method:: mouseReleaseEvent(event)


   .. py:method:: wheelEvent(event)


   .. py:method:: zoom_in()


   .. py:method:: zoom_out()


