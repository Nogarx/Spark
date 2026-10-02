spark.graph_editor.view.node_item
=================================

.. py:module:: spark.graph_editor.view.node_item


Classes
-------

.. autoapisummary::

   spark.graph_editor.view.node_item.PortItem
   spark.graph_editor.view.node_item.PropertyRowItem
   spark.graph_editor.view.node_item.OptionalDividerItem
   spark.graph_editor.view.node_item.NodeItem


Module Contents
---------------

.. py:class:: PortItem(model, parent = None)

   Bases: :py:obj:`PySide6.QtWidgets.QGraphicsObject`


   .. py:attribute:: model


   .. py:attribute:: radius


   .. py:attribute:: connected_pipes
      :type:  list[spark.graph_editor.view.pipe_item.PipeItem]
      :value: []



   .. py:method:: add_pipe(pipe)


   .. py:method:: remove_pipe(pipe)


   .. py:method:: get_pipe_at(pos)


   .. py:method:: hoverEnterEvent(event)


   .. py:method:: hoverLeaveEvent(event)


   .. py:method:: boundingRect()


   .. py:method:: paint(painter, option, widget)


.. py:class:: PropertyRowItem(name, input_port_model = None, output_port_model = None, parent = None)

   Bases: :py:obj:`PySide6.QtWidgets.QGraphicsItem`


   .. py:attribute:: name


   .. py:attribute:: width


   .. py:attribute:: height
      :value: 20.0



   .. py:attribute:: label


   .. py:method:: boundingRect()


   .. py:method:: paint(painter, option, widget)


.. py:class:: OptionalDividerItem(width, parent = None)

   Bases: :py:obj:`PySide6.QtWidgets.QGraphicsItem`


   Separator drawn above the optional ports.


   .. py:attribute:: width


   .. py:attribute:: height
      :value: 14.0



   .. py:method:: boundingRect()


   .. py:method:: paint(painter, option, widget)


.. py:class:: NodeItem(model, parent = None)

   Bases: :py:obj:`PySide6.QtWidgets.QGraphicsItem`


   .. py:attribute:: model


   .. py:attribute:: rows
      :type:  list[PySide6.QtWidgets.QGraphicsItem]
      :value: []



   .. py:attribute:: section_headers
      :type:  list[tuple[str, float]]
      :value: []



   .. py:attribute:: width


   .. py:attribute:: header_height


   .. py:attribute:: padding
      :value: 5.0



   .. py:attribute:: title_item


   .. py:attribute:: type_item


   .. py:method:: on_model_pos_changed(x, y)


   .. py:method:: on_model_selected_changed(is_selected)


   .. py:method:: on_model_name_changed(new_name)


   .. py:method:: mousePressEvent(event)


   .. py:method:: mouseReleaseEvent(event)


   .. py:method:: itemChange(change, value)


   .. py:method:: boundingRect()


   .. py:method:: paint(painter, option, widget)


