spark.graph_editor.view.pipe_item
=================================

.. py:module:: spark.graph_editor.view.pipe_item


Classes
-------

.. autoapisummary::

   spark.graph_editor.view.pipe_item.SegmentGizmo
   spark.graph_editor.view.pipe_item.PipeRouteContext
   spark.graph_editor.view.pipe_item.PipeItem


Module Contents
---------------

.. py:class:: SegmentGizmo(parent=None)

   Bases: :py:obj:`PySide6.QtWidgets.QGraphicsItem`


   .. py:attribute:: is_horizontal
      :value: True



   .. py:attribute:: draggable
      :value: True



   .. py:method:: boundingRect()


   .. py:method:: paint(painter, option, widget)


.. py:class:: PipeRouteContext(obstacles, bundles = None)

   Shared state of one routing pass.

   The node rectangles and the lanes already in use are collected once per pass rather than once per
   pipe, which keeps routing a whole scene linear in the number of pipes.


   .. py:attribute:: lanes
      :type:  list[tuple[float, float, float]]
      :value: []



   .. py:attribute:: channels
      :type:  list[tuple[float, float, float]]
      :value: []



   .. py:method:: for_scene(scene)
      :classmethod:



   .. py:method:: for_pipe(pipe)
      :classmethod:


      Context for a single pipe: the lanes of every older pipe are already claimed.



   .. py:method:: obstacles_excluding(*nodes)


   .. py:method:: claim(points)

      Registers the lanes and columns a route occupies.



   .. py:method:: bundle_index(pipe, source_node, target_node)

      Rank of a pipe among the ones joining the same two nodes, in both directions.



.. py:class:: PipeItem(source_port_item, target_port_item, model = None, parent = None)

   Bases: :py:obj:`PySide6.QtWidgets.QGraphicsPathItem`


   .. py:attribute:: source_port


   .. py:attribute:: target_port


   .. py:attribute:: model
      :value: None



   .. py:attribute:: pivots
      :value: []



   .. py:attribute:: gizmo


   .. py:attribute:: full_pts
      :value: []



   .. py:method:: scene()


   .. py:method:: on_waypoints_changed()


   .. py:method:: update_path(context = None)


   .. py:method:: pipe_color(active = False)

      Colour of the payload the pipe carries.

      :param active: Return the highlighted variant of the colour.
      :type active: bool, default False

      :returns: The colour of the payload.
      :rtype: QColor

      .. rubric:: Notes

      Both ends always share a type, connections between different payloads are rejected.



   .. py:method:: clean_loops()


   .. py:method:: split_segment(idx, scene_pos)


   .. py:method:: simplify_at(idx)


   .. py:method:: disconnect_pipe()


   .. py:method:: shape()


   .. py:method:: hoverEnterEvent(event)


   .. py:method:: hoverMoveEvent(event)


   .. py:method:: hoverLeaveEvent(event)


   .. py:method:: mousePressEvent(event)


   .. py:method:: mouseMoveEvent(event)


   .. py:method:: mouseReleaseEvent(event)


   .. py:method:: mouseDoubleClickEvent(event)


   .. py:method:: paint(painter, option, widget)


