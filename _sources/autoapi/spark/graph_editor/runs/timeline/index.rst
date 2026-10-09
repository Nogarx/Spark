spark.graph_editor.runs.timeline
================================

.. py:module:: spark.graph_editor.runs.timeline


Attributes
----------

.. autoapisummary::

   spark.graph_editor.runs.timeline.NOTABLE


Classes
-------

.. autoapisummary::

   spark.graph_editor.runs.timeline.Timeline


Functions
---------

.. autoapisummary::

   spark.graph_editor.runs.timeline.event_text


Module Contents
---------------

.. py:data:: NOTABLE
   :value: ('warning', 'error')


   Kinds of events drawn over the others, the full height of the row of events.

.. py:class:: Timeline(parent = None)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   Timeline of the steps recorded by each set of measurements, the values of the integer tags,
   the events and the cursor.

   Each set of measurements has one row, and each tag, such as ``episode``, a row marking where
   its values start. A left click or drag moves the cursor and emits
   `cursor_moved`. The wheel zooms around the mouse, and a right drag pans; both emit
   `view_changed`. The mouse over a row or an event shows what is there in a tooltip.

   Its colours and sizes are those of ``THEME.timeline``: the width of the labels, the heights
   of the ruler, of a row and of the events, and the colour of each kind of event. ``record``
   events take the colour of the cursor.


   .. py:attribute:: cursor_moved


   .. py:attribute:: view_changed


   .. py:attribute:: HOVER
      :value: 4


      Pixels from the mouse within which an event is shown in the tooltip.


   .. py:attribute:: total
      :value: 1



   .. py:attribute:: view
      :value: (0.0, 1.0)



   .. py:attribute:: rows
      :type:  list[tuple[str, list[tuple[int, int]]]]
      :value: []



   .. py:attribute:: events
      :type:  list[tuple[int, str]]
      :value: []



   .. py:attribute:: payloads
      :type:  list[dict[str, Any]]
      :value: []



   .. py:attribute:: tags
      :type:  list[tuple[str, numpy.ndarray, numpy.ndarray]]
      :value: []



   .. py:attribute:: cursor
      :value: 0



   .. py:method:: set_data(total, rows, events, payloads = None, tags = None)

      Sets the length of the run, the steps recorded and the events shown.

      A view showing the last step moves with the steps added.

      :param total: Steps of the run so far.
      :type total: int
      :param rows: Name of every set of measurements and the ``[start, end)`` spans of steps it
                   recorded, as a ``(spans, 2)`` array or a list of pairs.
      :type rows: list of (str, array or list of (int, int))
      :param events: Step and kind of every event.
      :type events: list of (int, str)
      :param payloads: Payload of every event, shown in the tooltip.
      :type payloads: list of dict, optional
      :param tags: Name of every integer tag, the steps at which it took its values, and the values.
      :type tags: list of (str, ndarray, ndarray), optional



   .. py:method:: set_cursor(step)

      Moves the cursor to ``step`` without emitting `cursor_moved`.



   .. py:method:: show_range(start, end)

      Shows the steps from ``start`` to ``end``, within the run and at least one step wide.



   .. py:method:: mousePressEvent(event)


   .. py:method:: mouseMoveEvent(event)


   .. py:method:: mouseReleaseEvent(event)


   .. py:method:: leaveEvent(event)


   .. py:method:: wheelEvent(event)


   .. py:method:: hover_text(x, y)

      Returns the tooltip at pixel ``(x, y)``: the events near the mouse in the row of events,
      or the span of steps under the mouse in the row of a set of measurements.



   .. py:method:: paintEvent(event)


.. py:function:: event_text(payload)

   Returns what an event says: its ``message`` or ``error`` whole, or else its other fields as
   JSON, cut past 120 characters.


