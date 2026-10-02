spark.graph_editor.runs.plots
=============================

.. py:module:: spark.graph_editor.runs.plots


Attributes
----------

.. autoapisummary::

   spark.graph_editor.runs.plots.MAX_IMAGE
   spark.graph_editor.runs.plots.MAX_MATRIX


Classes
-------

.. autoapisummary::

   spark.graph_editor.runs.plots.SeriesPlot
   spark.graph_editor.runs.plots.ImagePlot
   spark.graph_editor.runs.plots.BarPlot
   spark.graph_editor.runs.plots.MatrixPlot


Functions
---------

.. autoapisummary::

   spark.graph_editor.runs.plots.nice_ticks
   spark.graph_editor.runs.plots.format_value
   spark.graph_editor.runs.plots.tick_labels
   spark.graph_editor.runs.plots.value_range


Module Contents
---------------

.. py:data:: MAX_IMAGE
   :value: (4096, 1024)


   Largest raster or trace image, in steps and units. Larger values are reduced in blocks, with
   ``any`` for a raster and the mean otherwise.

.. py:data:: MAX_MATRIX
   :value: (1024, 1024)


   Largest matrix image, in rows and columns. Larger matrices are reduced in blocks, as for
   `MAX_IMAGE`.

.. py:function:: nice_ticks(lo, hi, count = 4)

   Returns round tick values within ``[lo, hi]``.

   The spacing is 1, 2, 2.5 or 5 times a power of ten, the smallest giving at most ``count``
   intervals. An empty range gives ``[lo]``, or no tick when ``lo`` is not finite.


.. py:function:: format_value(value)

   Formats a value for a label.

   Magnitudes from 1e-2 up to 1e4 are written with three significant digits, others in
   scientific notation with two. Zero is ``'0'``.


.. py:function:: tick_labels(values)

   Returns labels for evenly spaced ticks, with the digits that tell them apart.

   Ticks whose largest magnitude is from 1e-3 up to 1e9 are written in fixed point with
   thousands separators, such as ``12,250``. Others are written in scientific notation, such as
   ``1.25e+10``. Fewer than two ticks are written with `format_value`.


.. py:function:: value_range(values)

   Returns the lowest and the highest finite value, or ``(0.0, 1.0)`` without one.


.. py:class:: SeriesPlot(title = '', height = None, parent = None)

   Bases: :py:obj:`_Plot`


   Line plot of series against the step.

   Series take the series colours of `THEME` in order, unless given one. Bands, such as one
   standard deviation around a mean, are drawn under the series. With more than one series, a
   legend lists the first eight. The title ends with the values at the cursor, when there are
   at most `READOUT` of them: the value of each series in its color, or the values of a
   series and its bands by name.

   A drag with the right button zooms to the box drawn (`select`): its values become the
   vertical range, and its steps the horizontal range, or, with ``shares_view``, are asked
   for with `view_requested`, so that the plots sharing a view follow. A right click without
   a drag opens a menu that fits the values or shows every step. Ctrl and the wheel zoom the
   vertical axis around the mouse; a double click fits it to the values shown again.


   .. py:attribute:: view_requested


   .. py:attribute:: READOUT
      :value: 4


      Largest number of values at the cursor given in the title.


   .. py:attribute:: DRAG
      :value: 4


      Pixels the mouse moves, with the right button held, past which it draws a box to zoom to.


   .. py:attribute:: legend
      :value: True


      Whether the legend is drawn within the plot, with more than one series.


   .. py:attribute:: series
      :type:  list[tuple[str, numpy.ndarray, numpy.ndarray, PySide6.QtGui.QColor]]
      :value: []



   .. py:attribute:: bands
      :type:  list[tuple[str | tuple[str, str], numpy.ndarray, numpy.ndarray, numpy.ndarray, PySide6.QtGui.QColor]]
      :value: []



   .. py:attribute:: zoomed
      :type:  tuple[float, float] | None
      :value: None



   .. py:attribute:: shares_view
      :value: False



   .. py:attribute:: faint
      :type:  list[tuple[numpy.ndarray, numpy.ndarray, PySide6.QtGui.QColor]]
      :value: []



   .. py:method:: set_series(series, x_range = None, y_range = None, bands = None, faint = None)

      Sets the series drawn, each as ``(label, x, y)`` or ``(label, x, y, color)``.

      ``bands`` are drawn under the series in order, each as ``(label, x, low, high, color)``.
      The label names half the width of the band, such as ``'std'``, or both of its edges, as
      a pair such as ``('min', 'max')``; an empty label, neither. ``faint`` lines, each as
      ``(x, y, color)``, are drawn thin between the bands and the series, outside the legend
      and the values given. The horizontal range is ``x_range``, else that of the steps. The
      vertical range is ``y_range``, else that of the finite values within the horizontal
      range, with a margin. Without points, the plot shows a message.



   .. py:method:: set_view(x_range)

      Shows the steps of ``x_range``, or all steps for None, the vertical range following the
      values shown.



   .. py:method:: select(x0, x1, y0 = None, y1 = None)

      Zooms to the steps from ``x0`` to ``x1`` and, when given, the values from ``y0`` to
      ``y1``.

      With ``shares_view``, the steps are asked for with `view_requested` instead of shown.



   .. py:method:: show_every_step()

      Shows every step, asked for with `view_requested` with ``shares_view``.



   .. py:method:: fit()

      Fits the vertical range to the values shown again, after a zoom.



   .. py:method:: zoom(factor, at = None)

      Scales the vertical range by ``factor`` around the value ``at``, or its middle.



   .. py:method:: title_text()

      Returns the text drawn as the title.



   .. py:method:: values_at(x)

      Returns the label and value of every series at its last point at or before ``x``.

      A band gives half its width, or both of its edges, as its label names them.



   .. py:method:: set_cursor(x)

      Moves the cursor to ``x``, or hides it for None. Ignored without ``FOLLOWS_CURSOR``.



   .. py:method:: hover_text(x, y)

      Returns the tooltip at data coordinates ``(x, y)``, or None.

      Implemented by each plot.



   .. py:method:: mousePressEvent(event)


   .. py:method:: mouseMoveEvent(event)


   .. py:method:: mouseReleaseEvent(event)


   .. py:method:: wheelEvent(event)


   .. py:method:: mouseDoubleClickEvent(event)


   .. py:method:: shapes(rect)

      Returns the shapes each series draws in ``rect``, with their colors.

      A series of more than two points per pixel column draws one vertical line per column,
      from its lowest to its highest value there. Cached until the series, the ranges or the
      size change.



   .. py:method:: faint_shapes(rect)

      Returns the shapes the faint lines draw in ``rect``, as `shapes`.



   .. py:method:: band_shapes(rect)

      Returns the shapes each band draws in ``rect``, with their colors.

      A polygon is filled. A band of more than two points per pixel column draws one vertical
      line per column instead, from its lowest low to its highest high there. Cached as
      `shapes`.



   .. py:method:: draw(painter, rect)

      Draws the data in ``rect``, clipped to it.

      Implemented by each plot.



.. py:class:: ImagePlot(title = '', height = None, parent = None)

   Bases: :py:obj:`_Plot`


   Image of values over steps and units, such as a raster of spikes or a trace of many units.

   Steps run along the horizontal axis and units upwards. Booleans are drawn as events on the
   background of the plots, numbers with the colormap over their range, with a colour bar.


   .. py:attribute:: HEIGHT
      :value: 'image_height'


      Size of `THEME` giving the height of the plot when none is given.


   .. py:method:: set_image(times, values, x_range = None)

      Sets the values drawn, one row per step of ``times``, flattened past the first axis.

      The image spans from the first step to the step after the last. Values past `MAX_IMAGE`
      are reduced in blocks.



   .. py:method:: hover_text(x, y)

      Returns the tooltip at data coordinates ``(x, y)``, or None.

      Implemented by each plot.



   .. py:method:: draw(painter, rect)

      Draws the data in ``rect``, clipped to it.

      Implemented by each plot.



.. py:class:: BarPlot(title = '', height = None, parent = None)

   Bases: :py:obj:`_Plot`


   Bar plot of values, such as a histogram, the rate of every unit or an input vector.

   Each value has one bar. The horizontal axis is the index of the value, or spans ``x_range``,
   such as the edges of the bins. With more than one bar per pixel column, each column draws
   the range of its values.


   .. py:attribute:: FOLLOWS_CURSOR
      :value: False



   .. py:attribute:: HEIGHT
      :value: 'bar_height'


      Size of `THEME` giving the height of the plot when none is given.


   .. py:attribute:: values


   .. py:attribute:: labels
      :type:  list[str]
      :value: []



   .. py:attribute:: color
      :type:  PySide6.QtGui.QColor | None
      :value: None



   .. py:method:: set_values(values, x_range = None, labels = None, color = None)

      Sets the values drawn, flattened.

      ``labels`` are drawn over the bars when there are at most 12. The bars take ``color``,
      else the first color of the series.



   .. py:method:: hover_text(x, y)

      Returns the tooltip at data coordinates ``(x, y)``, or None.

      Implemented by each plot.



   .. py:method:: draw(painter, rect)

      Draws the data in ``rect``, clipped to it.

      Implemented by each plot.



.. py:class:: MatrixPlot(title = '', height = None, parent = None)

   Bases: :py:obj:`_Plot`


   Image of a matrix with the colormap over its range, such as the weights of a set of synapses.

   Row 0 is at the bottom. A colour bar gives the range of the values.


   .. py:attribute:: FOLLOWS_CURSOR
      :value: False



   .. py:attribute:: HEIGHT
      :value: 'matrix_height'


      Size of `THEME` giving the height of the plot when none is given.


   .. py:method:: set_matrix(matrix)

      Sets the matrix drawn.

      Arrays are flattened past the first axis, and a vector is one row.



   .. py:method:: hover_text(x, y)

      Returns the tooltip at data coordinates ``(x, y)``, or None.

      Implemented by each plot.



   .. py:method:: draw(painter, rect)

      Draws the data in ``rect``, clipped to it.

      Implemented by each plot.



