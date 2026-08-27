spark.graph_editor.widgets.scroll_utils
=======================================

.. py:module:: spark.graph_editor.widgets.scroll_utils


Classes
-------

.. autoapisummary::

   spark.graph_editor.widgets.scroll_utils.ScrollMarginBalancer


Module Contents
---------------

.. py:class:: ScrollMarginBalancer(scroll_area, layout, margins)

   Bases: :py:obj:`PySide6.QtCore.QObject`


   Keeps the content of a scroll area optically centered.

   A QScrollArea reserves the vertical scroll bar outside of its viewport, so the gap on the right of
   the content is the right margin plus the width of the bar. The right margin is reduced by the width
   of the bar while it is visible, leaving both gutters equal.


   .. py:method:: eventFilter(watched, event)


   .. py:method:: apply()

      Recomputes the right margin from the current scroll bar state.



