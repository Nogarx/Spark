spark.graph_editor.widgets.collapsible
======================================

.. py:module:: spark.graph_editor.widgets.collapsible


Classes
-------

.. autoapisummary::

   spark.graph_editor.widgets.collapsible.QCollapsible


Module Contents
---------------

.. py:class:: QCollapsible(title = '', expandedIcon = '▼', collapsedIcon = '▲')

   Bases: :py:obj:`PySide6.QtWidgets.QFrame`


   A collapsible widget to hide and unhide child widgets.
   A signal is emitted when the widget is expanded (True) or collapsed (False).


   .. py:attribute:: toggled


   .. py:method:: toggleButton()

      Return the toggle button.



   .. py:method:: setText(text)

      Set the text of the toggle button.



   .. py:method:: text()

      Return the text of the toggle button.



   .. py:method:: setContent(content)

      Replace central widget (the widget that gets expanded/collapsed).



   .. py:method:: content()

      Return the current content widget.



   .. py:method:: expandedIcon()

      Returns the icon used when the widget is expanded.



   .. py:method:: setExpandedIcon(icon = None)

      Set the icon on the toggle button when the widget is expanded.



   .. py:method:: collapsedIcon()

      Returns the icon used when the widget is collapsed.



   .. py:method:: setCollapsedIcon(icon = None)

      Set the icon on the toggle button when the widget is collapsed.



   .. py:method:: setDuration(msecs)

      Set duration of the collapse/expand animation.



   .. py:method:: setEasingCurve(easing)

      Set the easing curve for the collapse/expand animation.



   .. py:method:: addWidget(widget)

      Add a widget to the central content widget's layout.



   .. py:method:: removeWidget(widget)

      Remove widget from the central content widget's layout.



   .. py:method:: expand(animate = True)

      Expand (show) the collapsible section.



   .. py:method:: collapse(animate = True)

      Collapse (hide) the collapsible section.



   .. py:method:: isExpanded()

      Return whether the collapsible section is visible.



   .. py:method:: setLocked(locked = True)

      Set whether collapse/expand is disabled.



   .. py:method:: locked()

      Return True if collapse/expand is disabled.



   .. py:method:: eventFilter(a0, a1)

      If a child widget resizes, we need to update our expanded height.



