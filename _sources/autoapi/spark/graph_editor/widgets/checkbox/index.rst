spark.graph_editor.widgets.checkbox
===================================

.. py:module:: spark.graph_editor.widgets.checkbox


Classes
-------

.. autoapisummary::

   spark.graph_editor.widgets.checkbox.WarningFlag
   spark.graph_editor.widgets.checkbox.InheritToggleButton


Module Contents
---------------

.. py:class:: WarningFlag(value = False, parent = None)

   Bases: :py:obj:`PySide6.QtWidgets.QPushButton`


   Toggleable QPushButton used to indicate that an attribute in a SparkConfig may have conflicts.


   .. py:attribute:: warning_icon


   .. py:attribute:: tooltip_message
      :value: None



   .. py:attribute:: empty_icon


   .. py:attribute:: toggle_icon


   .. py:method:: set_error_status(messages)


   .. py:method:: mousePressEvent(event)


   .. py:method:: mouseReleaseEvent(event)


   .. py:method:: enterEvent(event)


   .. py:method:: leaveEvent(event)


.. py:class:: InheritToggleButton(value = False, interactable = True, parent = None)

   Bases: :py:obj:`PySide6.QtWidgets.QPushButton`


   Toggleable QPushButton used to indicate that an attribute in a SparkConfig is inheriting it's value to child attributes.


   .. py:attribute:: link_icon


   .. py:attribute:: lock_icon


   .. py:attribute:: unlink_icon


   .. py:attribute:: toggle_icon


