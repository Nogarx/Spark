spark.graph_editor.widgets.checkbox
===================================

.. py:module:: spark.graph_editor.widgets.checkbox


Classes
-------

.. autoapisummary::

   spark.graph_editor.widgets.checkbox.WarningFlag
   spark.graph_editor.widgets.checkbox.InitializerToggleButton
   spark.graph_editor.widgets.checkbox.InheritStatus
   spark.graph_editor.widgets.checkbox.InheritToggleButton


Module Contents
---------------

.. py:class:: WarningFlag(value = False, **kwargs)

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


.. py:class:: InitializerToggleButton(value = False, interactable = True, **kwargs)

   Bases: :py:obj:`PySide6.QtWidgets.QPushButton`


   Toggleable QPushButton used to indicate that an attribute in a SparkConfig can be set using an initializer.


   .. py:attribute:: simple_icon


   .. py:attribute:: complex_icon


   .. py:attribute:: toggle_icon


   .. py:method:: set_value(state)

      Set the current state of the checkbox.



.. py:class:: InheritStatus(*args, **kwds)

   Bases: :py:obj:`enum.Enum`


   Create a collection of name/value pairs.

   Example enumeration:

   >>> class Color(Enum):
   ...     RED = 1
   ...     BLUE = 2
   ...     GREEN = 3

   Access them by:

   - attribute access:

     >>> Color.RED
     <Color.RED: 1>

   - value lookup:

     >>> Color(1)
     <Color.RED: 1>

   - name lookup:

     >>> Color['RED']
     <Color.RED: 1>

   Enumerations can be iterated over, and know how many members they have:

   >>> len(Color)
   3

   >>> list(Color)
   [<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>]

   Methods can be added to enumerations, and members can have their own
   attributes -- see the documentation for details.


   .. py:attribute:: LINK


   .. py:attribute:: LOCK


   .. py:attribute:: FREE


.. py:class:: InheritToggleButton(value = False, interactable = True, **kwargs)

   Bases: :py:obj:`PySide6.QtWidgets.QPushButton`


   Toggleable QPushButton used to indicate that an attribute in a SparkConfig is inheriting it's value to child attributes.


   .. py:attribute:: link_icon


   .. py:attribute:: lock_icon


   .. py:attribute:: free_icon


   .. py:attribute:: interactable
      :value: True



   .. py:attribute:: toggle_icon


   .. py:method:: set_value(state)

      Set the current state of the checkbox.



