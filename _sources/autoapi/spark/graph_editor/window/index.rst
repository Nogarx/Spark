spark.graph_editor.window
=========================

.. py:module:: spark.graph_editor.window


Attributes
----------

.. autoapisummary::

   spark.graph_editor.window.CDockWidget


Classes
-------

.. autoapisummary::

   spark.graph_editor.window.DockPanels
   spark.graph_editor.window.EditorWindow


Module Contents
---------------

.. py:data:: CDockWidget

.. py:class:: DockPanels(*args, **kwds)

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


   .. py:attribute:: GRAPH


   .. py:attribute:: INSPECTOR


   .. py:attribute:: NODES


   .. py:attribute:: CONSOLE


.. py:class:: EditorWindow(*args, **kwargs)

   Bases: :py:obj:`PySide6.QtWidgets.QMainWindow`


   .. py:attribute:: __layout_file__
      :type:  str
      :value: 'layout.xml'



   .. py:attribute:: windowClosed


   .. py:attribute:: dock_manager


   .. py:method:: add_dock_widget(area, dock_widget)


   .. py:method:: closeEvent(event)


   .. py:method:: save_layout()

      Save current layout.



   .. py:method:: restore_layout()

      Try restoring the previous layout.



