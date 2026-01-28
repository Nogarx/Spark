spark.graph_editor.ui.menu_bar
==============================

.. py:module:: spark.graph_editor.ui.menu_bar


Classes
-------

.. autoapisummary::

   spark.graph_editor.ui.menu_bar.MenuActions
   spark.graph_editor.ui.menu_bar.MenuBar


Module Contents
---------------

.. py:class:: MenuActions(*args, **kwds)

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


   .. py:attribute:: FILE_NEW


   .. py:attribute:: FILE_LOAD_SESSION


   .. py:attribute:: FILE_LOAD_MODEL


   .. py:attribute:: FILE_SAVE_SESSION


   .. py:attribute:: FILE_SAVE_SESSION_AS


   .. py:attribute:: FILE_EXPORT


   .. py:attribute:: FILE_EXPORT_AS


   .. py:attribute:: FILE_EXIT


   .. py:attribute:: WINDOWS_INSPECTOR


   .. py:attribute:: WINDOWS_NODE_LIST


   .. py:attribute:: WINDOWS_CONSOLE


.. py:class:: MenuBar(editor, **kwargs)

   Bases: :py:obj:`PySide6.QtWidgets.QMenuBar`


   Graph Editor menu bar.


   .. py:attribute:: file_menu


   .. py:attribute:: windows_menu


