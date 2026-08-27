spark.graph_editor.view.graph_context_menu_view
===============================================

.. py:module:: spark.graph_editor.view.graph_context_menu_view


Attributes
----------

.. autoapisummary::

   spark.graph_editor.view.graph_context_menu_view.logger


Classes
-------

.. autoapisummary::

   spark.graph_editor.view.graph_context_menu_view.ContextMenuCommand
   spark.graph_editor.view.graph_context_menu_view.ActionData
   spark.graph_editor.view.graph_context_menu_view.GraphContextMenu


Module Contents
---------------

.. py:data:: logger

.. py:class:: ContextMenuCommand(*args, **kwds)

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


   .. py:attribute:: Undo


   .. py:attribute:: Redo


   .. py:attribute:: Create


   .. py:attribute:: Delete


   .. py:attribute:: Copy


   .. py:attribute:: Paste


   .. py:attribute:: Import


.. py:class:: ActionData

   .. py:attribute:: command
      :type:  ContextMenuCommand


   .. py:attribute:: cls
      :type:  type | None
      :value: None



   .. py:attribute:: entry
      :type:  spark.core.registry.RegistryEntry | None
      :value: None



.. py:class:: GraphContextMenu(profile = None, parent=None)

   Bases: :py:obj:`PySide6.QtWidgets.QMenu`


   Hierarchical context menu for adding nodes to the graph.


   .. py:attribute:: node_selected


   .. py:method:: set_profile(profile)

      Rebuilds the palette for a controller profile.



   .. py:method:: update_menu_state(has_selection, can_undo, can_redo, can_paste)


