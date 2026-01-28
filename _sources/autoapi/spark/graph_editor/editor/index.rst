spark.graph_editor.editor
=========================

.. py:module:: spark.graph_editor.editor


Attributes
----------

.. autoapisummary::

   spark.graph_editor.editor.logger


Classes
-------

.. autoapisummary::

   spark.graph_editor.editor.DockPanels
   spark.graph_editor.editor.SparkGraphEditor


Module Contents
---------------

.. py:data:: logger

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


.. py:class:: SparkGraphEditor

   .. py:attribute:: app


   .. py:method:: launch()

      Creates and shows the editor window without blocking.
      This method is safe to call multiple times.



   .. py:method:: exit_editor()

      Exit editor.



   .. py:method:: closeEvent(event)

      Overrides the default close event to check for unsaved changes.



   .. py:method:: new_session()

      Clears the current session after checking for unsaved changes.



   .. py:method:: save_session()

      Saves the current session to a Spark Graph Editor file.



   .. py:method:: save_session_as()

      Saves the current session to a new Spark Graph Editor file.



   .. py:method:: load_session()

      Loads a graph state from a Spark Graph Editor file after checking for unsaved changes.



   .. py:method:: load_from_model()

      Loads a graph state from a Spark configuration file after checking for unsaved changes.



   .. py:method:: export_model()

      Exports the graph state to a Spark configuration file.



   .. py:method:: export_model_as()

      Exports the graph state to a new Spark configuration file.



