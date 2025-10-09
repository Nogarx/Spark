spark.graph_editor.editor
=========================

.. py:module:: spark.graph_editor.editor


Classes
-------

.. autoapisummary::

   spark.graph_editor.editor.SparkGraphEditor


Module Contents
---------------

.. py:class:: SparkGraphEditor

   .. py:attribute:: app
      :value: None



   .. py:attribute:: window
      :value: None



   .. py:attribute:: graph
      :value: None



   .. py:attribute:: inspector
      :value: None



   .. py:method:: delete_node(graph, node)


   .. py:method:: delete_selected(graph)


   .. py:method:: validate_graph()


   .. py:method:: create_node(*args, id)

      Prompts the user for a node name using a dialog and then creates the node.



   .. py:method:: launch()

      Creates and shows the editor window without blocking.
      This method is safe to call multiple times.



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



   .. py:method:: closeEvent(event)

      Overrides the default close event to check for unsaved changes.



