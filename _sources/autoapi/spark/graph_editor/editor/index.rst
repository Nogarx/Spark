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

   spark.graph_editor.editor.EditorDocument
   spark.graph_editor.editor.SparkGraphEditor
   spark.graph_editor.editor.GraphEditorWindow


Module Contents
---------------

.. py:data:: logger

.. py:class:: EditorDocument

   One model open in the editor.


   .. py:attribute:: scene
      :type:  spark.graph_editor.view.graph_view.GraphScene


   .. py:attribute:: view
      :type:  spark.graph_editor.view.graph_view.GraphView


   .. py:attribute:: session_path
      :type:  pathlib.Path | None
      :value: None



   .. py:attribute:: model_path
      :type:  pathlib.Path | None
      :value: None



   .. py:attribute:: copy_index
      :type:  int
      :value: 0



   .. py:attribute:: copy_name
      :type:  str
      :value: ''



   .. py:property:: model


   .. py:property:: name
      :type: str



   .. py:property:: is_modified
      :type: bool



   .. py:property:: label
      :type: str


      Name shown on the tab, numbered when it is not the first of its name.


   .. py:property:: title
      :type: str



.. py:class:: SparkGraphEditor

   .. py:attribute:: app


   .. py:attribute:: window
      :type:  GraphEditorWindow | None
      :value: None



   .. py:method:: launch()

      Creates and shows the editor window without blocking.



   .. py:method:: exit_editor()

      Exits the editor.



.. py:class:: GraphEditorWindow

   Bases: :py:obj:`PySide6.QtWidgets.QMainWindow`


   .. py:attribute:: windowClosed


   .. py:property:: document
      :type: EditorDocument | None



   .. py:property:: view
      :type: spark.graph_editor.view.graph_view.GraphView



   .. py:method:: add_document(model = None)

      Opens a new tab and makes it current.



   .. py:method:: close_document(index)

      Closes a tab, offering to save it first when it holds unsaved work.



   .. py:method:: closeEvent(event)


   .. py:method:: new_graph(profile = None)

      Starts a new model. The controller is asked for whenever it was not provided.



   .. py:method:: save_session()


   .. py:method:: save_session_as()


   .. py:method:: load_session()


   .. py:method:: load_session_file(file_name)

      Opens a session from a path, in the session being edited when it is still empty.



   .. py:method:: open_path(path)

      Opens a file of either kind, telling them apart by their suffix.

      A path that is gone is dropped from the recent list rather than reported as an error.



   .. py:method:: check_model()

      Reports what the graph still needs before it can be exported, writing nothing.



   .. py:method:: open_model_file(path)

      Opens a model as a session of its own.

      Adding the modules to the session being edited is what "Import Model..." does instead.



   .. py:method:: export_model()


   .. py:method:: export_model_as()


   .. py:method:: import_model_file()

      Imports a model saved as a Spark configuration.

      A model can be opened as a session of its own, or its modules can be added to the session being
      edited. Importing does the second, and never adds a second controller.



   .. py:method:: add_model_to_library()

      Takes a model file into the library, making it available to every model built from now on.



   .. py:method:: open_preferences()


   .. py:method:: reload_ui()

      Rebuilds the widgets after a style change, for every open document.



