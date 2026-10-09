spark.graph_editor.models.model_library
=======================================

.. py:module:: spark.graph_editor.models.model_library


Attributes
----------

.. autoapisummary::

   spark.graph_editor.models.model_library.logger
   spark.graph_editor.models.model_library.SETTINGS_KEY
   spark.graph_editor.models.model_library.SUFFIX


Functions
---------

.. autoapisummary::

   spark.graph_editor.models.model_library.default_path
   spark.graph_editor.models.model_library.library_path
   spark.graph_editor.models.model_library.set_library_path
   spark.graph_editor.models.model_library.model_files
   spark.graph_editor.models.model_library.model_name
   spark.graph_editor.models.model_library.read_model
   spark.graph_editor.models.model_library.register_file
   spark.graph_editor.models.model_library.register_library
   spark.graph_editor.models.model_library.import_model


Module Contents
---------------

.. py:data:: logger

.. py:data:: SETTINGS_KEY
   :value: 'model_library_path'


.. py:data:: SUFFIX
   :value: '.scfg'


.. py:function:: default_path()

   Returns the location of the default editor's model library.


.. py:function:: library_path()

   Returns the location of the editor's model library.


.. py:function:: set_library_path(path)

   Sets the location of the editor's model library.

   :param path: The location. None restores the default.
   :type path: str or pathlib.Path or None


.. py:function:: model_files(path = None)

   Returns a list of model files in the editor's model library.

   :param path: A location to read instead of the chosen one.
   :type path: str or pathlib.Path, optional

   :returns: The files.
   :rtype: list of pathlib.Path


.. py:function:: model_name(path)

   Returns the the name of the file.


.. py:function:: read_model(path)

   Reads a file as a model.

   :param path: The file to read.
   :type path: str or pathlib.Path

   :returns: The configuration the file holds.
   :rtype: SparkConfig


.. py:function:: register_file(path)

   Registers a model from a file.

   :param path: The file to read.
   :type path: str or pathlib.Path

   :returns: The name the model answers to, or None if the name was already taken.
   :rtype: str or None


.. py:function:: register_library(path = None)

   Registers all models in the editor's model library.

   :param path: A location to read instead of the chosen one.
   :type path: str or pathlib.Path, optional

   :returns: * **registered** (*list of str*) -- The names now available.
             * **failed** (*list of tuple of (pathlib.Path, str)*) -- The files that could not be registered, with the reason.


.. py:function:: import_model(source, overwrite = False)

   Appends a model file to the editor's model library.

   :param source: The file to take in.
   :type source: str or pathlib.Path
   :param overwrite: Whether a file of that name already in the library may be replaced.
   :type overwrite: bool, default False

   :returns: Where the copy was left.
   :rtype: pathlib.Path


