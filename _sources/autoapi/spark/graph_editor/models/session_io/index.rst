spark.graph_editor.models.session_io
====================================

.. py:module:: spark.graph_editor.models.session_io


Attributes
----------

.. autoapisummary::

   spark.graph_editor.models.session_io.logger
   spark.graph_editor.models.session_io.SESSION_SUFFIX
   spark.graph_editor.models.session_io.MODEL_SUFFIX
   spark.graph_editor.models.session_io.SESSION_FORMAT
   spark.graph_editor.models.session_io.SESSION_FILTER
   spark.graph_editor.models.session_io.MODEL_FILTER


Classes
-------

.. autoapisummary::

   spark.graph_editor.models.session_io.LoadedSession


Functions
---------

.. autoapisummary::

   spark.graph_editor.models.session_io.save_session
   spark.graph_editor.models.session_io.load_session
   spark.graph_editor.models.session_io.check_model
   spark.graph_editor.models.session_io.export_model
   spark.graph_editor.models.session_io.model_layout
   spark.graph_editor.models.session_io.load_model


Module Contents
---------------

.. py:data:: logger

.. py:data:: SESSION_SUFFIX
   :value: '.sge'


.. py:data:: MODEL_SUFFIX
   :value: '.scfg'


.. py:data:: SESSION_FORMAT
   :value: 1


.. py:data:: SESSION_FILTER
   :value: 'Spark Graph Editor (*.sge);;All Files (*)'


.. py:data:: MODEL_FILTER
   :value: 'Spark Configuration (*.scfg);;All Files (*)'


.. py:class:: LoadedSession

   Contents of a session file.


   .. py:attribute:: profile
      :type:  Any
      :value: None



   .. py:attribute:: config
      :type:  Any
      :value: None



   .. py:attribute:: layout
      :type:  dict[str, tuple[float, float]]


.. py:function:: save_session(graph_model, path)

   Writes the graph as a session, however incomplete it is.

   :param graph_model: The graph to write.
   :type graph_model: GraphModel
   :param path: Where to write it. The session suffix is applied.
   :type path: str or pathlib.Path

   :returns: The path actually written.
   :rtype: pathlib.Path


.. py:function:: load_session(path)

   Reads a session file.


.. py:function:: check_model(graph_model)

   Reports what keeps the graph from being a model, without writing anything.

   :param graph_model: The graph to check.
   :type graph_model: GraphModel

   :returns: Every problem found. Empty when the graph can be exported as it stands.
   :rtype: list of str


.. py:function:: export_model(graph_model, path)

   Writes the graph as a model the framework can instantiate.

   :param graph_model: The graph to write.
   :type graph_model: GraphModel
   :param path: Where to write it. The model suffix is applied.
   :type path: str or pathlib.Path

   :returns: The path actually written.
   :rtype: pathlib.Path

   :raises ValueError: Listing everything that keeps the graph from being a valid model.


.. py:function:: model_layout(path)

   Node positions stored in a model file, by node name.

   Files not written by the editor carry none, and the model is laid out on import instead.

   :param path: The file to read.
   :type path: str or pathlib.Path

   :returns: The positions. Empty when the file carries none.
   :rtype: dict of str to tuple of float


.. py:function:: load_model(path)

   Reads a model file.


