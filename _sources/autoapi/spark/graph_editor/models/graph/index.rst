spark.graph_editor.models.graph
===============================

.. py:module:: spark.graph_editor.models.graph


Classes
-------

.. autoapisummary::

   spark.graph_editor.models.graph.ControllerType
   spark.graph_editor.models.graph.SparkNodeViewer
   spark.graph_editor.models.graph.SparkNodeGraph


Module Contents
---------------

.. py:class:: ControllerType

   Bases: :py:obj:`enum.IntFlag`


   Support for integer-based Flags

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:attribute:: BRAIN
      :value: 0



   .. py:attribute:: NEURON
      :value: 1



.. py:class:: SparkNodeViewer(parent=None, undo_stack=None)

   Bases: :py:obj:`NodeGraphQt.widgets.viewer.NodeViewer`


   Custom reimplementation of NodeViwer use to solve certain bugs.


   .. py:attribute:: BUGFIX_on_save


   .. py:method:: focusInEvent(event)


   .. py:method:: focusOutEvent(event)


   .. py:method:: keyPressEvent(event)

      Explicitly ignore Ctrl+S so it propagates to the Main Window.



.. py:class:: SparkNodeGraph(controller_type, parent=None, **kwargs)

   Bases: :py:obj:`NodeGraphQt.NodeGraph`


   NodeGraphQt object for building/managing Spark models.


   .. py:attribute:: context_menu_prompt


   .. py:attribute:: broadcast_message


   .. py:attribute:: on_update


   .. py:method:: set_controller_type(controller_type)


   .. py:method:: name_exist(name)


   .. py:method:: update_node_name(id, name)


   .. py:method:: create_node(node_type, name = None, selected = True, color = None, text_color = None, pos = None, push_undo = True, select_node = True)


   .. py:method:: delete_node(node, push_undo=True)


   .. py:method:: load_neuron_from_model(config)


   .. py:method:: load_from_model(config)


   .. py:method:: get_config_cls()


   .. py:method:: serialize_controller_config(is_partial = True, errors = None)


   .. py:method:: export_controller_config(errors = None)


   .. py:method:: build_nx_graph()


   .. py:method:: validate_graph()

      Simple graph validation.

      Ensures that graph has a single connected component and at least one source and one sink is present in the model.



   .. py:method:: spring_layout(scale = 500)


