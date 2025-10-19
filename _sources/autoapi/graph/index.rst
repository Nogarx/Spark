graph
=====

.. py:module:: graph


Classes
-------

.. autoapisummary::

   graph.SparkNodeViewer
   graph.SparkNodeGraph


Module Contents
---------------

.. py:class:: SparkNodeViewer(parent=None)

   Bases: :py:obj:`NodeGraphQt.widgets.viewer.NodeViewer`


   Custom reimplementation of NodeViwer use to solve certain visual bugs.


   .. py:method:: mousePressEvent(event)


.. py:class:: SparkNodeGraph(parent=None, **kwargs)

   Bases: :py:obj:`NodeGraphQt.NodeGraph`


   NodeGraphQt object for building/managing Spark models.


   .. py:attribute:: context_menu_prompt


   .. py:attribute:: broadcast_message


   .. py:attribute:: stateChanged


   .. py:method:: name_exist(name)


   .. py:method:: update_node_name(id, name)


   .. py:method:: create_node(node_type, name=None, selected=True, color=None, text_color=None, pos=None, push_undo=True)


   .. py:method:: delete_node(node, push_undo=True)


   .. py:method:: build_raw_graph()


   .. py:method:: get_nodes_by_map()


   .. py:method:: load_from_model(config)


   .. py:method:: validate_graph()

      Simple graph validation.

      Ensures that graph has a single connected component and at least one source and one sink is present in the model.



   .. py:method:: build_brain_config()

      Build the model from the graph state.



