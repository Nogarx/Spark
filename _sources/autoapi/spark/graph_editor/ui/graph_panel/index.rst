spark.graph_editor.ui.graph_panel
=================================

.. py:module:: spark.graph_editor.ui.graph_panel


Attributes
----------

.. autoapisummary::

   spark.graph_editor.ui.graph_panel.CDockWidget


Classes
-------

.. autoapisummary::

   spark.graph_editor.ui.graph_panel.GraphPanel


Module Contents
---------------

.. py:data:: CDockWidget

.. py:class:: GraphPanel(**kwargs)

   Bases: :py:obj:`CDockWidget`


   Container for the NodeGraphQt object.


   .. py:attribute:: broadcast_message


   .. py:attribute:: onWidgetUpdate


   .. py:attribute:: graph


   .. py:method:: delete_selected()


   .. py:method:: maybe_create_node(*args, nodegraph_cls)

      Prompts the user for a node name using a dialog and then creates the node.

      Input:
          id: str, node class id of the node to create.



