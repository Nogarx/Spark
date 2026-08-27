spark.graph_editor.widgets.hierarchy_view
=========================================

.. py:module:: spark.graph_editor.widgets.hierarchy_view


Classes
-------

.. autoapisummary::

   spark.graph_editor.widgets.hierarchy_view.HierarchyView


Module Contents
---------------

.. py:class:: HierarchyView(model, parent=None)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   .. py:attribute:: node_double_clicked


   .. py:attribute:: model


   .. py:attribute:: node_to_item
      :type:  dict[spark.graph_editor.models.node_model.NodeModel, tuple[PySide6.QtWidgets.QTreeWidget, PySide6.QtWidgets.QTreeWidgetItem]]


   .. py:attribute:: trees
      :type:  dict[str, PySide6.QtWidgets.QTreeWidget]


   .. py:attribute:: blocks
      :type:  dict[str, PySide6.QtWidgets.QWidget]


   .. py:attribute:: search_bar


   .. py:attribute:: content_widget


   .. py:attribute:: content_layout


   .. py:method:: on_node_added(node)


   .. py:method:: on_node_removed(node)


   .. py:method:: on_graph_cleared()


   .. py:method:: on_tree_selection_changed(changed_tree)


   .. py:method:: on_search_changed(text)


