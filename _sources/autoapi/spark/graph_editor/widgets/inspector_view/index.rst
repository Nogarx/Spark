spark.graph_editor.widgets.inspector_view
=========================================

.. py:module:: spark.graph_editor.widgets.inspector_view


Attributes
----------

.. autoapisummary::

   spark.graph_editor.widgets.inspector_view.logger


Classes
-------

.. autoapisummary::

   spark.graph_editor.widgets.inspector_view.NodeNameWidget
   spark.graph_editor.widgets.inspector_view.TreeDisplay
   spark.graph_editor.widgets.inspector_view.NodeHeaderWidget
   spark.graph_editor.widgets.inspector_view.ControllerHeaderWidget
   spark.graph_editor.widgets.inspector_view.InspectorView


Module Contents
---------------

.. py:data:: logger

.. py:class:: NodeNameWidget(name, **kwargs)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   .. py:attribute:: on_update


   .. py:method:: sizeHint()


.. py:class:: TreeDisplay(tree)

   Bases: :py:obj:`PySide6.QtWidgets.QPlainTextEdit`


   .. py:method:: showEvent(event)


   .. py:method:: resizeEvent(event)


   .. py:method:: changeEvent(event)


.. py:class:: NodeHeaderWidget(node_model, graph_model = None, config_tree = None, **kwargs)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   .. py:attribute:: node_model


   .. py:attribute:: graph_model
      :value: None



   .. py:attribute:: name_widget


   .. py:attribute:: error_label


   .. py:attribute:: class_label


.. py:class:: ControllerHeaderWidget(profile, config, **kwargs)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   Header of the controller settings, shown while no node is selected.


.. py:class:: InspectorView(parent=None)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   .. py:attribute:: current_node
      :value: None



   .. py:attribute:: state_model
      :value: None



   .. py:attribute:: graph_model
      :value: None



   .. py:attribute:: content_widget


   .. py:attribute:: content_layout


   .. py:method:: invalidate()

      Forces the next set_node() to rebuild, even when it names the same target.



   .. py:method:: set_node(node_model, graph_model = None)

      Populates the inspector with the properties and configuration of the selected node.



