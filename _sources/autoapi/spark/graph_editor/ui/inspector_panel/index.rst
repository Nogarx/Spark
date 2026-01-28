spark.graph_editor.ui.inspector_panel
=====================================

.. py:module:: spark.graph_editor.ui.inspector_panel


Classes
-------

.. autoapisummary::

   spark.graph_editor.ui.inspector_panel.InspectorPanel
   spark.graph_editor.ui.inspector_panel.QNodeConfig
   spark.graph_editor.ui.inspector_panel.QNodeIO
   spark.graph_editor.ui.inspector_panel.InspectorIdleWidget
   spark.graph_editor.ui.inspector_panel.NodeHeaderWidget
   spark.graph_editor.ui.inspector_panel.NodeNameWidget
   spark.graph_editor.ui.inspector_panel.TreeDisplay


Module Contents
---------------

.. py:class:: InspectorPanel(name = 'Inspector', **kwargs)

   Bases: :py:obj:`spark.graph_editor.widgets.dock_panel.QDockPanel`


   Dockable panel to show node configurations.


   .. py:attribute:: broadcast_message


   .. py:attribute:: on_update


   .. py:method:: clear_selection()


   .. py:method:: on_selection_update(new_selection, previous_selection)

      Event handler for graph selections.



   .. py:method:: set_node(node)

      Sets the node to be inspected. If the node is None, it clears the inspector.

      Input:
          node: The node to inspect, or None to clear.



   .. py:method:: set_dirty_flag(value)


   .. py:method:: update_node_metadata()


   .. py:method:: on_error_message(message)


.. py:class:: QNodeConfig(node, **kwargs)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   Constructs the UI to modify the SparkConfig associated with the a node.


   .. py:attribute:: error_detected


   .. py:attribute:: on_update


   .. py:attribute:: content


   .. py:method:: addWidget(widget)

      Add a widget to the central content widget's layout.



   .. py:method:: name_update(node, name)


   .. py:method:: on_inheritance_toggle(value, leaf)


   .. py:method:: set_dirty_flag(value)


.. py:class:: QNodeIO(node, **kwargs)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   Constructs the UI to modify the SparkConfig associated with the a inputs and outpus.


   .. py:attribute:: error_detected


   .. py:attribute:: on_update


   .. py:attribute:: content


   .. py:method:: addWidget(widget)

      Add a widget to the central content widget's layout.



   .. py:method:: name_update(node, name)


   .. py:method:: set_dirty_flag(value)


.. py:class:: InspectorIdleWidget(message, **kwargs)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   Constructs the UI shown when no valid node is selected.


.. py:class:: NodeHeaderWidget(name, node_cls, config_tree = None, **kwargs)

   Bases: :py:obj:`spark.graph_editor.widgets.base.QInput`


   QWidget used for the name of nodes in the SparkGraphEditor's Inspector.


   .. py:attribute:: name_widget


   .. py:attribute:: class_label


   .. py:method:: sizeHint()


   .. py:method:: get_value()

      Returns the widget value.



   .. py:method:: set_value(value)

      Sets the widget value.



.. py:class:: NodeNameWidget(name, **kwargs)

   Bases: :py:obj:`spark.graph_editor.widgets.base.QInput`


   QWidget used for the name of nodes in the SparkGraphEditor's Inspector.


   .. py:method:: sizeHint()


   .. py:method:: get_value()

      Returns the widget value.



   .. py:method:: set_value(value)

      Sets the widget value.



.. py:class:: TreeDisplay(tree)

   Bases: :py:obj:`PySide6.QtWidgets.QPlainTextEdit`


   .. py:method:: sizeHint()


