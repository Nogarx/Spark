spark.graph_editor.widgets.dock_panel
=====================================

.. py:module:: spark.graph_editor.widgets.dock_panel


Attributes
----------

.. autoapisummary::

   spark.graph_editor.widgets.dock_panel.CDockWidget


Classes
-------

.. autoapisummary::

   spark.graph_editor.widgets.dock_panel.QDockPanel


Module Contents
---------------

.. py:data:: CDockWidget

.. py:class:: QDockPanel(name, **kwargs)

   Bases: :py:obj:`CDockWidget`


   Generic dockable panel.


   .. py:method:: minimumSizeHint()


   .. py:method:: setContent(content)

      Replace central widget.



   .. py:method:: content()

      Return the current content widget.



   .. py:method:: addWidget(widget)

      Add a widget to the central content widget's layout.



   .. py:method:: removeWidget(widget)

      Remove widget from the central content widget's layout.



   .. py:method:: clearWidgets()

      Removes all widget from the central content widget's layout.



   .. py:method:: toggleView(dock_manager)


   .. py:method:: showEvent(event)


   .. py:method:: hideEvent(event)


