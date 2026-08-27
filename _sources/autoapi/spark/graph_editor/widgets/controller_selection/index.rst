spark.graph_editor.widgets.controller_selection
===============================================

.. py:module:: spark.graph_editor.widgets.controller_selection


Classes
-------

.. autoapisummary::

   spark.graph_editor.widgets.controller_selection.ControllerCard
   spark.graph_editor.widgets.controller_selection.ControllerChooser
   spark.graph_editor.widgets.controller_selection.RecentCard
   spark.graph_editor.widgets.controller_selection.StartView
   spark.graph_editor.widgets.controller_selection.NewModelDialog


Module Contents
---------------

.. py:class:: ControllerCard(profile, parent = None)

   Bases: :py:obj:`PySide6.QtWidgets.QPushButton`


   Selectable card describing a controller profile.


   .. py:attribute:: profile


   .. py:method:: hasHeightForWidth()


   .. py:method:: heightForWidth(width)


   .. py:method:: sizeHint()


   .. py:method:: minimumSizeHint()


.. py:class:: ControllerChooser(parent = None)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   Card list of every registered controller profile.


   .. py:attribute:: profile_selected


   .. py:method:: resizeEvent(event)


.. py:class:: RecentCard(path, parent = None)

   Bases: :py:obj:`PySide6.QtWidgets.QPushButton`


   One remembered file, shown on the start screen.


   .. py:attribute:: path


.. py:class:: StartView(parent = None)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   Start screen, shown on the canvas while no model is open.


   .. py:attribute:: model_requested


   .. py:attribute:: open_requested


   .. py:attribute:: recent_requested


   .. py:method:: set_recent_files(paths)

      Shows the remembered files, most recent first.



.. py:class:: NewModelDialog(parent = None)

   Bases: :py:obj:`PySide6.QtWidgets.QDialog`


   Controller selection dialog, used by "File > New".


   .. py:attribute:: selected_profile
      :type:  spark.graph_editor.models.controller_profile.ControllerProfile | None
      :value: None



