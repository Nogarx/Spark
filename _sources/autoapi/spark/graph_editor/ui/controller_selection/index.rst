spark.graph_editor.ui.controller_selection
==========================================

.. py:module:: spark.graph_editor.ui.controller_selection


Attributes
----------

.. autoapisummary::

   spark.graph_editor.ui.controller_selection.CDockWidget


Classes
-------

.. autoapisummary::

   spark.graph_editor.ui.controller_selection.ControllerSelectorDialog


Module Contents
---------------

.. py:data:: CDockWidget

.. py:class:: ControllerSelectorDialog(is_init_call, parent=None)

   Bases: :py:obj:`PySide6.QtWidgets.QDialog`


   .. py:attribute:: selected_choice
      :value: None



   .. py:attribute:: btn_brain


   .. py:attribute:: btn_neuron


   .. py:attribute:: info_label


   .. py:method:: confirm_selection(controller_name, controller_type)

      Shows the confirmation popup before accepting the dialog.



