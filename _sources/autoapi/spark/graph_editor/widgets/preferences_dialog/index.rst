spark.graph_editor.widgets.preferences_dialog
=============================================

.. py:module:: spark.graph_editor.widgets.preferences_dialog


Attributes
----------

.. autoapisummary::

   spark.graph_editor.widgets.preferences_dialog.SECTIONS
   spark.graph_editor.widgets.preferences_dialog.SYNONYMS
   spark.graph_editor.widgets.preferences_dialog.HINTS


Classes
-------

.. autoapisummary::

   spark.graph_editor.widgets.preferences_dialog.ColorButton
   spark.graph_editor.widgets.preferences_dialog.NumberListEdit
   spark.graph_editor.widgets.preferences_dialog.CssSizeEdit
   spark.graph_editor.widgets.preferences_dialog.PreferencesDialog


Module Contents
---------------

.. py:data:: SECTIONS
   :type:  tuple[tuple[str, str, tuple[str, ...]], ...]
   :value: (('Canvas', 'Background, grid and the area the graph lives in.', ('graph', 'viewer')), ('Nodes',...


.. py:data:: SYNONYMS
   :type:  dict[str, str]

.. py:data:: HINTS
   :type:  dict[str, str]

.. py:class:: ColorButton(color, is_hex = False, parent = None)

   Bases: :py:obj:`PySide6.QtWidgets.QPushButton`


   Swatch that opens a colour picker.


   .. py:attribute:: is_hex
      :value: False



   .. py:method:: get_value()


.. py:class:: NumberListEdit(values, parent = None)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   Editor for a fixed list of numbers, such as margins or a scene rectangle.


   .. py:attribute:: LABELS


   .. py:method:: get_value()


.. py:class:: CssSizeEdit(value, parent = None)

   Bases: :py:obj:`PySide6.QtWidgets.QSpinBox`


   Editor for the "12px" strings used by the stylesheet.


   .. py:attribute:: wheelEvent


   .. py:method:: get_value()


.. py:class:: PreferencesDialog(parent = None)

   Bases: :py:obj:`PySide6.QtWidgets.QDialog`


   Editor for the presentation of the graph editor.


   .. py:attribute:: applied


   .. py:attribute:: settings


   .. py:attribute:: config_data
      :type:  dict[str, Any]


   .. py:attribute:: widgets_map
      :type:  dict[tuple[str, ...], PySide6.QtWidgets.QWidget]


   .. py:attribute:: search


   .. py:attribute:: sidebar


   .. py:attribute:: pages


   .. py:attribute:: path_label


