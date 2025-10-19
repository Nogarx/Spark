spark.graph_editor.widgets.combobox
===================================

.. py:module:: spark.graph_editor.widgets.combobox


Classes
-------

.. autoapisummary::

   spark.graph_editor.widgets.combobox.QComboBoxEdit
   spark.graph_editor.widgets.combobox.QDtype
   spark.graph_editor.widgets.combobox.QBool
   spark.graph_editor.widgets.combobox.QGenericComboBox


Module Contents
---------------

.. py:class:: QComboBoxEdit(initial_value, options_list, parent = None)

   Bases: :py:obj:`PySide6.QtWidgets.QComboBox`


   A QComboBox widget for selecting a dtype from a predefined list.


   .. py:attribute:: options_list


   .. py:method:: set_value(value)

      Sets the current object selection.



   .. py:method:: get_value()

      Returns the currently selected object.



   .. py:method:: setEnabled(value)


.. py:class:: QDtype(initial_value, values_options, parent = None)

   Bases: :py:obj:`spark.graph_editor.widgets.base.SparkQWidget`


   Custom QWidget used for dtypes fields in the SparkGraphEditor's Inspector.


   .. py:method:: get_value()

      Returns the widget value.



   .. py:method:: set_value(value)

      Returns the widget value.



.. py:class:: QBool(initial_value = True, parent = None)

   Bases: :py:obj:`spark.graph_editor.widgets.base.SparkQWidget`


   Custom QWidget used for bool fields in the SparkGraphEditor's Inspector.


   .. py:method:: get_value()

      Returns the widget value.



   .. py:method:: set_value(value)

      Returns the widget value.



.. py:class:: QGenericComboBox(initial_value, values_options, parent = None)

   Bases: :py:obj:`spark.graph_editor.widgets.base.SparkQWidget`


   Custom QWidget used for arbitrary selectable fields fields in the SparkGraphEditor's Inspector.


   .. py:method:: get_value()

      Returns the widget value.



   .. py:method:: set_value(value)

      Returns the widget value.



