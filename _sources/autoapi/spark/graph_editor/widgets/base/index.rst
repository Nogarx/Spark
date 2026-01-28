spark.graph_editor.widgets.base
===============================

.. py:module:: spark.graph_editor.widgets.base


Classes
-------

.. autoapisummary::

   spark.graph_editor.widgets.base.QInput
   spark.graph_editor.widgets.base.QField
   spark.graph_editor.widgets.base.QMissing


Module Contents
---------------

.. py:class:: QInput(*args, value_options = None, **kwargs)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   QWidget interface class for the graph editor inputs.

   QInput streamlines updates by setting a common set/get interface.


   .. py:attribute:: on_update


   .. py:method:: get_value()
      :abstractmethod:


      Returns the widget value.



   .. py:method:: set_value(value)
      :abstractmethod:


      Sets the widget value.



.. py:class:: QField(attr_label, attr_value, attr_widget, value_options = None, **kwargs)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   Base QWidget class for the graph editor fields.


   .. py:attribute:: on_field_update


   .. py:attribute:: attr_label


   .. py:attribute:: attr_widget


   .. py:method:: get_value()

      Get the input widget value.



   .. py:method:: set_value(value)

      Sets the input widget value.



.. py:class:: QMissing(*args, **kwargs)

   Bases: :py:obj:`QInput`


   Custom QWidget used for int fields in the SparkGraphEditor's Inspector.


   .. py:method:: get_value()

      Returns the widget value.



