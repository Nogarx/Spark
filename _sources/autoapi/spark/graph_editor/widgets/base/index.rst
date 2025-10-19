spark.graph_editor.widgets.base
===============================

.. py:module:: spark.graph_editor.widgets.base


Classes
-------

.. autoapisummary::

   spark.graph_editor.widgets.base.SparkQWidget
   spark.graph_editor.widgets.base.QField
   spark.graph_editor.widgets.base.QMissing


Module Contents
---------------

.. py:class:: SparkQWidget(*args, **kwargs)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   Base QWidget class for the graph editor attributes.


   .. py:attribute:: on_update


   .. py:method:: get_value()
      :abstractmethod:


      Returns the widget value.



   .. py:method:: set_value(value)
      :abstractmethod:


      Returns the widget value.



.. py:class:: QField(attr_label, attr_widget, warning_value = False, inheritance_box = False, inheritance_interactable = True, inheritance_value = False, parent = None, **kwargs)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   Base QWidget class for the graph editor attributes.


   .. py:attribute:: inheritance_toggled


   .. py:attribute:: field_updated


   .. py:attribute:: warning_flag


   .. py:attribute:: label


   .. py:attribute:: attr_widget


   .. py:method:: on_inheritance_toggled(state)


   .. py:method:: on_field_update(value)


.. py:class:: QMissing(parent = None)

   Bases: :py:obj:`SparkQWidget`


   Custom QWidget used for int fields in the SparkGraphEditor's Inspector.


   .. py:method:: get_value()

      Returns the widget value.



