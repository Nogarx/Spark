spark.graph_editor.widgets.dict
===============================

.. py:module:: spark.graph_editor.widgets.dict


Classes
-------

.. autoapisummary::

   spark.graph_editor.widgets.dict.QKeyValueRow
   spark.graph_editor.widgets.dict.QDict


Module Contents
---------------

.. py:class:: QKeyValueRow(key, value, on_delete, parent = None)

   Bases: :py:obj:`Qt.QtWidgets.QWidget`


   A row widget for a key-value pair.


   .. py:attribute:: editingFinished


   .. py:attribute:: key_edit


   .. py:attribute:: value_edit


   .. py:attribute:: delete_button


   .. py:method:: on_update()


   .. py:method:: get_pair()


.. py:class:: QDict(initial_dict = None, parent = None)

   Bases: :py:obj:`spark.graph_editor.widgets.base.SparkQWidget`


   A dynamical list of key-value pairs.


   .. py:attribute:: rows
      :type:  list[QKeyValueRow]
      :value: []



   .. py:attribute:: rows_layout


   .. py:attribute:: add_button


   .. py:method:: add_row(key, value)

      Creates a new QKeyValueRow and adds it to the layout.



   .. py:method:: remove_row(row_to_delete)

      Removes a specific row from the list and the layout.



   .. py:method:: to_dict()

      Converts the current state of the editor to a dictionary.



   .. py:method:: get_value()

      Returns the widget value.



