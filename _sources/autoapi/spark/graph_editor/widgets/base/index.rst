spark.graph_editor.widgets.base
===============================

.. py:module:: spark.graph_editor.widgets.base


Classes
-------

.. autoapisummary::

   spark.graph_editor.widgets.base.SparkQWidgetMeta
   spark.graph_editor.widgets.base.SparkQWidget


Module Contents
---------------

.. py:class:: SparkQWidgetMeta

   Bases: :py:obj:`type`\ (\ :py:obj:`QtWidgets.QWidget`\ ), :py:obj:`abc.ABCMeta`


   Metaclass for defining Abstract Base Classes (ABCs).

   Use this metaclass to create an ABC.  An ABC can be subclassed
   directly, and then acts as a mix-in class.  You can also register
   unrelated concrete classes (even built-in classes) and unrelated
   ABCs as 'virtual subclasses' -- these and their descendants will
   be considered subclasses of the registering ABC by the built-in
   issubclass() function, but the registering ABC won't show up in
   their MRO (Method Resolution Order) nor will method
   implementations defined by the registering ABC be callable (not
   even via super()).


.. py:class:: SparkQWidget(*args, **kwargs)

   Bases: :py:obj:`Qt.QtWidgets.QFrame`, :py:obj:`abc.ABC`


   Base QWidget class for the graph editor attributes.


   .. py:attribute:: on_update


   .. py:method:: get_value()
      :abstractmethod:


      Returns the widget value.



