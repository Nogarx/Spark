spark.graph_editor.models.node_model
====================================

.. py:module:: spark.graph_editor.models.node_model


Attributes
----------

.. autoapisummary::

   spark.graph_editor.models.node_model.logger


Classes
-------

.. autoapisummary::

   spark.graph_editor.models.node_model.NodeModel
   spark.graph_editor.models.node_model.SourceNodeModel
   spark.graph_editor.models.node_model.SinkNodeModel
   spark.graph_editor.models.node_model.SelfPropertyNodeModel
   spark.graph_editor.models.node_model.InterfaceNodeModel
   spark.graph_editor.models.node_model.ControllerNodeModel
   spark.graph_editor.models.node_model.ComponentNodeModel


Module Contents
---------------

.. py:data:: logger

.. py:class:: NodeModel(name = None, type_name = 'BaseObject', pos=(0, 0), parent=None)

   Bases: :py:obj:`spark.graph_editor.models.base_model.BaseModel`


   Base class for all graph models.


   .. py:attribute:: position_changed


   .. py:attribute:: name_changed


   .. py:attribute:: type_changed


   .. py:attribute:: selected_changed


   .. py:attribute:: deleted


   .. py:attribute:: id
      :value: ''



   .. py:attribute:: config
      :type:  spark.core.config.SparkConfig | None
      :value: None



   .. py:attribute:: compartments
      :type:  list[spark.graph_editor.models.compartment_model.CompartmentModel]
      :value: []



   .. py:attribute:: call_section


   .. py:attribute:: props_section


   .. py:property:: name
      :type: str



   .. py:property:: type_name
      :type: str



   .. py:property:: pos
      :type: tuple[float, float]



   .. py:property:: is_selected
      :type: bool



   .. py:method:: add_compartment(compartment)


   .. py:method:: get_port_by_name(name, is_input = None)


   .. py:method:: get_all_ports()


   .. py:method:: delete()


   .. py:method:: to_dict()


   .. py:method:: from_dict(data)
      :classmethod:



.. py:class:: SourceNodeModel(name = None, type_name = 'Source Node', pos=(0, 0), parent=None)

   Bases: :py:obj:`NodeModel`


   Node standing for one input of the controller.


   .. py:attribute:: value_port


   .. py:method:: on_port_connected(edge)


.. py:class:: SinkNodeModel(name = None, type_name = 'Sink Node', pos=(0, 0), parent=None)

   Bases: :py:obj:`NodeModel`


   Node standing for one output of the controller.


   .. py:attribute:: value_port


   .. py:method:: on_port_connected(edge)


.. py:class:: SelfPropertyNodeModel(name = None, type_name = 'Controller Property', pos=(0, 0), parent=None, payload_type = None)

   Bases: :py:obj:`NodeModel`


   Node standing for one property the controller exposes to its modules.


   .. py:attribute:: value_port


.. py:class:: InterfaceNodeModel(name = None, type_name = None, pos=(0, 0), parent=None)

   Bases: :py:obj:`NodeModel`


   Node model of an Interface.


   .. py:attribute:: config


.. py:class:: ControllerNodeModel(name = None, type_name = None, pos=(0, 0), parent=None)

   Bases: :py:obj:`NodeModel`


   Node model of a nested controller (a Neuron placed inside a Brain).


   .. py:attribute:: config


.. py:class:: ComponentNodeModel(name = None, type_name = None, pos=(0, 0), parent=None)

   Bases: :py:obj:`NodeModel`


   Node model of a Component.


   .. py:attribute:: config


