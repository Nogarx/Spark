spark.graph_editor.models.port_model
====================================

.. py:module:: spark.graph_editor.models.port_model


Classes
-------

.. autoapisummary::

   spark.graph_editor.models.port_model.PortModel


Module Contents
---------------

.. py:class:: PortModel(name, is_input, port_type, is_optional = False, multi_connection = False, parent=None)

   Bases: :py:obj:`spark.graph_editor.models.base_model.BaseModel`


   Base class for all graph models.


   .. py:attribute:: connected


   .. py:attribute:: disconnected


   .. py:attribute:: type_changed


   .. py:attribute:: id
      :value: ''



   .. py:attribute:: name


   .. py:attribute:: is_input


   .. py:attribute:: is_optional
      :value: False



   .. py:attribute:: multi_connection
      :value: False



   .. py:attribute:: declared_type


   .. py:attribute:: node
      :type:  spark.graph_editor.models.node_model.NodeModel | None
      :value: None



   .. py:attribute:: compartment
      :type:  spark.graph_editor.models.compartment_model.CompartmentModel | None
      :value: None



   .. py:attribute:: edges
      :type:  list[spark.graph_editor.models.edge_model.EdgeModel]
      :value: []



   .. py:property:: port_type
      :type: type[spark.core.payloads.SparkPayload]



   .. py:property:: is_generic
      :type: bool


      True if the port is declared by a base payload type, and carries the type of what it is connected to.


   .. py:method:: add_edge(edge)


   .. py:method:: remove_edge(edge)


   .. py:method:: get_connected_nodes()


   .. py:method:: to_dict()


   .. py:method:: from_dict(data)
      :classmethod:



