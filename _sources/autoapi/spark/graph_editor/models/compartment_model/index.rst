spark.graph_editor.models.compartment_model
===========================================

.. py:module:: spark.graph_editor.models.compartment_model


Classes
-------

.. autoapisummary::

   spark.graph_editor.models.compartment_model.CompartmentModel


Module Contents
---------------

.. py:class:: CompartmentModel(name, parent=None)

   Bases: :py:obj:`spark.graph_editor.models.base_model.BaseModel`


   Base class for all graph models.


   .. py:attribute:: port_added


   .. py:attribute:: port_removed


   .. py:attribute:: id
      :value: ''



   .. py:attribute:: name


   .. py:attribute:: ports
      :type:  list[spark.graph_editor.models.port_model.PortModel]
      :value: []



   .. py:attribute:: node
      :type:  spark.graph_editor.models.node_model.NodeModel | None
      :value: None



   .. py:method:: add_port(port)


   .. py:method:: remove_port(port)


   .. py:method:: get_port(name)


   .. py:method:: to_dict()


   .. py:method:: from_dict(data)
      :classmethod:



