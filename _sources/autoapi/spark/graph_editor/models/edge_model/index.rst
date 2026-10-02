spark.graph_editor.models.edge_model
====================================

.. py:module:: spark.graph_editor.models.edge_model


Classes
-------

.. autoapisummary::

   spark.graph_editor.models.edge_model.EdgeModel


Module Contents
---------------

.. py:class:: EdgeModel(source_port, target_port, parent=None)

   Bases: :py:obj:`spark.graph_editor.models.base_model.BaseModel`


   Base class for all graph models.


   .. py:attribute:: waypoints_changed


   .. py:attribute:: deleted


   .. py:attribute:: id
      :value: ''



   .. py:attribute:: source_port


   .. py:attribute:: target_port


   .. py:property:: waypoints
      :type: list[tuple]



   .. py:method:: validate_connection(src_port, dst_port)
      :classmethod:


      Returns True if a connection between two ports is allowed.



   .. py:method:: delete()


   .. py:method:: to_dict()


   .. py:method:: from_dict(data, all_ports)
      :classmethod:



