spark.graph_editor.models.port_types
====================================

.. py:module:: spark.graph_editor.models.port_types


Functions
---------

.. autoapisummary::

   spark.graph_editor.models.port_types.refresh_port_types
   spark.graph_editor.models.port_types.connection_conflicts


Module Contents
---------------

.. py:function:: refresh_port_types(port)

   Sets the type of every port of the group of a generic port. A concrete port is left as it is.

   :param port: A port whose connections changed.
   :type port: PortModel


.. py:function:: connection_conflicts(source, target, carrier)

   The connections a new connection between two ports replaces, as their types would conflict.

   A connection bringing a type to a generic port gives that type to its node. The connections of the node that
   carry another type are dropped: the new connection sets the type the node is meant to carry. The type of a port
   of a concrete type stands. Between two generic ports, the type of the port the connection is drawn from stands.

   :param source: The output port of the new connection.
   :type source: PortModel
   :param target: The input port of the new connection.
   :type target: PortModel
   :param carrier: The port the connection is drawn from, ``source`` or ``target``.
   :type carrier: PortModel

   :returns: The existing connections to remove before the new one is added.
   :rtype: list of EdgeModel


