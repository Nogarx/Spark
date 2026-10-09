#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.graph_editor.models.node_model import NodeModel
    from spark.graph_editor.models.port_model import PortModel
    from spark.graph_editor.models.edge_model import EdgeModel
    from spark.core.payloads import SparkPayload

from spark.core.payloads import payload_types_match

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# NOTE: A generic port, declared by a base payload type (the ports of Concat and Sampler, the sources and sinks of the
# controller), carries the type of what it is connected to. Generic ports linked by a connection, or held by one node
# (see NodeModel.generic_ports_share_type), form a group that carries one type: the type of the ports of a concrete type
# connected to the group. A group connected to none carries its declared type. The types are derived from the
# connections, so they follow every change of the connections, undo and redo included.

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _far_end(edge: EdgeModel, port: PortModel) -> PortModel | None:
    """
        The port at the other end of an edge.
    """
    return edge.target_port if edge.source_port is port else edge.source_port

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _linked_ports(port: PortModel) -> list[PortModel]:
    """
        The generic ports of the node of a port that carry its type, the port included.
    """
    node = port.node
    if node is None or not node.generic_ports_share_type:
        return [port]
    return [other for other in node.get_all_ports() if other.is_generic]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _group(port: PortModel, exclude: NodeModel | None = None) -> list[PortModel]:
    """
        The generic ports that carry one type with a generic port, leaving out the ports of ``exclude``.
    """
    group: list[PortModel] = []
    seen: set[int] = set()
    pending = [port]
    while pending:
        current = pending.pop()
        if id(current) in seen or (exclude is not None and current.node is exclude):
            continue
        seen.add(id(current))
        group.append(current)
        pending.extend(_linked_ports(current))
        for edge in current.edges:
            far = _far_end(edge, current)
            if far is not None and far.is_generic:
                pending.append(far)
    return group

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _group_type(group: list[PortModel]) -> type[SparkPayload] | None:
    """
        The type the ports of a concrete type connected to a group give it, or None if the group is connected to none.
    """
    for port in group:
        for edge in port.edges:
            far = _far_end(edge, port)
            if far is not None and not far.is_generic:
                return far.declared_type
    return None

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def refresh_port_types(port: PortModel) -> None:
    """
        Sets the type of every port of the group of a generic port. A concrete port is left as it is.

        Parameters
        ----------
        port : PortModel
            A port whose connections changed.
    """
    if not port.is_generic:
        return
    group = _group(port)
    carried = _group_type(group)
    for member in group:
        member.port_type = carried if carried is not None else member.declared_type

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def connection_conflicts(source: PortModel, target: PortModel, carrier: PortModel) -> list[EdgeModel]:
    """
        The connections a new connection between two ports replaces, as their types would conflict.

        A connection bringing a type to a generic port gives that type to its node. The connections of the node that
        carry another type are dropped: the new connection sets the type the node is meant to carry. The type of a port
        of a concrete type stands. Between two generic ports, the type of the port the connection is drawn from stands.

        Parameters
        ----------
        source : PortModel
            The output port of the new connection.
        target : PortModel
            The input port of the new connection.
        carrier : PortModel
            The port the connection is drawn from, ``source`` or ``target``.

        Returns
        -------
        list of EdgeModel
            The existing connections to remove before the new one is added.
    """
    dropped = target if carrier is source else source
    if source.is_generic and target.is_generic:
        # The port the connection is dropped on takes the type of the group it is drawn from, if that has one.
        adapting = dropped
        new_type = _group_type(_group(carrier, exclude=adapting.node))
    elif source.is_generic or target.is_generic:
        adapting, fixed = (source, target) if source.is_generic else (target, source)
        new_type = fixed.declared_type
    else:
        return []
    if new_type is None:
        return []
    node = adapting.node
    conflicts: list[EdgeModel] = []
    for port in _linked_ports(adapting):
        for edge in port.edges:
            far = _far_end(edge, port)
            if far is None or far.node is node:
                continue
            far_type = _group_type(_group(far, exclude=node)) if far.is_generic else far.declared_type
            if far_type is not None and not payload_types_match(new_type, far_type) and edge not in conflicts:
                conflicts.append(edge)
    return conflicts

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
