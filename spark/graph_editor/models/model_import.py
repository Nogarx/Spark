#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.graph_editor.models.graph_model import GraphModel
    from spark.graph_editor.models.controller_profile import ControllerProfile

import copy
import logging
import typing as tp
import dataclasses as dc

from spark.core.specs import ModuleSpecs, PortMap
from spark.graph_editor.models.node_model import NodeModel, SourceNodeModel, SinkNodeModel, SelfPropertyNodeModel
from spark.graph_editor.models.port_model import PortModel
from spark.graph_editor.models.edge_model import EdgeModel
from spark.graph_editor.models.node_factory import NODE_REGISTRY
from spark.graph_editor.models import graph_layout

logger = logging.getLogger('spark')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# Importing a model is not the same thing as placing one, and the controller profile decides which of the
# two happens: a Neuron placed in a Brain is a single node, while the same Neuron opened in Neuron mode is
# the graph itself and is expanded into the components it is made of. This module implements the second
# case. It is additive: a model can be imported any number of times, and every import lands on free canvas.

_CALL_ORIGIN = '__call__'
_SELF_ORIGIN = '__self__'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@dc.dataclass
class ImportedGraph:
    """
        Result of expanding a controller configuration.
    """
    nodes: list[NodeModel] = dc.field(default_factory=list)
    edges: list[EdgeModel] = dc.field(default_factory=list)
    warnings: list[str] = dc.field(default_factory=list)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def _unique_name(name: str, taken: set[str]) -> str:
    """
        First free variation of a name.
    """
    candidate = name
    index = 0
    while candidate in taken:
        candidate = f'{name}{index}'
        index += 1
    taken.add(candidate)
    return candidate

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _as_position(pos: tp.Any) -> tuple[float, float] | None:
    if isinstance(pos, (list, tuple)) and len(pos) == 2:
        try:
            return (float(pos[0]), float(pos[1]))
        except (TypeError, ValueError):
            return None
    return None

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _controller_metadata(config: tp.Any) -> dict:
    """
        Editor metadata stored at the controller level, keyed by node name.
        """
    metadata = getattr(config, '__graph_editor_metadata__', None)
    return metadata if isinstance(metadata, dict) else {}

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _stored_position(config: tp.Any) -> tuple[float, float] | None:
    """
        Position saved with a module, when the model carries editor metadata.
        """
    return _as_position(_controller_metadata(config).get('pos', None))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _iter_port_maps(spec: ModuleSpecs) -> tp.Iterator[tuple[str, PortMap, bool]]:
    """
        Every incoming connection of a module, as (target port name, port map, targets a property).
    """
    for port_name, port_maps in (spec.inputs or {}).items():
        for port_map in port_maps or []:
            yield (port_name, port_map, False)
    for property_name, port_maps in (spec.effects or {}).items():
        for port_map in port_maps or []:
            yield (property_name, port_map, True)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def expand_controller_config(
        config: tp.Any,
        graph_model: GraphModel,
        profile: ControllerProfile | None = None,
        layout: dict[str, tuple[float, float]] | None = None,
    ) -> ImportedGraph:
    """
        Expands a controller configuration into the nodes and edges of its modules.

        Input:
            config: SparkConfig, configuration of the controller to expand.
            graph_model: GraphModel, graph the result is destined to. Only read, never modified.
            profile: ControllerProfile, active profile. Used to type the controller properties.
            layout: dict[str, tuple[float, float]], positions by node name. A configuration cannot carry
                the layout by itself (see session_io).

        Returns:
            ImportedGraph, the nodes and edges to add, already positioned.
    """
    specs: tuple[ModuleSpecs, ...] = tuple(getattr(config, 'modules_specs', ()) or ())
    if not specs:
        raise ValueError(f'"{type(config).__name__}" does not declare any module, there is nothing to import.')

    result = ImportedGraph()
    layout = layout or {}
    taken_names = {node.name for node in graph_model.nodes}
    # Modules are addressed by their name inside the configuration, which may be renamed on collision.
    module_nodes: dict[str, NodeModel] = {}
    call_nodes: dict[str, NodeModel] = {}
    self_nodes: dict[str, NodeModel] = {}
    positions: dict[str, tuple[float, float]] = {}

    # Modules.
    for spec in specs:
        node_cls = NODE_REGISTRY.get(spec.module_cls)
        if node_cls is None:
            result.warnings.append(f'No node is available for "{spec.module_cls.__name__}", module "{spec.name}" was skipped.')
            continue
        node = node_cls(name=_unique_name(spec.name, taken_names))
        # The configuration of the model is the configuration of the node.
        node.config = copy.deepcopy(spec.config)
        module_nodes[spec.name] = node
        result.nodes.append(node)
        stored = layout.get(spec.name, None) or _stored_position(spec.config)
        if stored is not None:
            positions[node.id] = stored

    # Controller level inputs and properties, which the modules refer to as "__call__" and "__self__".
    property_specs = profile.self_property_specs() if profile is not None else {}
    metadata = _controller_metadata(config)

    def _register(node: NodeModel, original_name: str) -> None:
        result.nodes.append(node)
        pos = layout.get(original_name, None)
        if pos is None:
            entry = metadata.get(original_name, None)
            pos = _as_position(entry.get('pos', None)) if isinstance(entry, dict) else None
        if pos is not None:
            positions[node.id] = pos

    for spec in specs:
        for _, port_map, _ in _iter_port_maps(spec):
            if port_map.origin == _CALL_ORIGIN and port_map.port not in call_nodes:
                node = SourceNodeModel(name=_unique_name(port_map.port, taken_names))
                call_nodes[port_map.port] = node
                _register(node, port_map.port)
            elif port_map.origin == _SELF_ORIGIN and port_map.port not in self_nodes:
                port_spec = property_specs.get(port_map.port, None)
                node = SelfPropertyNodeModel(
                    name=_unique_name(port_map.port, taken_names),
                    payload_type=getattr(port_spec, 'payload_type', None),
                )
                self_nodes[port_map.port] = node
                _register(node, port_map.port)

    def _origin_port(port_map: PortMap) -> PortModel | None:
        if port_map.origin == _CALL_ORIGIN:
            node = call_nodes.get(port_map.port, None)
            return node.value_port if node is not None else None
        if port_map.origin == _SELF_ORIGIN:
            node = self_nodes.get(port_map.port, None)
            return node.value_port if node is not None else None
        node = module_nodes.get(port_map.origin, None)
        if node is None:
            return None
        return node.get_port_by_name(port_map.port, is_input=False)

    # Connections.
    for spec in specs:
        target_node = module_nodes.get(spec.name, None)
        if target_node is None:
            continue
        for port_name, port_map, is_effect in _iter_port_maps(spec):
            target_port = target_node.get_port_by_name(port_name, is_input=True)
            if target_port is None:
                kind = 'property' if is_effect else 'input'
                result.warnings.append(f'Module "{spec.name}" has no {kind} port "{port_name}", the connection was skipped.')
                continue
            source_port = _origin_port(port_map)
            if source_port is None:
                result.warnings.append(
                    f'Unable to resolve "{port_map.origin}.{port_map.port}" for "{spec.name}.{port_name}", the connection was skipped.'
                )
                continue
            result.edges.append(EdgeModel(source_port, target_port))

    # Controller outputs, exposed as sinks.
    for spec in specs:
        source_node = module_nodes.get(spec.name, None)
        if source_node is None:
            continue
        for output_name, port_name in (spec.outputs or {}).items():
            source_port = source_node.get_port_by_name(port_name, is_input=False)
            if source_port is None:
                result.warnings.append(f'Module "{spec.name}" has no output port "{port_name}", the output "{output_name}" was skipped.')
                continue
            sink = SinkNodeModel(name=_unique_name(output_name, taken_names))
            _register(sink, output_name)
            result.edges.append(EdgeModel(source_port, sink.value_port))

    _place_nodes(result, graph_model, positions)
    return result

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _place_nodes(result: ImportedGraph, graph_model: GraphModel, stored: dict[str, tuple[float, float]]) -> None:
    """
        Positions the imported nodes.

        The rules, in order:
            1. A node saved with a position keeps it, untouched.
            2. The rest is laid out by dependency depth, under whatever was saved.
            3. If the canvas already holds nodes, the whole block is moved below them.
    """
    sizes = {node.id: graph_layout.estimate_node_size(node) for node in result.nodes}
    placed = [node for node in result.nodes if node.id in stored]
    missing = [node for node in result.nodes if node.id not in stored]

    if missing:
        edges = [
            (edge.source_port.node.id, edge.target_port.node.id)
            for edge in result.edges
            if edge.source_port is not None and edge.target_port is not None
            and edge.source_port.node is not None and edge.target_port.node is not None
        ]
        # The layout runs over the whole model, so connections to already placed nodes still order it.
        computed = graph_layout.layered_layout([node.id for node in result.nodes], edges, sizes)
        for node in missing:
            stored[node.id] = computed.get(node.id, (0.0, 0.0))

    for node in result.nodes:
        node.pos = stored.get(node.id, (0.0, 0.0))

    # Keep the computed nodes clear of the ones that came with a position of their own.
    if placed and missing:
        dx, dy = graph_layout.offset_below(placed, missing)
        for node in missing:
            node.pos = (node.pos[0] + dx, node.pos[1] + dy)

    if graph_model.nodes:
        # The canvas is occupied: move the block as a whole, preserving its internal geometry.
        dx, dy = graph_layout.offset_below(graph_model.nodes, result.nodes)
    elif placed:
        # Saved positions are absolute, an empty canvas takes them as they are.
        return
    else:
        dx, dy = graph_layout.offset_below((), result.nodes)
    if dx or dy:
        for node in result.nodes:
            node.pos = (node.pos[0] + dx, node.pos[1] + dy)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
