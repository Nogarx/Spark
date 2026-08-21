#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.graph_editor.models.node_model import NodeModel

import typing as tp
from spark.graph_editor.styles.manager import STYLES

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# NOTE: A Spark graph is a dataflow graph, not an arbitrary network: it has a direction, and the framework
# itself groups the modules in execution layers. A layered (left to right) placement therefore reproduces the
# way the model is read, which a force directed layout cannot do: a spring layout only minimizes edge length,
# so it scatters a pipeline into a blob and gives a different result on every run. Here the horizontal axis is
# the dependency depth and the vertical axis only separates modules that are independent of each other.

_ROW_HEIGHT = 20.0
_SECTION_HEADER = 18.0
_DIVIDER = 16.0

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def estimate_node_size(node: NodeModel) -> tuple[float, float]:
    """
        Approximates the rendered size of a node from its ports.

        NOTE: The real geometry belongs to the view (NodeItem), which does not exist yet while a model is
        being imported. The estimate follows the same layout rules so that the placement leaves enough room.
    """
    width = float(STYLES.get_val('node', 'width', default=180))
    height = float(STYLES.get_val('node', 'header_height', default=45))

    def _rows(ports) -> int:
        # An input and an output sharing a name are drawn on the same row (see NodeItem._group_ports).
        rows, groups = 0, {}
        for port in ports:
            slot = 'in' if port.is_input else 'out'
            group = groups.get(port.name)
            if group is None or slot in group:
                groups[port.name] = {slot}
                rows += 1
            else:
                group.add(slot)
        return rows

    call_ports = list(node.call_section.ports)
    prop_ports = list(node.props_section.ports)
    if call_ports:
        height += _SECTION_HEADER + _rows(call_ports) * _ROW_HEIGHT + 10.0
        if any(port.is_optional for port in call_ports):
            height += _DIVIDER
    if prop_ports:
        height += _SECTION_HEADER + _rows(prop_ports) * _ROW_HEIGHT
    return (width, height + 10.0)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def _break_cycles(keys: list[str], successors: dict[str, list[str]]) -> set[tuple[str, str]]:
    """
        Finds the back edges of the graph with a depth first search.

        Recurrent models are legal in Spark, so the dependency graph is not acyclic. Back edges are ignored
        while layering, which places a recurrent module after the modules it feeds forward into.
    """
    back_edges: set[tuple[str, str]] = set()
    state = {key: 0 for key in keys}  # 0 unvisited, 1 in progress, 2 done

    def _visit(node: str) -> None:
        state[node] = 1
        for target in successors.get(node, []):
            if state.get(target, 2) == 1:
                back_edges.add((node, target))
            elif state.get(target, 2) == 0:
                _visit(target)
        state[node] = 2

    for key in keys:
        if state[key] == 0:
            _visit(key)
    return back_edges

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _assign_layers(keys: list[str], edges: list[tuple[str, str]]) -> dict[str, int]:
    """
        Longest path layering: every node sits one layer after its deepest dependency.
    """
    successors: dict[str, list[str]] = {key: [] for key in keys}
    for source, target in edges:
        if source in successors and target in successors and source != target:
            successors[source].append(target)
    back_edges = _break_cycles(keys, successors)
    predecessors: dict[str, list[str]] = {key: [] for key in keys}
    for source, target in edges:
        if source == target or (source, target) in back_edges:
            continue
        if source in predecessors and target in predecessors:
            predecessors[target].append(source)

    layer: dict[str, int] = {}

    def _depth(node: str) -> int:
        if node in layer:
            return layer[node]
        # Guards against any residual cycle.
        layer[node] = 0
        parents = predecessors.get(node, [])
        layer[node] = max((_depth(parent) + 1 for parent in parents), default=0)
        return layer[node]

    for key in keys:
        _depth(key)
    return layer

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _order_layers(
        layers: dict[int, list[str]],
        edges: list[tuple[str, str]],
        sweeps: int = 4,
    ) -> None:
    """
        Reduces edge crossings by repeatedly sorting each layer on the barycenter of its neighbours.
    """
    predecessors: dict[str, list[str]] = {}
    successors: dict[str, list[str]] = {}
    for source, target in edges:
        predecessors.setdefault(target, []).append(source)
        successors.setdefault(source, []).append(target)

    for sweep in range(sweeps):
        indices = {key: i for layer in layers.values() for i, key in enumerate(layer)}
        forward = sweep % 2 == 0
        neighbours = predecessors if forward else successors
        for index in sorted(layers.keys(), reverse=not forward):
            layer = layers[index]
            # NOTE: The current order has to be captured up front. CPython empties a list while sorting it,
            # so the key function cannot look the element up in the very list being sorted.
            current = {key: position for position, key in enumerate(layer)}
            def _barycenter(key: str, current=current) -> float:
                related = [indices[n] for n in neighbours.get(key, []) if n in indices]
                return sum(related) / len(related) if related else float(current[key])
            layer.sort(key=_barycenter)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def layered_layout(
        keys: list[str],
        edges: list[tuple[str, str]],
        sizes: dict[str, tuple[float, float]],
        h_gap: float | None = None,
        v_gap: float | None = None,
    ) -> dict[str, tuple[float, float]]:
    """
        Places a dataflow graph left to right, by dependency depth.

        Input:
            keys: list[str], node identifiers, in a stable order.
            edges: list[tuple[str, str]], directed (source, target) dependencies.
            sizes: dict[str, tuple[float, float]], rendered size of every node.
            h_gap: float, free space between two columns.
            v_gap: float, free space between two nodes of the same column.

        Returns:
            dict[str, tuple[float, float]], top-left position of every node.
    """
    if not keys:
        return {}
    h_gap = float(STYLES.get_val('graph', 'layout_h_gap', default=90) if h_gap is None else h_gap)
    v_gap = float(STYLES.get_val('graph', 'layout_v_gap', default=40) if v_gap is None else v_gap)

    layer_of = _assign_layers(keys, edges)
    layers: dict[int, list[str]] = {}
    for key in keys:
        layers.setdefault(layer_of[key], []).append(key)
    _order_layers(layers, edges)

    positions: dict[str, tuple[float, float]] = {}
    x = 0.0
    for index in sorted(layers.keys()):
        column = layers[index]
        column_width = max(sizes.get(key, (180.0, 100.0))[0] for key in column)
        total_height = sum(sizes.get(key, (180.0, 100.0))[1] for key in column) + v_gap * (len(column) - 1)
        y = -total_height / 2.0
        for key in column:
            width, height = sizes.get(key, (180.0, 100.0))
            # Nodes are centered inside their column so that columns of different widths stay aligned.
            positions[key] = (x + (column_width - width) / 2.0, y)
            y += height + v_gap
        x += column_width + h_gap
    return positions

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def bounding_box(nodes: tp.Iterable[NodeModel]) -> tuple[float, float, float, float] | None:
    """
        Bounding box (left, top, right, bottom) of a collection of nodes.
    """
    boxes = []
    for node in nodes:
        width, height = estimate_node_size(node)
        x, y = node.pos
        boxes.append((x, y, x + width, y + height))
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def offset_below(existing: tp.Iterable[NodeModel], incoming: tp.Iterable[NodeModel], margin: float | None = None) -> tuple[float, float]:
    """
        Translation that drops a set of new nodes under everything already on the canvas.

        Importing is additive, so a second import must not land on top of the first one. The relative
        placement of the incoming nodes is preserved, only the block as a whole is moved.
    """
    margin = float(STYLES.get_val('graph', 'layout_import_margin', default=120) if margin is None else margin)
    incoming = list(incoming)
    occupied = bounding_box(existing)
    block = bounding_box(incoming)
    if block is None:
        return (0.0, 0.0)
    if occupied is None:
        # First import: center the block on the origin of the scene.
        return (-(block[0] + block[2]) / 2.0, -(block[1] + block[3]) / 2.0)
    return (occupied[0] - block[0], occupied[3] + margin - block[1])

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
