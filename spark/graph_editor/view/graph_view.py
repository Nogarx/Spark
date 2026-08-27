#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import logging
import typing as tp

from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QFrame, QGraphicsPathItem, QGraphicsSceneMouseEvent
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPainter, QBrush, QColor, QPen, QWheelEvent, QMouseEvent, QPainterPath, QKeyEvent, QContextMenuEvent
from shiboken6 import isValid

from spark.graph_editor.styles.manager import STYLES
from spark.graph_editor.view.node_item import PortItem
from spark.graph_editor.view.node_item import NodeItem, PropertyRowItem
from spark.graph_editor.view.pipe_item import PipeItem, SegmentGizmo, PipeRouteContext
from spark.graph_editor.models import GraphModel, PortModel, EdgeModel, NodeModel
from spark.graph_editor.models.node_factory import NodeFactory
from spark.graph_editor.models.model_import import expand_controller_config
from spark.graph_editor.widgets.console_view import MessageLevel
import spark.core.utils as utils
from spark.graph_editor.view.graph_context_menu_view import GraphContextMenu
from spark.graph_editor.models.node_factory import NODE_REGISTRY
from spark.graph_editor.commands.graph_commands import (
    AddNodeCommand, AddEdgeCommand, RemoveNodeCommand, RemoveEdgeCommand, MoveNodeCommand, ChangeEdgeWaypointsCommand
)
from spark.graph_editor.view.graph_context_menu_view import ActionData, ContextMenuCommand

logger = logging.getLogger('spark')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class TempPipeItem(QGraphicsPathItem):
    """
        Visual feedback for dragging a new connection or a disconnected one.
    """

    def __init__(self, start_pos, is_disconnecting=False, port_type=None) -> None:
        super().__init__()
        if is_disconnecting:
            color = QColor(STYLES.get_color('temp_pipe', 'color_disconnecting'))
        elif port_type is not None:
            # NOTE: Dragging shows the colour of the payload being connected, like the finished pipe will.
            color = QColor(STYLES.get_port_style(port_type).get('color'))
        else:
            color = QColor(255, 255, 255)
        color.setAlpha(STYLES.get_val('temp_pipe', 'alpha', default=200))
        self.setPen(QPen(color, STYLES.get_val('temp_pipe', 'width', default=2), Qt.PenStyle.DashLine))
        self.start_pos = start_pos
        self.update_path(start_pos)

    def update_path(self, end_pos) -> None:
        path = QPainterPath()
        path.moveTo(self.start_pos)
        mid_x = (self.start_pos.x() + end_pos.x()) / 2
        path.lineTo(mid_x, self.start_pos.y())
        path.lineTo(mid_x, end_pos.y())
        path.lineTo(end_pos)
        self.setPath(path)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class GraphScene(QGraphicsScene):

    def __init__(self, model=None, parent=None) -> None:
        super().__init__(parent)
        # Create a model if none was provided
        self.model = model if model else GraphModel()
        # Style setup
        self.setBackgroundBrush(QBrush(STYLES.get_color('graph', 'background_color')))
        self._grid_size = STYLES.get_val('graph', 'grid_size')
        self.setSceneRect(-5000, -5000, 10000, 10000)
        # Placeholder variables
        self.active_port = None
        self.drag_source_port = None
        self.temp_pipe = None
        self.is_disconnecting = False
        self.original_pipe_model = None
        # Callbacks
        self.model.node_added.connect(self.on_node_added)
        self.model.node_removed.connect(self.on_node_removed)
        self.model.edge_added.connect(self.on_edge_added)
        self.model.edge_removed.connect(self.on_edge_removed)
        self.model.graph_cleared.connect(self.on_graph_cleared)
        # Populate existing items from the model
        for node in self.model.nodes:
            self.on_node_added(node)
        for edge in self.model.edges:
            self.on_edge_added(edge)

    def on_graph_cleared(self) -> None:
        self.clear()
        # Reset any temporary drag/panning states
        self.active_port = None
        self.drag_source_port = None
        if self.temp_pipe:
            self.removeItem(self.temp_pipe)
            self.temp_pipe = None
        self.is_disconnecting = False
        self._pending_disconnect_item = None
        # Reset view-level tracking for multi-move
        for view in self.views():
            if isinstance(view, GraphView):
                view._move_start_positions = {}

    def find_port_item(self, port_id: str) -> PortItem | None:
        for item in self.items():
            if isinstance(item, NodeItem):
                for row in getattr(item, 'rows', []):
                    if isinstance(row, PropertyRowItem) and row.input_port and row.input_port.model.id == port_id:
                        return row.input_port
                    if isinstance(row, PropertyRowItem) and row.output_port and row.output_port.model.id == port_id:
                        return row.output_port
        return None

    def on_node_added(self, node_model: NodeModel) -> None:
        item = NodeItem(node_model)
        self.addItem(item)
        item.setPos(*node_model.pos)

    def on_node_removed(self, node_model: NodeModel) -> None:
        for item in self.items():
            if isinstance(item, NodeItem) and item.model == node_model:
                self.removeItem(item)
                break

    def on_edge_added(self, edge_model: EdgeModel) -> None:
        src_port = self.find_port_item(edge_model.source_port.id)
        dst_port = self.find_port_item(edge_model.target_port.id)
        if src_port and dst_port:
            pipe = PipeItem(src_port, dst_port, edge_model)
            self.addItem(pipe)
            src_port.add_pipe(pipe)
            dst_port.add_pipe(pipe)
            # If waypoints exist in the model, use them
            if edge_model.waypoints:
                pipe.pivots = [QPointF(x, y) for x, y in edge_model.waypoints]
            pipe.update_path()

    def on_edge_removed(self, edge_model: EdgeModel) -> None:
        for item in self.items():
            if isinstance(item, PipeItem) and item.model == edge_model:
                if item.source_port and isValid(item.source_port):
                    item.source_port.remove_pipe(item)
                if item.target_port and isValid(item.target_port):
                    item.target_port.remove_pipe(item)
                self.removeItem(item)
                break

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        item = self.itemAt(event.scenePos(), self.views()[0].transform())
        if isinstance(item, SegmentGizmo):
            item = item.parentItem()
        self._pending_disconnect_item = None
        if isinstance(item, PortItem) and event.button() == Qt.MouseButton.LeftButton:
            # ALWAYS start a new pipe when dragging from a port
            self.is_disconnecting = False
            self.active_port = item
            self.drag_source_port = item
            self.temp_pipe = TempPipeItem(item.scenePos(), port_type=item.model.port_type)
            self.addItem(self.temp_pipe)
            return
        if isinstance(item, PipeItem) and event.button() == Qt.MouseButton.LeftButton:
            idx, is_horiz = item._get_segment_at(event.scenePos())
            if idx == 0 or idx == len(item.full_pts) - 2:
                # Prepare for disconnect logic via dragging the endpoint segment of a pipe
                # We wait for mouseMoveEvent to actually disconnect, allowing double-clicks to pass through.
                self._pending_disconnect_item = item
                self._pending_disconnect_idx = idx
                self._pending_disconnect_pos = event.scenePos()
        super().mousePressEvent(event)

    def update_all_pipes(self) -> None:
        # NOTE: Pipes are routed oldest first: each one avoids the lanes the previous ones already claimed, so
        # the result is stable instead of depending on the order the scene happens to return its items in.
        pipes = [item for item in self.items() if isinstance(item, PipeItem) and isValid(item)]
        context = PipeRouteContext.for_scene(self)
        for item in sorted(pipes, key=lambda pipe: pipe._route_priority):
            item.update_path(context)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if hasattr(self, '_pending_disconnect_item') and self._pending_disconnect_item:
            if (event.scenePos() - self._pending_disconnect_pos).manhattanLength() > 5.0:
                item = self._pending_disconnect_item
                idx = self._pending_disconnect_idx
                self.is_disconnecting = True
                if idx == 0:
                    self.active_port = item.target_port
                    self.drag_source_port = item.source_port
                    start_pos = item.target_port.scenePos() if item.target_port else event.scenePos()
                else:
                    self.active_port = item.source_port
                    self.drag_source_port = item.target_port
                    start_pos = item.source_port.scenePos() if item.source_port else event.scenePos()
                self.original_pipe_model = item.model
                self.temp_pipe = TempPipeItem(start_pos, is_disconnecting=True)
                self.temp_pipe.update_path(event.scenePos())
                self.addItem(self.temp_pipe)
                item.disconnect_pipe()
                self._pending_disconnect_item = None
                return
        if self.temp_pipe:
            self.temp_pipe.update_path(event.scenePos())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self._pending_disconnect_item = None
        if self.temp_pipe:
            self.removeItem(self.temp_pipe)
            self.temp_pipe = None
            target_item = self.itemAt(event.scenePos(), self.views()[0].transform())
            if isinstance(target_item, SegmentGizmo):
                target_item = target_item.parentItem()
            if isinstance(target_item, PortItem) and target_item != self.active_port:
                src_port_model = self.active_port.model if not self.active_port.model.is_input else target_item.model
                dst_port_model = target_item.model if target_item.model.is_input else self.active_port.model
                is_valid, msg = EdgeModel.validate_connection(src_port_model, dst_port_model)
                if is_valid:
                    src = self.active_port if not self.active_port.model.is_input else target_item
                    dst = target_item if target_item.model.is_input else self.active_port
                    # Remove existing pipes if multi_connection is False
                    if not src.model.multi_connection and src.connected_pipes:
                        for p in list(src.connected_pipes): p.disconnect_pipe()
                    if not dst.model.multi_connection and dst.connected_pipes:
                        for p in list(dst.connected_pipes): p.disconnect_pipe()
                    edge_model = EdgeModel(src.model, dst.model)
                    self.model.undo_stack.push(AddEdgeCommand(self.model, edge_model))
                else:
                    print(f'Connection Denied: {msg}')
            elif self.is_disconnecting:
                pass # Pipe already removed during mousePressEvent
            self.active_port = None
            self.drag_source_port = None
            self.is_disconnecting = False
        super().mouseReleaseEvent(event)
        # Trigger loop cleaning on all valid pipes after interaction
        for item in self.items():
            if isinstance(item, PipeItem) and isValid(item):
                try:
                    item.clean_loops()
                except RuntimeError:
                    pass

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        super().drawBackground(painter, rect)
        pen = QPen(STYLES.get_color('graph', 'grid_color'), 1)
        painter.setPen(pen)
        left = int(rect.left()) - (int(rect.left()) % self._grid_size)
        top = int(rect.top()) - (int(rect.top()) % self._grid_size)
        lines = []
        for x in range(left, int(rect.right()), self._grid_size):
            lines.append(QPointF(x, rect.top()))
            lines.append(QPointF(x, rect.bottom()))
        for y in range(top, int(rect.bottom()), self._grid_size):
            lines.append(QPointF(rect.left(), y))
            lines.append(QPointF(rect.right(), y))
        painter.drawLines(lines)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class GraphClipboard:

    def __init__(self,) -> None:
        self._nodes: list[NodeModel] = []
        self._nodes_pos: list[QPointF] = []
        self._pipes: list[dict[str, tp.Any]] = []

    def add_node(self, node: NodeModel, pos: QPointF) -> None:
        self._nodes.append(node)
        self._nodes_pos.append(pos)

    def add_pipe(self, pipe_dict: dict) -> None:
        self._pipes.append(pipe_dict)

    def get_nodes_data(self,) -> tuple[list[NodeModel], list[QPointF]]:
        return self._nodes, self._nodes_pos

    def get_pipes_data(self,) -> list[dict[str, tp.Any]]:
        return self._pipes

    def shift_pos(self, shift: QPointF) -> None:
        self._nodes_pos = [pos + shift for pos in self._nodes_pos]

    def __len__(self) -> int:
        return len(self._nodes)

    def clear(self,) -> None:
        self._nodes = []
        self._nodes_pos = []
        self._pipes = []

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class GraphView(QGraphicsView):

    def __init__(self, scene, parent=None) -> None:
        super().__init__(scene, parent)
        self.setObjectName('graph')
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._zoom = 1.0
        self._zoom_step = 1.1
        self._is_panning = False
        self._last_mouse_pos = QPointF()
        self._clipboard = GraphClipboard()
        self._move_start_positions: dict[NodeItem, QPointF] = {}
        # The palette is dictated by the controller being built.
        self._context_menu = GraphContextMenu(self.scene().model.profile, self)
        self.scene().model.profile_changed.connect(self._context_menu.set_profile)

    # Signature overwrite
    def scene(self) -> GraphScene:
        return super().scene()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            self.delete_selected()
        elif event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_C:
            self.copy_selected()
        elif event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_V:
            self.paste_selected()
        super().keyPressEvent(event)

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        self._context_menu.update_menu_state(
            has_selection = len(self.scene().selectedItems()) > 0, 
            can_paste = len(self._clipboard) > 0,
            can_undo = self.scene().model.undo_stack.canUndo(), 
            can_redo = self.scene().model.undo_stack.canRedo(), 
        )
        action = self._context_menu.exec(event.globalPos())
        if not action is None:
            action_data: ActionData = action.data()
            if action_data.command == ContextMenuCommand.Create:
                # Instantiate node
                try:
                    new_node: NodeModel = action_data.cls()
                except Exception as error:
                    logger.error(f'Unable to create a "{action_data.cls.__name__}" node: {error}')
                    return
                self._place_node(new_node, self.mapToScene(event.pos()))
            elif action_data.command == ContextMenuCommand.Import:
                self.import_model(action_data.entry)
            elif action_data.command == ContextMenuCommand.Copy:
                self.copy_selected()
            elif action_data.command == ContextMenuCommand.Paste:
                self.paste_selected()
            elif action_data.command == ContextMenuCommand.Delete:
                self.delete_selected()
            else:
                logger.warning(f'Action: {action_data.command} is not tied to any action.')



    def _place_node(self, node: NodeModel, scene_pos, label: str = 'Add Node') -> None:
        """
            Names a node so it does not collide, snaps it to the grid and adds it as one undoable step.
        """
        model = self.scene().model
        node.name = model.get_next_free_name(node.name)
        pos_x, pos_y = scene_pos.x(), scene_pos.y()
        if STYLES.get_val('graph', 'snapping', 'enabled'):
            grid = float(STYLES.get_val('graph', 'snapping', 'node_grid'))
            pos_x = round(pos_x / grid) * grid
            pos_y = round(pos_y / grid) * grid
        node.pos = (pos_x, pos_y)
        model.undo_stack.push(AddNodeCommand(model, node, description=label))

    def add_node_for(self, module_cls: type, label: str = 'Add Node') -> None:
        """
            Places a node for a module class at the centre of the view.

            This is the menu driven counterpart of dropping a node from the context menu, which knows
            where the pointer was. It is what makes an imported model a node of the graph.
        """
        node_cls = NODE_REGISTRY.get(module_cls)
        if node_cls is None:
            raise RuntimeError(f'No node model is available for "{module_cls.__name__}".')
        self._place_node(node_cls(), self.mapToScene(self.viewport().rect().center()), label=label)

    def import_model(self, entry) -> None:
        """
            Expands a registered model into the graph.
        """
        try:
            config = entry.get_cls().get_config_spec().partial()
        except Exception as error:
            logger.error(f'Unable to read the configuration of "{entry.name}": {error}')
            return
        self.import_config(config, label=utils.to_human_readable(entry.get_cls().__name__))

    def import_config(self, config, label: str = 'Model', layout: dict | None = None) -> None:
        """
            Expands a controller configuration into nodes and edges, as a single undoable step.

            Input:
                layout: dict[str, tuple[float, float]], node positions by name. Supplied when a session is
                    reopened, since a configuration cannot carry the layout by itself.
        """
        model = self.scene().model
        # NOTE: Importing pre-populates the controller with reasonable data, it does not add a second model.
        # The controller keeps a single set of settings, so those are only taken from the file when there is
        # nothing on the canvas yet; a later import only contributes modules.
        adopt_settings = not model.nodes
        try:
            imported = expand_controller_config(config, model, model.profile, layout=layout)
        except Exception as error:
            logger.error(f'Unable to import "{label}": {error}')
            return
        for message in imported.warnings:
            logger.warning(message)
        if not imported.nodes:
            logger.warning(f'"{label}" produced no node, nothing was imported.')
            return
        stack = model.undo_stack
        stack.beginMacro(f'Import {label}')
        try:
            for node in imported.nodes:
                stack.push(AddNodeCommand(model, node))
            for edge in imported.edges:
                stack.push(AddEdgeCommand(model, edge))
        finally:
            stack.endMacro()
        if adopt_settings:
            model.adopt_controller_config(config)
        # NOTE: Pipes route themselves as they are created, so the first ones are laid out before the last
        # nodes of the import exist. One pass over the finished scene gives every pipe the same information.
        self.scene().update_all_pipes()
        # Select what was just added, so it is obvious where it landed.
        self.scene().clearSelection()
        for item in self.scene().items():
            if isinstance(item, NodeItem) and item.model in imported.nodes:
                item.setSelected(True)
        logger.log(
            MessageLevel.SUCCESS.value,
            f'Imported "{label}": {len(imported.nodes)} nodes and {len(imported.edges)} connections.',
        )

    def delete_selected(self) -> None:
        stack = self.scene().model.undo_stack
        stack.beginMacro('Delete Selection')
        # Collect items to avoid iterator invalidation issues if any
        selected_items = self.scene().selectedItems()
        # First remove pipes that are explicitly selected
        for item in selected_items:
            if isinstance(item, PipeItem) and item.model:
                # Check if it's still in the model (might have been removed if its node was processed)
                if item.model in self.scene().model.edges:
                    stack.push(RemoveEdgeCommand(self.scene().model, item.model))
        # Then remove nodes
        for item in selected_items:
            if isinstance(item, NodeItem) and item.model:
                if item.model in self.scene().model.nodes:
                    stack.push(RemoveNodeCommand(self.scene().model, item.model))
        stack.endMacro()

    def copy_selected(self) -> None:
        self._clipboard.clear()
        selected_nodes = []
        for item in self.scene().selectedItems():
            if isinstance(item, NodeItem):
                selected_nodes.append(item)
                self._clipboard.add_node(item.model, item.scenePos())
        # Find shared pipes
        for item in self.scene().selectedItems():
            if isinstance(item, PipeItem):
                src_node = item.source_port.parentItem().parentItem() if item.source_port else None
                dst_node = item.target_port.parentItem().parentItem() if item.target_port else None
                if src_node in selected_nodes and dst_node in selected_nodes:
                    self._clipboard.add_pipe({
                        'src_idx': selected_nodes.index(src_node),
                        'src_port_name': item.source_port.model.name,
                        'src_is_input': item.source_port.model.is_input,
                        'dst_idx': selected_nodes.index(dst_node),
                        'dst_port_name': item.target_port.model.name,
                        'dst_is_input': item.target_port.model.is_input
                    })

    def paste_selected(self) -> None:
        if len(self._clipboard) == 0:
            return
        self.scene().clearSelection()
        stack = self.scene().model.undo_stack
        stack.beginMacro('Paste Selection')
        nodes_model, nodes_pos = self._clipboard.get_nodes_data()
        # Calculate center of copied items
        min_x = min(pos.x() for pos in nodes_pos)
        max_x = max(pos.x() for pos in nodes_pos)
        min_y = min(pos.y() for pos in nodes_pos)
        max_y = max(pos.y() for pos in nodes_pos)
        orig_center = QPointF((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
        # Get center of current view
        view_center = self.mapToScene(self.viewport().rect().center())
        offset = view_center - orig_center
        new_models = []
        for orig_model, orig_pos in zip(nodes_model, nodes_pos):
            # Clone model using its specific class
            new_model: NodeModel = orig_model.__class__(orig_model.name, orig_model.type_name)
            for p in orig_model.call_section.ports:
                new_model.call_section.add_port(PortModel(p.name, p.is_input, p.port_type, p.is_optional, p.multi_connection))
            for p in orig_model.props_section.ports:
                new_model.props_section.add_port(PortModel(p.name, p.is_input, p.port_type, p.is_optional, p.multi_connection))
            # Offset position to view center
            new_pos = orig_pos + offset
            # Snap pasted items if snapping is enabled
            if STYLES.get_val('graph', 'snapping', 'enabled'):
                grid = float(STYLES.get_val('graph', 'snapping', 'node_grid'))
                new_pos.setX(round(new_pos.x() / grid) * grid)
                new_pos.setY(round(new_pos.y() / grid) * grid)
            new_model.pos = (new_pos.x(), new_pos.y())
            # Rename to prevent collisions
            name = self.scene().model.get_next_free_name(new_model.name)
            new_model.name = name
            # Use command
            stack.push(AddNodeCommand(self.scene().model, new_model))
            new_models.append(new_model)
        # Update clipboard so pasting again shifts further
        self._clipboard.shift_pos(QPointF(20, 20))
        # Select newly pasted items
        for item in self.scene().items():
            if isinstance(item, NodeItem) and item.model in new_models:
                item.setSelected(True)
        # Paste pipes
        for pipe_data in self._clipboard.get_pipes_data():
            src_model = new_models[pipe_data['src_idx']]
            dst_model = new_models[pipe_data['dst_idx']]
            src_port = src_model.get_port_by_name(pipe_data['src_port_name'], pipe_data['src_is_input'])
            dst_port = dst_model.get_port_by_name(pipe_data['dst_port_name'], pipe_data['dst_is_input'])
            if src_port and dst_port:
                edge_model = EdgeModel(src_port, dst_port)
                stack.push(AddEdgeCommand(self.scene().model, edge_model))
        # Select newly pasted pipes
        for item in self.scene().items():
            if isinstance(item, PipeItem):
                if item.model and item.model.source_port.node in new_models:
                    item.setSelected(True)
        stack.endMacro()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._is_panning = True
            self._last_mouse_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        elif event.button() == Qt.MouseButton.LeftButton:
            # Check what is under the mouse before passing the event
            item = self.itemAt(event.position().toPoint())
            if not item:
                self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
            # Capture nodes that are selected before the click
            pre_selected_nodes = [it for it in self.scene().selectedItems() if isinstance(it, NodeItem)]
            super().mousePressEvent(event)
            # NOTE: After super() the selection is updated.
            self._move_start_positions = {}
            # We only track movement if we clicked on something that is or belongs to a NodeItem
            curr = item
            is_node_interaction = False
            clicked_node = None
            while curr:
                if isinstance(curr, NodeItem):
                    is_node_interaction = True
                    clicked_node = curr
                    break
                curr = curr.parentItem()
            if is_node_interaction:
                # The set of nodes to track is:
                # 1. The clicked node
                # 2. Any nodes that were ALREADY selected if the clicked node was part of that selection
                # 3. Any nodes that are NOW selected (in case Qt updated selection)
                nodes_to_track: set[NodeItem] = set()
                if clicked_node:
                    nodes_to_track.add(clicked_node)
                # If we clicked on an item that was already part of a multi-selection, 
                # keep tracking the whole group.
                if clicked_node in pre_selected_nodes:
                    nodes_to_track.update(pre_selected_nodes)
                for it in self.scene().selectedItems():
                    if isinstance(it, NodeItem):
                        nodes_to_track.add(it)
                for it in nodes_to_track:
                    self._move_start_positions[it] = it.pos()
                # Track waypoints for pipes that might be shifted (both ends in the tracked set)
                self._pipe_start_waypoints = {}
                for it in self.scene().items():
                    if isinstance(it, PipeItem) and it.model:
                        src_node = it.source_port.parentItem().parentItem() if it.source_port else None
                        dst_node = it.target_port.parentItem().parentItem() if it.target_port else None
                        if src_node in nodes_to_track and dst_node in nodes_to_track:
                            self._pipe_start_waypoints[it] = list(it.model.waypoints)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._is_panning:
            delta = event.position() - self._last_mouse_pos
            self._last_mouse_pos = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        elif event.button() == Qt.MouseButton.LeftButton:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            # Check for movement in any selected nodes
            if self._move_start_positions:
                # Filter for nodes that actually moved
                moved_nodes = []
                for it, old_pos in self._move_start_positions.items():
                    if it.pos() != old_pos:
                        moved_nodes.append((it.model, (old_pos.x(), old_pos.y()), (it.x(), it.y())))
                # Filter for pipes that actually moved their waypoints
                moved_pipes = []
                if hasattr(self, '_pipe_start_waypoints'):
                    for it, old_waypoints in self._pipe_start_waypoints.items():
                        new_waypoints = list(it.model.waypoints)
                        if new_waypoints != old_waypoints:
                            moved_pipes.append((it.model, old_waypoints, new_waypoints))
                if moved_nodes or moved_pipes:
                    stack = self.scene().model.undo_stack
                    stack.beginMacro('Move Items')
                    # Push pipe routing commands first so they are undone LAST (after node restoration)
                    for model, ow, nw in moved_pipes:
                        stack.push(ChangeEdgeWaypointsCommand(model, ow, nw))
                    for model, op, np in moved_nodes:
                        stack.push(MoveNodeCommand(model, op, np))
                    stack.endMacro()
                self._move_start_positions = {}
                self._pipe_start_waypoints = {}
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
        else:
            super().wheelEvent(event)

    def zoom_in(self) -> None:
        self._zoom *= self._zoom_step
        self.scale(self._zoom_step, self._zoom_step)

    def zoom_out(self) -> None:
        self._zoom /= self._zoom_step
        self.scale(1/self._zoom_step, 1/self._zoom_step)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################