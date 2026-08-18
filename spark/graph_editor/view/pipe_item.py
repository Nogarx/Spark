#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.graph_editor.view.node_item import PortItem
    from spark.graph_editor.models.edge_model import EdgeModel
    from spark.graph_editor.view.graph_view import GraphScene

import itertools
from shiboken6 import isValid
from PySide6.QtWidgets import QGraphicsPathItem, QGraphicsItem, QWidget, QStyleOption, QGraphicsSceneMouseEvent
from PySide6.QtCore import Qt, QPointF, QRectF, QLineF
from PySide6.QtGui import QPainterPath, QPen, QPainter, QPainterPathStroker, QBrush, QPolygonF
from spark.graph_editor.styles.manager import STYLES
from spark.graph_editor.view.node_item import NodeItem
from spark.graph_editor.commands.graph_commands import ChangeEdgeWaypointsCommand, RemoveEdgeCommand

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class SegmentGizmo(QGraphicsItem):

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setZValue(10)
        self.is_horizontal = True
        self.draggable = True
        self.setVisible(False)
        self.setAcceptHoverEvents(True)

    def boundingRect(self) -> QRectF:
        return QRectF(-12, -12, 24, 24)

    def paint(self, painter: QPainter, option: QStyleOption, widget: QWidget) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = STYLES.get_color('pipe', 'active_color')
        color.setAlpha(150)
        painter.setBrush(color)
        painter.setPen(QPen(color, 1))
        painter.drawEllipse(-8, -8, 16, 16)
        if not self.draggable: 
            return
        painter.setPen(QPen(Qt.GlobalColor.black, 2))
        if self.is_horizontal:
            painter.drawLine(0, -5, 0, 5)
            painter.drawLine(-3, -2, 0, -5); painter.drawLine(3, -2, 0, -5)
            painter.drawLine(-3, 2, 0, 5); painter.drawLine(3, 2, 0, 5)
        else:
            painter.drawLine(-5, 0, 5, 0)
            painter.drawLine(-2, -3, -5, 0); painter.drawLine(-2, 3, -5, 0)
            painter.drawLine(2, -3, 5, 0); painter.drawLine(2, 3, 5, 0)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class PipeRouteContext:
    """
        Shared state of one routing pass.

        Collecting the node rectangles and the lanes already in use once per pass, instead of once per pipe,
        is what keeps routing a whole scene linear in the number of pipes.
    """

    def __init__(self, obstacles: list[tuple[NodeItem, QRectF]], bundles: dict[tuple[int, int], list[int]] | None = None) -> None:
        self._obstacles = obstacles
        self.lanes: list[tuple[float, float, float]] = []
        self.channels: list[tuple[float, float, float]] = []
        self._bundles = bundles if bundles is not None else {}

    @classmethod
    def for_scene(cls, scene) -> 'PipeRouteContext':
        margin = float(STYLES.get_val('pipe', 'node_margin', default=16))
        obstacles, bundles = [], {}
        if scene is not None:
            for item in scene.items():
                if isinstance(item, NodeItem) and isValid(item):
                    obstacles.append((item, item.sceneBoundingRect().adjusted(-margin, -margin, margin, margin)))
            # Pipes joining the same two nodes form a bundle and are given one lane each. The ranking is
            # computed from the whole scene so that it does not depend on what is being redrawn.
            for item in scene.items():
                if not isinstance(item, PipeItem) or not isValid(item):
                    continue
                source_node, target_node = item._end_nodes()
                if source_node is None or target_node is None:
                    continue
                key = tuple(sorted((id(source_node), id(target_node))))
                bundles.setdefault(key, []).append(item._route_priority)
            for priorities in bundles.values():
                priorities.sort()
        return cls(obstacles, bundles)

    @classmethod
    def for_pipe(cls, pipe: 'PipeItem') -> 'PipeRouteContext':
        """
            Context for a single pipe: the lanes of every older pipe are already claimed.
        """
        scene = pipe.scene()
        context = cls.for_scene(scene)
        if scene is None:
            return context
        for item in scene.items():
            if not isinstance(item, PipeItem) or item is pipe or not isValid(item):
                continue
            if item._route_priority > pipe._route_priority:
                continue
            context.claim(item.full_pts)
        return context

    def obstacles_excluding(self, *nodes) -> list[QRectF]:
        excluded = {id(node) for node in nodes if node is not None}
        return [rect for item, rect in self._obstacles if id(item) not in excluded]

    def claim(self, points: list[QPointF]) -> None:
        """
            Registers the lanes and columns a route occupies.
        """
        for index in range(len(points) - 1):
            a, b = points[index], points[index + 1]
            if abs(a.y() - b.y()) < 0.5:
                self.lanes.append((a.y(), min(a.x(), b.x()), max(a.x(), b.x())))
            elif abs(a.x() - b.x()) < 0.5:
                self.channels.append((a.x(), min(a.y(), b.y()), max(a.y(), b.y())))

    def bundle_index(self, pipe: 'PipeItem', source_node: NodeItem, target_node: NodeItem) -> int:
        """
            Rank of a pipe among the ones joining the same two nodes, in both directions.
        """
        key = tuple(sorted((id(source_node), id(target_node))))
        bundle = self._bundles.get(key, [])
        return bundle.index(pipe._route_priority) if pipe._route_priority in bundle else 0

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class PipeItem(QGraphicsPathItem):

    # NOTE: Routing priority. A pipe only avoids the lanes claimed by older pipes, never the other way round,
    # so routing the scene is a fixed point instead of an endless negotiation.
    _priority_counter = itertools.count()

    def __init__(self, source_port_item: PortItem, target_port_item: PortItem, model: EdgeModel | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.source_port = source_port_item
        self.target_port = target_port_item
        self.model = model
        self.setPen(QPen(STYLES.get_color('pipe', 'color'), STYLES.get_val('pipe', 'width')))
        self.setZValue(-1)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self._hovered = False
        self.pivots = []
        # NOTE: A pipe routes itself until the user edits it. From then on its waypoints are respected as they
        # are, which is why manual overlaps stay untouched while automatic ones are avoided.
        self._auto_routed = True
        self._route_priority = next(PipeItem._priority_counter)
        self.gizmo = SegmentGizmo(self)
        self._dragging_segment_idx = -1
        self.full_pts = []
        if self.model:
            self.model.waypoints_changed.connect(self.on_waypoints_changed)
            # Initialize pivots from model if they exist
            if self.model.waypoints:
                self.pivots = [QPointF(x, y) for x, y in self.model.waypoints]
                self._auto_routed = False
        self.update_path()

    # Signature overwrite
    def scene(self) -> GraphScene:
        return super().scene()

    def on_waypoints_changed(self) -> None:
        new_pivots = [QPointF(x, y) for x, y in self.model.waypoints]
        if self.pivots != new_pivots:
            self.pivots = new_pivots
            self._auto_routed = not new_pivots
            self.update_path()

    def update_path(self, context: PipeRouteContext | None = None) -> None:
        if self.source_port and not isValid(self.source_port): 
            return
        if self.target_port and not isValid(self.target_port): 
            return
        self.prepareGeometryChange()
        p1 = self.source_port.scenePos() if self.source_port else QPointF(0,0)
        p2 = self.target_port.scenePos() if self.target_port else QPointF(100,0)
        # Initialize Pivots
        if self._auto_routed or not self.pivots:
            # NOTE: Routing the whole scene shares one context, which is what keeps it O(pipes) instead of
            # rescanning the scene for every pipe.
            owned_context = context is None
            if owned_context:
                context = PipeRouteContext.for_pipe(self)
            self.pivots = self._auto_pivots(p1, p2, context)
        # Enforce Manhattan Snapping
        pts = [p1] + self.pivots + [p2]
        for i in range(1, len(pts)-1):
            prev = pts[i-1]
            if (i-1) % 2 == 0: pts[i].setY(prev.y())
            else: pts[i].setX(prev.x())
        pts[-2].setY(p2.y())
        self.full_pts = pts
        if context is not None and self._auto_routed:
            context.claim(pts)
        # Sync back to model
        if self.model and not self._auto_routed:
            model_pts = [(float(p.x()), float(p.y())) for p in self.pivots]
            if self.model.waypoints != model_pts:
                # We don't want to trigger signals here to avoid loops, but we need consistency
                self.model._waypoints = model_pts 
        # Path Generation with Jumps
        path = QPainterPath()
        path.moveTo(self.full_pts[0])
        for i in range(len(self.full_pts) - 1):
            seg_start, seg_end = self.full_pts[i], self.full_pts[i+1]
            # Vertical segment
            if abs(seg_start.x() - seg_end.x()) < 0.1: 
                crossings = self._find_crossings(seg_start, seg_end)
                crossings.sort(key=lambda p: p.y(), reverse=(seg_start.y() > seg_end.y()))
                merged_groups = []
                if crossings:
                    curr_group = [crossings[0]]
                    for k in range(1, len(crossings)):
                        if abs(crossings[k].y() - curr_group[-1].y()) < 15.0: curr_group.append(crossings[k])
                        else: merged_groups.append(curr_group); curr_group = [crossings[k]]
                    merged_groups.append(curr_group)
                for group in merged_groups:
                    min_y, max_y = min(p.y() for p in group), max(p.y() for p in group)
                    center_y = (min_y + max_y) / 2.0
                    jump_r = STYLES.get_val('pipe', 'jump_radius')
                    eff_r = jump_r + ((max_y - min_y) / 2.0)
                    path.lineTo(seg_start.x(), center_y - (eff_r if seg_start.y() < seg_end.y() else -eff_r))
                    path.arcTo(seg_start.x() - jump_r, center_y - eff_r, 2*jump_r, 2*eff_r, 
                               90 if seg_start.y() < seg_end.y() else 270, 180)
            path.lineTo(seg_end)
        self.setPath(path)

    #-------------------------------------------------------------------------------------------------------#
    # Automatic routing
    #-------------------------------------------------------------------------------------------------------#

    # NOTE: Pipes are routed around the nodes instead of through them, and never along a lane another pipe is
    # already using. Only the automatic route is affected: a pipe the user has edited keeps its waypoints,
    # overlaps included.

    def _end_nodes(self) -> tuple[NodeItem | None, NodeItem | None]:
        def _node_of(port) -> NodeItem | None:
            if port is None or not isValid(port):
                return None
            item = port.parentItem()
            item = item.parentItem() if item is not None else None
            return item if isinstance(item, NodeItem) and isValid(item) else None
        return (_node_of(self.source_port), _node_of(self.target_port))

    @staticmethod
    def _segment_hits(a: QPointF, b: QPointF, rect: QRectF) -> bool:
        """
            True if an axis aligned segment enters a rectangle.
        """
        x0, x1 = sorted((a.x(), b.x()))
        y0, y1 = sorted((a.y(), b.y()))
        return x0 < rect.right() and x1 > rect.left() and y0 < rect.bottom() and y1 > rect.top()

    def _collisions(self, points: list[QPointF], obstacles: list[QRectF]) -> int:
        return sum(
            1
            for index in range(len(points) - 1)
            for rect in obstacles
            if self._segment_hits(points[index], points[index + 1], rect)
        )

    @staticmethod
    def _track_is_free(value: float, low: float, high: float, tracks: list[tuple[float, float, float]], separation: float) -> bool:
        """
            True if a lane/channel does not run alongside one that is already taken.
        """
        low, high = min(low, high), max(low, high)
        for position, start, end in tracks:
            if abs(position - value) >= separation:
                continue
            # Only an actual shared stretch counts, touching at a corner does not.
            if min(high, end) - max(low, start) > separation:
                return False
        return True

    def _lane_offset(self, context: PipeRouteContext) -> float:
        """
            Separation given to pipes that share the same pair of nodes.

            Two modules can be connected twice in opposite directions (a plasticity rule reading a kernel and
            writing it back), and both connections would otherwise be drawn along the very same lane.
        """
        source_node, target_node = self._end_nodes()
        if source_node is None or target_node is None:
            return 0.0
        spacing = float(STYLES.get_val('pipe', 'lane_spacing', default=14))
        return spacing * context.bundle_index(self, source_node, target_node)

    def _free_column(
            self,
            start_x: float,
            y_a: float,
            y_b: float,
            direction: float,
            obstacles: list[QRectF],
            channels: list[tuple[float, float, float]],
            margin: float,
            step: float,
            separation: float,
        ) -> float:
        """
            First x, walking away from a node, where a vertical run between two heights is free.

            Searching by fixed increments is not enough: a column can be blocked by a whole node, so the
            search jumps straight past whatever is in the way.
        """
        low, high = min(y_a, y_b), max(y_a, y_b)
        x = start_x
        for _ in range(8):
            blocked = next(
                (rect for rect in obstacles
                 if rect.left() < x < rect.right() and rect.top() < high and rect.bottom() > low),
                None,
            )
            if blocked is not None:
                x = blocked.right() if direction > 0 else blocked.left()
                continue
            if not self._track_is_free(x, low, high, channels, separation):
                x += direction * step
                continue
            break
        return x

    def _auto_pivots(self, p1: QPointF, p2: QPointF, context: PipeRouteContext) -> list[QPointF]:
        """
            Computes an obstacle free Manhattan route between two ports.
        """
        margin = float(STYLES.get_val('pipe', 'node_margin', default=16))
        step = float(STYLES.get_val('pipe', 'lane_spacing', default=14))
        separation = step * 0.75
        source_node, target_node = self._end_nodes()
        lane = self._lane_offset(context)

        # A self connection leaves on the right and comes back on the left.
        if source_node is not None and source_node is target_node:
            rect = source_node.sceneBoundingRect()
            offset = margin + lane
            return [
                QPointF(rect.right() + offset, p1.y()),
                QPointF(rect.right() + offset, rect.bottom() + offset),
                QPointF(rect.left() - offset, rect.bottom() + offset),
                QPointF(rect.left() - offset, p2.y()),
            ]

        obstacles = context.obstacles_excluding(source_node, target_node)
        lanes, channels = context.lanes, context.channels
        # Best effort fallback: the least colliding route seen, used only if nothing is fully free.
        best_score, best_route = None, None

        def _consider(route: list[QPointF]) -> list[QPointF] | None:
            nonlocal best_score, best_route
            collisions = self._collisions(route, obstacles)
            length = sum(
                abs(route[i + 1].x() - route[i].x()) + abs(route[i + 1].y() - route[i].y())
                for i in range(len(route) - 1)
            )
            score = (collisions, length)
            if best_score is None or score < best_score:
                best_score, best_route = score, route
            return route if collisions == 0 else None

        # 1) Straight ahead: a single vertical channel between the two ports.
        exit_x, entry_x = p1.x() + margin, p2.x() - margin
        if entry_x > exit_x:
            candidates = [(p1.x() + p2.x()) / 2.0 + lane, exit_x + lane, entry_x - lane]
            # Any gap between the obstacles standing in the way is a valid channel too. Their rectangles are
            # already inflated by the margin, so their own edges are the closest safe position.
            for rect in obstacles:
                candidates.extend((rect.left(), rect.right(), rect.left() - margin, rect.right() + margin))
            for channel_x in candidates:
                # NOTE: The window is the whole span between the two ports, not the comfortable one. A
                # channel hugging a node is still better than a pipe drawn straight through it.
                if not (p1.x() <= channel_x <= p2.x()):
                    continue
                if not self._track_is_free(channel_x, p1.y(), p2.y(), channels, separation):
                    continue
                route = _consider([p1, QPointF(channel_x, p1.y()), QPointF(channel_x, p2.y()), p2])
                if route is not None:
                    return route[1:-1]

        # 2) Around: leave on the right, travel along a free lane, come back on the left.
        involved = [rect for rect in (
            source_node.sceneBoundingRect() if source_node else None,
            target_node.sceneBoundingRect() if target_node else None,
        ) if rect is not None]
        exit_x, entry_x = p1.x() + margin + lane, p2.x() - margin - lane
        span_low, span_high = min(exit_x, entry_x), max(exit_x, entry_x)
        # NOTE: The free lanes are the borders of the obstacles themselves. Stepping blindly away from the
        # graph is not enough: a second imported model sits right below the first one, and a fixed number of
        # steps cannot clear it.
        blocking = [rect for rect in obstacles if rect.right() > span_low and rect.left() < span_high]
        spread = involved + blocking
        lane_candidates = [
            min([rect.top() for rect in spread], default=min(p1.y(), p2.y())) - margin,
            max([rect.bottom() for rect in spread], default=max(p1.y(), p2.y())) + margin,
        ]
        for rect in blocking:
            lane_candidates.extend((rect.top(), rect.bottom(), rect.top() - margin, rect.bottom() + margin))
        # Closest detour first.
        lane_candidates.sort(key=lambda y: abs(y - p1.y()) + abs(y - p2.y()))
        for lane_y in lane_candidates:
            for nudge in (0.0, step, -step, 2.0 * step, -2.0 * step):
                y = lane_y + nudge + (lane if lane_y >= p1.y() else -lane)
                if not self._track_is_free(y, exit_x, entry_x, lanes, separation):
                    continue
                # The columns used to leave and to enter are resolved independently, so that a crowded side
                # does not invalidate the lane, and two pipes reaching the same node do not share the stub.
                column_out = self._free_column(exit_x, p1.y(), y, 1.0, obstacles, channels, margin, step, separation)
                column_in = self._free_column(entry_x, p2.y(), y, -1.0, obstacles, channels, margin, step, separation)
                route = _consider([
                    p1,
                    QPointF(column_out, p1.y()),
                    QPointF(column_out, y),
                    QPointF(column_in, y),
                    QPointF(column_in, p2.y()),
                    p2,
                ])
                if route is not None:
                    return route[1:-1]

        # 3) Nothing is completely free: keep the least colliding route rather than a blind one.
        if best_route is not None:
            return best_route[1:-1]
        mid_x = (p1.x() + p2.x()) / 2.0 + lane
        return [QPointF(mid_x, p1.y()), QPointF(mid_x, p2.y())]

    def _find_crossings(self, v_start: QPointF, v_end: QPointF) -> list:
        crossings = []
        if not self.scene(): 
            return crossings
        x, y_min, y_max = v_start.x(), min(v_start.y(), v_end.y()), max(v_start.y(), v_end.y())
        # Use BSP tree to only check items in the segment's vicinity
        rect = QRectF(x - 1, y_min, 2, y_max - y_min)
        for item in self.scene().items(rect):
            if isinstance(item, PipeItem) and item != self:
                o_pts = item.full_pts
                for j in range(len(o_pts) - 1):
                    op1, op2 = o_pts[j], o_pts[j+1]
                    if abs(op1.y() - op2.y()) < 0.1: # Horizontal
                        if min(op1.x(), op2.x()) < x < max(op1.x(), op2.x()):
                            if y_min < op1.y() < y_max: crossings.append(QPointF(x, op1.y()))
        return crossings

    def _get_segment_at(self, pos: QPointF) -> tuple[int, bool]:
        eps = 10.0
        for i in range(len(self.full_pts) - 1):
            p1, p2 = self.full_pts[i], self.full_pts[i+1]
            if abs(p1.y() - p2.y()) < 0.1: 
                # Horizontal
                if min(p1.x(), p2.x()) - eps <= pos.x() <= max(p1.x(), p2.x()) + eps:
                    if abs(pos.y() - p1.y()) <= eps: 
                        return i, True
            else: 
                # Verttical
                if min(p1.y(), p2.y()) - eps <= pos.y() <= max(p1.y(), p2.y()) + eps:
                    if abs(pos.x() - p1.x()) <= eps: 
                        return i, False
        return -1, False

    def _refresh_gizmo(self, scene_pos: QPointF) -> None:
        idx, is_horiz = self._get_segment_at(scene_pos)
        if idx != -1:
            self.gizmo.is_horizontal = is_horiz
            self.gizmo.draggable = (0 < idx < len(self.full_pts) - 2)
            mid = (self.full_pts[idx] + self.full_pts[idx+1]) / 2.0
            self.gizmo.setPos(self.mapFromScene(mid)); self.gizmo.setVisible(True)
        else: self.gizmo.setVisible(False)

    def clean_loops(self) -> None:
        # Disable loop cleaning for self connections to prevent erratic behavior
        if self.source_port and self.target_port:
            if not isValid(self.source_port) or not isValid(self.target_port):
                return
            try:
                src_node = self.source_port.parentItem().parentItem()
                dst_node = self.target_port.parentItem().parentItem()
                if src_node == dst_node and src_node is not None:
                    return
            except RuntimeError:
                return
        pts = self.full_pts
        if len(pts) < 4: 
            return
        for i in range(len(pts) - 1):
            l1 = QLineF(pts[i], pts[i+1])
            for j in range(i + 2, len(pts) - 1):
                res, ipt = l1.intersects(QLineF(pts[j], pts[j+1]))
                if res == QLineF.BoundedIntersection:
                    # To maintain Manhattan parity, we must insert an even number of pivots.
                    self.pivots = self.pivots[:i] + [ipt, ipt] + self.pivots[j:]
                    self.update_path(); return

    def split_segment(self, idx: int, scene_pos: QPointF) -> None:
        self._auto_routed = False
        if self.model: old_waypoints = list(self.model.waypoints)
        if STYLES.get_val('snapping', 'enabled', True):
            grid = float(STYLES.get_val('snapping', 'pipe_grid'))
            scene_pos = QPointF(round(scene_pos.x() / grid) * grid, round(scene_pos.y() / grid) * grid)
        self.pivots.insert(idx, QPointF(scene_pos))
        self.pivots.insert(idx, QPointF(scene_pos))
        self.update_path()
        self._refresh_gizmo(scene_pos)
        if self.model:
            new_waypoints = [(float(p.x()), float(p.y())) for p in self.pivots]
            self.scene().model.undo_stack.push(
                ChangeEdgeWaypointsCommand(self.model, old_waypoints, new_waypoints)
            )

    def simplify_at(self, idx: int) -> None:
        self._auto_routed = False
        if self.model: old_waypoints = list(self.model.waypoints)
        src_node = self.source_port.parentItem().parentItem() if self.source_port else None
        dst_node = self.target_port.parentItem().parentItem() if self.target_port else None
        is_self = (src_node == dst_node and src_node is not None)
        # Default self-connection has 4 pivots. We shouldn't reduce below this.
        if is_self and len(self.pivots) <= 4:
            return 
        if len(self.pivots) > 2:
            pop_idx = max(0, idx - 1)
            self.pivots.pop(pop_idx)
            if pop_idx < len(self.pivots): self.pivots.pop(pop_idx)
        elif not is_self:
            self.pivots = []
        self.update_path()
        self.gizmo.setVisible(False)
        if self.model:
            new_waypoints = [(float(p.x()), float(p.y())) for p in self.pivots]
            self.scene().model.undo_stack.push(
                ChangeEdgeWaypointsCommand(self.model, old_waypoints, new_waypoints)
            )

    def disconnect_pipe(self) -> None:
        if self.source_port and isValid(self.source_port): self.source_port.remove_pipe(self)
        if self.target_port and isValid(self.target_port): self.target_port.remove_pipe(self)
        if self.model and self.scene() and hasattr(self.scene(), 'model'):
            self.scene().model.undo_stack.push(
                RemoveEdgeCommand(self.scene().model, self.model)
            )
        elif self.scene() and isValid(self.scene()):
            # Fallback if no model (temp pipes etc, though they shouldn't call this)
            self.scene().removeItem(self)

    def shape(self) -> QPainterPath:
        stroker = QPainterPathStroker()
        stroker.setWidth(15)
        return stroker.createStroke(self.path())

    def hoverEnterEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self._hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self._refresh_gizmo(event.scenePos())
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self._hovered = False
        self.update()
        self.gizmo.setVisible(False)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        idx, is_horiz = self._get_segment_at(event.scenePos())
        if event.button() == Qt.MouseButton.LeftButton and idx != -1:
            if self.model: self._old_waypoints = list(self.model.waypoints)
            if 0 < idx < len(self.full_pts) - 2:
                self._dragging_segment_idx = idx
                self._is_horiz_drag = is_horiz
                return
        elif event.button() == Qt.MouseButton.RightButton:
            if idx != -1: self.simplify_at(idx); return
            self.disconnect_pipe()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._dragging_segment_idx != -1:
            self._auto_routed = False
        if self._dragging_segment_idx != -1:
            pos = event.scenePos()
            if STYLES.get_val('snapping', 'enabled'):
                grid = float(STYLES.get_val('snapping', 'pipe_grid'))
                pos.setX(round(pos.x() / grid) * grid)
                pos.setY(round(pos.y() / grid) * grid)
            idx = self._dragging_segment_idx
            if self._is_horiz_drag:
                if idx > 0: self.pivots[idx-1].setY(pos.y())
                if idx < len(self.full_pts) - 2: self.pivots[idx].setY(pos.y())
            else:
                if idx > 0: self.pivots[idx-1].setX(pos.x())
                if idx < len(self.full_pts) - 2: self.pivots[idx].setX(pos.x())
            if self.scene() and hasattr(self.scene(), 'update_all_pipes'): self.scene().update_all_pipes()
            else: self.update_path()
            mid = (self.full_pts[idx] + self.full_pts[idx+1]) / 2.0
            self.gizmo.setPos(self.mapFromScene(mid))
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._dragging_segment_idx != -1 and self.model and hasattr(self, '_old_waypoints'):
            new_waypoints = [(float(p.x()), float(p.y())) for p in self.pivots]
            if self._old_waypoints != new_waypoints:
                self.scene().model.undo_stack.push(
                    ChangeEdgeWaypointsCommand(self.model, self._old_waypoints, new_waypoints)
                )
        self._dragging_segment_idx = -1
        self.clean_loops()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        idx, _ = self._get_segment_at(event.scenePos())
        if idx != -1: self.split_segment(idx, event.scenePos())
        super().mouseDoubleClickEvent(event)

    def paint(self, painter: QPainter, option: QStyleOption, widget: QWidget) -> None:
        is_active = self.isSelected() or getattr(self, '_hovered', False)
        pen = QPen(STYLES.get_color('pipe', 'active_color') if is_active else STYLES.get_color('pipe', 'color'))
        pen.setWidth(STYLES.get_val('pipe', 'active_width') if is_active else STYLES.get_val('pipe', 'width'))
        painter.setPen(pen)
        painter.drawPath(self.path())
        # Draw clear port stubs so it's visually obvious the port is in use
        if self.full_pts:
            stub_len = 10.0
            p1 = self.full_pts[0]
            p2 = self.full_pts[-1]
            stub_pen = QPen(pen.color(), pen.width(), Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap)
            painter.setPen(stub_pen)
            p1_dir = -1.0 if (self.source_port and self.source_port.model.is_input) else 1.0
            p2_dir = -1.0 if (self.target_port and self.target_port.model.is_input) else 1.0
            # If target_port is missing (during temp creation or somehow detached), default to input behavior (-1)
            painter.drawLine(p1, QPointF(p1.x() + (stub_len * p1_dir), p1.y()))
            painter.drawLine(p2, QPointF(p2.x() + (stub_len * p2_dir), p2.y()))
            # Draw directional arrows
            painter.setBrush(QBrush(pen.color()))
            painter.setPen(Qt.PenStyle.NoPen)
            arrow_size = 5.0
            for i in range(len(self.full_pts) - 1):
                p_start = self.full_pts[i]
                p_end = self.full_pts[i+1]
                if (p_end - p_start).manhattanLength() > 60.0:
                    mid = (p_start + p_end) / 2.0
                    poly = QPolygonF()
                    # Horizontal
                    if abs(p_start.y() - p_end.y()) < 0.1: 
                        direction = 1.0 if p_start.x() < p_end.x() else -1.0
                        poly.append(QPointF(mid.x() + (arrow_size * direction), mid.y()))
                        poly.append(QPointF(mid.x() - (arrow_size * direction), mid.y() - arrow_size))
                        poly.append(QPointF(mid.x() - (arrow_size * direction), mid.y() + arrow_size))
                    # Vertical
                    else:
                        direction = 1.0 if p_start.y() < p_end.y() else -1.0
                        poly.append(QPointF(mid.x(), mid.y() + (arrow_size * direction)))
                        poly.append(QPointF(mid.x() - arrow_size, mid.y() - (arrow_size * direction)))
                        poly.append(QPointF(mid.x() + arrow_size, mid.y() - (arrow_size * direction)))
                    painter.drawPolygon(poly)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################