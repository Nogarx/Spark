#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.graph_editor.view.node_item import PortItem
    from spark.graph_editor.models.edge_model import EdgeModel
    from spark.graph_editor.view.graph_view import GraphScene

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

class PipeItem(QGraphicsPathItem):

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
        self.gizmo = SegmentGizmo(self)
        self._dragging_segment_idx = -1
        self.full_pts = []
        if self.model:
            self.model.waypoints_changed.connect(self.on_waypoints_changed)
            # Initialize pivots from model if they exist
            if self.model.waypoints:
                self.pivots = [QPointF(x, y) for x, y in self.model.waypoints]
        self.update_path()

    # Signature overwrite
    def scene(self) -> GraphScene:
        return super().scene()

    def on_waypoints_changed(self) -> None:
        new_pivots = [QPointF(x, y) for x, y in self.model.waypoints]
        if self.pivots != new_pivots:
            self.pivots = new_pivots
            self.update_path()

    def update_path(self) -> None:
        if self.source_port and not isValid(self.source_port): 
            return
        if self.target_port and not isValid(self.target_port): 
            return
        self.prepareGeometryChange()
        p1 = self.source_port.scenePos() if self.source_port else QPointF(0,0)
        p2 = self.target_port.scenePos() if self.target_port else QPointF(100,0)
        # Initialize Pivots
        if not self.pivots:
            src_node = self.source_port.parentItem().parentItem() if self.source_port else None
            dst_node = self.target_port.parentItem().parentItem() if self.target_port else None
            if src_node and not isValid(src_node): 
                src_node = None
            if dst_node and not isValid(dst_node): 
                dst_node = None
            if src_node == dst_node and isinstance(src_node, NodeItem):
                rect = src_node.sceneBoundingRect()
                self.pivots = [
                    QPointF(rect.right() + 30, p1.y()),
                    QPointF(rect.right() + 30, rect.bottom() + 30),
                    QPointF(rect.left() - 30, rect.bottom() + 30),
                    QPointF(rect.left() - 30, p2.y())
                ]
            else:
                mid_x = (p1.x() + p2.x()) / 2
                self.pivots = [QPointF(mid_x, p1.y()), QPointF(mid_x, p2.y())]
        # Enforce Manhattan Snapping
        pts = [p1] + self.pivots + [p2]
        for i in range(1, len(pts)-1):
            prev = pts[i-1]
            if (i-1) % 2 == 0: pts[i].setY(prev.y())
            else: pts[i].setX(prev.x())
        pts[-2].setY(p2.y())
        self.full_pts = pts
        # Sync back to model
        if self.model:
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