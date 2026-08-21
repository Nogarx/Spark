#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.graph_editor.models.node_model import NodeModel
    from spark.graph_editor.models.port_model import PortModel
    from spark.graph_editor.view.pipe_item import PipeItem
    from spark.graph_editor.view.graph_view import GraphScene

import math
import typing as tp
from shiboken6 import isValid
from PySide6.QtWidgets import QGraphicsItem, QGraphicsTextItem, QGraphicsObject, QStyleOption, QWidget, QGraphicsSceneMouseEvent
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPainter, QBrush, QColor, QPen, QFont, QFontMetrics, QPolygonF
from spark.graph_editor.styles.manager import STYLES

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class PortItem(QGraphicsObject):

    def __init__(self, model: PortModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = model
        self.radius = STYLES.get_val('port', 'radius')
        self.setAcceptHoverEvents(True)
        self._hovered = False
        self.connected_pipes: list[PipeItem] = []

    def add_pipe(self, pipe: PipeItem) -> None:
        if pipe not in self.connected_pipes: 
            self.connected_pipes.append(pipe)

    def remove_pipe(self, pipe: PipeItem) -> None:
        if pipe in self.connected_pipes: 
            self.connected_pipes.remove(pipe)

    def get_pipe_at(self, pos: QPointF) -> PipeItem | None:
        if not self.connected_pipes: 
            return None
        return self.connected_pipes[-1]

    def hoverEnterEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self._hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self._hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def boundingRect(self) -> QRectF:
        r = self.radius
        # Expand bounding rect by 4 pixels to account for pen width and custom shapes (stars/polygons)
        return QRectF(-r - 4.0, -r - 4.0, 2.0 * r + 8.0, 2.0 * r + 8.0)

    def paint(self, painter: QPainter, option: QStyleOption, widget: QWidget) -> None:
        r = self.radius
        rect = QRectF(-r, -r, 2.0 * r, 2.0 * r)
        hovered = getattr(self, '_hovered', False)
        connected = len(self.connected_pipes) > 0
        style = STYLES.get_port_style(self.model.port_type)
        base_color = QColor(style.get('color'))
        shape_type = style.get('shape', 'circle')
        sides = style.get('sides', 4)
        # Unconnected: Dark fill, Colored border
        # Connected: Colored fill (semi-transparent), Colored border
        # Hovered: Bright colored fill, Colored border
        if hovered:
            fill_color = QColor(base_color)
            fill_color.setAlpha(230)
            border_color = QColor(base_color)
        elif connected:
            fill_color = QColor(base_color)
            fill_color.setAlpha(180)
            border_color = QColor(base_color)
        else:
            fill_color = QColor(10, 10, 10, 150)
            border_color = QColor(base_color)
        painter.setBrush(QBrush(fill_color))
        pen = QPen(border_color, 1.8)
        pen.setJoinStyle(Qt.MiterJoin)
        painter.setPen(pen)
        cx, cy = 0.0, 0.0
        if shape_type == 'polygon':
            poly = QPolygonF()
            for i in range(sides):
                theta = 2 * math.pi * i / sides - math.pi/4
                x = cx + r * math.cos(theta)
                y = cy + r * math.sin(theta)
                poly.append(QPointF(x, y))
            painter.drawPolygon(poly)
        elif shape_type == 'star':
            outer = r + 1.0 # Slight boost for stars
            inner = outer * 0.5
            poly = QPolygonF()
            total = sides * 2
            for i in range(total):
                rad = outer if (i % 2 == 0) else inner
                theta = math.pi/2 + math.pi * i / sides
                x = cx + rad * math.cos(theta)
                y = cy - rad * math.sin(theta)
                poly.append(QPointF(x, y))
            painter.drawPolygon(poly)
        else:
            painter.drawEllipse(rect)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class PropertyRowItem(QGraphicsItem):

    def __init__(self, name: str, input_port_model: PortModel | None = None, output_port_model: PortModel | None = None, parent: QWidget | None = None,) -> None:
        super().__init__(parent)
        self.name = name
        self.width = parent.width if parent else 180.0
        self.height = 20.0
        if input_port_model:
            self.input_port = PortItem(input_port_model, self)
            self.input_port.setPos(0, self.height/2)
        else:
            self.input_port = None
        if output_port_model:
            self.output_port = PortItem(output_port_model, self)
            self.output_port.setPos(self.width, self.height/2)
        else:
            self.output_port = None
        self.label = QGraphicsTextItem(name, self)
        self.label.setDefaultTextColor(STYLES.get_color('port', 'label_color'))
        self.label.setFont(QFont('Segoe UI', 8))
        self._center_label()

    def _center_label(self) -> None:
        fm = QFontMetrics(self.label.font())
        tw = fm.horizontalAdvance(self.name)
        self.label.setPos((self.width - tw)/2, -2)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.width, self.height)

    def paint(self, painter: QPainter, option: QStyleOption, widget: QWidget) -> None:
        pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class OptionalDividerItem(QGraphicsItem):
    """
        A custom separator for optional ports, drawing '-- Optional --'.
    """

    def __init__(self, width: float, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.width = width
        self.height = 14.0

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.width, self.height)

    def paint(self, painter: QPainter, option: QStyleOption, widget: QWidget) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        y = self.height / 2.0
        text = 'Optional'
        font = QFont('Segoe UI', 7)
        font.setItalic(True)
        painter.setFont(font)
        fm = QFontMetrics(font)
        tw = fm.horizontalAdvance(text)
        # Dashed lines
        painter.setPen(QPen(QColor(255, 255, 255, 30), 1, Qt.PenStyle.DashLine))
        spacing = 6
        painter.drawLine(20, y, self.width/2 - tw/2 - spacing, y)
        painter.drawLine(self.width/2 + tw/2 + spacing, y, self.width - 20, y)
        # Text
        painter.setPen(QPen(QColor(150, 150, 150, 150)))
        painter.drawText(QRectF(0, 0, self.width, self.height), Qt.AlignmentFlag.AlignCenter, text)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class NodeItem(QGraphicsItem):

    def __init__(self, model: NodeModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Model
        self.model = model
        self.rows: list[QGraphicsItem] = []
        self.section_headers: list[tuple[str, float]] = [] # (text, y_pos)
        # Style
        self.width = STYLES.get_val('node', 'width')
        self.header_height = STYLES.get_val('node', 'header_height')
        self.padding = 5.0
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        # Headers
        # Title
        self.title_item = QGraphicsTextItem(self.model.name, self)
        self.title_item.setDefaultTextColor(STYLES.get_color('node', 'text_color'))
        self.title_item.setFont(QFont('Segoe UI', 10, QFont.Weight.Bold))
        self.title_item.setPos(5, 2)
        # Class
        self.type_item = QGraphicsTextItem(self.model.type_name.upper(), self)
        self.type_item.setDefaultTextColor(STYLES.get_color('node', 'type_text_color'))
        self.type_item.setFont(QFont('Segoe UI', 7, QFont.Weight.Bold))
        self.type_item.setPos(7, 22)
        # Callbacks
        self._setup_content()
        self.model.position_changed.connect(self.on_model_pos_changed)
        self.model.selected_changed.connect(self.on_model_selected_changed)
        self.model.name_changed.connect(self.on_model_name_changed)

    def on_model_pos_changed(self, x: float, y: float) -> None:
        if self.pos() != QPointF(x, y):
            self.setPos(x, y)

    def on_model_selected_changed(self, is_selected: bool) -> None:
        if self.isSelected() != is_selected:
            self.setSelected(is_selected)
            
    def on_model_name_changed(self, new_name: str) -> None:
        self.title_item.setPlainText(new_name)

    def _setup_content(self) -> None:
        self.section_headers = []
        self.rows = []
        current_y = self.header_height
        # Call Section
        if self.model.call_section.ports:
            self.section_headers.append(('Call', current_y))
            current_y += 18
            mandatory = [p for p in self.model.call_section.ports if not p.is_optional]
            optional = [p for p in self.model.call_section.ports if p.is_optional]
            # Required ports
            for group in self._group_ports(mandatory):
                row = PropertyRowItem(name=group['name'], input_port_model=group['in'], output_port_model=group['out'], parent=self)
                row.setPos(0, current_y)
                self.rows.append(row)
                current_y += row.height
            # Optional ports
            if optional:
                current_y += 2
                divider = OptionalDividerItem(self.width, self)
                divider.setPos(0, current_y)
                self.rows.append(divider)
                current_y += divider.height + 2
                for group in self._group_ports(optional):
                    row = PropertyRowItem(name=group['name'], input_port_model=group['in'], output_port_model=group['out'], parent=self)
                    # Dim the label slightly for optional ports
                    row.label.setDefaultTextColor(QColor(150, 150, 150, 150))
                    row.setPos(0, current_y)
                    self.rows.append(row)
                    current_y += row.height
            current_y += 10
        # Properties
        if self.model.props_section.ports:
            self.section_headers.append(('Properties', current_y))
            current_y += 18
            for group in self._group_ports(self.model.props_section.ports):
                row = PropertyRowItem(name=group['name'], input_port_model=group['in'], output_port_model=group['out'], parent=self)
                row.setPos(0, current_y)
                self.rows.append(row)
                current_y += row.height
        self.total_height = current_y + 10

    @staticmethod
    def _group_ports(ports: list[PortModel]) -> list[dict]:
        groups: list[dict] = []
        index: dict[str, dict] = {}
        for port in ports:
            key = port.name.rstrip('*')
            slot = 'in' if port.is_input else 'out'
            group = index.get(key)
            if group is None or group[slot] is not None:
                group = {'name': key, 'in': None, 'out': None}
                groups.append(group)
                index[key] = group
            group[slot] = port
        return groups

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = self.pos()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        super().mouseReleaseEvent(event)

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value: QPointF) -> QPointF:
        if not isValid(self):
            return super().itemChange(change, value)
        scene: GraphScene | None = self.scene()
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and scene:
            if STYLES.get_val('graph', 'snapping', 'enabled'):
                grid = float(STYLES.get_val('graph', 'snapping', 'node_grid'))
                x = round(value.x() / grid) * grid
                y = round(value.y() / grid) * grid
                value = QPointF(x, y)
            # Shift shared pipes if both ends are selected and moving
            # ONLY during an active mouse drag (not during undo/redo)
            if scene.mouseGrabberItem() == self:
                delta = value - self.pos()
                if delta.x() != 0 or delta.y() != 0:
                    for row in self.rows:
                        ports: list[PortItem] = []
                        if isinstance(row, PropertyRowItem) and row.input_port: 
                            ports.append(row.input_port)
                        if isinstance(row, PropertyRowItem) and row.output_port: 
                            ports.append(row.output_port)
                        for port in ports:
                            for pipe in port.connected_pipes:
                                src_node = pipe.source_port.parentItem().parentItem() if pipe.source_port else None
                                dst_node = pipe.target_port.parentItem().parentItem() if pipe.target_port else None
                                if src_node and dst_node and src_node.isSelected() and dst_node.isSelected():
                                    # We apply half the delta per node; both nodes will trigger this.
                                    for p in pipe.pivots:
                                        p.setX(p.x() + delta.x() / 2.0)
                                        p.setY(p.y() + delta.y() / 2.0)
            return value
        elif change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self.model.is_selected = value
        elif change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            if scene:
                scene.update_all_pipes()
        return super().itemChange(change, value)

    def boundingRect(self) -> QRectF:
        h = getattr(self, 'total_height', 150.0)
        return QRectF(-5, -5, self.width + 10, h + 10)

    def paint(self, painter: QPainter, option: QStyleOption, widget: QWidget) -> None:
        h = getattr(self, 'total_height', 150.0)
        rect = QRectF(0, 0, self.width, h)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Body
        painter.setBrush(STYLES.get_color('node', 'background_color'))
        border_color = STYLES.get_color('node', 'selected_border_color' if self.isSelected() else 'border_color')
        painter.setPen(QPen(border_color, 1.2))
        painter.drawRoundedRect(rect, 4, 4)
        # Header background
        header_rect = QRectF(0, 0, self.width, self.header_height)
        painter.setBrush(STYLES.get_color('node', 'header_color'))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(header_rect, 4, 4)
        painter.drawRect(0, self.header_height-10, self.width, 10)
        # Section Headers
        for text, y in self.section_headers:
            bar_rect = QRectF(0, y, self.width, 16)
            painter.setBrush(STYLES.get_color('node', 'section_header_bg'))
            painter.setPen(Qt.NoPen)
            painter.drawRect(bar_rect)
            painter.setPen(QPen(STYLES.get_color('node', 'section_header_label'), 1))
            painter.setFont(QFont('Segoe UI', 7, QFont.Weight.Bold))
            # Center the text both horizontally and vertically
            painter.drawText(bar_rect, Qt.AlignmentFlag.AlignCenter, text.upper())
            # Subtle section divider line
            painter.setPen(QPen(QColor(0, 0, 0, 80), 1))
            painter.drawLine(bar_rect.bottomLeft(), bar_rect.bottomRight())
        # Main Header separator line
        painter.setPen(QPen(QColor(0, 0, 0, 100), 1.5))
        painter.drawLine(0, self.header_height, self.width, self.header_height)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################