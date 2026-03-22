#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import enum
import math
import typing as tp
from PySide6 import QtGui, QtCore
import spark.core.payloads as vars
import spark.core.utils as utils
from spark.core.payloads import SparkPayload

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class PortColorStyle(enum.Enum):
    DEFAULT = enum.auto()
    OPTIONAL = enum.auto()
    PROPERTY = enum.auto()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

#DEFAULT_COLOR = QtGui.QColor(3, 92, 186, 255)
#DEFAULT_HOVER_COLOR = QtGui.QColor(3, 92, 186, 230)
#DEFAULT_CONNECTED_COLOR = QtGui.QColor(3, 92, 186, 180)

DEFAULT_COLOR = QtGui.QColor(3, 199, 186, 255)
DEFAULT_HOVER_COLOR = QtGui.QColor(3, 199, 186, 230)
DEFAULT_CONNECTED_COLOR = QtGui.QColor(3, 199, 186, 180)

PROPERTY_COLOR = QtGui.QColor(102, 70, 125, 255)
PROPERTY_HOVER_COLOR = QtGui.QColor(102, 70, 125, 230)
PROPERTY_CONNECTED_COLOR = QtGui.QColor(102, 70, 125, 180)

OPTIONAL_COLOR = QtGui.QColor(236, 164, 52, 255)
OPTIONAL_HOVER_COLOR = QtGui.QColor(236, 164, 52, 230)
OPTIONAL_CONNECTED_COLOR = QtGui.QColor(236, 164, 52, 180)

MISSING_COLOR = QtGui.QColor(154, 68, 68, 255)
NOT_CONNECTED_FILL = QtGui.QColor(10, 10, 10, 150)

def _get_colors(info, color_style: PortColorStyle = PortColorStyle.DEFAULT) -> tuple[QtGui.QColor, QtGui.QColor]:
    if color_style == PortColorStyle.DEFAULT:
        # Standard color map
        if info['hovered']:
            return DEFAULT_HOVER_COLOR, DEFAULT_COLOR
        elif info['connected']:
            return DEFAULT_CONNECTED_COLOR, DEFAULT_COLOR
        else:
            return NOT_CONNECTED_FILL, MISSING_COLOR
    elif color_style == PortColorStyle.OPTIONAL:
        # Optional color map
        if info['hovered']:
            return OPTIONAL_HOVER_COLOR, OPTIONAL_COLOR
        elif info['connected']:
            return OPTIONAL_CONNECTED_COLOR, OPTIONAL_COLOR
        else:
            return NOT_CONNECTED_FILL, OPTIONAL_COLOR
    elif color_style == PortColorStyle.PROPERTY:
        # Property color map
        if info['hovered']:
            return PROPERTY_HOVER_COLOR, PROPERTY_COLOR
        elif info['connected']:
            # Using fixed purple tones for connected properties
            return PROPERTY_CONNECTED_COLOR, PROPERTY_COLOR
        else:
            # Default unconnected property state
            return NOT_CONNECTED_FILL, PROPERTY_COLOR
        
#-----------------------------------------------------------------------------------------------------------------------------------------------#

def make_port_painter(shape_builder, color_style: PortColorStyle = PortColorStyle.DEFAULT) -> tp.Callable:
    def draw_port(painter, rect, info) -> None:
        painter.save()
        fill_color, border_color = _get_colors(info, color_style)
        pen = QtGui.QPen(border_color, 1.8)
        pen.setJoinStyle(QtCore.Qt.MiterJoin)
        painter.setPen(pen)
        painter.setBrush(fill_color)

        shape = shape_builder(rect)
        if isinstance(shape, QtGui.QPolygonF):
            painter.drawPolygon(shape)
        else:
            painter.drawPath(shape)
        painter.restore()
    return draw_port

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def regular_polygon_builder(sides) -> tp.Callable:
    def _builder(rect):
        cx, cy = rect.center().x(), rect.center().y()
        r = min(rect.width(), rect.height()) / 2.0
        poly = QtGui.QPolygonF()
        for i in range(sides):
            theta = 2 * math.pi * i / sides - math.pi/4
            x = cx + r * math.cos(theta)
            y = cy + r * math.sin(theta)
            poly.append(QtCore.QPointF(x, y))
        return poly
    return _builder

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def star_builder(points, inner_ratio=0.5) -> tp.Callable:
    """
    Builds a star (spark) with `points` spikes.
    inner_ratio: fraction of outer radius for the inner vertices.
    """
    def _builder(rect):
        cx, cy = rect.center().x(), rect.center().y()
        outer = min(rect.width(), rect.height()) / 2.0
        inner = outer * inner_ratio
        poly = QtGui.QPolygonF()
        total = points * 2
        for i in range(total):
            r = outer if (i % 2 == 0) else inner
            theta = math.pi/2 + math.pi * i / points
            x = cx + r * math.cos(theta)
            y = cy - r * math.sin(theta)
            poly.append(QtCore.QPointF(x, y))
        return poly
    return _builder

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class Palette:
    def __init__(self) -> None:

        self.builders = {
            # Triangle
            utils.normalize_str(vars.IntegerArray.__name__): regular_polygon_builder(3),
            # Square
            utils.normalize_str(vars.FloatArray.__name__): regular_polygon_builder(4),
            # Pentagon
            utils.normalize_str(vars.PotentialArray.__name__): regular_polygon_builder(5),
            # Hexagon
            utils.normalize_str(vars.CurrentArray.__name__): regular_polygon_builder(6),
            # Hexagon
            utils.normalize_str(vars.BooleanMask.__name__): regular_polygon_builder(7),
            # Star
            utils.normalize_str(vars.SpikeArray.__name__): star_builder(points=5, inner_ratio=0.5),
            # Circle
            'default': lambda rect: QtGui.QPainterPath().addEllipse(rect) or QtGui.QPainterPath(),
        }

    def __call__(self, payload: type[SparkPayload], color_style: PortColorStyle = PortColorStyle.DEFAULT):
        # Fetch the builder, defaulting to the circle builder
        builder = self.builders.get(utils.normalize_str(payload), self.builders['default'])
        # Generate and return the painter closure with the property flag applied
        return make_port_painter(builder, color_style)

DEFAULT_PALETTE = Palette()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################