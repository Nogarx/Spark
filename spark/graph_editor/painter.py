#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import math
from PySide6 import QtGui, QtCore
import spark.core.payloads as vars
from spark.graph_editor.utils import _normalize_name
from spark.core.payloads import SparkPayload

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def _get_colors(info):
    if info['hovered']:
        return QtGui.QColor(14, 45, 59), QtGui.QColor(136, 255, 35, 255)
    elif info['connected']:
        return QtGui.QColor(*info['color']), QtGui.QColor(*info['border_color'])
    else:
        return QtGui.QColor(195, 60, 60), QtGui.QColor(200, 130, 70)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def make_port_painter(shape_builder):
    def draw_port(painter, rect, info):
        painter.save()
        fill_color, border_color = _get_colors(info)
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

def regular_polygon_builder(sides):
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

def star_builder(points, inner_ratio=0.5):
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

class Pallete:

    pallete = {
        # Triangle
        _normalize_name(vars.IntegerArray.__name__): make_port_painter(regular_polygon_builder(3)),
        # Square
        _normalize_name(vars.FloatArray.__name__):   make_port_painter(regular_polygon_builder(4)),
        # Pentagon
        _normalize_name(vars.PotentialArray.__name__): make_port_painter(regular_polygon_builder(5)),
        # Hexagon
        _normalize_name(vars.CurrentArray.__name__):  make_port_painter(regular_polygon_builder(6)),
        # Star
        _normalize_name(vars.SpikeArray.__name__):     make_port_painter(star_builder(points=5, inner_ratio=0.5)),
        # Circle
        'default': make_port_painter(lambda rect: QtGui.QPainterPath().addEllipse(rect) or QtGui.QPainterPath()),
    }

    def __call__(self, payload: type[SparkPayload]):
        painter = self.pallete.get(_normalize_name(payload), None)
        if painter is None:
            painter = self.pallete['default']
        return painter

DEFAULT_PALLETE = Pallete()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################