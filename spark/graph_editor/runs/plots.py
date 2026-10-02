#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp

import math

import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QImage, QPolygonF, QPainterPath
from PySide6.QtWidgets import QWidget, QSizePolicy, QToolTip, QMenu

from spark.graph_editor.styles.run_viewer import THEME

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

MAX_IMAGE = (4096, 1024)
"""
    Largest raster or trace image, in steps and units. Larger values are reduced in blocks, with
    ``any`` for a raster and the mean otherwise.
"""

MAX_MATRIX = (1024, 1024)
"""
    Largest matrix image, in rows and columns. Larger matrices are reduced in blocks, as for
    `MAX_IMAGE`.
"""

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def nice_ticks(lo: float, hi: float, count: int = 4) -> list[float]:
    """
        Returns round tick values within ``[lo, hi]``.

        The spacing is 1, 2, 2.5 or 5 times a power of ten, the smallest giving at most ``count``
        intervals. An empty range gives ``[lo]``, or no tick when ``lo`` is not finite.
    """
    if not np.isfinite(hi - lo) or hi <= lo:
        return [lo] if np.isfinite(lo) else []
    raw = (hi - lo) / count
    magnitude = 10 ** math.floor(math.log10(raw))
    step = min((m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw), default=raw)
    first = math.ceil(lo / step) * step
    return [first + i * step for i in range(int((hi - first) / step) + 1)]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def format_value(value: float) -> str:
    """
        Formats a value for a label.

        Magnitudes from 1e-2 up to 1e4 are written with three significant digits, others in
        scientific notation with two. Zero is ``'0'``.
    """
    if value == 0:
        return '0'
    if abs(value) >= 1e4 or abs(value) < 1e-2:
        return f'{value:.1e}'
    return f'{value:.3g}'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _decimals(value: float) -> int:
    """
        Returns the decimals needed to write ``value``, up to 12.
    """
    for decimals in range(12):
        if abs(round(value, decimals) - value) <= abs(value) * 1e-9:
            return decimals
    return 12

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def tick_labels(values: list[float]) -> list[str]:
    """
        Returns labels for evenly spaced ticks, with the digits that tell them apart.

        Ticks whose largest magnitude is from 1e-3 up to 1e9 are written in fixed point with
        thousands separators, such as ``12,250``. Others are written in scientific notation, such as
        ``1.25e+10``. Fewer than two ticks are written with `format_value`.
    """
    if len(values) < 2:
        return [format_value(v) for v in values]
    spacing = abs(values[1] - values[0])
    largest = max(abs(v) for v in values)
    if not math.isfinite(spacing) or spacing <= 0:
        return [format_value(v) for v in values]
    spacing = float(f'{spacing:.6g}')
    if 1e-3 <= largest < 1e9:
        decimals = _decimals(spacing)
        return [f'{v + 0.0:,.{decimals}f}' for v in values]
    scale = math.floor(math.log10(spacing))
    digits = max(math.floor(math.log10(largest)) - scale, 0) + _decimals(spacing / 10 ** scale)
    return [f'{v:.{digits}e}' for v in values]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _shrink(values: np.ndarray, axis: int, limit: int) -> np.ndarray:
    """
        Reduces ``values`` along ``axis`` in equal blocks to at most ``limit`` entries.

        Booleans are reduced with ``any``, others with the mean. The last block may be shorter.
    """
    n = values.shape[axis]
    if n <= limit:
        return values
    block = -(-n // limit)
    reduce = (lambda a: a.any(axis=axis + 1)) if values.dtype == np.bool_ else (lambda a: a.mean(axis=axis + 1))
    whole = n - n % block
    # Whole blocks reduced in place, the last one, shorter, on its own: no padded copy of the values.
    head = np.take(values, np.arange(whole), axis=axis) if whole < n else values
    shape = list(head.shape)
    shape[axis:axis + 1] = [whole // block, block]
    reduced = reduce(head.reshape(shape))
    if whole == n:
        return reduced
    tail = np.take(values, np.arange(whole, n), axis=axis)
    shape = list(tail.shape)
    shape[axis:axis + 1] = [1, n - whole]
    return np.concatenate([reduced, reduce(tail.reshape(shape))], axis=axis)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def value_range(values: np.ndarray) -> tuple[float, float]:
    """
        Returns the lowest and the highest finite value, or ``(0.0, 1.0)`` without one.
    """
    finite = values[np.isfinite(values)]
    return (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _at_or_before(steps: np.ndarray, x: float) -> int:
    """
        Returns the index of the last of the ordered ``steps`` at or before ``x``, or -1.
    """
    return int(np.searchsorted(steps, x, side='right')) - 1

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _image(values: np.ndarray) -> tuple[QImage, np.ndarray]:
    """
        Converts a 2D array to an image, with row 0 at the bottom.

        Booleans are drawn as the plots are, True as an event of `THEME`. Numbers are drawn with the
        colormap of `THEME` over their finite range. Returns the image and the buffer it reads,
        which must outlive the image.
    """
    if values.dtype == np.bool_:
        event, silent = np.uint32(THEME.raster_event.rgba()), np.uint32(THEME.area.rgba())
        pixels = np.where(values, event, silent).astype(np.uint32)
    else:
        values = values.astype(np.float64)
        lo, hi = value_range(values)
        scaled = np.clip((values - lo) / max(hi - lo, 1e-12) * 255, 0, 255)
        pixels = THEME.colormap_pixels[np.nan_to_num(scaled).astype(np.int64)]
    buffer = np.ascontiguousarray(pixels[::-1])
    height, width = buffer.shape
    return QImage(buffer.data, width, height, width * 4, QImage.Format.Format_ARGB32), buffer

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class _Plot(QWidget):
    """
        Base of the plots, drawing their frame, axes, title and cursor.

        A plot with ``FOLLOWS_CURSOR`` has steps on its horizontal axis and draws the cursor. A
        press or a drag with the left button emits `cursor_moved` with the step under the mouse.
        Over the data, the mouse shows `hover_text` in a tooltip, and over the title, ``detail``.
        ``readout`` is drawn at the right of the title, in the colors of ``readout_colors`` when
        given, else in the color of the cursor. With ``colorbar``, a ``(low, high)`` range, a colour
        bar of the colormap is drawn there instead, between its lowest and highest value. The axes
        are named with `set_labels`. Every plot has the same margins left and right, so that plots of
        the same width have axes of the same width; a name of the horizontal axis makes the plot
        taller. Colours, fonts, margins and sizes are those of `THEME`, such as its ``plot_margins``.
    """

    cursor_moved = Signal(float)
    FOLLOWS_CURSOR = True
    HEIGHT = 'series_height'
    """
        Size of `THEME` giving the height of the plot when none is given.
    """

    def __init__(self, title: str = '', height: int | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title = title
        self.subtitle = ''
        self.x_label = ''
        self.y_label = ''
        self._height = getattr(THEME, self.HEIGHT) if height is None else height
        self.readout = ''
        self.readout_colors: list[QColor] = []
        self.detail = ''
        self.colorbar: tuple[float, float] | None = None
        self.x_range: tuple[float, float] = (0.0, 1.0)
        self.y_range: tuple[float, float] = (0.0, 1.0)
        self.cursor: float | None = None
        self.message = ''
        self.setMinimumHeight(self._height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)

    def plot_rect(self) -> QRectF:
        """
            Returns the area inside the margins, where the data is drawn.
        """
        left, top, right, bottom = THEME.plot_margins
        bottom += THEME.axis_label_size if self.x_label else 0
        return QRectF(left, top, max(self.width() - left - right, 1), max(self.height() - top - bottom, 1))

    def set_labels(self, x: str | None = None, y: str | None = None) -> None:
        """
            Names the horizontal axis ``x`` and the vertical axis ``y``. A name not given is kept;
            an empty one is removed.
        """
        self.x_label = self.x_label if x is None else x
        self.y_label = self.y_label if y is None else y
        self.setMinimumHeight(self._height + (THEME.axis_label_size if self.x_label else 0))
        self.update()

    def set_cursor(self, x: float | None) -> None:
        """
            Moves the cursor to ``x``, or hides it for None. Ignored without ``FOLLOWS_CURSOR``.
        """
        self.cursor = x if self.FOLLOWS_CURSOR else None
        self.update()

    def _px(self, x: np.ndarray | float, rect: QRectF) -> tp.Any:
        x0, x1 = self.x_range
        with np.errstate(over='ignore', invalid='ignore'):
            return rect.left() + (np.asarray(x, dtype=np.float64) - x0) / max(x1 - x0, 1e-12) * rect.width()

    def _py(self, y: np.ndarray | float, rect: QRectF) -> tp.Any:
        y0, y1 = self.y_range
        # Ranges past the float64 range map to NaN, which is not drawn.
        with np.errstate(over='ignore', invalid='ignore'):
            return rect.bottom() - (np.asarray(y, dtype=np.float64) - y0) / max(y1 - y0, 1e-12) * rect.height()

    def to_px(self, x: np.ndarray | float, y: np.ndarray | float, rect: QRectF) -> tuple[tp.Any, tp.Any]:
        """
            Maps data coordinates to pixel coordinates in ``rect``.
        """
        return self._px(x, rect), self._py(y, rect)

    def from_px(self, px: float, py: float, rect: QRectF) -> tuple[float, float]:
        """
            Maps pixel coordinates in ``rect`` to data coordinates.
        """
        x0, x1 = self.x_range
        y0, y1 = self.y_range
        return (x0 + (px - rect.left()) / max(rect.width(), 1e-12) * (x1 - x0),
                y0 + (rect.bottom() - py) / max(rect.height(), 1e-12) * (y1 - y0))

    def hover_text(self, x: float, y: float) -> str | None:
        """
            Returns the tooltip at data coordinates ``(x, y)``, or None.

            Implemented by each plot.
        """
        return None

    def mousePressEvent(self, event) -> None:
        if self.FOLLOWS_CURSOR and event.button() == Qt.MouseButton.LeftButton:
            self._emit_cursor(event)

    def mouseMoveEvent(self, event) -> None:
        if self.FOLLOWS_CURSOR and event.buttons() & Qt.MouseButton.LeftButton:
            self._emit_cursor(event)
            return
        position, rect = event.position(), self.plot_rect()
        text = None
        if position.y() < THEME.plot_margins[1]:
            text = self.detail
        elif not self.message and rect.contains(position):
            text = self.hover_text(*self.from_px(position.x(), position.y(), rect))
        if text:
            QToolTip.showText(event.globalPosition().toPoint(), text, self)
        else:
            QToolTip.hideText()

    def leaveEvent(self, event) -> None:
        QToolTip.hideText()
        super().leaveEvent(event)

    def _emit_cursor(self, event) -> None:
        rect = self.plot_rect()
        fraction = (event.position().x() - rect.left()) / rect.width()
        if 0 <= fraction <= 1:
            x0, x1 = self.x_range
            self.cursor_moved.emit(x0 + fraction * (x1 - x0))

    # Drawing.

    def title_text(self) -> str:
        """
            Returns the text drawn as the title.
        """
        return self.title + self.subtitle

    def paintEvent(self, event) -> None:
        # A painter left active when drawing raises takes the process down with its paint device.
        painter = QPainter(self)
        try:
            self._paint(painter)
        finally:
            painter.end()

    def _paint(self, painter: QPainter) -> None:
        painter.fillRect(self.rect(), THEME.background)
        rect = self.plot_rect()
        painter.setFont(THEME.font())
        self._draw_title(painter)
        if self.message:
            painter.setPen(THEME.muted)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.message)
            return
        painter.fillRect(rect, THEME.area)
        self._draw_axes(painter, rect)
        painter.save()
        painter.setClipRect(rect)
        self.draw(painter, rect)
        painter.restore()
        if self.cursor is not None:
            px = self._px(self.cursor, rect)
            if rect.left() <= px <= rect.right():
                painter.setPen(QPen(THEME.cursor, 1))
                painter.drawLine(QPointF(px, rect.top()), QPointF(px, rect.bottom()))

    def _draw_title(self, painter: QPainter) -> None:
        title = QRectF(4, 1, self.width() - 8, THEME.plot_margins[1] - 2)
        metrics = painter.fontMetrics()
        if self.colorbar is not None:
            taken = self._draw_colorbar(painter, title)
            self._draw_title_text(painter, title, taken)
            return
        parts = self.readout.split('   ') if self.readout_colors else [self.readout]
        colors = self.readout_colors if self.readout_colors else [THEME.cursor]
        widths = [metrics.horizontalAdvance(part) for part in parts] if self.readout else []
        taken = sum(widths) + 10 * max(len(widths) - 1, 0) + (12 if widths else 0)
        self._draw_title_text(painter, title, taken)
        x = title.right()
        for part, width, part_color in reversed(list(zip(parts, widths, colors))):
            x -= width
            painter.setPen(part_color)
            painter.drawText(QRectF(x, title.top(), width + 1, title.height()), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, part)
            x -= 10

    def _draw_title_text(self, painter: QPainter, title: QRectF, taken: float) -> None:
        """
            Draws the title in ``title``, elided before the ``taken`` pixels at its right, in the
            title font of `THEME`.
        """
        font = painter.font()
        painter.setFont(THEME.font(THEME.title_font_size))
        painter.setPen(THEME.text)
        painter.drawText(title, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         painter.fontMetrics().elidedText(self.title_text(), Qt.TextElideMode.ElideRight, int(title.width() - taken)))
        painter.setFont(font)

    def _draw_axes(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QPen(THEME.frame, 1))
        labelled = bool(getattr(self, 'labels', None))
        painter.drawRect(rect)
        ticks = nice_ticks(*self.y_range, count=3)
        for value, label in zip(ticks, tick_labels(ticks)):
            py = self._py(value, rect)
            painter.setPen(QPen(THEME.grid, 1))
            painter.drawLine(QPointF(rect.left(), py), QPointF(rect.right(), py))
            painter.setPen(THEME.text)
            painter.drawText(QRectF(0, py - 7, THEME.plot_margins[0] - 4, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, label)
        ticks = [] if labelled else nice_ticks(*self.x_range, count=5)
        for value, label in zip(ticks, tick_labels(ticks)):
            px = self._px(value, rect)
            painter.setPen(QPen(THEME.grid, 1))
            painter.drawLine(QPointF(px, rect.top()), QPointF(px, rect.bottom()))
            painter.setPen(THEME.text)
            painter.drawText(QRectF(px - 40, rect.bottom() + 2, 80, 14), Qt.AlignmentFlag.AlignCenter, label)
        painter.setPen(THEME.text)
        if self.x_label:
            painter.drawText(QRectF(rect.left(), rect.bottom() + 15, rect.width(), THEME.axis_label_size), Qt.AlignmentFlag.AlignCenter, self.x_label)
        if self.y_label:
            # Read upwards, along the left edge.
            painter.save()
            painter.translate(1, rect.center().y())
            painter.rotate(-90)
            painter.drawText(QRectF(-rect.height() / 2, 0, rect.height(), THEME.axis_label_size), Qt.AlignmentFlag.AlignCenter, self.y_label)
            painter.restore()

    def _draw_colorbar(self, painter: QPainter, title: QRectF) -> float:
        """
            Draws the colour bar of ``colorbar`` at the right of the title line, between its lowest
            and its highest value. Returns the width it takes.
        """
        lo, hi = self.colorbar
        metrics = painter.fontMetrics()
        low, high = format_value(lo), format_value(hi)
        right = title.right() - metrics.horizontalAdvance(high)
        bar = QRectF(right - 4 - THEME.colorbar_width, title.center().y() - 3, THEME.colorbar_width, 6)
        left = bar.left() - 4 - metrics.horizontalAdvance(low)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(bar, THEME.colorbar_image)
        painter.setPen(THEME.text)
        painter.drawText(QRectF(left, title.top(), bar.left() - left, title.height()), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, low)
        painter.drawText(QRectF(right, title.top(), title.right() - right + 1, title.height()), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, high)
        return title.right() - left + 12

    def draw(self, painter: QPainter, rect: QRectF) -> None:
        """
            Draws the data in ``rect``, clipped to it.

            Implemented by each plot.
        """

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SeriesPlot(_Plot):
    """
        Line plot of series against the step.

        Series take the series colours of `THEME` in order, unless given one. Bands, such as one
        standard deviation around a mean, are drawn under the series. With more than one series, a
        legend lists the first eight. The title ends with the values at the cursor, when there are
        at most `READOUT` of them: the value of each series in its color, or the values of a
        series and its bands by name.

        A drag with the right button zooms to the box drawn (`select`): its values become the
        vertical range, and its steps the horizontal range, or, with ``shares_view``, are asked
        for with `view_requested`, so that the plots sharing a view follow. A right click without
        a drag opens a menu that fits the values or shows every step. Ctrl and the wheel zoom the
        vertical axis around the mouse; a double click fits it to the values shown again.
    """

    view_requested = Signal(float, float)
    READOUT = 4
    """
        Largest number of values at the cursor given in the title.
    """
    DRAG = 4
    """
        Pixels the mouse moves, with the right button held, past which it draws a box to zoom to.
    """
    legend = True
    """
        Whether the legend is drawn within the plot, with more than one series.
    """

    def __init__(self, title: str = '', height: int | None = None, parent: QWidget | None = None) -> None:
        super().__init__(title, height, parent)
        self.series: list[tuple[str, np.ndarray, np.ndarray, QColor]] = []
        self.bands: list[tuple[str | tuple[str, str], np.ndarray, np.ndarray, np.ndarray, QColor]] = []
        # A vertical range set by zooming, kept until fitted again.
        self.zoomed: tuple[float, float] | None = None
        self.shares_view = False
        # Corners of the box being drawn with the right button, in pixels.
        self._box: tuple[QPointF, QPointF] | None = None
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        self._x_given: tuple[float, float] | None = None
        self._y_given: tuple[float, float] | None = None
        # Thin lines under the series, outside the legend, the readout and the tooltip.
        self.faint: list[tuple[np.ndarray, np.ndarray, QColor]] = []
        self._shapes: tuple[tp.Any, list[tuple[QPainterPath | QPolygonF | QPointF, QColor]]] | None = None
        self._bands: tuple[tp.Any, list[tuple[QPainterPath | QPolygonF, QColor]]] | None = None
        self._faints: tuple[tp.Any, list[tuple[QPainterPath | QPolygonF | QPointF, QColor]]] | None = None

    def set_series(self, series: list[tuple], x_range: tuple[float, float] | None = None, y_range: tuple[float, float] | None = None,
                   bands: list[tuple[str | tuple[str, str], np.ndarray, np.ndarray, np.ndarray, QColor]] | None = None,
                   faint: list[tuple[np.ndarray, np.ndarray, QColor]] | None = None) -> None:
        """
            Sets the series drawn, each as ``(label, x, y)`` or ``(label, x, y, color)``.

            ``bands`` are drawn under the series in order, each as ``(label, x, low, high, color)``.
            The label names half the width of the band, such as ``'std'``, or both of its edges, as
            a pair such as ``('min', 'max')``; an empty label, neither. ``faint`` lines, each as
            ``(x, y, color)``, are drawn thin between the bands and the series, outside the legend
            and the values given. The horizontal range is ``x_range``, else that of the steps. The
            vertical range is ``y_range``, else that of the finite values within the horizontal
            range, with a margin. Without points, the plot shows a message.
        """
        self.faint = [(np.asarray(x, np.float64), np.asarray(y, np.float64), QColor(c)) for x, y, c in (faint or [])]
        self.series = [
            (s[0], np.asarray(s[1], np.float64), np.asarray(s[2], np.float64), QColor(s[3]) if len(s) > 3 else THEME.series_color(i))
            for i, s in enumerate(series)
        ]
        self.bands = [(label, np.asarray(x, np.float64), np.asarray(lo, np.float64), np.asarray(hi, np.float64), QColor(c))
                      for label, x, lo, hi, c in (bands or [])]
        self._x_given, self._y_given = x_range, y_range
        self._ranges()

    def set_view(self, x_range: tuple[float, float] | None) -> None:
        """
            Shows the steps of ``x_range``, or all steps for None, the vertical range following the
            values shown.
        """
        self._x_given = x_range
        self._ranges()

    def select(self, x0: float, x1: float, y0: float | None = None, y1: float | None = None) -> None:
        """
            Zooms to the steps from ``x0`` to ``x1`` and, when given, the values from ``y0`` to
            ``y1``.

            With ``shares_view``, the steps are asked for with `view_requested` instead of shown.
        """
        if y0 is not None and y1 is not None and y1 > y0:
            self.zoomed = (float(y0), float(y1))
        if x1 > x0:
            if self.shares_view:
                self.view_requested.emit(float(x0), float(x1))
            else:
                self._x_given = (float(x0), float(x1))
        self._ranges()

    def show_every_step(self) -> None:
        """
            Shows every step, asked for with `view_requested` with ``shares_view``.
        """
        if self.shares_view:
            self.view_requested.emit(-math.inf, math.inf)
        else:
            self.set_view(None)

    def fit(self) -> None:
        """
            Fits the vertical range to the values shown again, after a zoom.
        """
        self.zoomed = None
        self._ranges()

    def zoom(self, factor: float, at: float | None = None) -> None:
        """
            Scales the vertical range by ``factor`` around the value ``at``, or its middle.
        """
        a, b = self.y_range
        at = (a + b) / 2 if at is None else at
        self.zoomed = (at - (at - a) * factor, at + (b - at) * factor)
        self._ranges()

    def _ranges(self) -> None:
        points = [s for s in self.series if len(s[1])]
        self.message = '' if points else 'Nothing recorded yet'
        if points:
            xs = np.concatenate([s[1] for s in points])
            self.x_range = self._x_given or (float(xs.min()), float(max(xs.max(), xs.min() + 1)))
            a, b = self.x_range
            shown = [y[(x >= a) & (x <= b)] for _, x, y, _ in points]
            shown += [v[(x >= a) & (x <= b)] for _, x, lo, hi, _ in self.bands for v in (lo, hi)]
            shown += [y[(x >= a) & (x <= b)] for x, y, _ in self.faint]
            ys = np.concatenate(shown)
            if not np.isfinite(ys).any():
                ys = np.concatenate([y for _, _, y, _ in points])
            if self.zoomed is not None:
                self.y_range = self.zoomed
            elif self._y_given is not None:
                self.y_range = self._y_given
            else:
                lo, hi = value_range(ys) if np.isfinite(ys).any() else (0.0, 0.0)
                if hi <= lo:
                    lo, hi = lo - 1, hi + 1
                pad = (hi - lo) * 0.06
                self.y_range = (lo - pad, hi + pad)
        self._shapes = self._bands = self._faints = None
        self.set_cursor(self.cursor)

    def title_text(self) -> str:
        return self.title + self.subtitle + ('  ·  zoomed' if self.zoomed is not None else '')

    # Values.

    def values_at(self, x: float) -> list[tuple[str, float]]:
        """
            Returns the label and value of every series at its last point at or before ``x``.

            A band gives half its width, or both of its edges, as its label names them.
        """
        found = []
        for label, xs, ys, _ in self.series:
            index = _at_or_before(xs, x)
            if index >= 0:
                found.append((label, float(ys[index])))
        for label, xs, lo, hi, _ in self.bands:
            index = _at_or_before(xs, x)
            if not label or index < 0:
                continue
            if isinstance(label, tuple):
                found += [(label[0], float(lo[index])), (label[1], float(hi[index]))]
            else:
                found.append((label, float(hi[index] - lo[index]) / 2))
        return found

    def set_cursor(self, x: float | None) -> None:
        super().set_cursor(x)
        found = [] if self.cursor is None else self.values_at(self.cursor)
        self.readout_colors = []
        if len(found) == 1:
            self.readout = format_value(found[0][1])
        elif len(found) > self.READOUT:
            self.readout = ''
        elif len(self.series) > 1:
            # One value per series, in its color.
            at = {label: value for label, value in found}
            shown = [(format_value(at[label]), series_color) for label, _, _, series_color in self.series if label in at]
            self.readout = '   '.join(text for text, _ in shown)
            self.readout_colors = [series_color for _, series_color in shown]
        else:
            self.readout = '   '.join(f'{label} {format_value(value)}' for label, value in found)

    def hover_text(self, x: float, y: float) -> str | None:
        found = self.values_at(x)
        if not found:
            return None
        lines = [f'{label}: {format_value(value)}' if label else format_value(value) for label, value in found[:8]]
        return '\n'.join([f'step {int(round(x)):,}', *lines])

    # Interaction.

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            self._box = (event.position(), event.position())
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._box is not None:
            self._box = (self._box[0], event.position())
            self.update()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._box is None or event.button() != Qt.MouseButton.RightButton:
            return
        (start, end), self._box = self._box, None
        rect = self.plot_rect()
        wide, tall = abs(end.x() - start.x()) > self.DRAG, abs(end.y() - start.y()) > self.DRAG
        if (wide or tall) and not self.message:
            (x0, y0), (x1, y1) = self.from_px(start.x(), start.y(), rect), self.from_px(end.x(), end.y(), rect)
            x0, x1 = (min(x0, x1), max(x0, x1)) if wide else (0.0, 0.0)
            self.select(x0, x1, *((min(y0, y1), max(y0, y1)) if tall else (None, None)))
        else:
            self._menu(event.globalPosition().toPoint())
        self.update()

    def _menu(self, at) -> None:
        menu = QMenu(self)
        fit = menu.addAction('Fit the values shown')
        fit.setEnabled(self.zoomed is not None)
        fit.triggered.connect(self.fit)
        every = menu.addAction('Show every step')
        every.triggered.connect(self.show_every_step)
        menu.exec(at)

    def wheelEvent(self, event) -> None:
        if not event.modifiers() & Qt.KeyboardModifier.ControlModifier or self.message:
            event.ignore()                                              # the panel scrolls
            return
        rect = self.plot_rect()
        _, at = self.from_px(event.position().x(), event.position().y(), rect)
        self.zoom(0.8 if event.angleDelta().y() > 0 else 1.25, at)
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        self.fit()

    # Drawing.

    def _columns(self, px: np.ndarray, values: tuple[np.ndarray, ...], rect: QRectF) -> tuple[np.ndarray, list[np.ndarray]] | None:
        """
            Groups the points of ``px`` by pixel column of ``rect``.

            Returns the columns and, for the first and the last array of ``values``, their lowest
            and highest value in each column, ordered by column. None when no point is inside
            ``rect``.
        """
        columns = max(int(rect.width()), 1)
        column = np.floor(px - rect.left()).astype(np.int64)
        inside = (column >= 0) & (column < columns)
        column, values = column[inside], [v[inside] for v in values]
        if not len(column):
            return None
        if np.any(column[1:] < column[:-1]):
            order = np.argsort(column, kind='stable')
            column, values = column[order], [v[order] for v in values]
        starts = np.flatnonzero(np.r_[True, column[1:] != column[:-1]])
        return column[starts], [np.minimum.reduceat(values[0], starts), np.maximum.reduceat(values[-1], starts)]

    def shapes(self, rect: QRectF) -> list[tuple[QPainterPath | QPolygonF | QPointF, QColor]]:
        """
            Returns the shapes each series draws in ``rect``, with their colors.

            A series of more than two points per pixel column draws one vertical line per column,
            from its lowest to its highest value there. Cached until the series, the ranges or the
            size change.
        """
        key = (rect.width(), rect.height(), self.x_range, self.y_range)
        if self._shapes is None or self._shapes[0] != key:
            self._shapes = (key, self._line_shapes([(x, y, pen_color) for _, x, y, pen_color in self.series], rect))
        return self._shapes[1]

    def faint_shapes(self, rect: QRectF) -> list[tuple[QPainterPath | QPolygonF | QPointF, QColor]]:
        """
            Returns the shapes the faint lines draw in ``rect``, as `shapes`.
        """
        key = (rect.width(), rect.height(), self.x_range, self.y_range)
        if self._faints is None or self._faints[0] != key:
            self._faints = (key, self._line_shapes(self.faint, rect))
        return self._faints[1]

    def _line_shapes(self, lines: list[tuple[np.ndarray, np.ndarray, QColor]], rect: QRectF) -> list[tuple[QPainterPath | QPolygonF | QPointF, QColor]]:
        shapes = []
        columns = max(int(rect.width()), 1)
        for x, y, pen_color in lines:
            px, py = self.to_px(x, y, rect)
            keep = np.isfinite(py)
            px, py = px[keep], py[keep]
            if not len(px):
                continue
            if len(px) > 2 * columns:
                # One vertical line per pixel column, from the lowest to the highest value in it.
                grouped = self._columns(px, (py,), rect)
                if grouped is None:
                    continue
                path = QPainterPath()
                for c, a, b in zip(grouped[0], *grouped[1]):
                    path.moveTo(rect.left() + c + 0.5, a)
                    path.lineTo(rect.left() + c + 0.5, b + 0.5)
                shapes.append((path, pen_color))
            elif len(px) == 1:
                shapes.append((QPointF(px[0], py[0]), pen_color))
            else:
                shapes.append((QPolygonF([QPointF(a, b) for a, b in zip(px, py)]), pen_color))
        return shapes

    def band_shapes(self, rect: QRectF) -> list[tuple[QPainterPath | QPolygonF, QColor]]:
        """
            Returns the shapes each band draws in ``rect``, with their colors.

            A polygon is filled. A band of more than two points per pixel column draws one vertical
            line per column instead, from its lowest low to its highest high there. Cached as
            `shapes`.
        """
        key = (rect.width(), rect.height(), self.x_range, self.y_range)
        if self._bands is not None and self._bands[0] == key:
            return self._bands[1]
        shapes = []
        columns = max(int(rect.width()), 1)
        for _, x, lo, hi, band_color in self.bands:
            px, low = self.to_px(x, lo, rect)
            _, high = self.to_px(x, hi, rect)
            keep = np.isfinite(low) & np.isfinite(high)
            px, low, high = px[keep], low[keep], high[keep]
            if not len(px):
                continue
            if len(px) > 2 * columns:
                grouped = self._columns(px, (high, low), rect)
                if grouped is None:
                    continue
                path = QPainterPath()
                for c, a, b in zip(grouped[0], *grouped[1]):
                    path.moveTo(rect.left() + c + 0.5, a)
                    path.lineTo(rect.left() + c + 0.5, b + 0.5)
                shapes.append((path, band_color))
            else:
                upper = [QPointF(a, b) for a, b in zip(px, high)]
                lower = [QPointF(a, b) for a, b in zip(px[::-1], low[::-1])]
                shapes.append((QPolygonF(upper + lower), band_color))
        self._bands = (key, shapes)
        return shapes

    def draw(self, painter: QPainter, rect: QRectF) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        for shape, band_color in self.band_shapes(rect):
            if isinstance(shape, QPainterPath):
                painter.setPen(QPen(band_color, 1))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(shape)
            else:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(band_color)
                painter.drawPolygon(shape)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for width, shapes in ((THEME.faint_line_width, self.faint_shapes(rect)), (THEME.line_width, self.shapes(rect))):
            for shape, pen_color in shapes:
                painter.setPen(QPen(pen_color, width))
                if isinstance(shape, QPainterPath):
                    painter.drawPath(shape)
                elif isinstance(shape, QPointF):
                    painter.drawEllipse(shape, 2, 2)
                else:
                    painter.drawPolyline(shape)
        if self._box is not None:
            box = QRectF(self._box[0], self._box[1]).normalized()
            painter.setPen(QPen(THEME.cursor, 1, Qt.PenStyle.DashLine))
            painter.setBrush(THEME.with_alpha(THEME.cursor, THEME.zoom_box_alpha))
            painter.drawRect(box)
            painter.setBrush(Qt.BrushStyle.NoBrush)
        if self.legend and len(self.series) > 1:
            # The first series that fit, labels cut past the legend width of `THEME`, aligned to the right.
            painter.setFont(THEME.font())
            metrics = painter.fontMetrics()
            swatch = THEME.legend_swatch
            entries, room = [], rect.width() - 8
            for label, _, _, pen_color in self.series[:8]:
                label = metrics.elidedText(label, Qt.TextElideMode.ElideRight, THEME.legend_width)
                width = metrics.horizontalAdvance(label) + swatch + 6
                if width > room:
                    break
                entries.append((label, width, pen_color))
                room -= width
            x = rect.right() - 4
            for label, width, pen_color in reversed(entries):
                x -= width
                painter.fillRect(QRectF(x, rect.top() + 4, swatch, swatch), pen_color)
                painter.setPen(THEME.text)
                painter.drawText(QPointF(x + swatch + 3, rect.top() + 12), label)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ImagePlot(_Plot):
    """
        Image of values over steps and units, such as a raster of spikes or a trace of many units.

        Steps run along the horizontal axis and units upwards. Booleans are drawn as events on the
        background of the plots, numbers with the colormap over their range, with a colour bar.
    """

    HEIGHT = 'image_height'

    def __init__(self, title: str = '', height: int | None = None, parent: QWidget | None = None) -> None:
        super().__init__(title, height, parent)
        self._image: QImage | None = None
        self._buffer: np.ndarray | None = None
        self._times = (0.0, 1.0)
        # Steps and rows drawn, for the tooltip.
        self._steps = np.zeros(0)
        self._rows = np.zeros((0, 0))

    def set_image(self, times: np.ndarray, values: np.ndarray, x_range: tuple[float, float] | None = None) -> None:
        """
            Sets the values drawn, one row per step of ``times``, flattened past the first axis.

            The image spans from the first step to the step after the last. Values past `MAX_IMAGE`
            are reduced in blocks.
        """
        values = np.asarray(values)
        if not len(times) or not values.size:
            self.message, self._image, self.colorbar = 'Nothing recorded here', None, None
            self.update()
            return
        self.message = ''
        values = values.reshape(values.shape[0], -1)
        self._steps, self._rows = np.asarray(times), values
        units = values.shape[1]
        self.colorbar = None if values.dtype == np.bool_ else value_range(values.astype(np.float64))
        values = _shrink(_shrink(values, 0, MAX_IMAGE[0]), 1, MAX_IMAGE[1])
        self._image, self._buffer = _image(values.T)
        t = np.asarray(times, np.float64)
        self._times = (float(t.min()), float(t.max() + 1))
        self.x_range = x_range or self._times
        self.y_range = (0.0, float(units))
        self.update()

    def hover_text(self, x: float, y: float) -> str | None:
        index, unit = _at_or_before(self._steps, x), int(y)
        if index < 0 or not 0 <= unit < self._rows.shape[1]:
            return None
        value = self._rows[index, unit]
        shown = ('spike' if value else 'no spike') if self._rows.dtype == np.bool_ else format_value(float(value))
        return f'step {int(self._steps[index]):,}\nunit {unit}: {shown}'

    def draw(self, painter: QPainter, rect: QRectF) -> None:
        if self._image is None:
            return
        left, right = self._px(self._times[0], rect), self._px(self._times[1], rect)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawImage(QRectF(left, rect.top(), right - left, rect.height()), self._image)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class BarPlot(_Plot):
    """
        Bar plot of values, such as a histogram, the rate of every unit or an input vector.

        Each value has one bar. The horizontal axis is the index of the value, or spans ``x_range``,
        such as the edges of the bins. With more than one bar per pixel column, each column draws
        the range of its values.
    """

    FOLLOWS_CURSOR = False
    HEIGHT = 'bar_height'

    def __init__(self, title: str = '', height: int | None = None, parent: QWidget | None = None) -> None:
        super().__init__(title, height, parent)
        self.values = np.zeros(0)
        self.labels: list[str] = []
        self.color: QColor | None = None

    def set_values(
            self, values: np.ndarray, x_range: tuple[float, float] | None = None, labels: list[str] | None = None,
            color: QColor | None = None,
        ) -> None:
        """
            Sets the values drawn, flattened.

            ``labels`` are drawn over the bars when there are at most 12. The bars take ``color``,
            else the first color of the series.
        """
        self.values = np.asarray(values, np.float64).reshape(-1)
        self.labels = labels or []
        self.color = None if color is None else QColor(color)
        finite = self.values[np.isfinite(self.values)]
        self.message = '' if len(self.values) else 'Nothing recorded here'
        if len(self.values):
            self.x_range = x_range or (0.0, float(len(self.values)))
            lo = min(0.0, float(finite.min())) if finite.size else 0.0
            hi = max(0.0, float(finite.max())) if finite.size else 1.0
            self.y_range = (lo, hi if hi > lo else lo + 1)
        self.update()

    def hover_text(self, x: float, y: float) -> str | None:
        n = len(self.values)
        x0, x1 = self.x_range
        index = int((x - x0) / max(x1 - x0, 1e-12) * n)
        if not 0 <= index < n:
            return None
        if index < len(self.labels):
            name = self.labels[index]
        elif (x0, x1) != (0.0, float(n)):
            width = (x1 - x0) / n
            name = f'[{format_value(x0 + index * width)}, {format_value(x0 + (index + 1) * width)})'
        else:
            name = str(index)
        return f'{name}: {format_value(self.values[index])}'

    def draw(self, painter: QPainter, rect: QRectF) -> None:
        values = np.nan_to_num(self.values)
        n = len(values)
        if not n:
            return
        columns = max(int(rect.width()), 1)
        x0, x1 = self.x_range
        if n > columns:
            # Values of one pixel column drawn as one bar from their lowest to their highest, zero included.
            block = -(-n // columns)
            padded = np.pad(values, (0, (-n) % block), mode='edge').reshape(-1, block)
            low, high = np.minimum(padded.min(axis=1), 0), np.maximum(padded.max(axis=1), 0)
            edges = np.linspace(x0, x0 + (x1 - x0) * len(padded) * block / n, len(padded) + 1)
            gap = 0
        else:
            low, high = np.minimum(values, 0), np.maximum(values, 0)
            edges = np.linspace(x0, x1, n + 1)
            gap = 1 if n < 80 else 0
        left, bottom = self.to_px(edges[:-1], low, rect)
        right, top = self.to_px(edges[1:], high, rect)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.color or THEME.series_color(0))
        for a, b, y0, y1 in zip(left, right, top, bottom):
            if y1 > y0:
                painter.drawRect(QRectF(a, y0, max(b - a - gap, 1), y1 - y0))
        if self.labels and n <= 12:
            painter.setPen(THEME.text)
            painter.setFont(THEME.font())
            for a, b, label in zip(left, right, self.labels):
                painter.drawText(QRectF(a, rect.top() + 2, b - a, 12), Qt.AlignmentFlag.AlignCenter, label)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class MatrixPlot(_Plot):
    """
        Image of a matrix with the colormap over its range, such as the weights of a set of synapses.

        Row 0 is at the bottom. A colour bar gives the range of the values.
    """

    FOLLOWS_CURSOR = False
    HEIGHT = 'matrix_height'

    def __init__(self, title: str = '', height: int | None = None, parent: QWidget | None = None) -> None:
        super().__init__(title, height, parent)
        self._image: QImage | None = None
        self._buffer: np.ndarray | None = None
        self._matrix = np.zeros((0, 0))

    def set_matrix(self, matrix: np.ndarray) -> None:
        """
            Sets the matrix drawn.

            Arrays are flattened past the first axis, and a vector is one row.
        """
        matrix = np.asarray(matrix, np.float64)
        if not matrix.size:
            self.message, self._image, self.subtitle, self.colorbar = 'Nothing recorded here', None, '', None
            self.update()
            return
        self.message = ''
        matrix = matrix.reshape(matrix.shape[0], -1) if matrix.ndim > 1 else matrix.reshape(1, -1)
        self._matrix = matrix
        rows, columns = matrix.shape
        self.colorbar = value_range(matrix)
        self._image, self._buffer = _image(_shrink(_shrink(matrix, 0, MAX_MATRIX[0]), 1, MAX_MATRIX[1]))
        self.x_range = (0.0, float(columns))
        self.y_range = (0.0, float(rows))
        self.update()

    def hover_text(self, x: float, y: float) -> str | None:
        row, column = int(y), int(x)
        if not (0 <= row < self._matrix.shape[0] and 0 <= column < self._matrix.shape[1]):
            return None
        return f'row {row}, column {column}: {format_value(self._matrix[row, column])}'

    def draw(self, painter: QPainter, rect: QRectF) -> None:
        if self._image is not None:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
            painter.drawImage(rect, self._image)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
