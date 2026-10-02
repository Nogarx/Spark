#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp

import json

import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import QPainter, QPen
from PySide6.QtWidgets import QWidget, QSizePolicy, QToolTip

from spark.graph_editor.styles.run_viewer import THEME
from spark.graph_editor.runs.plots import nice_ticks, tick_labels

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

NOTABLE = ('warning', 'error')
"""
    Kinds of events drawn over the others, the full height of the row of events.
"""

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Timeline(QWidget):
    """
        Timeline of the steps recorded by each set of measurements, the values of the integer tags,
        the events and the cursor.

        Each set of measurements has one row, and each tag, such as ``episode``, a row marking where
        its values start. A left click or drag moves the cursor and emits
        `cursor_moved`. The wheel zooms around the mouse, and a right drag pans; both emit
        `view_changed`. The mouse over a row or an event shows what is there in a tooltip.

        Its colours and sizes are those of ``THEME.timeline``: the width of the labels, the heights
        of the ruler, of a row and of the events, and the colour of each kind of event. ``record``
        events take the colour of the cursor.
    """

    cursor_moved = Signal(int)
    view_changed = Signal(float, float)
    HOVER = 4
    """
        Pixels from the mouse within which an event is shown in the tooltip.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.total = 1
        self.view = (0.0, 1.0)
        self.rows: list[tuple[str, list[tuple[int, int]]]] = []
        self.events: list[tuple[int, str]] = []
        self.payloads: list[dict[str, tp.Any]] = []
        # Every tag shown: its name, and the steps it took its values at, with the values.
        self.tags: list[tuple[str, np.ndarray, np.ndarray]] = []
        self._spans: list[tuple[np.ndarray, np.ndarray]] = []
        self._events: dict[str, np.ndarray] = {}
        self._steps = np.zeros(0)
        self.cursor = 0
        self._pan_from: tuple[float, tuple[float, float]] | None = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMouseTracking(True)
        self._resize()

    @property
    def _rows(self) -> int:
        return len(self.rows) + len(self.tags)

    def _resize(self) -> None:
        t = THEME.timeline
        self.setMinimumHeight(t.ruler_height + t.row_height * max(self._rows, 1) + t.events_height + t.padding)

    def set_data(self, total: int, rows: list[tuple[str, list[tuple[int, int]]]], events: list[tuple[int, str]],
                 payloads: list[dict[str, tp.Any]] | None = None, tags: list[tuple[str, np.ndarray, np.ndarray]] | None = None) -> None:
        """
            Sets the length of the run, the steps recorded and the events shown.

            A view showing the last step moves with the steps added.

            Parameters
            ----------
            total : int
                Steps of the run so far.
            rows : list of (str, array or list of (int, int))
                Name of every set of measurements and the ``[start, end)`` spans of steps it
                recorded, as a ``(spans, 2)`` array or a list of pairs.
            events : list of (int, str)
                Step and kind of every event.
            payloads : list of dict, optional
                Payload of every event, shown in the tooltip.
            tags : list of (str, ndarray, ndarray), optional
                Name of every integer tag, the steps at which it took its values, and the values.
        """
        # A view reaching the last step follows the steps added.
        following = self.view[1] >= self.total - 1
        was_whole = self.view == (0.0, float(self.total))
        self.total = max(int(total), 1)
        self.rows, self.events = rows, events
        self.tags = [(name, np.asarray(steps, np.float64), np.asarray(values)) for name, steps, values in (tags or [])]
        self.payloads = payloads if payloads is not None else [{} for _ in events]
        # Spans are sorted and disjoint; their starts and their ends are both ordered.
        self._spans = []
        for _, spans in rows:
            edges = np.asarray(spans, np.float64).reshape(-1, 2)
            self._spans.append((edges[:, 0], edges[:, 1]))
        by_kind: dict[str, list[float]] = {}
        for step, kind in events:
            by_kind.setdefault(kind, []).append(step)
        self._events = {kind: np.sort(np.asarray(steps, np.float64)) for kind, steps in by_kind.items()}
        self._steps = np.asarray([step for step, _ in events], np.float64)
        view = self.view
        if was_whole or self.view == (0.0, 1.0):
            self.view = (0.0, float(self.total))
        elif following:
            width = self.view[1] - self.view[0]
            self.view = (max(self.total - width, 0.0), float(self.total))
        self._resize()
        self.update()
        if self.view != view:
            self.view_changed.emit(*self.view)

    def set_cursor(self, step: int) -> None:
        """
            Moves the cursor to ``step`` without emitting `cursor_moved`.
        """
        self.cursor = int(step)
        self.update()

    def show_range(self, start: float, end: float) -> None:
        """
            Shows the steps from ``start`` to ``end``, within the run and at least one step wide.
        """
        start = min(max(float(start), 0.0), self.total - 1.0)
        self._set_view((start, min(max(float(end), start + 1.0), float(self.total))))

    def _set_view(self, view: tuple[float, float]) -> None:
        changed = view != self.view
        self.view = view
        self.update()
        if changed:
            self.view_changed.emit(*view)

    # Geometry.

    def _area(self) -> QRectF:
        t = THEME.timeline
        return QRectF(t.labels_width, 0, max(self.width() - t.labels_width - t.padding, 1), self.height())

    def _x(self, step: float) -> float:
        area = self._area()
        a, b = self.view
        return area.left() + (step - a) / max(b - a, 1e-9) * area.width()

    def _step(self, x: float) -> float:
        area = self._area()
        a, b = self.view
        return a + (x - area.left()) / area.width() * (b - a)

    # Interaction.

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._move_cursor(event.position().x())
        elif event.button() == Qt.MouseButton.RightButton:
            self._pan_from = (event.position().x(), self.view)

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._move_cursor(event.position().x())
        elif self._pan_from is not None:
            x, (a, b) = self._pan_from
            shift = (x - event.position().x()) / self._area().width() * (b - a)
            shift = min(max(shift, -a), self.total - b)
            self._set_view((a + shift, b + shift))
        else:
            text = self.hover_text(event.position().x(), event.position().y())
            if text:
                QToolTip.showText(event.globalPosition().toPoint(), text, self)
            else:
                QToolTip.hideText()

    def mouseReleaseEvent(self, event) -> None:
        self._pan_from = None

    def leaveEvent(self, event) -> None:
        QToolTip.hideText()
        super().leaveEvent(event)

    def wheelEvent(self, event) -> None:
        a, b = self.view
        at = min(max(self._step(event.position().x()), a), b)
        factor = 0.8 if event.angleDelta().y() > 0 else 1.25
        width = min(max((b - a) * factor, 10.0), float(self.total))
        start = min(max(at - (at - a) * width / max(b - a, 1e-9), 0.0), self.total - width)
        self._set_view((start, start + width))

    def hover_text(self, x: float, y: float) -> str | None:
        """
            Returns the tooltip at pixel ``(x, y)``: the events near the mouse in the row of events,
            or the span of steps under the mouse in the row of a set of measurements.
        """
        t = THEME.timeline
        if x < t.labels_width:
            return None
        step = self._step(x)
        events_top = t.ruler_height + self._rows * t.row_height
        if y >= events_top:
            near = np.flatnonzero(np.abs(self._x(self._steps) - x) <= self.HOVER) if len(self._steps) else []
            lines = [_describe(self.events[i], self.payloads[i]) for i in near[:6]]
            if len(near) > 6:
                lines.append(f'and {len(near) - 6} more')
            return '\n'.join(lines) or None
        index = int((y - t.ruler_height) // t.row_height)
        if len(self.rows) <= index < self._rows:
            name, steps, values = self.tags[index - len(self.rows)]
            at = int(np.searchsorted(steps, step, side='right')) - 1
            if at < 0:
                return f'{name}\nnot set at step {int(step):,}'
            end = steps[at + 1] if at + 1 < len(steps) else self.total
            return f'{name} {values[at]}\nsteps {int(steps[at]):,} to {int(end):,}'
        if not 0 <= index < len(self.rows):
            return None
        starts, ends = self._spans[index]
        at = int(np.searchsorted(starts, step, side='right')) - 1
        name = self.rows[index][0]
        if at >= 0 and step < ends[at]:
            return f'{name}\nsteps {int(starts[at]):,} to {int(ends[at]):,}'
        return f'{name}\nnot recorded at step {int(step):,}'

    def _move_cursor(self, x: float) -> None:
        if x < THEME.timeline.labels_width:
            return
        step = int(round(min(max(self._step(x), 0), self.total)))
        self.cursor = step
        self.update()
        self.cursor_moved.emit(step)

    # Drawing.

    def paintEvent(self, event) -> None:
        # A painter left active when drawing raises takes the process down with its paint device.
        painter = QPainter(self)
        try:
            self._paint(painter)
        finally:
            painter.end()

    def _paint(self, painter: QPainter) -> None:
        t = THEME.timeline
        row, pad = t.row_height, t.row_padding
        painter.fillRect(self.rect(), THEME.background)
        area = self._area()
        painter.setFont(THEME.font())
        metrics = painter.fontMetrics()
        names = [name for name, _ in self.rows] + [name for name, _, _ in self.tags]
        for index, name in enumerate(names):
            painter.setPen(THEME.text if index < len(self.rows) else THEME.muted)
            painter.drawText(QRectF(4, t.ruler_height + index * row, t.labels_width - 8, row), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                             metrics.elidedText(name, Qt.TextElideMode.ElideRight, t.labels_width - 10))
        # Ruler.
        ticks = nice_ticks(*self.view, count=8)
        for value, label in zip(ticks, tick_labels(ticks)):
            x = self._x(value)
            painter.setPen(QPen(t.tick, 1))
            painter.drawLine(QPointF(x, t.ruler_height), QPointF(x, self.height()))
            painter.setPen(THEME.text)
            painter.drawText(QRectF(x - 40, 2, 80, t.ruler_height - 4), Qt.AlignmentFlag.AlignCenter, label)
        # What follows is drawn within the area right of the labels.
        painter.setClipRect(area)
        # Measurements. Spans closer than a pixel are merged.
        for index, (starts, ends) in enumerate(self._spans):
            top = t.ruler_height + index * row
            first, last = np.searchsorted(ends, self.view[0], 'left'), np.searchsorted(starts, self.view[1], 'right')
            if first >= last:
                continue
            x0, x1 = self._x(starts[first:last]), self._x(ends[first:last])
            x1 = np.maximum(x1, x0 + 1.5)
            opens = np.flatnonzero(np.r_[True, x0[1:] > x1[:-1] + 1])
            closes = np.r_[opens[1:], len(x0)] - 1
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(THEME.series_color(index))
            for a, b in zip(x0[opens], x1[closes]):
                painter.drawRect(QRectF(a, top + pad, b - a, row - 2 * pad))
        # Tags: every value from where it starts, in alternating shades, numbered when there is room.
        for index, (_, steps, values) in enumerate(self.tags, start=len(self.rows)):
            top = t.ruler_height + index * row
            ends = np.r_[steps[1:], self.total]
            first, last = np.searchsorted(ends, self.view[0], 'right'), np.searchsorted(steps, self.view[1], 'right')
            x0, x1 = self._x(steps[first:last]), self._x(ends[first:last])
            wide = np.flatnonzero(x1 - x0 >= 3)
            if len(wide) < len(x0):
                # Values narrower than a few pixels: one faint band.
                painter.fillRect(QRectF(max(x0[0] if len(x0) else area.left(), area.left()), top + pad, area.width(), row - 2 * pad), t.tag_dense)
            for i in wide[:2000]:
                shade = t.tag if (first + i) % 2 else t.tag_alt
                box = QRectF(x0[i], top + pad, x1[i] - x0[i] - 1, row - 2 * pad)
                painter.fillRect(box, shade)
                label = str(values[first + i])
                if box.width() > metrics.horizontalAdvance(label) + 6:
                    painter.setPen(t.tag_text)
                    painter.drawText(box, Qt.AlignmentFlag.AlignCenter, label)
        # Events, one line per pixel column and kind: the lower half of the row, warnings and errors over the others
        # and the full height.
        top = t.ruler_height + self._rows * row + 2
        kinds = {'record': THEME.cursor, 'checkpoint': t.checkpoint, 'warning': t.warning, 'error': t.error}
        for kind in sorted(self._events, key=lambda kind: kind in NOTABLE):
            steps = self._events[kind]
            visible = steps[(steps >= self.view[0]) & (steps <= self.view[1])]
            if not len(visible):
                continue
            notable = kind in NOTABLE
            painter.setPen(QPen(kinds.get(kind, t.event), t.notable_event_width if notable else t.event_width))
            start = top if notable else top + (t.events_height - 2) / 2
            for x in np.unique(np.round(self._x(visible))):
                painter.drawLine(QPointF(x, start), QPointF(x, top + t.events_height - 2))
        # Cursor.
        x = self._x(self.cursor)
        if area.left() <= x <= area.right():
            painter.setPen(QPen(THEME.cursor, t.cursor_width))
            painter.drawLine(QPointF(x, 0), QPointF(x, self.height()))
            painter.setPen(THEME.cursor)
            label = f'step {self.cursor:,}'
            width = metrics.horizontalAdvance(label) + 6
            left = x + 3 if x + 3 + width <= area.right() else x - 3 - width
            painter.drawText(QRectF(left, self.height() - 14, width, 12), Qt.AlignmentFlag.AlignLeft, label)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def event_text(payload: dict[str, tp.Any]) -> str:
    """
        Returns what an event says: its ``message`` or ``error`` whole, or else its other fields as
        JSON, cut past 120 characters.
    """
    message = payload.get('message') or payload.get('error')
    if message:
        return str(message)
    fields = {k: v for k, v in payload.items() if k not in ('t', 'kind', 'wall')}
    text = json.dumps(fields, default=str)[1:-1] if fields else ''
    return text if len(text) <= 120 else text[:117] + '...'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _describe(event: tuple[int, str], payload: dict[str, tp.Any]) -> str:
    """
        Returns one line describing an event: its kind, step and `event_text`.
    """
    step, kind = event
    text = event_text(payload)
    return f'{kind} at step {int(step):,}' + (f': {text}' if text else '')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
