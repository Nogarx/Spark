#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp

import sys
import math
import html
import time
import warnings
import pathlib
import datetime

import numpy as np
from shiboken6 import isValid
from PySide6.QtCore import Qt, QRectF, QTimer, Signal, QByteArray
from PySide6.QtGui import QPainter, QColor, QPen, QImage, QPixmap, QKeySequence, QShortcut, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QScrollArea, QPushButton,
    QSpinBox, QCheckBox, QGraphicsItem, QGraphicsView, QGraphicsScene, QFormLayout, QFrame, QStatusBar, QTabWidget, QFileDialog,
    QMessageBox,
)

from spark.graph_editor.styles.manager import STYLES
from spark.graph_editor.view.graph_view import GraphScene, GraphView
from spark.graph_editor.view.node_item import NodeItem
from spark.graph_editor.runs.data import RunData, READ_ERRORS
from spark.graph_editor.styles.run_viewer import THEME
from spark.graph_editor.runs.plots import SeriesPlot, ImagePlot, BarPlot, MatrixPlot, format_value
from spark.graph_editor.runs.timeline import Timeline, NOTABLE, event_text
from spark.graph_editor.runs.workspace import Project, Selection, SeriesStore, Spaces, Line, Exploration, STEP, SUFFIX, axis_label
from spark.graph_editor.runs.workspace_view import RunsTable, WorkspaceView
from spark.recording.probe import (
    Probe, SummaryProbe, TraceProbe, RasterProbe, SnapshotProbe, DeltaProbe, SummaryReduction, DeltaReduction, CALL,
)
from spark.recording.measurements import Measurements, scalar_key
from spark.recording.run import Run

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

PREFERENCES = ('Canvas', 'Nodes', 'Ports & Payloads', 'Edges', 'Run Viewer', 'Window')
"""
    Sections of the preferences of the editor that change the viewer.
"""

_SHOWN: list[RunViewerWindow] = []
"""
    Every run viewer window shown, kept referenced while it is open. A closed window is dropped when
    the next one is shown.
"""

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class _ReadOnlyScene(GraphScene):
    """
        Scene of the graph of the recorded model, in which clicks only select.

        Nothing can be connected, moved or deleted.
    """

    def mousePressEvent(self, event) -> None:
        QGraphicsScene.mousePressEvent(self, event)

    def mouseMoveEvent(self, event) -> None:
        QGraphicsScene.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event) -> None:
        QGraphicsScene.mouseReleaseEvent(self, event)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class _ReadOnlyView(GraphView):
    """
        View of a `_ReadOnlyScene`, without the editing keys and the context menu of the editor.

        The whole graph is kept in view as the view is resized, as when the window is first laid
        out, until the view is zoomed, scrolled or panned by hand.
    """

    MARGIN = 60
    """
        Margin around the graph when it is fitted, in units of the scene.
    """

    _MOVING_KEYS = {
        Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_PageUp, Qt.Key.Key_PageDown,
        Qt.Key.Key_Home, Qt.Key.Key_End,
    }
    """
        Keys with which `QGraphicsView` scrolls the view.
    """

    def __init__(self, scene: QGraphicsScene, parent: QWidget | None = None) -> None:
        super().__init__(scene, parent)
        self.fitting = True

    def fit(self) -> None:
        """
            Fits the whole graph in the view, and keeps it fitted as the view is resized.
        """
        self.fitting = True
        self._fit()

    def _fit(self) -> None:
        if self.scene() is not None and self.scene().items():
            margin = self.MARGIN
            self.fitInView(self.scene().itemsBoundingRect().adjusted(-margin, -margin, margin, margin), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.fitting:
            self._fit()

    def wheelEvent(self, event) -> None:
        self.fitting = False
        super().wheelEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self.fitting = False
        super().mousePressEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() in self._MOVING_KEYS:
            self.fitting = False
        QGraphicsView.keyPressEvent(self, event)

    def contextMenuEvent(self, event) -> None:
        event.ignore()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ActivityBadge(QGraphicsItem):
    """
        Badge showing the firing rate of a node at the cursor.

        Drawn above the top right corner of the node, at the same size at any zoom. Its colour goes
        from the low to the high colour of ``THEME.badge`` as the rate rises to its full scale, in Hz.
    """

    def __init__(self, node: NodeItem) -> None:
        super().__init__(node)
        self.value: float | None = None
        # Anchored by its bottom right corner, in pixels of the view.
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setPos(node.width, -3)
        self.setZValue(10)

    def set_value(self, hz: float | None) -> None:
        """
            Sets the rate shown, in Hz, or hides the badge for None.
        """
        self.value = hz
        self.update()

    def boundingRect(self) -> QRectF:
        badge = THEME.badge
        return QRectF(-badge.width, -badge.height, badge.width, badge.height)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        if self.value is None:
            return
        badge = THEME.badge
        level = min(max(self.value / badge.full_scale, 0.0), 1.0)
        # From the low colour to the high one, through their hues.
        lerp = lambda a, b: a + (b - a) * level
        low, high = badge.low, badge.high
        fill = QColor.fromHsvF(lerp(low.hsvHueF(), high.hsvHueF()), lerp(low.hsvSaturationF(), high.hsvSaturationF()), lerp(low.valueF(), high.valueF()))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(badge.border, 1))
        painter.setBrush(fill)
        painter.drawRoundedRect(self.boundingRect(), badge.radius, badge.radius)
        painter.setPen(badge.dark_text if level > 0.5 else badge.light_text)
        painter.setFont(THEME.font(badge.font_size, bold=True))
        painter.drawText(self.boundingRect(), Qt.AlignmentFlag.AlignCenter, f'{self.value:.1f} Hz')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _scroll(content: QWidget) -> QScrollArea:
    """
        Returns a scroll area around a panel, scrolling down only.
    """
    area = QScrollArea()
    area.setObjectName('runViewerScroll')
    area.viewport().setObjectName('runViewerViewport')
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    content.setObjectName('runViewerPanel')
    content.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    area.setWidget(content)
    return area

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _heading(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName('runViewerHeading')
    return label

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _section(text: str) -> QLabel:
    """
        Returns the header of a group of plots.
    """
    label = QLabel(text)
    label.setObjectName('runViewerSection')
    label.setWordWrap(True)
    return label

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _note(text: str = '') -> QLabel:
    label = QLabel(text)
    label.setObjectName('runViewerNote')
    label.setWordWrap(True)
    return label

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _value(text: str) -> QLabel:
    """
        Returns a label of a value of the Details panel, wrapped and selectable.
    """
    label = QLabel(text)
    label.setWordWrap(True)
    label.setToolTip(text)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _block() -> QWidget:
    """
        Returns a framed block of the panels.
    """
    block = QWidget()
    block.setObjectName('runViewerBlock')
    block.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    return block

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _form() -> QFormLayout:
    form = QFormLayout()
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    form.setHorizontalSpacing(THEME.layout.form_horizontal_spacing)
    form.setVerticalSpacing(THEME.layout.form_vertical_spacing)
    return form

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _recorded_at(step: int, cursor: int) -> str:
    """
        Returns the step a value was recorded at, with how far it lies before the cursor.
    """
    before = cursor - step
    return f'at step {step:,}' + (f', {before:,} steps earlier' if before > 0 else '')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _when(text: str | None) -> str:
    """
        Returns a time written by the recorder as ``YYYY-MM-DD HH:MM:SS``, or as written.
    """
    try:
        return datetime.datetime.fromisoformat(str(text)).strftime('%Y-%m-%d %H:%M:%S')
    except ValueError:
        return str(text or '')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _description(measurements: Measurements) -> str:
    """
        Returns the trigger, the groups and the number of probes of a set of measurements.
    """
    parts = [type(measurements.trigger).__name__ + (f' by {measurements.trigger.tag}' if measurements.trigger.tag else '')]
    if isinstance(measurements.group, str):
        parts.append(f'groups of {measurements.group}')
    elif measurements.group is not None:
        parts.append(f'groups of {measurements.group:,} steps')
    count = len(measurements.probes)
    parts.append(f'{count} probe' + ('' if count == 1 else 's'))
    return ', '.join(parts)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _module(probe: Probe, node: str) -> str:
    """
        Returns the module of ``node`` a probe reads, such as ``'soma'``: ``'inputs'`` for the inputs
        of the node, and ``''`` for the node itself.
    """
    if probe.path == (CALL,):
        return ''
    within = probe.path[1:] if probe.path[:1] == (node,) else probe.path
    return 'inputs' if within == (CALL,) else '.'.join(within)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ProbePanel(QWidget):
    """
        Panel of plots of what was recorded for one node of the graph.

        Each set of measurements recording the node has a tab. Within a tab, plots are grouped by
        module and titled by port or attribute and reduction. Without a node, the panel plots the
        scalars logged with `Recorder.log`, in tabs by the prefix of their names. ``inputs``, when
        given, is the last tab. The tab chosen last is shown again for the next node that has a tab
        of that name.

        Scalar series are drawn in their natural space, the tag they were written per, or in the
        space ``selection`` chooses, over the steps of `view`; the ``mean``, ``std``, ``min`` and
        ``max`` of a summary share one plot. The cursor, a step, is drawn at the tag value it falls
        in, and a click moves it to the first step of the value clicked. The runs and groups
        ``selection`` shows besides this run are drawn with it, each in its color, and listed under
        the name of the node. A box drawn with the right button in a scalar plot asks for its steps
        with `view_requested`, for every plot.
        Values recorded once per group, such as histograms, active fractions per unit, snapshots
        and changes, show the last written group starting at or before the cursor. Traces and
        rasters show the consecutive recorded steps around the cursor.
    """

    cursor_moved = Signal(int)
    view_requested = Signal(float, float)

    LINES = 16
    """
        Largest number of units of a trace drawn as lines. Wider traces are drawn as images.
    """

    SPREAD = ('mean', 'std', 'min', 'max')
    """
        Reductions of a summary drawn in one plot: the mean as a line, the band of one standard
        deviation around it, and the band from the lowest to the highest value, fainter. Runs
        compared draw their means.
    """

    def __init__(self, data: RunData, selection: Selection | None = None, store: SeriesStore | None = None,
                 inputs: QWidget | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.data = data
        self.selection = selection
        if store is None:
            store = SeriesStore()
            store.set_reader(data.run, data.scalar)
        self.store = store
        self._inputs = None if inputs is None else _scroll(inputs)
        self.node: str | None = None
        self.cursor = 0
        self.view: tuple[float, float] | None = None
        self._logged: list[str] = []
        # Scalar plots and the keys of the series each draws: under 'value', or under the reductions of `SPREAD`.
        self._series: list[tuple[SeriesPlot, dict[str, str]]] = []
        self._at_cursor: list[tp.Callable[[int], None]] = []
        self._plots: list[QWidget] = []
        # Tab chosen last among those of nodes (False) and those of logged scalars (True), and whether the tabs are
        # being built, which changes the tab shown.
        self._chosen: dict[bool, str] = {}
        self._building = False
        self.setObjectName('runViewerPanel')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(*THEME.layout.probes_margins)
        layout.setSpacing(THEME.layout.probes_spacing)
        self._title = _heading('')
        self._legend = QLabel()
        self._legend.setObjectName('runViewerLegend')
        self._legend.setWordWrap(True)
        self._legend.setTextFormat(Qt.TextFormat.RichText)
        self._empty = _note()
        self.tabs = QTabWidget()
        self.tabs.setObjectName('runViewerTabs')
        self.tabs.setDocumentMode(True)
        self.tabs.tabBar().setDrawBase(False)
        layout.addWidget(self._title)
        layout.addWidget(self._legend)
        layout.addWidget(self._empty)
        layout.addWidget(self.tabs, 1)
        self.tabs.currentChanged.connect(self._remember_tab)
        if selection is not None:
            selection.changed.connect(self.refresh)
        self.show_node(None)

    def release(self) -> None:
        """
            Stops following the lines shown, before the panel is dropped.
        """
        if self.selection is not None:
            try:
                self.selection.changed.disconnect(self.refresh)
            except (RuntimeError, TypeError):
                pass

    def _remember_tab(self, index: int) -> None:
        if not self._building and index >= 0:
            self._chosen[self.node is None] = self.tabs.tabText(index)

    def plots(self) -> list[QWidget]:
        """
            Returns the plots shown, in the order of the tabs and within them.
        """
        return list(self._plots)

    def _clear(self) -> None:
        while self.tabs.count():
            page = self.tabs.widget(0)
            self.tabs.removeTab(0)
            if page is self._inputs:
                continue
            # Detached before deleteLater; the panel stops drawing it at once.
            page.setParent(None)
            page.deleteLater()
        self._series, self._at_cursor, self._plots = [], [], []

    def _page(self, name: str, description: str = '') -> QVBoxLayout:
        """
            Adds a tab and returns the layout of its page.
        """
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(*THEME.layout.probes_page_margins)
        layout.setSpacing(THEME.layout.probes_page_spacing)
        if description:
            layout.addWidget(_note(description))
        self.tabs.addTab(_scroll(page), name)
        return layout

    def _add(self, layout: QVBoxLayout, widget: QWidget, detail: str = '') -> QWidget:
        layout.addWidget(widget)
        widget.detail = detail
        self._plots.append(widget)
        if isinstance(widget, (SeriesPlot, ImagePlot)):
            widget.cursor_moved.connect(lambda x, w=widget: self._pick(w, x))
        return widget

    def _add_scalars(self, layout: QVBoxLayout, plot: SeriesPlot, keys: dict[str, str], detail: str = '') -> None:
        self._add(layout, plot, (detail or ', '.join(keys.values())) + '\nDrag with the right button to zoom; double click to fit.')
        self._series.append((plot, keys))
        plot.shares_view = True
        plot.view_requested.connect(lambda a, b, w=plot: self._request_view(w, a, b))

    # Spaces.

    def _spaces(self) -> Spaces:
        return self.store.spaces(self.data.run)

    def _space_of(self, keys: dict[str, str]) -> str:
        """
            Returns the space the series of ``keys`` are drawn in.
        """
        key = keys.get('value', keys.get('mean'))
        if self.selection is not None:
            return self.selection.space_of(self.store, self.data.run, key)
        return self._spaces().natural(key)

    def _position(self, step: float, space: str) -> float | None:
        """
            Returns where ``step`` falls in ``space``, or None.
        """
        x = float(self._spaces().to_space(np.array([step]), space)[0])
        return x if np.isfinite(x) else None

    def _pick(self, plot: QWidget, x: float) -> None:
        # A click in a plot of another space than steps moves the cursor to the first step of the value clicked.
        space = getattr(plot, 'space', STEP)
        found = self._spaces().to_steps(x, space) if space != STEP else (int(round(x)), 0)
        if found is not None:
            self.cursor_moved.emit(int(found[0]))

    def _request_view(self, plot: SeriesPlot, a: float, b: float) -> None:
        space = getattr(plot, 'space', STEP)
        if space == STEP or not (np.isfinite(a) and np.isfinite(b)):
            self.view_requested.emit(a, b)
            return
        first, last = self._spaces().to_steps(math.ceil(a - 0.5), space), self._spaces().to_steps(math.floor(b + 0.5), space)
        if first is not None and last is not None:
            self.view_requested.emit(float(first[0]), float(last[1]))

    def _x_range(self, space: str) -> tuple[float, float] | None:
        """
            Returns the steps of `view` in ``space``, or None to show every value.
        """
        if self.view is None:
            return None
        if space == STEP:
            return self.view
        a, b = self._position(self.view[0], space), self._position(max(self.view[1] - 1, self.view[0]), space)
        if b is None:
            return None
        a = b if a is None else a
        return (a, b) if b > a else (a - 0.5, b + 0.5)

    def show_node(self, node: str | None) -> None:
        """
            Shows the plots of ``node``, or of the logged scalars for None.
        """
        self._building = True
        self.node = node
        self._clear()
        if node is None:
            self._logged = self.data.logged_keys()
            self._title.setText('Logged scalars')
            self._empty.setText('Nothing was logged with Recorder.log. Select a node to see what was recorded for it.')
            groups: dict[str, list[str]] = {}
            for key in self._logged:
                groups.setdefault(key.split('/', 1)[0] if '/' in key else 'logged', []).append(key)
            for name, keys in groups.items():
                layout = self._page(name)
                for key in keys:
                    self._add_scalars(layout, SeriesPlot(key.split('/', 1)[-1] if name != 'logged' else key), {'value': key}, key)
                layout.addStretch(1)
        else:
            probes = self.data.probes_of(node)
            self._title.setText(node)
            self._empty.setText('Nothing is recorded for this node.')
            for name in dict.fromkeys(measurements for measurements, _ in probes):
                layout = self._page(name, _description(self.data.measurements[name]))
                modules: dict[str, list[Probe]] = {}
                for measurements, probe in probes:
                    if measurements == name:
                        modules.setdefault(_module(probe, node), []).append(probe)
                for module, members in modules.items():
                    if module:
                        layout.addWidget(_section(module))
                    for probe in members:
                        self._add_probe(layout, name, probe)
                layout.addStretch(1)
        self._empty.setVisible(not self.tabs.count())
        if self._inputs is not None:
            self.tabs.addTab(self._inputs, 'Inputs')
        self.tabs.setVisible(bool(self.tabs.count()))
        for index in range(self.tabs.count()):
            if self.tabs.tabText(index) == self._chosen.get(node is None):
                self.tabs.setCurrentIndex(index)
        self._building = False
        self.refresh()

    def _add_probe(self, layout: QVBoxLayout, measurements: str, probe: Probe) -> None:
        name, key = probe.name, probe.key
        detail = f'{measurements} · {key}'
        if isinstance(probe, (SummaryProbe, DeltaProbe)):
            reductions = SummaryReduction if isinstance(probe, SummaryProbe) else DeltaReduction
            spread = [r for r in self.SPREAD if r in probe.reduce] if isinstance(probe, SummaryProbe) and 'mean' in probe.reduce else []
            if spread:
                title = 'mean' + (' ± std' if 'std' in spread else '') + (', min to max' if {'min', 'max'} <= set(spread) else '')
                self._add_scalars(layout, SeriesPlot(f'{name} · {title}'), {r: scalar_key(measurements, key, r) for r in spread}, detail)
            for reduction in probe.reduce:
                if reduction in spread:
                    continue
                if reductions(reduction).scalar:
                    label = reduction if isinstance(probe, SummaryProbe) else f'change {reduction}'
                    self._add_scalars(layout, SeriesPlot(f'{name} · {label}'), {'value': scalar_key(measurements, key, reduction)}, detail)
                elif reduction in ('active_fraction_per_unit', 'hist'):
                    plot = self._add(layout, BarPlot(f'{name} · {reduction}'), detail)
                    plot.set_labels(x='value' if reduction == 'hist' else 'unit')
                    x_range = probe.range if reduction == 'hist' else None
                    self._at_cursor.append(self._group_bars(plot, measurements, f'{key}#{reduction}', x_range))
                elif reduction == 'full':
                    plot = self._add(layout, MatrixPlot(f'{name} · change'), detail)
                    self._at_cursor.append(self._group_matrix(plot, measurements, f'{key}#full'))
        elif isinstance(probe, SnapshotProbe):
            plot = self._add(layout, MatrixPlot(f'{name} · snapshot'), detail)
            self._at_cursor.append(self._group_matrix(plot, measurements, key))
        elif isinstance(probe, TraceProbe):
            units = len(probe.units) if probe.units is not None else self._trace_width(measurements, key)
            if units is not None and units > self.LINES:
                plot = self._add(layout, ImagePlot(f'{name} · trace of {units} units'), detail)
                plot.set_labels(x='step', y='unit')
                self._at_cursor.append(self._rows(plot, measurements, key, image=True))
            else:
                plot = self._add(layout, SeriesPlot(f'{name} · trace'), detail)
                plot.set_labels(x='step')
                self._at_cursor.append(self._rows(plot, measurements, key, image=False))
        elif isinstance(probe, RasterProbe):
            plot = self._add(layout, ImagePlot(f'{name} · raster'), detail)
            plot.set_labels(x='step', y='unit')
            self._at_cursor.append(self._rows(plot, measurements, key, image=True))

    def _trace_width(self, measurements: str, key: str) -> int | None:
        """
            Returns the values per step of a trace, read from its first span.

            None while that span is not written.
        """
        t0, _, _ = self.data.recorded(measurements)
        found = self.data.rows(measurements, key, int(t0[0])) if len(t0) else None
        return None if found is None or not len(found[0]) else int(np.prod(found[1].shape[1:]))

    # Updates at the cursor.

    def _at_group(self, measurements: str, key: str, show: tp.Callable[[tp.Any], None]) -> tp.Callable[[int], None]:
        """
            Returns an update that calls ``show`` when the group at the cursor changes.

            ``show`` receives ``(t0, value)`` of ``key`` in the last written group starting at or
            before the cursor, or None.
        """
        shown: tp.Any = False
        def update(t: int) -> None:
            nonlocal shown
            group = self.data.group_at(measurements, t)
            if group != shown:
                shown = group
                show(None if group is None else self.data.group_value(measurements, key, t))
        return update

    def _group_bars(self, plot: BarPlot, measurements: str, key: str, x_range: tuple[float, float] | None) -> tp.Callable[[int], None]:
        def show(found) -> None:
            plot.subtitle = '' if found is None else f'  ·  group from step {found[0]:,}'
            plot.set_values(np.zeros(0) if found is None else found[1], x_range=x_range, color=self._color())
        return self._at_group(measurements, key, show)

    def _group_matrix(self, plot: MatrixPlot, measurements: str, key: str) -> tp.Callable[[int], None]:
        def show(found) -> None:
            plot.set_matrix(np.zeros(0) if found is None else found[1])
            plot.subtitle = '' if found is None else f'  ·  group from step {found[0]:,}'
        return self._at_group(measurements, key, show)

    def _rows(self, plot: SeriesPlot | ImagePlot, measurements: str, key: str, image: bool) -> tp.Callable[[int], None]:
        """
            Returns an update that draws the rows of a trace or raster around the cursor.

            The rows are read again only when `RunData.rows_at` changes.
        """
        shown: tp.Any = False
        def update(t: int) -> None:
            nonlocal shown
            at = self.data.rows_at(measurements, t)
            if at != shown:
                shown = at
                found = self.data.rows(measurements, key, t)
                if found is None or not len(found[0]):
                    if image:
                        plot.set_image(np.zeros(0), np.zeros(0))
                    else:
                        plot.set_series([])
                elif image:
                    times, values = found
                    plot.set_image(times, values if values.dtype == np.bool_ else values.astype(np.float64))
                else:
                    times, values = found
                    values = values.reshape(len(values), -1).astype(np.float64)
                    # A trace whose width was not known when the plot was chosen draws its first units.
                    lines = min(values.shape[1], self.LINES)
                    plot.subtitle = '' if lines == values.shape[1] else f' (first {lines} of {values.shape[1]} units)'
                    plot.set_series([(f'unit {u}', times, values[:, u]) for u in range(lines)])
            plot.set_cursor(t)
        return update

    def refresh(self) -> None:
        """
            Draws the scalar series again with the rows read since, and the runs compared.

            Without a node, logged scalars that appeared since are added.
        """
        if self.node is None and self.data.logged_keys() != self._logged:
            self.show_node(None)
            return
        lines = self._lines()
        for plot, keys in self._series:
            self._draw(plot, keys, lines)
        self._legend.setText('Compared: ' + '&nbsp;&nbsp; '.join(
            f'<span style="color: {line.color.name()};">■</span>&nbsp;{html.escape(line.label)}' for line in lines
        ))
        self._legend.setVisible(bool(lines))
        self.set_cursor(self.cursor)

    def _color(self) -> QColor:
        """
            Returns the color of this run, as the table of runs gives it.
        """
        if self.selection is None:
            return THEME.series_color(0)
        run = self.selection.project.run(str(self.data.run.path.resolve())) or self.data.run
        return self.selection.color(str(run.path))

    def _lines(self) -> list[Line]:
        """
            Returns the lines drawn with this run: none when the selection shows no other run, else
            its lines, with this run as a line of its own when it is not shown.
        """
        if self.selection is None:
            return []
        shown = str(self.data.run.path.resolve())
        if not any(path != shown for path in self.selection.visible):
            return []
        lines = self.selection.lines()
        if not any(str(run.path.resolve()) == shown for line in lines for run in line.runs):
            run = self.selection.project.run(shown) or self.data.run
            lines.insert(0, Line(f'{self.selection.label(run)} · shown', self._color(), [run], False))
        return lines

    def _draw(self, plot: SeriesPlot, keys: dict[str, str], lines: list[Line]) -> None:
        """
            Draws the series of ``keys`` in ``plot``, in their space: one series, or the spread of a
            summary, of this run alone, in its color, or of every line compared.
        """
        space = self._space_of(keys)
        plot.space = space
        plot.set_labels(x=axis_label(space))
        x_range = self._x_range(space)
        # Lines compared are listed above the tabs, not over each plot.
        plot.legend = not lines
        if lines:
            self._draw_compared(plot, keys, lines, space, x_range)
            return
        color = self._color()
        if 'value' in keys:
            x, y = self.store.series(self.data.run, keys['value'], space)
            plot.set_series([(keys['value'].split('/')[-1], x, y, color)], x_range=x_range)
            return
        spaces = self._spaces()
        at = lambda steps: spaces.to_space(steps, space)
        t, mean = self.data.scalar(keys['mean'])
        bands = []
        if 'min' in keys and 'max' in keys:
            (tl, low), (th, high) = self.data.scalar(keys['min']), self.data.scalar(keys['max'])
            # The rows of both series written for the same groups.
            common, first, second = np.intersect1d(tl, th, return_indices=True)
            bands.append((('min', 'max'), at(common), low[first], high[second], THEME.with_alpha(color, THEME.band_alpha)))
        if 'std' in keys:
            ts, std = self.data.scalar(keys['std'])
            common, first, second = np.intersect1d(t, ts, return_indices=True)
            bands.append(('std', at(common), mean[first] - std[second], mean[first] + std[second], THEME.with_alpha(color, THEME.spread_alpha)))
        plot.set_series([('mean', at(t), mean, color)], x_range=x_range, bands=bands)

    def _draw_compared(self, plot: SeriesPlot, keys: dict[str, str], lines: list[Line], space: str,
                       x_range: tuple[float, float] | None) -> None:
        """
            Draws the series of ``keys`` for every line: a run as its series, a group as the mean of
            its runs over the band from their lowest to their highest value, with its runs as faint
            lines when the selection draws them.
        """
        key = keys.get('value', keys.get('mean'))
        series, bands, faint = [], [], []
        for line in lines:
            _, band, members = self.selection.draw(self.store, line, key, space)
            if not len(band.x):
                continue
            series.append((line.label, band.x, band.mean, line.color))
            if line.group and len(members) > 1:
                bands.append(('', band.x, band.low, band.high, THEME.with_alpha(line.color, THEME.band_alpha)))
                if self.selection.members:
                    faint += [(x, y, THEME.with_alpha(line.color, THEME.faint_alpha)) for x, y in members]
        plot.set_series(series, x_range=x_range, bands=bands, faint=faint)

    def set_view(self, start: float, end: float) -> None:
        """
            Shows the steps from ``start`` to ``end`` in the scalar plots, each in its space.
        """
        self.view = (float(start), float(end))
        for plot, _ in self._series:
            plot.set_view(self._x_range(getattr(plot, 'space', STEP)))

    def set_cursor(self, t: int) -> None:
        """
            Moves the cursor of every plot to step ``t``, in the space of each, and updates the
            values shown at it.
        """
        self.cursor = int(t)
        for plot, _ in self._series:
            space = getattr(plot, 'space', STEP)
            plot.set_cursor(self.cursor if space == STEP else self._position(self.cursor, space))
        for update in self._at_cursor:
            update(self.cursor)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class _Entry:
    """
        What `InputsPanel` shows of one input or raw stream at the cursor.

        A single value is a row of the list of single values; a vector is a bar plot. One of the two
        is shown.
    """

    def __init__(self, plot: BarPlot, name: QLabel, value: QLabel, step: QLabel) -> None:
        self.plot, self.name, self.value, self.step = plot, name, value, step

    def show(self, found: tuple[int, np.ndarray] | None, cursor: int, labels: list[str] | None = None) -> None:
        """
            Shows the step and the values found at or before ``cursor``, or that none were.
        """
        values = np.zeros(0) if found is None else np.asarray(found[1], np.float64).reshape(-1)
        single = values.size <= 1
        self.plot.setVisible(not single)
        for label in (self.name, self.value, self.step):
            label.setVisible(single)
        if found is None or not values.size:
            self.value.setText('—')
            self.step.setText('nothing recorded at or before the cursor')
            self.plot.set_values(np.zeros(0))
        elif single:
            self.value.setText(format_value(float(values[0])))
            self.step.setText(_recorded_at(found[0], cursor))
        else:
            self.plot.subtitle = f'  ·  {_recorded_at(found[0], cursor)}'
            self.plot.set_values(values, labels=labels)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class InputsPanel(QWidget):
    """
        Panel of what the model received at the cursor.

        Shows the recorded inputs of the model and the raw streams of every set of measurements, as
        last recorded at or before the cursor, with the step they were recorded at. Single values
        are listed together; vectors are drawn as bars, and a raw stream viewed as an image as an
        image, following the views of its measurements.
    """

    def __init__(self, data: RunData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.data = data
        self._updates: list[tp.Callable[[int], None]] = []
        self._entries: list[_Entry] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(*THEME.layout.inputs_margins)
        layout.setSpacing(THEME.layout.inputs_spacing)
        # Single values, one row each: name, value and step.
        self._values = _block()
        self._grid = QGridLayout(self._values)
        self._grid.setContentsMargins(*THEME.layout.inputs_grid_margins)
        self._grid.setHorizontalSpacing(THEME.layout.inputs_grid_spacing)
        self._grid.setColumnStretch(2, 1)
        layout.addWidget(self._values)
        found = False
        for name, measurements in data.measurements.items():
            for stream in measurements.raw:
                found = True
                view = measurements.views.get(stream, {})
                if view.get('kind') == 'image':
                    label = QLabel()
                    label.setMinimumHeight(THEME.layout.inputs_image_height)
                    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    note = _note()
                    layout.addWidget(_section(f'{stream} ({name})'))
                    layout.addWidget(label)
                    layout.addWidget(note)
                    self._updates.append(self._image(label, note, name, stream, view.get('shape')))
                else:
                    self._updates.append(self._vector(self._entry(layout, f'{stream} ({name})'), name, stream, view.get('labels')))
            for probe in measurements.probes:
                if probe.path == (CALL,) and isinstance(probe, TraceProbe):
                    found = True
                    self._updates.append(self._input(self._entry(layout, f'{probe.name} ({name})'), name, probe.key))
        if not found:
            layout.addWidget(_note('No measurements keep the inputs or raw streams of the model.'))
        layout.addStretch(1)

    def _entry(self, layout: QVBoxLayout, title: str) -> _Entry:
        plot = BarPlot(title)
        layout.addWidget(plot)
        row = self._grid.rowCount()
        name, value, step = QLabel(title), QLabel('—'), _note()
        for column, label in enumerate((name, value, step)):
            self._grid.addWidget(label, row, column)
        entry = _Entry(plot, name, value, step)
        self._entries.append(entry)
        return entry

    def _vector(self, entry: _Entry, measurements: str, stream: str, labels: list[str] | None) -> tp.Callable[[int], None]:
        def update(t: int) -> None:
            entry.show(self.data.raw_at(measurements, stream, t), t, labels)
        return update

    def _image(self, label: QLabel, note: QLabel, measurements: str, stream: str, shape: tp.Sequence[int] | None) -> tp.Callable[[int], None]:
        def update(t: int) -> None:
            found = self.data.raw_at(measurements, stream, t)
            note.setText('' if found is None else _recorded_at(found[0], t))
            if found is None:
                label.setText('Nothing recorded at or before the cursor')
                return
            frame = np.asarray(found[1])
            if shape is not None and frame.size == int(np.prod(shape)):
                frame = frame.reshape(tuple(shape))
            if frame.ndim == 3 and frame.shape[2] < 3:
                frame = frame[:, :, 0]
            if frame.ndim not in (2, 3):
                label.setText(f'A frame of shape {frame.shape} is not an image; give its shape in the views of the measurements')
                return
            if frame.dtype != np.uint8:
                frame = frame.astype(np.float64)
                finite = frame[np.isfinite(frame)]
                lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
                frame = (np.nan_to_num(np.clip((frame - lo) / max(hi - lo, 1e-12), 0, 1)) * 255).astype(np.uint8)
            if frame.ndim == 2:
                frame = np.repeat(frame[:, :, None], 3, axis=2)
            frame = np.ascontiguousarray(frame[:, :, :3])
            h, w, _ = frame.shape
            image = QImage(frame.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            label.setPixmap(QPixmap.fromImage(image).scaled(max(label.width(), 64), 160, Qt.AspectRatioMode.KeepAspectRatio))
        return update

    def _input(self, entry: _Entry, measurements: str, key: str) -> tp.Callable[[int], None]:
        def update(t: int) -> None:
            found = self.data.rows(measurements, key, t)
            if found is None or not len(found[0]):
                entry.show(None, t)
                return
            times, values = found
            index = min(max(int(np.searchsorted(times, t, side='right')) - 1, 0), len(times) - 1)
            entry.show((int(times[index]), values[index]), t)
        return update

    def set_cursor(self, t: int) -> None:
        """
            Shows the inputs and raw frames at step ``t``.
        """
        for update in self._updates:
            update(int(t))
        self._values.setVisible(any(not entry.name.isHidden() for entry in self._entries))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class _RecordRow(tp.NamedTuple):
    """
        Widgets of the row of `RunPanel` that records a set of measurements.

        Holds the spin box of the steps to record, the record button and the status of the request.
    """
    steps: QSpinBox
    button: QPushButton
    note: QLabel

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class RunPanel(QWidget):
    """
        Panel of the status, step, environment and parameters of the run, of its warnings and
        errors, and of its measurements.

        While the run is written, each set of measurements can be recorded for a number of steps.
        The request is made with `Run.record`, and its status is shown under the row. Clicking the
        step of a warning or an error emits `cursor_requested` with that step.
    """

    cursor_requested = Signal(int)

    WARNINGS = 50
    """
        Largest number of warnings and errors listed. The others are counted.
    """

    def __init__(self, data: RunData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.data = data
        layout = QVBoxLayout(self)
        layout.setContentsMargins(*THEME.layout.run_margins)
        layout.setSpacing(THEME.layout.run_spacing)
        self._form = _form()
        layout.addLayout(self._form)
        # Coloured by its status, in the stylesheet.
        self._status = QLabel()
        self._status.setObjectName('runViewerStatus')
        self._step = QLabel()
        self._form.addRow('Status', self._status)
        self._form.addRow('Step', self._step)
        info = data.run.info
        git = info.get('git') or {}
        self._form.addRow('Run', _value(info.get('id', data.path.name)))
        self._form.addRow('Created', _value(_when(info.get('created'))))
        if git:
            self._form.addRow('Commit', _value(f"{git.get('sha', '')[:10]} {'(dirty)' if git.get('dirty') else ''}".strip()))
        devices = (info.get('environment') or {}).get('devices') or []
        if devices:
            self._form.addRow('Devices', _value('\n'.join(devices)))
        # Warnings and errors, listed by `refresh`.
        self._notable_heading = _heading('')
        self._notable = QVBoxLayout()
        self._notable.setSpacing(THEME.layout.notable_spacing)
        self._notable_shown = -1
        layout.addWidget(self._notable_heading)
        layout.addLayout(self._notable)
        if data.run.hparams:
            layout.addWidget(_heading('Parameters'))
            params = _form()
            for key, value in data.run.hparams.items():
                params.addRow(str(key), _value(str(value)))
            layout.addLayout(params)
        layout.addWidget(_heading('Measurements'))
        self.record_rows: dict[str, _RecordRow] = {}
        for name, measurements in data.measurements.items():
            block = _block()
            block.setToolTip('\n'.join(p.key for p in measurements.probes))
            column = QVBoxLayout(block)
            column.setContentsMargins(*THEME.layout.block_margins)
            column.setSpacing(THEME.layout.block_spacing)
            title = QLabel(name)
            title.setObjectName('runViewerBlockTitle')
            steps = QSpinBox()
            steps.setRange(1, 2 ** 31 - 1)
            steps.setValue(1000)
            steps.setSuffix(' steps')
            steps.setGroupSeparatorShown(True)
            steps.setToolTip('Steps to record')
            button = QPushButton('Record')
            button.setToolTip(f'Ask the recorder to record "{name}"')
            button.clicked.connect(lambda _=False, n=name, s=steps: self._request(n, s.value()))
            row = QHBoxLayout()
            row.addWidget(steps, 1)
            row.addWidget(button)
            note = _note()
            column.addWidget(title)
            column.addWidget(_note(_description(measurements)))
            column.addLayout(row)
            column.addWidget(note)
            layout.addWidget(block)
            self.record_rows[name] = _RecordRow(steps, button, note)
        layout.addStretch(1)
        self.refresh()

    def _request(self, name: str, steps: int) -> None:
        try:
            self.data.run.record(name, steps)
        except READ_ERRORS as error:
            self.record_rows[name].note.setText(f'request failed: {error}')
            return
        self.record_rows[name].note.setText('request: pending')

    def refresh(self) -> None:
        """
            Shows the status and step of the run and the status of the last request of each row.

            The rows can record only while the run is written.
        """
        status = self.data.status
        self._status.setText(status)
        if self._status.property('status') != status:
            # The stylesheet colours the label by this property, read again once it changes.
            self._status.setProperty('status', status)
            self._status.style().unpolish(self._status)
            self._status.style().polish(self._status)
        self._step.setText(f'{self.data.step:,}')
        live = status == 'running'
        for row in self.record_rows.values():
            row.steps.setEnabled(live)
            row.button.setEnabled(live)
        latest = {request.get('measurements'): request for request in self.data.run.requests()}
        for name, row in self.record_rows.items():
            if name in latest:
                row.note.setText(f"request: {latest[name]['status']}")
        self._list_notable()

    def _list_notable(self) -> None:
        """
            Lists the warnings and errors of the run, when their number changed.
        """
        notable = [event for event in self.data.events if event['kind'] in NOTABLE]
        if len(notable) == self._notable_shown:
            return
        self._notable_shown = len(notable)
        while self._notable.count():
            widget = self._notable.takeAt(0).widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        kinds = sorted({event['kind'] for event in notable}, key=NOTABLE.index)
        self._notable_heading.setText(f"{' and '.join(f'{kind}s' for kind in kinds).capitalize()} ({len(notable)})")
        self._notable_heading.setVisible(bool(notable))
        for event in notable[:self.WARNINGS]:
            step = int(event['t'])
            label = QLabel(f'<a href="{step}" style="color: {THEME.link.name()}; text-decoration: none;">step {step:,}</a>&nbsp; '
                           f'{html.escape(event_text(event))}')
            label.setObjectName('runViewerWarning' if event['kind'] == 'warning' else 'runViewerError')
            label.setWordWrap(True)
            label.setTextFormat(Qt.TextFormat.RichText)
            label.linkActivated.connect(lambda href: self.cursor_requested.emit(int(href)))
            self._notable.addWidget(label)
        if len(notable) > self.WARNINGS:
            self._notable.addWidget(_note(f'and {len(notable) - self.WARNINGS} more'))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class RunViewerWindow(QMainWindow):
    """
        Window comparing the runs of a directory, and showing one of them over the graph of its
        model.

        Opened on a directory, the window shows its Workspace: a panel for every scalar series of its
        runs, drawn for every run or group the Runs table shows, each in the space it was recorded
        in. Opened on a run, it shows that run, compared with nothing until other runs are shown. A
        double click on a run of the table shows it on the Graph tab, the Details and Probes panels
        and the timeline. Runs added to the directory are listed as they appear.

        The exploration of the directory, what the window shows and how, is kept as it changes, in a
        file of the viewer rather than within the directory, and resumed when the directory is opened
        again. File > Save Exploration As keeps a copy elsewhere, which File > Open Exploration
        resumes.

        The graph shows at the cursor the firing rate of every node whose spikes are summarized.
        Selecting a node shows what was recorded for it. The timeline moves the cursor through the
        run, and its zoom sets the steps the scalar plots show. The toolbar plays the run forward
        and moves the cursor to the previous or next recorded span. A run still being written is
        read again periodically. Its measurements can be recorded from the window, and Follow keeps
        the cursor at the last step. The Probes panel shows what was recorded for the node selected,
        and what the model received, in tabs. The Probes panel and the timeline belong to the Graph
        tab: both are collapsed while the Workspace is shown, and open again with the Graph tab
        unless closed there. The Runs and Details panels span the height of the window; the timeline lies
        under the graph and the Probes panel. On the Graph tab, the Probes panel can take all but a
        sliver of the graph.

        The runs and groups the Runs table shows besides the run shown are drawn with it in the scalar
        plots of the Probes panel. The menu bar opens other directories and runs, the preferences of
        the look of the viewer, and the panels closed.

        Keys: Left and Right move the cursor by 1% of the steps the timeline shows, one step with
        Shift. Home and End go to the first and the last step, PageUp and PageDown to the start of
        the previous and next recorded span. Space plays and pauses.

        Parameters
        ----------
        path : str or path-like
            Directory of a run, or of runs.
        refresh_every : int, default 1000
            Milliseconds between two reads of the run. Once a read finds a change while the run is
            not being written, the interval is five times as long.
        parent : QWidget, optional
            Parent widget.

        Attributes
        ----------
        project : Project
            The runs of the directory.
        exploration : Exploration
            Where the state of the window is kept for the directory.
        selection : Selection
            The runs shown, their groups and how they are drawn.
        store : SeriesStore
            The series read from the runs.
        data : RunData
            What the window read from the run shown.
        cursor : int
            Step at the cursor.
        badges : dict of str to ActivityBadge
            Badge of every node of the graph, by name.
        refresh_every : int
            Milliseconds between two reads of a run being written.

        Notes
        -----
        When the model of the run cannot be loaded, the window shows the error in place of the
        graph. Custom neurons must be registered before the run is opened, as for the editor.
        Closing the window drops what it read from the run.

        See Also
        --------
        SparkRunViewer : Opens runs in windows of their own from a script or a notebook.
        RunData : What the viewer reads from a run.
        spark.recording.Run : A run written by a `Recorder`, read back.
    """

    def __init__(self, path: str | pathlib.Path, refresh_every: int = 1000, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        STYLES.init()
        THEME.read()
        self.setObjectName('runViewer')
        self.resize(THEME.layout.window_width, THEME.layout.window_height)
        path = pathlib.Path(path)
        opened_run = (path / 'run.json').exists()
        self.project = Project(path.parent if opened_run else path, self)
        if not opened_run and not self.project.runs:
            raise FileNotFoundError(f'No run at "{path}", nor runs within it.')
        self.selection = Selection(self.project, self)
        self.store = SeriesStore()
        self.data = RunData(path if opened_run else self.project.runs[-1].path)
        self.store.set_reader(self.data.run, self.data.scalar)
        if opened_run:
            # A run opened on its own is compared with nothing until other runs are shown.
            self.selection.show_only([str(path.resolve())])
        self.cursor = 0
        self.badges: dict[str, ActivityBadge] = {}
        self.setDockNestingEnabled(True)
        # Panels sharing a place show their tabs above them.
        self.setTabPosition(Qt.DockWidgetArea.AllDockWidgetAreas, QTabWidget.TabPosition.North)
        self.center = QTabWidget()
        self.center.setObjectName('runViewerCenter')
        self.center.setDocumentMode(True)
        self.setCentralWidget(self.center)
        self.workspace = WorkspaceView(self.project, self.selection, self.store)
        self.center.addTab(self.workspace, 'Workspace')
        self._build_graph()
        self._build_docks()
        self._build_menus()
        self._build_toolbar()
        self._build_keys()
        STYLES.reloaded.connect(self.restyle)
        status = QStatusBar()
        status.setObjectName('statusBar')
        self.setStatusBar(status)
        self.refresh_every = int(refresh_every)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(self.refresh_every)
        # Runs added to the directory are looked for less often.
        self._runs_timer = QTimer(self)
        self._runs_timer.timeout.connect(self._refresh_runs)
        self._runs_timer.start(5 * self.refresh_every)
        self._player = QTimer(self)
        self._player.timeout.connect(self._advance)
        self._update_timeline()
        self.set_cursor(0)
        self._name()
        # The panels of the graph: collapsed on the Workspace, open again on the Graph tab unless closed there.
        self._graph_docks = {'Probes': self.dock_probes, 'Timeline': self.dock_timeline}
        self._docks_open = dict.fromkeys(self._graph_docks, True)
        self._tab: QWidget | None = None
        self.center.currentChanged.connect(self._on_tab)
        self.center.setCurrentIndex(self._graph_tab() if opened_run else self.center.indexOf(self.workspace))
        self._on_tab(self.center.currentIndex())
        (self.dock_run if opened_run else self.dock_runs).raise_()
        self.statusBar().showMessage(f'{self.data.path}')
        self.exploration = Exploration.of(self.project.root)
        state = self.exploration.read()
        if state is not None:
            # A run opened on its own stays shown; the rest of the exploration resumes.
            self.resume(state, shown=not opened_run)
        # Kept once it changes: a directory opened and left as it was leaves no file.
        self._kept: dict[str, tp.Any] | None = self.state()

    # Exploration.

    def state(self) -> dict[str, tp.Any]:
        """
            Returns the state of the exploration: the directory, the run shown, its node selected,
            the cursor and the view of the timeline, the tab shown, the runs shown and how, the
            table, the workspace and the layout of the panels.
        """
        a, b = self.timeline.view
        shown = self.data.path.resolve()
        return {
            'root': str(self.project.root),
            'shown': shown.name if shown.parent == self.project.root else str(shown),
            'node': self.probe_panel.node,
            'cursor': self.cursor,
            'view': [float(a), float(b)],
            'tab': self.center.tabText(self.center.currentIndex()),
            'docks': self._docks_open if self.center.currentWidget() is self.workspace else {
                name: not dock.isHidden() for name, dock in self._graph_docks.items()
            },
            'project': self.project.state(),
            'selection': self.selection.state(),
            'table': self.runs_table.state(),
            'workspace': self.workspace.state(),
            'layout': bytes(self.saveState().toBase64()).decode(),
        }

    def resume(self, state: dict[str, tp.Any], shown: bool = True) -> None:
        """
            Resumes the exploration ``state``, with the run it shows when ``shown``.

            What no longer applies, such as a run deleted since, is left out. A state that cannot be
            resumed whole is reported in the status bar.
        """
        try:
            self.project.restore(state.get('project') or {})
            self.selection.restore(state.get('selection') or {})
            self.runs_table.restore(state.get('table') or {})
            self.workspace.restore(state.get('workspace') or {})
            layout = state.get('layout')
            if isinstance(layout, str):
                self.restoreState(QByteArray.fromBase64(layout.encode()))
                # The layout kept holds the corners too, as they were when it was kept.
                self._set_corners()
            run = self.project.root / str(state.get('shown', ''))
            if shown and state.get('shown') and (run / 'run.json').exists():
                self.show_run(run)
            node = state.get('node')
            if node and self.scene is not None:
                for item in self.scene.items():
                    if isinstance(item, NodeItem) and item.model.name == node:
                        item.setSelected(True)
            view = state.get('view')
            if isinstance(view, list) and len(view) == 2 and view[1] > view[0]:
                self.timeline.show_range(float(view[0]), float(view[1]))
            self.set_cursor(int(state.get('cursor', self.cursor)))
            tab = next((i for i in range(self.center.count()) if self.center.tabText(i) == state.get('tab')), -1)
            if tab >= 0:
                self.center.setCurrentIndex(tab)
            # The panels of the graph open on the Graph tab, which the layout restored may not say.
            docks = state.get('docks') or {}
            self._docks_open = {name: bool(docks.get(name, True)) for name in self._graph_docks}
            self._on_tab(self.center.currentIndex())
        except (TypeError, ValueError, KeyError, AttributeError) as error:
            self.statusBar().showMessage(f'The exploration could not be resumed whole: {type(error).__name__}: {error}')

    def keep_exploration(self) -> None:
        """
            Writes the state of the window to its exploration when it changed since last written.

            A failure is shown in the status bar.
        """
        state = self.state()
        if state == self._kept:
            return
        try:
            self.exploration.write(state)
        except (OSError, ValueError) as error:
            self.statusBar().showMessage(f'The exploration could not be kept: {error}')
            return
        self._kept = state

    def save_exploration(self, path: str | pathlib.Path) -> bool:
        """
            Writes the state of the window to the file ``path``, which cannot be within a directory
            of runs.

            Returns whether it was written; a refusal or a failure is reported in a message box.
        """
        path = pathlib.Path(path)
        if not path.name.endswith(SUFFIX):
            path = path.with_name(path.name.removesuffix('.json') + SUFFIX)
        try:
            Exploration(path).write(self.state())
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, 'Save Exploration', str(error))
            return False
        return True

    def open_exploration(self, path: str | pathlib.Path) -> RunViewerWindow | None:
        """
            Resumes the exploration saved in the file ``path``: in this window when it explores the
            same directory, else in a window of its own.

            Returns the window, or None when the file cannot be read or its directory opened, which
            is reported in a message box.
        """
        state = Exploration(path).read()
        if state is None or not isinstance(state.get('root'), str):
            QMessageBox.warning(self, 'Open Exploration', f'"{path}" is not an exploration this viewer can read.')
            return None
        root = pathlib.Path(state['root'])
        window = self if root.resolve() == self.project.root else self.open_run(root)
        if window is not None:
            window.resume(state)
        return window

    def _save_exploration_dialog(self, checked: bool = False) -> None:
        path, _ = QFileDialog.getSaveFileName(self, 'Save Exploration As', str(pathlib.Path.home() / f'{self.project.root.name}{SUFFIX}'),
                                              f'Explorations (*{SUFFIX})')
        if path:
            self.save_exploration(path)

    def _open_exploration_dialog(self, checked: bool = False) -> None:
        path, _ = QFileDialog.getOpenFileName(self, 'Open Exploration', str(pathlib.Path.home()), f'Explorations (*{SUFFIX})')
        if path:
            self.open_exploration(path)

    def _name(self) -> None:
        self.setWindowTitle(f'Spark Run Viewer - {self.project.root.name} - {self.data.path.name}')

    def _graph_tab(self) -> int:
        return next((i for i in range(self.center.count()) if self.center.tabText(i) == 'Graph'), -1)

    def _on_tab(self, index: int) -> None:
        """
            Collapses the Probes panel and the timeline on the Workspace tab, and opens each again
            on the Graph tab unless it was closed there.
        """
        widget = self.center.widget(index)
        # The tabs are as wide as their widest page, the Workspace; the graph alone may be narrowed to a sliver.
        self.center.setMinimumWidth(0 if widget is self.workspace else THEME.layout.graph_min_width)
        if widget is self.workspace:
            leaving = self._tab is not None and self._tab is not self.workspace
            for name, dock in self._graph_docks.items():
                if leaving:
                    self._docks_open[name] = not dock.isHidden()
                dock.hide()
        elif widget is not None:
            for name, dock in self._graph_docks.items():
                dock.setVisible(self._docks_open[name])
        self._tab = widget

    # Construction.

    def _build_graph(self) -> None:
        if self.data.config is None:
            label = QLabel(
                'The model of this run could not be loaded, so its graph is not shown.\n'
                f'{self.data.config_error}\n\nCustom neurons must be registered before the run is opened, '
                'as for the editor (spark.register_neuron_from_config_file).'
            )
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setWordWrap(True)
            self.scene, self.view = None, None
            self._show_graph(label)
            return
        from spark.graph_editor.models import session_io
        from spark.graph_editor.models.controller_profile import profile_for_config
        self.scene = _ReadOnlyScene(parent=self)
        self.view = _ReadOnlyView(self.scene)
        profile = profile_for_config(self.data.config)
        self.scene.model.set_profile(profile, force=True)
        self.view.import_config(self.data.config, label=self.data.path.name, layout=session_io.model_layout(self.data.path / 'model.scfg'))
        self.scene.model.undo_stack.clear()
        # import_config leaves the imported nodes selected.
        self.scene.clearSelection()
        nodes = []
        for item in self.scene.items():
            if isinstance(item, NodeItem):
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
                nodes.append(item)
            else:
                # Pipes and ports take no mouse buttons; clicks reach the node below.
                item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                item.setAcceptHoverEvents(False)
        self.badges = {node.model.name: ActivityBadge(node) for node in nodes}
        self.scene.selectionChanged.connect(self._on_selection)
        self._show_graph(self.view)
        # Fitted again as the window is laid out: its size is not final until it is shown.
        self.view.fit()

    def _show_graph(self, widget: QWidget) -> None:
        """
            Shows ``widget`` in the Graph tab of the center, in place of the graph shown before.
        """
        current, index = self.center.currentIndex(), self._graph_tab()
        if index >= 0:
            old = self.center.widget(index)
            self.center.removeTab(index)
            old.setParent(None)
            old.deleteLater()
        self.center.insertTab(1, widget, 'Graph')
        self.center.setCurrentIndex(max(current, 0))

    def _dock(self, title: str, widget: QWidget, area: Qt.DockWidgetArea, name: str | None = None) -> QDockWidget:
        dock = QDockWidget(title, self)
        label = QLabel(f' {title.upper()}')
        label.setObjectName('dockTitle')
        dock.setTitleBarWidget(label)
        dock.setWidget(widget)
        # The layouts kept by the explorations name the panels by it.
        dock.setObjectName(f'dock{name or title}')
        self.addDockWidget(area, dock)
        return dock

    def _set_corners(self) -> None:
        # The left panels take their corner, so that the timeline, collapsed with the Workspace, leaves them as they are;
        # the timeline takes the other, under the Probes panel.
        self.setCorner(Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.BottomDockWidgetArea)

    def _build_docks(self) -> None:
        self._set_corners()
        self.runs_table = RunsTable(self.project, self.selection)
        self.runs_table.run_activated.connect(self.show_run)
        self.timeline = Timeline()
        self.timeline.cursor_moved.connect(self.set_cursor)
        self.dock_runs = self._dock('Runs', self.runs_table, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.dock_run = self._dock('Details', QWidget(), Qt.DockWidgetArea.LeftDockWidgetArea, name='Run')
        self.tabifyDockWidget(self.dock_runs, self.dock_run)
        # Their tabs name them.
        for dock in (self.dock_runs, self.dock_run):
            dock.setTitleBarWidget(QWidget())
        self.dock_probes = self._dock('Probes', QWidget(), Qt.DockWidgetArea.RightDockWidgetArea)
        self.dock_timeline = self._dock('Timeline', self.timeline, Qt.DockWidgetArea.BottomDockWidgetArea)
        self._build_run()
        self.resizeDocks([self.dock_runs, self.dock_probes], [THEME.layout.runs_width, THEME.layout.probes_width], Qt.Orientation.Horizontal)

    def _build_run(self) -> None:
        """
            Builds the Details and Probes panels of the run shown, in place of those of the run before.
        """
        old = getattr(self, 'probe_panel', None)
        if old is not None:
            old.release()
            self.timeline.view_changed.disconnect(old.set_view)
        self.run_panel = RunPanel(self.data)
        self.run_panel.cursor_requested.connect(self._go)
        self.inputs_panel = InputsPanel(self.data)
        # The probe panel scrolls within its tabs, the inputs being the last.
        self.probe_panel = ProbePanel(self.data, self.selection, self.store, self.inputs_panel)
        self.probe_panel.cursor_moved.connect(self.set_cursor)
        self.probe_panel.view_requested.connect(self.timeline.show_range)
        self.timeline.view_changed.connect(self.probe_panel.set_view)
        for dock, widget in ((self.dock_run, _scroll(self.run_panel)), (self.dock_probes, self.probe_panel)):
            old = dock.widget()
            dock.setWidget(widget)
            if old is not None:
                old.setParent(None)
                old.deleteLater()

    def show_run(self, path: str | pathlib.Path) -> None:
        """
            Shows the run at ``path`` on the Graph tab, the Details and Probes panels and the timeline.

            A run that cannot be read is reported in a message box.
        """
        path = pathlib.Path(path)
        if path.resolve() != self.data.path.resolve():
            try:
                data = RunData(path)
            except (*READ_ERRORS, FileNotFoundError, TypeError) as error:
                QMessageBox.warning(self, 'Show Run', f'"{path}" cannot be read.\n\n{type(error).__name__}: {error}')
                return
            self.store.set_reader(self.data.run, None)
            self.data.release()
            self.data = data
            self.store.set_reader(self.data.run, self.data.scalar)
            old = self.scene
            if old is not None:
                old.selectionChanged.disconnect(self._on_selection)
            self._build_graph()
            if old is not None:
                old.deleteLater()
            self._build_run()
            self.follow.setChecked(self.data.status == 'running')
            # The timeline shows the whole of the run.
            self.timeline.view = (0.0, 1.0)
            self._update_timeline()
            self.set_cursor(0)
            self._name()
        self.center.setCurrentIndex(self._graph_tab())
        self.dock_run.raise_()

    def _build_menus(self) -> None:
        # Menus are kept as attributes: reached through temporaries, their PySide wrappers are invalidated.
        bar = self.menuBar()
        self._file_menu = bar.addMenu('&File')
        self._action(self._file_menu, 'Open...', 'Ctrl+O', self._open_run_dialog)
        self._action(self._file_menu, 'Open Exploration...', None, self._open_exploration_dialog)
        self._action(self._file_menu, 'Save Exploration As...', None, self._save_exploration_dialog)
        self._file_menu.addSeparator()
        self._action(self._file_menu, 'Close Window', 'Ctrl+W', self.close)
        self._action(self._file_menu, 'Quit', 'Ctrl+Q', self.quit)
        self._edit_menu = bar.addMenu('&Edit')
        self._action(self._edit_menu, 'Preferences...', None, self.open_preferences)
        self._window_menu = bar.addMenu('&Window')
        for dock in (self.dock_runs, self.dock_run, self.dock_probes, self.dock_timeline):
            action = dock.toggleViewAction()
            action.setText(dock.windowTitle())
            self._window_menu.addAction(action)

    def _action(self, menu, text: str, shortcut: str | None, slot: tp.Callable) -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(shortcut)
        # A bound method: a lambda holding the window would keep it alive once closed.
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    def _build_toolbar(self) -> None:
        toolbar = self.addToolBar('Run')
        toolbar.setObjectName('runToolbar')
        toolbar.setMovable(False)
        self.play = QPushButton('Play')
        self.play.setCheckable(True)
        self.play.setToolTip('Play the run forward through the steps the timeline shows (Space)')
        self.play.toggled.connect(self._toggle_play)
        self.follow = QCheckBox('Follow')
        self.follow.setToolTip('Keep the cursor at the last step while the run is written')
        self.follow.setChecked(self.data.status == 'running')
        whole = QPushButton('Whole run')
        whole.setToolTip('Show every step in the timeline and the plots')
        whole.clicked.connect(self._show_whole_run)
        previous = QPushButton('◀  Previous recording')
        previous.setToolTip('Move the cursor to the start of the previous recorded span (Page Up)')
        previous.clicked.connect(self._previous_span)
        following = QPushButton('Next recording  ▶')
        following.setToolTip('Move the cursor to the start of the next recorded span (Page Down)')
        following.clicked.connect(self._next_span)
        for widget in (self.play, self.follow, whole, previous, following):
            toolbar.addWidget(widget)

    def _build_keys(self) -> None:
        # Bound methods: a lambda holding the window would keep it alive once closed.
        keys = {
            'Left': self._back, 'Right': self._forward, 'Shift+Left': self._back_one, 'Shift+Right': self._forward_one,
            'Home': self._first, 'End': self._last, 'PgUp': self._previous_span, 'PgDown': self._next_span,
            'Space': self.play.toggle,
        }
        for sequence, action in keys.items():
            QShortcut(QKeySequence(sequence), self).activated.connect(action)

    def _show_whole_run(self) -> None:
        self.timeline.show_range(0, self.data.step)

    # Menus.

    def open_run(self, path: str | pathlib.Path) -> RunViewerWindow | None:
        """
            Opens the run, or the directory of runs, at ``path`` in a window of its own.

            Returns the window, or None when nothing at ``path`` can be opened, which is reported in
            a message box.
        """
        try:
            window = RunViewerWindow(path)
        except (*READ_ERRORS, FileNotFoundError, TypeError) as error:
            QMessageBox.warning(self, 'Open', f'"{path}" cannot be opened.\n\n{type(error).__name__}: {error}')
            return None
        window.show()
        return window

    def _open_run_dialog(self, checked: bool = False) -> None:
        path = QFileDialog.getExistingDirectory(self, 'Open a Directory of Runs, or a Run', str(self.project.root))
        if path:
            self.open_run(path)

    def quit(self, checked: bool = False) -> None:
        """
            Closes every window of the run viewer.

            The application goes on when it holds other windows, such as the editor's, or runs in a
            notebook.
        """
        for window in [w for w in _SHOWN if isValid(w) and w.isVisible()] + [self]:
            if isValid(window):
                window.close()

    def open_preferences(self, checked: bool = False) -> None:
        """
            Opens the preferences of the editor, limited to what changes the viewer.

            Applying them redraws every window of the viewer.
        """
        from spark.graph_editor.widgets.preferences_dialog import PreferencesDialog
        dialog = PreferencesDialog(self, sections=PREFERENCES, library=False)
        dialog.exec()

    def restyle(self) -> None:
        """
            Draws the window again with the style: the plots, the timeline and the graph.

            Called when the style is reloaded, as by the preferences, once `THEME` has read it.
        """
        node = self.probe_panel.node
        self._rebuild_graph()
        self.probe_panel.show_node(node)
        self.workspace.redraw()
        self.set_cursor(self.cursor)
        for widget in self.findChildren(QWidget):
            widget.update()

    def _rebuild_graph(self) -> None:
        """
            Builds the graph again, with the node selected before still selected.
        """
        if self.scene is None:
            return
        selected = [item.model.name for item in self.scene.selectedItems() if isinstance(item, NodeItem)]
        old = self.scene
        old.selectionChanged.disconnect(self._on_selection)
        self._build_graph()
        old.deleteLater()
        for item in self.scene.items():
            if isinstance(item, NodeItem) and item.model.name in selected:
                item.setSelected(True)

    # Navigation.

    def _go(self, step: int) -> None:
        """
            Moves the cursor to ``step``, and the timeline to show it.
        """
        self.set_cursor(step)
        a, b = self.timeline.view
        if not a <= self.cursor <= b:
            self.timeline.show_range(self.cursor - (b - a) / 2, self.cursor + (b - a) / 2)

    def _step_by(self, direction: int, one: bool) -> None:
        """
            Moves the cursor by one step, or by 1% of the steps the timeline shows.
        """
        a, b = self.timeline.view
        self._go(self.cursor + direction * (1 if one else max(int(round((b - a) / 100)), 1)))

    def _jump(self, direction: int) -> None:
        """
            Moves the cursor to the start of the next recorded span, or of the previous one.

            Spans of every set of measurements count. The cursor stays without one.
        """
        starts = [self.data.spans(name)[:, 0] for name in self.data.measurements]
        starts = np.unique(np.concatenate(starts)) if starts else np.zeros(0, np.int64)
        found = starts[starts > self.cursor][:1] if direction > 0 else starts[starts < self.cursor][-1:]
        if len(found):
            self._go(int(found[0]))

    def _back(self) -> None:
        self._step_by(-1, False)

    def _forward(self) -> None:
        self._step_by(1, False)

    def _back_one(self) -> None:
        self._step_by(-1, True)

    def _forward_one(self) -> None:
        self._step_by(1, True)

    def _first(self) -> None:
        self._go(0)

    def _last(self) -> None:
        self._go(self.data.step)

    def _previous_span(self) -> None:
        self._jump(-1)

    def _next_span(self) -> None:
        self._jump(1)

    # Cursor.

    def set_cursor(self, step: int) -> None:
        """
            Moves the cursor to a step and updates the timeline, the plots and the badges.

            The status bar shows the step and the error of the last read that failed.

            Parameters
            ----------
            step : int
                Step, clipped to the steps of the run.
        """
        self.cursor = int(min(max(step, 0), max(self.data.step, 0)))
        self.timeline.set_cursor(self.cursor)
        self.probe_panel.set_cursor(self.cursor)
        self.inputs_panel.set_cursor(self.cursor)
        try:
            with self.data.run.reading():
                for name, badge in self.badges.items():
                    badge.set_value(self.data.activity(name, self.cursor))
        except READ_ERRORS as error:
            self.data.error = f'{type(error).__name__}: {error}'
        failed = f'   Reading the run failed: {self.data.error}' if self.data.error else ''
        self.statusBar().showMessage(f'Step {self.cursor:,} of {self.data.step:,}{failed}')

    def _toggle_play(self, playing: bool) -> None:
        if playing:
            self._player.start(40)
        else:
            self._player.stop()

    def _advance(self) -> None:
        a, b = self.timeline.view
        stride = max(int((b - a) / 400), 1)
        if self.cursor + stride > self.data.step:
            self.play.setChecked(False)
            return
        self.set_cursor(self.cursor + stride)

    def _on_selection(self) -> None:
        if self.scene is None or not isValid(self.scene):
            return
        nodes = [item.model.name for item in self.scene.selectedItems() if isinstance(item, NodeItem)]
        self.probe_panel.show_node(nodes[0] if len(nodes) == 1 else None)
        self.dock_probes.raise_()

    # Live.

    def _update_timeline(self) -> None:
        rows = [(name, self.data.spans(name)) for name in self.data.measurements]
        events = [(event['t'], event['kind']) for event in self.data.events]
        spaces = self.store.spaces(self.data.run)
        if self.data.status == 'running':
            spaces.forget()
        tags = [(name, *spaces.timeline(name)) for name in spaces.tags()]
        self.timeline.set_data(self.data.step, rows, events, self.data.events, tags)

    def refresh(self) -> None:
        """
            Reads what the run added since the last refresh and redraws what changed.

            Called by a timer. A read that fails is shown in the status bar and tried again on the
            next refresh. With Follow checked, the cursor moves to the last step.
        """
        changed = self.data.refresh()
        if not changed:
            return
        try:
            self._timer.setInterval(self.refresh_every if self.data.status == 'running' else 5 * self.refresh_every)
            self.run_panel.refresh()
            if changed & {'windows', 'events', 'info'}:
                self._update_timeline()
            if 'scalars' in changed:
                # The series of the run and its tags as the viewer read them, not as read before.
                self.store.forget(self.data.run)
                self.probe_panel.refresh()
                self.workspace.refresh(redraw=str(self.data.run.path.resolve()) in self.selection.visible)
        except READ_ERRORS as error:
            self.data.error = f'{type(error).__name__}: {error}'
        finally:
            self.set_cursor(self.data.step if self.follow.isChecked() else self.cursor)

    def _refresh_runs(self) -> None:
        """
            Lists the runs added to the directory, reads again the workspace shown while a run it
            draws is written, and keeps the exploration.
        """
        self.project.refresh()
        self.keep_exploration()
        shown = str(self.data.run.path.resolve())
        written = any(
            run.info.get('status') == 'running' and str(run.path) in self.selection.visible and str(run.path) != shown
            for run in self.project.runs
        )
        if written and self.center.currentWidget() is self.workspace:
            self.workspace.refresh()

    def showEvent(self, event) -> None:
        _keep(self)
        super().showEvent(event)

    def closeEvent(self, event) -> None:
        self.keep_exploration()
        self._timer.stop()
        self._runs_timer.stop()
        self._player.stop()
        try:
            STYLES.reloaded.disconnect(self.restyle)
        except (RuntimeError, TypeError):
            pass
        if self.scene is not None:
            try:
                self.scene.selectionChanged.disconnect(self._on_selection)
            except (RuntimeError, TypeError):
                pass
        self.probe_panel.release()
        self.data.release()
        self.store.release()
        super().closeEvent(event)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _keep(window: RunViewerWindow) -> None:
    _SHOWN[:] = [w for w in _SHOWN if w is not window and isValid(w) and w.isVisible()] + [window]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SparkRunViewer:
    """
        Opens directories of runs, and runs, in windows of their own from a script or a notebook.

        Available as ``spark.RunViewer``. In a notebook, the Qt event loop runs within the kernel
        and `open` returns at once. In a script, `open` runs the event loop until the windows are
        closed.

        Attributes
        ----------
        app : QApplication
            The application, created when none exists.
        windows : list of RunViewerWindow
            Windows opened, those closed dropped at the next `open`.

        See Also
        --------
        RunViewerWindow : The window of a directory of runs.
        SparkGraphEditor : The graph editor, opened in the same way.

        Examples
        --------
        >>> viewer = spark.RunViewer()
        >>> viewer.open('runs')                                         # every run, compared
        >>> viewer.open('runs/20260923-091240_cartpole_4784ed')        # one run
    """

    def __init__(self) -> None:
        self._is_interactive = 'ipykernel' in sys.modules
        if self._is_interactive:
            from IPython import get_ipython
            get_ipython().enable_gui('qt')
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.windows: list[RunViewerWindow] = []

    def open(self, path: str | pathlib.Path) -> RunViewerWindow:
        """
            Opens the directory of runs, or the run, at ``path`` in a new window.

            Outside a notebook, blocks until the windows are closed.

            Parameters
            ----------
            path : str or path-like
                Directory of runs, opened on its workspace, or of a run.

            Returns
            -------
            RunViewerWindow
                The window.

            Raises
            ------
            FileNotFoundError
                When ``path`` is not a run and holds none.
            ValueError
                When the run was written with another version of the index.
        """
        STYLES.init()
        STYLES.apply(self.app)
        window = RunViewerWindow(path)
        window.show()
        self.windows = [w for w in self.windows if w.isVisible() and w is not window] + [window]
        if not self._is_interactive:
            self.app.exec()
        return window

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
