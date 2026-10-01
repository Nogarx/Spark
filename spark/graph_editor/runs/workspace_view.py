#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp

import re
import html

from PySide6.QtCore import Qt, Signal, QTimer, QEvent, QObject
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton, QToolButton, QMenu, QTreeWidget,
    QTreeWidgetItem, QHeaderView, QInputDialog, QColorDialog, QComboBox, QSlider, QCheckBox, QDialog, QListWidget,
    QListWidgetItem, QScrollArea, QFrame, QSizePolicy,
)

from spark.graph_editor.styles.run_viewer import THEME
from spark.graph_editor.runs.data import READ_ERRORS
from spark.graph_editor.runs.plots import SeriesPlot
from spark.graph_editor.runs.workspace import (
    Project, Selection, SeriesStore, NATURAL, STEP, WALL, axis_label, short_name,
)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def _note(text: str = '') -> QLabel:
    label = QLabel(text)
    label.setObjectName('runViewerNote')
    label.setWordWrap(True)
    return label

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class _Item(QTreeWidgetItem):
    """
        Row of the table of runs, sorted by the number a column holds when it holds one.
    """

    def __lt__(self, other: QTreeWidgetItem) -> bool:
        column = self.treeWidget().sortColumn() if self.treeWidget() is not None else 0
        mine, theirs = self.data(column, Qt.ItemDataRole.UserRole), other.data(column, Qt.ItemDataRole.UserRole)
        if column > 0 and isinstance(mine, (int, float)) and isinstance(theirs, (int, float)):
            return mine < theirs
        return self.text(column).lower() < other.text(column).lower()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class RunsTable(QWidget):
    """
        Table of the runs of a project, setting what the workspace and the probe panel draw.

        Every run has an eye, a check box that shows or hides it, and the color it is drawn in.
        Grouped by fields of the runs (Group by), the table is a tree of groups, each with its eye,
        its color and its runs under it. The search field keeps the runs whose name, experiment or
        parameters hold its text. Columns give the experiment, the steps and status, and the
        parameters in which the runs differ; more are chosen from Columns. Columns are resized by
        dragging the edges of their headers; until it is, the first takes the width the others leave,
        and the last fills what is left after it. A double click on a run
        emits `run_activated` with its path. A right click on a run sets its experiment or its color.

        Parameters
        ----------
        project : Project
            The runs.
        selection : Selection
            What is shown, set here.
        parent : QWidget, optional
            Parent widget.
    """

    run_activated = Signal(str)

    FIXED = ('Run', 'Experiment', 'Steps')

    def __init__(self, project: Project, selection: Selection, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project = project
        self.selection = selection
        # Parameters shown as columns: those in which the runs differ, until chosen.
        self.columns: list[str] = self._differing()
        self._expanded: set[str] = set()
        # Widths the columns were resized to, by header, kept as the table is built again.
        self._widths: dict[str, int] = {}
        self._building = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(*THEME.layout.runs_margins)
        layout.setSpacing(THEME.layout.runs_spacing)
        self.search = QLineEdit()
        self.search.setPlaceholderText('Search runs...')
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        tools = QHBoxLayout()
        self._group_button = QToolButton()
        self._group_button.setText('Group by')
        self._group_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._group_menu = QMenu(self._group_button)
        self._group_menu.aboutToShow.connect(self._fill_group_menu)
        self._group_button.setMenu(self._group_menu)
        self._columns_button = QToolButton()
        self._columns_button.setText('Columns')
        self._columns_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._columns_menu = QMenu(self._columns_button)
        self._columns_menu.aboutToShow.connect(self._fill_columns_menu)
        self._columns_button.setMenu(self._columns_menu)
        tools.addWidget(self._group_button)
        tools.addWidget(self._columns_button)
        tools.addStretch(1)
        for text, tip, slot in (
            ('All', 'Show every run', self._show_all), ('None', 'Hide every run', self._hide_all),
            (f'Newest {Selection.NEWEST}', f'Show the {Selection.NEWEST} newest runs only', self._show_newest),
        ):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            tools.addWidget(button)
        self.tree = QTreeWidget()
        self.tree.setObjectName('runViewerRuns')
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setSortingEnabled(True)
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self.tree.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        self.tree.itemChanged.connect(self._on_item)
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        self.tree.header().sectionResized.connect(self._on_resized)
        self.tree.viewport().installEventFilter(self)
        self.tree.itemExpanded.connect(lambda item: self._expanded.add(item.data(0, Qt.ItemDataRole.UserRole + 1)))
        self.tree.itemCollapsed.connect(lambda item: self._expanded.discard(item.data(0, Qt.ItemDataRole.UserRole + 1)))
        self._summary = _note()
        layout.addWidget(self.search)
        layout.addLayout(tools)
        layout.addWidget(self.tree, 1)
        layout.addWidget(self._summary)
        selection.changed.connect(self.rebuild)
        project.changed.connect(self.rebuild)
        self.rebuild()

    def _differing(self) -> list[str]:
        """
            Returns the parameters in which the runs differ.
        """
        fields = [field for field in self.project.fields() if field not in ('experiment', 'name')]
        return [field for field in fields if len({repr(self.project.value(run, field)) for run in self.project.runs}) > 1]

    # Building.

    def rebuild(self) -> None:
        """
            Lists the runs again, grouped as the selection says, keeping what is expanded.
        """
        self._building = True
        self.tree.setSortingEnabled(False)
        self.tree.clear()
        headers = [*self.FIXED, *self.columns]
        self.tree.setHeaderLabels(headers)
        header = self.tree.header()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        grouped = bool(self.selection.group_by)
        colors = {str(run.path): line.color for line in self.selection.lines() for run in line.runs}
        for label, runs in self.selection.groups():
            if grouped and label:
                parent = _Item([f'{label}  ({len(runs)})'])
                parent.setData(0, Qt.ItemDataRole.UserRole + 1, f'group:{label}')
                parent.setFlags(parent.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsAutoTristate)
                parent.setData(0, Qt.ItemDataRole.DecorationRole, self.selection.color(f'group:{label}'))
                parent.setToolTip(0, f'{label}: {len(runs)} runs')
                bold = parent.font(0)
                bold.setBold(True)
                parent.setFont(0, bold)
                self.tree.addTopLevelItem(parent)
                parent.setFirstColumnSpanned(True)
                for run in runs:
                    parent.addChild(self._item(run, colors))
                parent.setExpanded(f'group:{label}' in self._expanded)
            else:
                for run in runs:
                    self.tree.addTopLevelItem(self._item(run, colors))
        self.tree.setSortingEnabled(True)
        for column, name in enumerate(headers[:-1]):
            if name in self._widths:
                self.tree.setColumnWidth(column, self._widths[name])
            elif column:
                self.tree.resizeColumnToContents(column)
        self._fit()
        self._building = False
        self._filter(self.search.text())
        shown = len(self.selection.visible)
        self._summary.setText(f'{shown} of {len(self.project.runs)} runs shown' + (
            f', grouped by {", ".join(self.selection.group_by)}' if grouped else ''))

    def _item(self, run, colors: dict[str, QColor]) -> QTreeWidgetItem:
        path = str(run.path)
        try:
            status, steps = run.status, f'{run.step:,}'
        except READ_ERRORS:
            status, steps = 'cannot be read', ''
        experiment = self.project.experiment(run) or ''
        values = [self.project.value(run, field) for field in self.columns]
        item = _Item([short_name(run.path.name), experiment, steps + ('' if status == 'finished' else f' {status}'),
                      *('' if value is None else str(value) for value in values)])
        for column, value in enumerate(values, start=len(self.FIXED)):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                item.setData(column, Qt.ItemDataRole.UserRole, value)
        item.setData(0, Qt.ItemDataRole.UserRole, path)
        item.setData(0, Qt.ItemDataRole.UserRole + 1, f'run:{path}')
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(0, Qt.CheckState.Checked if path in self.selection.visible else Qt.CheckState.Unchecked)
        item.setData(0, Qt.ItemDataRole.DecorationRole, colors.get(path, self.selection.color(path)))
        item.setToolTip(0, f'{run.path.name}\n{run.path}\nDouble click to show it on the Run tab.')
        item.setToolTip(2, status)
        # Sorted by the number of steps, not by its text.
        item.setData(2, Qt.ItemDataRole.UserRole, run.step if steps else -1)
        return item

    # Interaction.

    def _on_item(self, item: QTreeWidgetItem, column: int) -> None:
        if self._building or column != 0:
            return
        # A group turns half shown when one of its runs is shown or hidden; that run says what changed.
        if item.checkState(0) == Qt.CheckState.PartiallyChecked:
            return
        paths, visible = self._paths(item), item.checkState(0) == Qt.CheckState.Checked
        QTimer.singleShot(0, lambda: self.selection.set_visible(paths, visible))

    def _fit(self) -> None:
        """
            Gives the first column, until it is resized by hand, the width the other columns leave, at
            least the first column width of ``THEME.layout``.
        """
        header = self.tree.header()
        if header.count() < 2 or self.tree.headerItem().text(0) in self._widths:
            return
        last = header.count() - 1
        building, self._building = self._building, True
        # The last column measured as wide as its contents, not as the width it fills.
        header.setStretchLastSection(False)
        self.tree.setColumnWidth(last, max(self.tree.sizeHintForColumn(last), header.sectionSizeHint(last)))
        others = sum(self.tree.columnWidth(column) for column in range(1, last + 1))
        self.tree.setColumnWidth(0, max(self.tree.viewport().width() - others, THEME.layout.first_column_width))
        header.setStretchLastSection(True)
        self._building = building

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Resize and watched is self.tree.viewport():
            self._fit()
        return super().eventFilter(watched, event)

    def _on_resized(self, column: int, old: int, new: int) -> None:
        # The last column fills the width left, which is not a width it was given.
        if not self._building and column < self.tree.header().count() - 1:
            self._widths[self.tree.headerItem().text(column)] = new

    def _paths(self, item: QTreeWidgetItem) -> list[str]:
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if path:
            return [path]
        return [item.child(i).data(0, Qt.ItemDataRole.UserRole) for i in range(item.childCount())]

    def _on_double_click(self, item: QTreeWidgetItem, column: int) -> None:
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if path:
            self.run_activated.emit(path)

    def _filter(self, text: str) -> None:
        needle = text.strip().lower()
        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            if item.childCount():
                shown = 0
                for child in (item.child(i) for i in range(item.childCount())):
                    match = self._matches(child, needle)
                    child.setHidden(not match)
                    shown += match
                item.setHidden(bool(needle) and not shown and needle not in item.text(0).lower())
            else:
                item.setHidden(not self._matches(item, needle))

    def _matches(self, item: QTreeWidgetItem, needle: str) -> bool:
        if not needle:
            return True
        run = self.project.run(item.data(0, Qt.ItemDataRole.UserRole))
        texts = [item.text(column) for column in range(item.columnCount())]
        if run is not None:
            texts += [run.path.name, *(f'{k}={v}' for k, v in run.hparams.items())]
        return any(needle in text.lower() for text in texts)

    def _show_all(self) -> None:
        self.selection.show_only(str(run.path) for run in self.project.runs)

    def _hide_all(self) -> None:
        self.selection.show_only([])

    def _show_newest(self) -> None:
        self.selection.show_newest()

    def _fill_group_menu(self) -> None:
        self._group_menu.clear()
        none = self._group_menu.addAction('No groups')
        none.setCheckable(True)
        none.setChecked(not self.selection.group_by)
        none.triggered.connect(lambda: self.selection.set_group_by([]))
        self._group_menu.addSeparator()
        for field in self.project.fields():
            action = self._group_menu.addAction(field)
            action.setCheckable(True)
            action.setChecked(field in self.selection.group_by)
            action.triggered.connect(lambda checked, f=field: self._toggle_group(f, checked))

    def _toggle_group(self, field: str, checked: bool) -> None:
        fields = [f for f in self.selection.group_by if f != field] + ([field] if checked else [])
        self.selection.set_group_by(fields)

    def _fill_columns_menu(self) -> None:
        self._columns_menu.clear()
        for field in self.project.fields():
            if field in ('experiment', 'name'):
                continue
            action = self._columns_menu.addAction(field)
            action.setCheckable(True)
            action.setChecked(field in self.columns)
            action.triggered.connect(lambda checked, f=field: self.set_columns(
                [c for c in self.columns if c != f] + ([f] if checked else [])))

    def set_columns(self, fields: tp.Sequence[str]) -> None:
        """
            Shows the parameters ``fields`` as columns.
        """
        self.columns = list(fields)
        self.rebuild()

    def state(self) -> dict[str, tp.Any]:
        """
            Returns the columns and their widths, the search, the groups expanded and the sorting
            of the table, as an `Exploration` keeps them.
        """
        header = self.tree.header()
        return {
            'columns': list(self.columns),
            'search': self.search.text(),
            'expanded': sorted(key for key in self._expanded if key and key.startswith('group:')),
            'widths': dict(self._widths),
            'sort': [header.sortIndicatorSection(), header.sortIndicatorOrder().value],
        }

    def restore(self, state: dict[str, tp.Any]) -> None:
        """
            Shows the table as ``state`` gives it, without the parameters the runs no longer have.
        """
        fields = set(self.project.fields())
        self.columns = [field for field in state.get('columns', self.columns) if field in fields]
        self._expanded = {key for key in state.get('expanded', ()) if isinstance(key, str)}
        self._widths = {name: int(width) for name, width in (state.get('widths') or {}).items() if isinstance(width, int) and width > 0}
        self.rebuild()
        column, order = (list(state.get('sort') or ()) + [None, None])[:2]
        if isinstance(column, int) and 0 <= column < self.tree.columnCount() and order in (0, 1):
            self.tree.sortByColumn(column, Qt.SortOrder(order))
        self.search.setText(str(state.get('search', '')))

    def _context_menu(self, at) -> None:
        item = self.tree.itemAt(at)
        if item is None:
            return
        key = item.data(0, Qt.ItemDataRole.UserRole + 1)
        path = item.data(0, Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        if path:
            show = menu.addAction('Show on the Run tab')
            show.triggered.connect(lambda: self.run_activated.emit(path))
            experiment = menu.addAction('Set experiment...')
            experiment.triggered.connect(lambda: self._set_experiment([p for p in self._selected_paths() or [path]]))
        color = menu.addAction('Color...')
        color.triggered.connect(lambda: self._choose_color(key[len('run:'):] if key.startswith('run:') else key))
        menu.exec(self.tree.viewport().mapToGlobal(at))

    def _selected_paths(self) -> list[str]:
        return [item.data(0, Qt.ItemDataRole.UserRole) for item in self.tree.selectedItems() if item.data(0, Qt.ItemDataRole.UserRole)]

    def _set_experiment(self, paths: list[str]) -> None:
        runs = [run for run in (self.project.run(path) for path in paths) if run is not None]
        if not runs:
            return
        current = self.project.experiment(runs[0]) or ''
        name, accepted = QInputDialog.getText(self, 'Set Experiment', f'Experiment of the {len(runs)} runs selected (empty for none):', text=current)
        if accepted:
            self.set_experiment(paths, name)

    def set_experiment(self, paths: tp.Iterable[str], name: str | None) -> None:
        """
            Sets the experiment of the runs at ``paths`` in the viewer, or none for an empty name.
        """
        for path in paths:
            run = self.project.run(path)
            if run is not None:
                self.project.set_experiment(run, name)

    def _choose_color(self, key: str) -> None:
        color = QColorDialog.getColor(self.selection.color(key), self, 'Color')
        if color.isValid():
            self.selection.set_color(key, color)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def section_of(key: str) -> str:
    """
        Returns the section of the workspace of the series ``key``: the prefix of its name, with the
        node of a summary, such as ``'summary · A_excitatory'`` for
        ``'summary/A_excitatory.soma:spikes/active_fraction'``, or ``'logged'`` without a prefix.
    """
    parts = key.split('/')
    if len(parts) >= 3:
        return f'{parts[0]} · {re.split(r"[.:]", parts[1])[0]}'
    return parts[0] if len(parts) == 2 else 'logged'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def in_section(key: str) -> str:
    """
        Returns the name of the series ``key`` within its section, without the prefix the section
        is named by, such as ``'soma:spikes/active_fraction'`` for
        ``'summary/A_excitatory.soma:spikes/active_fraction'`` and ``'steps'`` for
        ``'episode/steps'``.
    """
    parts = key.split('/')
    if len(parts) >= 3:
        node = re.split(r'[.:]', parts[1])[0]
        return '/'.join([parts[1][len(node) + 1:] or parts[1], *parts[2:]])
    return parts[1] if len(parts) == 2 else key

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Panel(QWidget):
    """
        A panel of the workspace: scalar series drawn for every line of a `Selection`.

        Drawn in the space the selection chooses, or else in the natural space of its first series.
        A group is a strong line, the mean of its runs, over a faint band from their lowest to their
        highest value, with its runs as faint lines when the selection draws them. With several
        series, every line of every series has a color of its own. A box drawn with the right button
        zooms the panel; a double click fits it again.

        Parameters
        ----------
        keys : sequence of str
            Names of the series.
        selection : Selection
            What is drawn.
        store : SeriesStore
            Where the series are read.
        pinned : bool, default False
            Whether the panel is in the Pinned section: it has a button removing it, else one
            pinning it.
        title : str, optional
            Title of the plot, the names of the series by default.
    """

    pin_requested = Signal(object)
    removed = Signal(object)

    def __init__(self, keys: tp.Sequence[str], selection: Selection, store: SeriesStore, pinned: bool = False,
                 title: str | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.keys = list(keys)
        self.selection = selection
        self.store = store
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        bar = QHBoxLayout()
        bar.setContentsMargins(*THEME.layout.panel_bar_margins)
        # The legend, above the plot rather than over it.
        self.legend = QLabel()
        self.legend.setObjectName('runViewerLegend')
        self.legend.setTextFormat(Qt.TextFormat.RichText)
        self.legend.setWordWrap(True)
        button = QToolButton()
        button.setText('×' if pinned else 'Pin')
        button.setObjectName('runViewerRemove' if pinned else 'runViewerPin')
        button.setToolTip('Remove the panel' if pinned else 'Pin the panel on top of the workspace')
        button.clicked.connect(self._removed if pinned else self._pin)
        bar.addWidget(self.legend, 1)
        bar.addWidget(button, 0, Qt.AlignmentFlag.AlignTop)
        self.plot = SeriesPlot(', '.join(self.keys) if title is None else title, height=THEME.panel_height)
        self.plot.legend = False
        self.plot.detail = '\n'.join(self.keys) + '\nDrag with the right button to zoom; double click to fit.'
        layout.addLayout(bar)
        layout.addWidget(self.plot)

    def _pin(self) -> None:
        self.pin_requested.emit(self)

    def zoom(self) -> dict[str, list[float] | None] | None:
        """
            Returns the ranges the panel is zoomed to, ``x`` and ``y``, or None when it is not.
        """
        x, y = self.plot._x_given, self.plot.zoomed
        if x is None and y is None:
            return None
        return {'x': None if x is None else [float(v) for v in x], 'y': None if y is None else [float(v) for v in y]}

    def set_zoom(self, x: tp.Sequence[float] | None, y: tp.Sequence[float] | None) -> None:
        """
            Zooms the panel to the range ``x`` of its axis and ``y`` of its values, each fitted for
            None.
        """
        self.plot.zoomed = None if y is None else (float(y[0]), float(y[1]))
        self.plot.set_view(None if x is None else (float(x[0]), float(x[1])))

    def _removed(self) -> None:
        self.removed.emit(self)

    def space(self) -> str:
        """
            Returns the space the panel is drawn in.
        """
        if self.selection.space != NATURAL:
            return self.selection.space
        for line in self.selection.lines():
            for run in line.runs:
                if len(self.store.rows(run, self.keys[0])[0]):
                    return self.store.spaces(run).natural(self.keys[0])
        return STEP

    def draw(self) -> None:
        """
            Draws the series of the panel for every line of the selection.
        """
        space = self.space()
        many = len(self.keys) > 1
        series, bands, faint = [], [], []
        for line in self.selection.lines():
            for key in self.keys:
                _, band, members = self.selection.draw(self.store, line, key, space)
                if not len(band.x):
                    continue
                line_color = THEME.series_color(len(series)) if many else line.color
                series.append((f'{line.label} · {key}' if many else line.label, band.x, band.mean, line_color))
                if line.group and len(members) > 1:
                    bands.append(('', band.x, band.low, band.high, THEME.with_alpha(line_color, THEME.band_alpha)))
                    if self.selection.members:
                        faint += [(x, y, THEME.with_alpha(line_color, THEME.faint_alpha)) for x, y in members]
        self.plot.set_labels(x=axis_label(space))
        self.plot.set_series(series, x_range=self.plot._x_given, bands=bands, faint=faint)
        self.legend.setText('&nbsp;&nbsp; '.join(
            f'<span style="color: {c.name()};">■</span>&nbsp;{html.escape(label)}' for label, _, _, c in series
        ))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Section(QWidget):
    """
        A section of the workspace: panels of series sharing a prefix, open or collapsed.

        Its panels are made, and their series read, when it is first open. The search of the
        workspace shows the panels whose series match, and hides a section without any.
    """

    pin_requested = Signal(object)
    removed = Signal(object)

    def __init__(self, name: str, keys: tp.Sequence[str], workspace: WorkspaceView, opened: bool, pinned: bool = False,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.name = name
        self.keys = list(keys)
        self.workspace = workspace
        self.pinned = pinned
        self.panels: dict[str, Panel] = {}
        self._needle = ''
        layout = QVBoxLayout(self)
        layout.setContentsMargins(*THEME.layout.section_margins)
        layout.setSpacing(THEME.layout.section_spacing)
        self.header = QToolButton()
        self.header.setObjectName('runViewerSectionHeader')
        self.header.setCheckable(True)
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.header.toggled.connect(self.set_open)
        self._body = QWidget()
        self._grid = QGridLayout(self._body)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(THEME.layout.panel_spacing)
        layout.addWidget(self.header)
        layout.addWidget(self._body)
        self.header.setChecked(opened)
        self.set_open(opened)

    @property
    def opened(self) -> bool:
        return self.header.isChecked()

    def _matching(self) -> list[str]:
        return [key for key in self.keys if not self._needle or self._needle in key.lower()]

    def set_open(self, opened: bool) -> None:
        """
            Opens the section, making its panels, or collapses it.
        """
        if self.header.isChecked() != opened:
            self.header.setChecked(opened)
            return
        matching = self._matching()
        self.header.setArrowType(Qt.ArrowType.DownArrow if opened else Qt.ArrowType.RightArrow)
        self.header.setText(f'{self.name}   ({len(matching)} of {len(self.keys)})' if self._needle else f'{self.name}   ({len(self.keys)})')
        self._body.setVisible(opened)
        if opened:
            self._make()
        self.arrange()

    def _make(self) -> None:
        for key in self._matching():
            if key not in self.panels:
                keys = key.split('\n') if self.pinned else [key]
                # A section names its panels without its own prefix; pinned panels, gathered from any, name theirs whole.
                panel = Panel(keys, self.workspace.selection, self.workspace.store, self.pinned, None if self.pinned else in_section(key))
                panel.pin_requested.connect(self.pin_requested)
                panel.removed.connect(self.removed)
                self.panels[key] = panel
                panel.draw()

    def arrange(self) -> None:
        """
            Places the panels matching the search, in rows of the columns of the workspace.
        """
        for panel in self.panels.values():
            self._grid.removeWidget(panel)
            panel.setVisible(False)
        columns = self.workspace.columns
        for index, key in enumerate(key for key in self._matching() if key in self.panels):
            panel = self.panels[key]
            self._grid.addWidget(panel, index // columns, index % columns)
            panel.setVisible(True)
        for column in range(3):
            self._grid.setColumnStretch(column, 1 if column < columns else 0)

    def set_filter(self, needle: str) -> None:
        """
            Shows the panels whose series hold ``needle``, and the section when it has any.
        """
        self._needle = needle.strip().lower()
        self.setVisible(bool(self._matching()))
        self.set_open(self.opened or bool(self._needle))

    def draw(self) -> None:
        for panel in self.panels.values():
            panel.draw()

    def remove(self, key: str) -> None:
        panel = self.panels.pop(key, None)
        if key in self.keys:
            self.keys.remove(key)
        if panel is not None:
            panel.setParent(None)
            panel.deleteLater()
        self.setVisible(bool(self._matching()))
        self.set_open(self.opened)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class KeyPicker(QDialog):
    """
        Dialog choosing scalar series by name, with the number of runs that have each.
    """

    def __init__(self, counts: dict[str, int], runs: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle('Add Panel')
        self.resize(THEME.layout.picker_width, THEME.layout.picker_height)
        layout = QVBoxLayout(self)
        self._filter = QLineEdit()
        self._filter.setPlaceholderText('Filter series...')
        self._filter.setClearButtonEnabled(True)
        self._list = QListWidget()
        for key, count in counts.items():
            item = QListWidgetItem(key)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            item.setToolTip(f'{count} of the {runs} runs')
            self._list.addItem(item)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        for text, slot in (('Add', self.accept), ('Cancel', self.reject)):
            button = QPushButton(text)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        layout.addWidget(_note('Series checked share one panel, drawn for every run or group shown.'))
        layout.addWidget(self._filter)
        layout.addWidget(self._list, 1)
        layout.addLayout(buttons)
        self._filter.textChanged.connect(self._apply_filter)

    def _apply_filter(self, text: str) -> None:
        needle = text.strip().lower()
        for index in range(self._list.count()):
            item = self._list.item(index)
            item.setHidden(bool(needle) and needle not in item.text().lower())

    def keys(self) -> list[str]:
        """
            Returns the names checked.
        """
        return [self._list.item(i).text() for i in range(self._list.count()) if self._list.item(i).checkState() == Qt.CheckState.Checked]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class WorkspaceView(QWidget):
    """
        The workspace: a panel for every scalar series of the runs of a project, in sections named by
        the prefix of the series, drawn for every run or group shown.

        The settings bar sets, for every panel, the x axis (the natural space of each series, steps,
        wall time, or an integer tag), the smoothing, whether the runs of groups are drawn faintly,
        and the panels per row. The search field shows the panels whose series match. Panels added by
        hand (Add panel), of one or more series, and panels pinned from a section are in the Pinned
        section, on top. Sections of more than `OPEN` panels start collapsed.

        Parameters
        ----------
        project : Project
            The runs.
        selection : Selection
            What is drawn.
        store : SeriesStore
            Where the series are read.
        parent : QWidget, optional
            Parent widget.
    """

    OPEN = 12
    """
        Largest number of panels of a section open from the start.
    """

    def __init__(self, project: Project, selection: Selection, store: SeriesStore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project = project
        self.selection = selection
        self.store = store
        self.columns = 2
        self.sections: dict[str, Section] = {}
        self._opened: dict[str, bool] = {}
        self._keys: list[str] = []
        self.setObjectName('runViewerPanel')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(*THEME.layout.workspace_margins)
        layout.setSpacing(THEME.layout.workspace_spacing)
        bar = QHBoxLayout()
        self._space = QComboBox()
        self._space.setToolTip('X axis of every panel: the space each series was recorded in, steps, wall time or a tag')
        self._space.currentIndexChanged.connect(self._on_space)
        self._smoothing = QSlider(Qt.Orientation.Horizontal)
        self._smoothing.setRange(0, 99)
        self._smoothing.setFixedWidth(THEME.layout.smoothing_width)
        self._smoothing.setToolTip('Smoothing of every series: an exponential moving average')
        self._smoothing_label = QLabel('0')
        self._smoothing_label.setFixedWidth(THEME.layout.smoothing_label_width)
        self._smoothing.valueChanged.connect(self._on_smoothing)
        self._members = QCheckBox('Runs of groups')
        self._members.setToolTip('Draw every run of a group as a faint line')
        self._members.toggled.connect(lambda checked: self.selection.set_settings(members=checked))
        self._per_row = QComboBox()
        self._per_row.addItems(['1', '2', '3'])
        self._per_row.setCurrentText(str(self.columns))
        self._per_row.setToolTip('Panels per row')
        self._per_row.currentTextChanged.connect(lambda text: self.set_columns(int(text)))
        self.search = QLineEdit()
        self.search.setPlaceholderText('Search panels...')
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.set_search)
        add = QPushButton('Add panel...')
        add.setToolTip('A panel of one or more series, pinned on top')
        add.clicked.connect(self._pick)
        for widget in (QLabel('X axis'), self._space, QLabel('Smoothing'), self._smoothing, self._smoothing_label, self._members,
                       QLabel('Per row'), self._per_row):
            bar.addWidget(widget)
        bar.addWidget(self.search, 1)
        bar.addWidget(add)
        page = QWidget()
        page.setObjectName('runViewerPanel')
        self._sections = QVBoxLayout(page)
        self._sections.setContentsMargins(0, 0, 0, 0)
        self._sections.setSpacing(THEME.layout.sections_spacing)
        self.pinned = Section('Pinned', [], self, True, pinned=True)
        self.pinned.removed.connect(self._unpin)
        self._sections.addWidget(self.pinned)
        self._sections.addStretch(1)
        area = QScrollArea()
        area.setObjectName('runViewerScroll')
        area.viewport().setObjectName('runViewerViewport')
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        area.setWidget(page)
        layout.addLayout(bar)
        layout.addWidget(area, 1)
        self._list_spaces()
        self.pinned.setVisible(False)
        selection.changed.connect(self._on_selection)
        project.changed.connect(self.rebuild)
        self.rebuild()

    # Contents.

    def keys(self) -> list[str]:
        """
            Returns the names of the scalar series of the runs of the project.
        """
        return sorted({key for run in self.project.runs for key in self.store.keys(run)})

    def rebuild(self) -> None:
        """
            Makes the sections again from the series of the runs, keeping which are open.
        """
        for name, section in self.sections.items():
            self._opened[name] = section.opened
            self._sections.removeWidget(section)
            section.setParent(None)
            section.deleteLater()
        by_section: dict[str, list[str]] = {}
        self._keys = self.keys()
        for key in self._keys:
            by_section.setdefault(section_of(key), []).append(key)
        self.sections = {}
        for index, (name, keys) in enumerate(by_section.items(), start=1):
            section = Section(name, keys, self, self._opened.get(name, len(keys) <= self.OPEN))
            section.pin_requested.connect(self.pin)
            self._sections.insertWidget(index, section)
            self.sections[name] = section
        self.set_search(self.search.text())

    def _list_spaces(self) -> None:
        tags = sorted({tag for run in self.project.runs for tag in self.store.spaces(run).tags()})
        choices = [(NATURAL, 'Natural'), (STEP, 'Steps'), (WALL, 'Wall time'), *((tag, f'per {tag}') for tag in tags)]
        if [self._space.itemData(i) for i in range(self._space.count())] == [space for space, _ in choices]:
            return
        self._space.blockSignals(True)
        self._space.clear()
        for space, text in choices:
            self._space.addItem(text, space)
        self._space.setCurrentIndex(max(self._space.findData(self.selection.space), 0))
        self._space.blockSignals(False)

    # Settings.

    def _on_space(self, index: int) -> None:
        self.selection.set_settings(space=self._space.itemData(index))

    def _on_smoothing(self, value: int) -> None:
        self._smoothing_label.setText(f'{value / 100:.2f}'.rstrip('0').rstrip('.') or '0')
        self.selection.set_settings(smoothing=value / 100)

    def set_columns(self, columns: int) -> None:
        """
            Places ``columns`` panels per row.
        """
        self.columns = min(max(int(columns), 1), 3)
        self._per_row.setCurrentText(str(self.columns))
        for section in (self.pinned, *self.sections.values()):
            section.arrange()

    def set_search(self, text: str) -> None:
        """
            Shows the panels whose series hold ``text``.
        """
        for section in self.sections.values():
            section.set_filter(text)

    def _on_selection(self) -> None:
        self._list_spaces()
        self._show_settings()
        self.redraw()

    def _show_settings(self) -> None:
        """
            Shows the settings of the selection in the bar, as they may be set elsewhere.
        """
        for widget, value in ((self._space, max(self._space.findData(self.selection.space), 0)),
                              (self._smoothing, round(self.selection.smoothing * 100)), (self._members, self.selection.members)):
            widget.blockSignals(True)
            if isinstance(widget, QComboBox):
                widget.setCurrentIndex(value)
            elif isinstance(widget, QSlider):
                widget.setValue(value)
            else:
                widget.setChecked(value)
            widget.blockSignals(False)
        self._smoothing_label.setText(f'{self.selection.smoothing:.2f}'.rstrip('0').rstrip('.') or '0')

    def refresh(self, redraw: bool = True) -> None:
        """
            Makes the sections again when the runs hold series not listed yet, else draws every
            panel again with ``redraw``.
        """
        if self.keys() != self._keys:
            self.rebuild()
        elif redraw:
            self.redraw()

    def redraw(self) -> None:
        """
            Draws every panel made again, as after rows are written.
        """
        for section in (self.pinned, *self.sections.values()):
            section.draw()

    # Exploration.

    def _panels(self) -> tp.Iterator[tuple[str, Panel]]:
        # Pinned panels are told apart from those of the sections, which may draw the same series.
        for section in (self.pinned, *self.sections.values()):
            for key, panel in section.panels.items():
                yield ('pinned:' if section is self.pinned else '') + key, panel

    def state(self) -> dict[str, tp.Any]:
        """
            Returns the panels per row, the search, the sections open, the pinned panels and the
            zooms of the workspace, as an `Exploration` keeps them.
        """
        return {
            'columns': self.columns,
            'search': self.search.text(),
            'opened': {**self._opened, **{name: section.opened for name, section in self.sections.items()}},
            'pinned': [key.split('\n') for key in self.pinned.keys],
            'zooms': {key: zoom for key, panel in self._panels() if (zoom := panel.zoom()) is not None},
        }

    def restore(self, state: dict[str, tp.Any]) -> None:
        """
            Shows the workspace as ``state`` gives it.
        """
        self.set_columns(int(state.get('columns', self.columns)))
        opened = {name: bool(value) for name, value in (state.get('opened') or {}).items()}
        self._opened.update(opened)
        for name, section in self.sections.items():
            if name in opened:
                section.set_open(opened[name])
        for keys in state.get('pinned', ()):
            if keys and all(isinstance(key, str) for key in keys):
                self.add_panel(keys)
        self.search.setText(str(state.get('search', '')))
        zooms = state.get('zooms') or {}
        for key, panel in self._panels():
            zoom = zooms.get(key)
            if isinstance(zoom, dict):
                panel.set_zoom(zoom.get('x'), zoom.get('y'))

    # Pinned panels.

    def add_panel(self, keys: tp.Sequence[str]) -> Panel:
        """
            Adds a panel of the series ``keys`` to the Pinned section.
        """
        name = '\n'.join(keys)
        if name not in self.pinned.keys:
            self.pinned.keys.append(name)
        self.pinned.setVisible(True)
        self.pinned.set_open(True)
        return self.pinned.panels[name]

    def pin(self, panel: Panel) -> None:
        """
            Pins a copy of ``panel`` on top.
        """
        self.add_panel(panel.keys)

    def _unpin(self, panel: Panel) -> None:
        self.pinned.remove('\n'.join(panel.keys))
        self.pinned.setVisible(bool(self.pinned.keys))

    def _pick(self, checked: bool = False) -> None:
        counts: dict[str, int] = {}
        for run in self.project.runs:
            for key in self.store.keys(run):
                counts[key] = counts.get(key, 0) + 1
        picker = KeyPicker(dict(sorted(counts.items())), len(self.project.runs), self)
        if picker.exec() and picker.keys():
            self.add_panel(picker.keys())

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
