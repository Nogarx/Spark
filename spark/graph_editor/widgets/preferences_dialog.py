#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import os
import re
import json
import typing as tp
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QMessageBox, QWidget, QScrollArea, QFormLayout, QListWidget,
    QListWidgetItem, QStackedWidget, QSpinBox, QDoubleSpinBox, QCheckBox,
    QColorDialog, QComboBox, QApplication, QSizePolicy
)
from PySide6.QtCore import Qt, QSettings, Signal, QSize
from PySide6.QtGui import QColor
from spark.graph_editor.styles.manager import STYLES

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# NOTE: The dialog is generated from the style configuration, so a new token shows up without touching this
# file. What is declared here is only what the configuration cannot say about itself: how the categories are
# grouped and named for a human ("pipe" is an edge), and what a value means when its type is ambiguous (a
# list of four numbers is a set of margins, not a colour).

SECTIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ('Canvas', 'Background, grid and the area the graph lives in.', ('graph', 'viewer')),
    ('Nodes', 'Body, header and section headers of a node.', ('node',)),
    ('Ports & Payloads', 'One colour and shape per payload type. Edges inherit them.', ('port',)),
    ('Edges', 'Connections between nodes, and the one being dragged.', ('pipe', 'temp_pipe')),
    ('Inspector', 'The configuration panel on the right.', ('inspector',)),
    ('Hierarchy', 'The node list on the left.', ('hierarchy',)),
    ('Console', 'The message log at the bottom.', ('console',)),
    ('Start Screen', 'The controller picker shown when no model is open.', ('start',)),
    ('Window', 'Menus, status bar and scroll bars.', ('main', 'menu_bar', 'menu', 'status_bar', 'scrollbar')),
    ('Preferences', 'This dialog.', ('preferences',)),
)

# Words a user is likely to type for a category the configuration names differently.
SYNONYMS: dict[str, str] = {
    'pipe': 'edge edges connection connections link wire',
    'temp_pipe': 'edge connection dragging',
    'port': 'payload payloads type types socket',
    'graph': 'canvas scene grid snapping layout',
    'viewer': 'canvas background',
    'node': 'module block',
    'start': 'welcome home controller picker',
}

# What a setting does, when the name alone does not say it.
HINTS: dict[str, str] = {
    'graph.layout_h_gap': 'Horizontal space between two columns when a model is laid out automatically.',
    'graph.layout_v_gap': 'Vertical space between two nodes of the same column.',
    'graph.layout_import_margin': 'Distance kept below the existing nodes when a model is imported.',
    'graph.snapping.node_grid': 'Grid a node snaps to while dragged.',
    'graph.snapping.pipe_grid': 'Grid an edge segment snaps to while dragged.',
    'pipe.brightness': 'Lift applied to the payload colour when drawing an edge. 100 keeps the port colour.',
    'pipe.active_boost': 'Extra brightness of a hovered or selected edge.',
    'pipe.node_margin': 'Clearance an automatically routed edge keeps from any node.',
    'pipe.lane_spacing': 'Distance between two edges sharing a detour lane.',
    'pipe.jump_radius': 'Size of the hop drawn where two edges cross.',
    'pipe.color': 'Fallback only. Edges take the colour of the payload they carry.',
    'port.type_styles': 'Colour and shape of each payload type. Edges inherit the colour of their ports.',
    'inspector.input_min_width': 'Smallest width a field may shrink to before the panel scrolls.',
    'inspector.combo_min_chars': 'Characters a drop down reserves, so a long entry cannot widen the panel.',
    'inspector.icon_idle_opacity': 'Opacity of an available but inactive cascade link.',
    'inspector.dims_min_width': 'Smallest width of one dimension in a shape field.',
    'console.min_height': 'Smallest height of the console panel.',
    'inspector.min_width': 'Smallest width of the inspector panel.',
    'hierarchy.min_width': 'Smallest width of the hierarchy panel.',
}

# Keys whose value is a set of edge distances rather than anything else.
_MARGIN_PATTERN = re.compile(r'(margins|rect|scene_rect)$')
_CSS_SIZE_PATTERN = re.compile(r'^(-?\d+)px$')
_SHAPE_OPTIONS = ('circle', 'polygon', 'star')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _is_colour(path: tuple[str, ...], value: tp.Any) -> bool:
    """
        True if a value is meant to be picked with a colour dialog.
    """
    if isinstance(value, str):
        return bool(re.fullmatch(r'#[0-9a-fA-F]{3,8}', value))
    # A list is only a colour when it has colour components and the key says so.
    if isinstance(value, (list, tuple)) and len(value) in (3, 4):
        if not all(isinstance(x, int) and 0 <= x <= 255 for x in value):
            return False
        return bool(re.search(r'(color|colour|bg|fg)$', path[-1]))
    return False

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class ColorButton(QPushButton):
    """
        Swatch that opens a colour picker.
    """

    def __init__(self, color: tp.Any, is_hex: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName('prefsColorBtn')
        self.is_hex = is_hex
        self._color = color
        self.clicked.connect(self._choose_color)
        self.setFixedWidth(STYLES.get_val('preferences', 'color_btn_width', default=80))
        self._update_style()

    def _qcolor(self) -> QColor:
        if self.is_hex:
            return QColor(self._color)
        values = list(self._color) + [255]
        return QColor(values[0], values[1], values[2], values[3])

    def _update_style(self) -> None:
        color = self._qcolor()
        self.setStyleSheet(f'background-color: rgba({color.red()}, {color.green()}, {color.blue()}, {color.alpha()});')
        self.setText(self._color if self.is_hex else '')
        self.setToolTip(color.name(QColor.NameFormat.HexArgb))

    def _choose_color(self) -> None:
        chosen = QColorDialog.getColor(
            self._qcolor(), self, 'Select Color',
            QColorDialog.ColorDialogOption.ShowAlphaChannel | QColorDialog.ColorDialogOption.DontUseNativeDialog,
        )
        if not chosen.isValid():
            return
        if self.is_hex:
            self._color = chosen.name(QColor.NameFormat.HexArgb) if chosen.alpha() < 255 else chosen.name()
        else:
            self._color = [chosen.red(), chosen.green(), chosen.blue()] if len(self._color) == 3 else \
                          [chosen.red(), chosen.green(), chosen.blue(), chosen.alpha()]
        self._update_style()

    def get_value(self) -> tp.Any:
        return self._color

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class NumberListEdit(QWidget):
    """
        Editor for a fixed list of numbers, such as margins or a scene rectangle.

        NOTE: These used to be handed to the colour picker, because a list of four numbers looks like an RGBA
        tuple. Margins were therefore impossible to edit and were shown as a meaningless swatch.
    """

    LABELS = {4: ('left', 'top', 'right', 'bottom'), 3: ('x', 'y', 'z'), 2: ('x', 'y')}

    def __init__(self, values: list, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._spins: list[QSpinBox | QDoubleSpinBox] = []
        labels = self.LABELS.get(len(values), ())
        for index, value in enumerate(values):
            spin = QDoubleSpinBox() if isinstance(value, float) else QSpinBox()
            if isinstance(spin, QDoubleSpinBox):
                spin.setDecimals(3)
            spin.setRange(-999999, 999999)
            spin.setValue(value)
            spin.setMinimumWidth(58)
            spin.wheelEvent = lambda event: event.ignore()
            if index < len(labels):
                spin.setToolTip(labels[index])
            layout.addWidget(spin)
            self._spins.append(spin)
        layout.addStretch(1)

    def get_value(self) -> list:
        return [spin.value() if isinstance(spin, QDoubleSpinBox) else int(spin.value()) for spin in self._spins]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class CssSizeEdit(QSpinBox):
    """
        Editor for the "12px" strings used by the stylesheet.
    """

    def __init__(self, value: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setRange(-9999, 9999)
        self.setSuffix('px')
        self.setValue(int(_CSS_SIZE_PATTERN.fullmatch(value).group(1)))
        self.wheelEvent = lambda event: event.ignore()

    def get_value(self) -> str:
        return f'{self.value()}px'

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class PreferencesDialog(QDialog):
    """
        Editor for the presentation of the graph editor.
    """

    applied = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle('Preferences')
        self.resize(
            STYLES.get_val('preferences', 'dialog_width', default=820),
            STYLES.get_val('preferences', 'dialog_height', default=600),
        )
        self.settings = QSettings()
        self.config_data: dict[str, tp.Any] = json.loads(json.dumps(STYLES._config))
        self.widgets_map: dict[tuple[str, ...], QWidget] = {}
        # (page index, form, row, searchable text)
        self._rows: list[tuple[int, QFormLayout, int, str]] = []

        layout = QVBoxLayout(self)
        layout.setSpacing(STYLES.get_val('preferences', 'layout_spacing', default=8))

        self.search = QLineEdit()
        self.search.setObjectName('prefsSearch')
        self.search.setPlaceholderText('Search settings...')
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._apply_filter)
        layout.addWidget(self.search)

        body = QHBoxLayout()
        body.setSpacing(STYLES.get_val('preferences', 'layout_spacing', default=8))
        self.sidebar = QListWidget()
        self.sidebar.setObjectName('prefsSidebar')
        self.sidebar.setFixedWidth(STYLES.get_val('preferences', 'sidebar_width', default=170))
        # NOTE: Rounded highlights that touch read as bleeding into the neighbouring entry, and a row must
        # never be shorter than the text plus the padding the stylesheet asks for.
        self.sidebar.setSpacing(STYLES.get_val('preferences', 'sidebar_spacing', default=3))
        self.pages = QStackedWidget()
        self.sidebar.currentRowChanged.connect(self.pages.setCurrentIndex)
        body.addWidget(self.sidebar)
        body.addWidget(self.pages, 1)
        layout.addLayout(body, 1)

        self._build_pages()

        controls = QHBoxLayout()
        self.path_label = QLabel()
        self.path_label.setObjectName('prefsPathLabel')
        self._update_path_label()
        controls.addWidget(self.path_label)
        controls.addStretch(1)
        for text, slot, name in (
            ('Load from...', self._load_custom, None),
            ('Export as...', self._export_custom, None),
            ('Reset to Default', self._reset_default, None),
            ('Apply', self._apply_now, None),
            ('Save & Close', self._save_and_close, 'prefsSaveBtn'),
            ('Cancel', self.reject, None),
        ):
            button = QPushButton(text)
            if name:
                button.setObjectName(name)
                button.setDefault(True)
            button.clicked.connect(slot)
            controls.addWidget(button)
        layout.addLayout(controls)

    #-------------------------------------------------------------------------------------------------------#
    # Construction
    #-------------------------------------------------------------------------------------------------------#

    def _build_pages(self) -> None:
        declared = {key for _, _, keys in SECTIONS for key in keys}
        # Anything not placed by hand still gets a home, so a new category is never invisible.
        leftovers = tuple(key for key in self.config_data if key not in declared)
        sections = list(SECTIONS) + ([('Other', 'Categories with no section of their own.', leftovers)] if leftovers else [])

        for title, description, keys in sections:
            present = [key for key in keys if isinstance(self.config_data.get(key), dict)]
            if not present:
                continue
            index = self.pages.count()
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(0, 0, 0, 0)
            page_layout.setSpacing(4)
            caption = QLabel(description)
            caption.setObjectName('prefsSectionCaption')
            caption.setWordWrap(True)
            page_layout.addWidget(caption)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            container = QWidget()
            form = QFormLayout(container)
            form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
            form.setSpacing(STYLES.get_val('preferences', 'form_spacing', default=10))
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            for key in present:
                # The category name is only worth showing when a section holds several of them.
                if len(present) > 1:
                    header = QLabel(key.replace('_', ' ').title())
                    header.setObjectName('prefsGroupLabel')
                    form.addRow(header)
                    self._rows.append((index, form, form.rowCount() - 1, ''))
                self._build_form(index, form, self.config_data[key], [key], title)
            scroll.setWidget(container)
            page_layout.addWidget(scroll, 1)
            self.pages.addWidget(page)
            item = QListWidgetItem(title, self.sidebar)
            padding = STYLES.get_val('preferences', 'sidebar_item_padding', default=14)
            item.setSizeHint(QSize(0, self.sidebar.fontMetrics().height() + padding))
        self.sidebar.setCurrentRow(0)

    def _build_form(self, page: int, form: QFormLayout, items: dict[str, tp.Any], current_path: list[str], section: str) -> None:
        for key, value in items.items():
            path = tuple(current_path + [key])
            dotted = '.'.join(path)
            if isinstance(value, dict):
                header = QLabel(key.replace('_', ' ').title())
                header.setObjectName('prefsGroupLabel')
                hint = HINTS.get(dotted, '')
                if hint:
                    header.setToolTip(hint)
                form.addRow(header)
                self._rows.append((page, form, form.rowCount() - 1, ''))
                self._build_form(page, form, value, current_path + [key], section)
                continue
            widget = self._build_widget(path, value)
            label_text = key.replace('_', ' ').title()
            label = QLabel(label_text)
            hint = HINTS.get(dotted, '')
            if hint:
                label.setToolTip(hint)
                widget.setToolTip(hint)
            form.addRow(label, widget)
            # NOTE: The section name and its synonyms are searchable too, so looking for "edge" finds the
            # settings the configuration calls "pipe".
            haystack = ' '.join((label_text, dotted, section, SYNONYMS.get(path[0], ''), hint)).lower()
            self._rows.append((page, form, form.rowCount() - 1, haystack))
            self.widgets_map[path] = widget

    def _build_widget(self, path: tuple[str, ...], value: tp.Any) -> QWidget:
        """
            Chooses the editor a value deserves.
        """
        if _is_colour(path, value):
            return ColorButton(value, is_hex=isinstance(value, str))
        if isinstance(value, bool):
            widget = QCheckBox()
            widget.setChecked(value)
            return widget
        if isinstance(value, (list, tuple)) and value and all(isinstance(x, (int, float)) for x in value):
            return NumberListEdit(list(value))
        if isinstance(value, str) and _CSS_SIZE_PATTERN.fullmatch(value):
            return CssSizeEdit(value)
        if path[-1] == 'shape':
            widget = QComboBox()
            widget.addItems(_SHAPE_OPTIONS)
            if value in _SHAPE_OPTIONS:
                widget.setCurrentText(value)
            widget.wheelEvent = lambda event: event.ignore()
            return widget
        if isinstance(value, bool):
            widget = QCheckBox()
            widget.setChecked(value)
            return widget
        if isinstance(value, int):
            widget = QSpinBox()
            widget.setRange(-999999, 999999)
            widget.setValue(value)
            widget.wheelEvent = lambda event: event.ignore()
            return widget
        if isinstance(value, float):
            widget = QDoubleSpinBox()
            widget.setRange(-999999.0, 999999.0)
            widget.setDecimals(4)
            widget.setValue(value)
            widget.wheelEvent = lambda event: event.ignore()
            return widget
        return QLineEdit('' if value is None else str(value))

    #-------------------------------------------------------------------------------------------------------#
    # Search
    #-------------------------------------------------------------------------------------------------------#

    def _apply_filter(self, text: str) -> None:
        """
            Hides every row that does not match, and dims the sections left empty.
        """
        needle = text.strip().lower()
        matches_per_page: dict[int, int] = {}
        for page, form, row, haystack in self._rows:
            # Group headers carry no searchable text: they follow their section.
            visible = (not needle) or (bool(haystack) and needle in haystack)
            form.setRowVisible(row, visible)
            if haystack and visible:
                matches_per_page[page] = matches_per_page.get(page, 0) + 1
        for index in range(self.sidebar.count()):
            item = self.sidebar.item(index)
            has_match = (not needle) or matches_per_page.get(index, 0) > 0
            item.setHidden(bool(needle) and not has_match)
        if needle:
            for index in range(self.sidebar.count()):
                if not self.sidebar.item(index).isHidden():
                    self.sidebar.setCurrentRow(index)
                    break

    #-------------------------------------------------------------------------------------------------------#
    # Values and files
    #-------------------------------------------------------------------------------------------------------#

    def _update_path_label(self) -> None:
        active = STYLES._active_path
        if active == STYLES._default_path:
            self.path_label.setText('Active: built-in defaults')
            self.path_label.setToolTip(str(active))
        else:
            self.path_label.setText(f'Active: {os.path.basename(active)}')
            self.path_label.setToolTip(str(active))

    def _get_form_values(self) -> dict[str, tp.Any]:
        data = json.loads(json.dumps(self.config_data))
        for path, widget in self.widgets_map.items():
            if hasattr(widget, 'get_value'):
                value = widget.get_value()
            elif isinstance(widget, QCheckBox):
                value = widget.isChecked()
            elif isinstance(widget, QComboBox):
                value = widget.currentText()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                value = widget.value()
            elif isinstance(widget, QLineEdit):
                value = widget.text()
            else:
                continue
            node = data
            for part in path[:-1]:
                node = node[part]
            node[path[-1]] = value
        return data

    def _write(self, path: str) -> bool:
        try:
            with open(path, 'w') as handle:
                json.dump(self._get_form_values(), handle, indent=4)
            return True
        except Exception as error:
            QMessageBox.critical(self, 'Error', f'Failed to save styles: {error}')
            return False

    def _target_path(self) -> str | None:
        """
            File the changes are written to, asking for one the first time.
        """
        active = STYLES._active_path
        if active != STYLES._default_path:
            return str(active)
        QMessageBox.information(
            self, 'Save Custom Style',
            'The built-in defaults are read only.\nChoose where to keep your customised style.',
        )
        file_path, _ = QFileDialog.getSaveFileName(self, 'Save Custom Style', 'custom_style.json', 'JSON Files (*.json)')
        if not file_path:
            return None
        self.settings.setValue('style_config_path', file_path)
        return file_path

    def _apply_now(self) -> None:
        path = self._target_path()
        if path and self._write(path):
            self._apply_styles()
            self._update_path_label()

    def _save_and_close(self) -> None:
        path = self._target_path()
        if path and self._write(path):
            self._apply_styles()
            self.accept()

    def _load_custom(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, 'Load Style JSON', '', 'JSON Files (*.json)')
        if not file_path:
            return
        self.settings.setValue('style_config_path', file_path)
        self._apply_styles()
        self._reload_values()

    def _export_custom(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(self, 'Export Styles As', 'custom_style.json', 'JSON Files (*.json)')
        if not file_path:
            return
        if self._write(file_path):
            self.settings.setValue('style_config_path', file_path)
            self._apply_styles()
            self._update_path_label()

    def _reset_default(self) -> None:
        self.settings.remove('style_config_path')
        self._apply_styles()
        self._reload_values()

    def _reload_values(self) -> None:
        """
            Rebuilds the whole dialog from the style that is now active.

            NOTE: Loading or resetting used to ask the user to close and reopen the dialog to see the values.
        """
        self.config_data = json.loads(json.dumps(STYLES._config))
        self.widgets_map.clear()
        self._rows.clear()
        while self.pages.count():
            widget = self.pages.widget(0)
            self.pages.removeWidget(widget)
            widget.deleteLater()
        self.sidebar.clear()
        self._build_pages()
        self._update_path_label()
        self._apply_filter(self.search.text())

    def _apply_styles(self) -> None:
        STYLES.reload()
        app = QApplication.instance()
        if app:
            app.setStyleSheet(STYLES.stylesheet())
        self.applied.emit()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
