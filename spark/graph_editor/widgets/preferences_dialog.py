#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import json
import os
import typing as tp
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, 
    QFileDialog, QMessageBox, QTabWidget, QWidget, QScrollArea, QFormLayout,
    QSpinBox, QDoubleSpinBox, QCheckBox, QColorDialog, QApplication
)
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QColor, QPalette
from spark.graph_editor.styles.manager import STYLES

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class ColorButton(QPushButton):

    def __init__(self, color, is_hex: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName('prefsColorBtn')
        self.is_hex = is_hex
        self._color = color
        self.clicked.connect(self._choose_color)
        self._update_style()
        self.setFixedWidth(STYLES.get_val('preferences', 'color_btn_width'))

    def _update_style(self) -> None:
        self.setStyleSheet(f'background-color: {self.get_css_color()};')

    def get_css_color(self) -> str:
        if self.is_hex:
            # Check if it's #AARRGGBB from QColor.name(HexArgb) and convert to rgba
            if len(self._color) == 9:
                c = QColor(self._color)
                return f'rgba({c.red()}, {c.green()}, {c.blue()}, {c.alpha()})'
            return self._color
        else:
            c = self._color
            if len(c) == 3:
                return f'rgba({c[0]}, {c[1]}, {c[2]}, 255)'
            else:
                return f'rgba({c[0]}, {c[1]}, {c[2]}, {c[3]})'
                
    def _choose_color(self) -> None:
        if self.is_hex:
            initial = QColor(self._color)
        else:
            c = self._color
            if len(c) == 3:
                initial = QColor(c[0], c[1], c[2])
            else:
                initial = QColor(c[0], c[1], c[2], c[3])
        color = QColorDialog.getColor(initial, self, 'Select Color', QColorDialog.ColorDialogOption.ShowAlphaChannel | QColorDialog.ColorDialogOption.DontUseNativeDialog)
        if color.isValid():
            if self.is_hex:
                if color.alpha() < 255:
                    self._color = color.name(QColor.NameFormat.HexArgb)
                else:
                    self._color = color.name()
            else:
                if len(self._color) == 3:
                    self._color = [color.red(), color.green(), color.blue()]
                else:
                    self._color = [color.red(), color.green(), color.blue(), color.alpha()]
            self._update_style()
            
    def get_value(self) -> str | list[int]:
        return self._color

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class PreferencesDialog(QDialog):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle('Preferences & Styling')
        self.resize(
            STYLES.get_val('preferences', 'dialog_width'),
            STYLES.get_val('preferences', 'dialog_height'),
        )
        self.settings = QSettings()
        self.config_data: dict[str, tp.Any] = json.loads(json.dumps(STYLES._config))
        self.widgets_map: dict[tuple[str], QWidget] = {} # tuple(path) -> widget
        layout = QVBoxLayout(self)
        # Tabs for categories
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        for category, items in self.config_data.items():
            if isinstance(items, dict):
                tab = self._create_category_tab(category, items)
                self.tabs.addTab(tab, category.title())
        # Bottom controls
        btn_layout = QHBoxLayout()
        self.path_label = QLabel()
        self.path_label.setObjectName('prefsPathLabel')
        self._update_path_label()
        load_btn = QPushButton('Load from...')
        load_btn.clicked.connect(self._load_custom)
        export_btn = QPushButton('Export as...')
        export_btn.clicked.connect(self._export_custom)
        reset_btn = QPushButton('Reset to Default')
        reset_btn.clicked.connect(self._reset_default)
        save_btn = QPushButton('Save & Apply')
        save_btn.setObjectName('prefsSaveBtn')
        save_btn.clicked.connect(self._save_and_apply)
        save_btn.setDefault(True)
        close_btn = QPushButton('Cancel')
        close_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.path_label)
        btn_layout.addStretch()
        btn_layout.addWidget(load_btn)
        btn_layout.addWidget(export_btn)
        btn_layout.addWidget(reset_btn)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def _update_path_label(self) -> None:
        active = STYLES._active_path
        if active == STYLES._default_path:
            self.path_label.setText('Active: Built-in Defaults')
        else:
            filename = os.path.basename(active)
            self.path_label.setText(f'Active: {filename}')
            self.path_label.setToolTip(active)

    def _create_category_tab(self, category: str, items: dict[str, tp.Any]) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        container = QWidget()
        form = QFormLayout(container)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setSpacing(STYLES.get_val('preferences', 'form_spacing'))
        self._build_form(form, items, [category])
        scroll.setWidget(container)
        return scroll
        
    def _build_form(self, form: QFormLayout, items: dict[str, tp.Any], current_path: list[str]) -> None:
        for k, v in items.items():
            path = tuple(current_path + [k])
            if isinstance(v, dict):
                lbl = QLabel(f'<br><b>{k.replace('_', ' ').title()}</b>')
                form.addRow(lbl)
                self._build_form(form, v, current_path + [k])
                continue
            widget = None
            if isinstance(v, bool):
                widget = QCheckBox()
                widget.setChecked(v)
            elif isinstance(v, int):
                widget = QSpinBox()
                widget.setRange(-999999, 999999)
                widget.setValue(v)
            elif isinstance(v, float):
                widget = QDoubleSpinBox()
                widget.setRange(-999999.0, 999999.0)
                widget.setDecimals(4)
                widget.setValue(v)
            elif isinstance(v, str):
                if v.startswith('#') and len(v) in (4, 7, 9):
                    widget = ColorButton(v, is_hex=True)
                else:
                    widget = QLineEdit(v)
            elif isinstance(v, list) and len(v) in (3, 4) and all(isinstance(x, (int, float)) for x in v):
                # Ensure they are ints
                v_ints = [int(x) for x in v]
                widget = ColorButton(v_ints, is_hex=False)
            else:
                widget = QLineEdit(str(v) if v is not None else '')
            if widget:
                label_text = k.replace('_', ' ').title()
                form.addRow(label_text, widget)
                self.widgets_map[path] = widget

    def _get_form_values(self) -> tp.Any:
        new_data = json.loads(json.dumps(self.config_data)) # deepcopy
        for path, widget in self.widgets_map.items():
            val = None
            if isinstance(widget, QCheckBox): val = widget.isChecked()
            elif isinstance(widget, QSpinBox): val = widget.value()
            elif isinstance(widget, QDoubleSpinBox): val = widget.value()
            elif isinstance(widget, ColorButton): val = widget.get_value()
            elif isinstance(widget, QLineEdit): val = widget.text()
            if val is not None:
                d = new_data
                for p in path[:-1]:
                    d = d[p]
                d[path[-1]] = val
        return new_data

    def _load_custom(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, 'Load Style JSON', '', 'JSON Files (*.json)')
        if file_path:
            self.settings.setValue('style_config_path', file_path)
            self._apply_styles()
            QMessageBox.information(self, 'Loaded', 'Styles loaded. Please close and reopen Preferences to see the updated values.')
            self.accept()

    def _export_custom(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(self, 'Export Styles As', 'custom_style.json', 'JSON Files (*.json)')
        if file_path:
            try:
                data = self._get_form_values()
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=4)
                self.settings.setValue('style_config_path', file_path)
                self._apply_styles()
                self._update_path_label()
                QMessageBox.information(self, 'Exported', f'Successfully exported and set as active style.')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to export: {e}')

    def _reset_default(self) -> None:
        self.settings.remove('style_config_path')
        self._apply_styles()
        QMessageBox.information(self, 'Reset', 'Reverted to default built-in styles. Please close and reopen Preferences to see the updated values.')
        self.accept()

    def _save_and_apply(self) -> None:
        data = self._get_form_values()
        active = STYLES._active_path
        if active == STYLES._default_path:
            QMessageBox.information(self, 'Save Custom Style', 'You are modifying default styles.\nPlease choose a location to save your custom style configuration.')
            file_path, _ = QFileDialog.getSaveFileName(self, 'Save Custom Style', 'custom_style.json', 'JSON Files (*.json)')
            if not file_path:
                return # Cancelled
            active = file_path
            self.settings.setValue('style_config_path', active)
        try:
            with open(active, 'w') as f:
                json.dump(data, f, indent=4)
            self._apply_styles()
            QMessageBox.information(self, 'Saved', 'Styles successfully saved and applied.')
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to save styles: {e}')

    def _apply_styles(self) -> None:
        STYLES.reload()
        app = QApplication.instance()
        if app:
            app.setStyleSheet(STYLES.stylesheet())

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################