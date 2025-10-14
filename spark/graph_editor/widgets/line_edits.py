#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

from PySide6 import QtWidgets, QtGui, QtCore
from spark.graph_editor.widgets.base import SparkQWidget
from spark.graph_editor.editor_config import GRAPH_EDITOR_CONFIG

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class QStrLineEdit(QtWidgets.QLineEdit):
    """
        A simple QLineEdit with a placeholder.
    """
    def __init__(
            self,
            initial_value: str,
            placeholder: str = '',
            parent: QtWidgets.QWidget | None = None,
            **kwargs,
        ) -> None:
        super().__init__(str(initial_value), parent)
        self.setPlaceholderText(placeholder)
        self.setTextMargins(GRAPH_EDITOR_CONFIG.input_field_margin)
        self.setStyleSheet(
            f"""
                background-color: {GRAPH_EDITOR_CONFIG.input_field_bg_color};
                border-radius: {GRAPH_EDITOR_CONFIG.input_field_border_radius}px;
            """
        )

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QIntLineEdit(QtWidgets.QLineEdit):
    """
        A QLineEdit that only accepts integers.
    """
    def __init__(
            self, 
            initial_value: int,
            bottom_value: int | None = None,
            placeholder: str = '',
            parent: QtWidgets.QWidget = None,
            **kwargs,
        ) -> None:
        super().__init__(str(initial_value), parent)
        validator = QtGui.QIntValidator()
        if bottom_value:
            validator.setBottom(bottom_value)
        self.setValidator(validator)
        self.setPlaceholderText(placeholder)
        self.setTextMargins(GRAPH_EDITOR_CONFIG.input_field_margin)
        self.setStyleSheet(
            f"""
                background-color: {GRAPH_EDITOR_CONFIG.input_field_bg_color};
                border-radius: {GRAPH_EDITOR_CONFIG.input_field_border_radius}px;
            """
        )

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QFloatLineEdit(QtWidgets.QLineEdit):
    """
        A QLineEdit that only accepts floating-point numbers with flexible precision.
    """
    def __init__(
            self, 
            initial_value: float, 
            bottom_value: float | None = None, 
            placeholder: str = '', 
            parent: QtWidgets.QWidget = None,
            **kwargs,
        ) -> None:
        super().__init__(str(initial_value), parent)
        validator = QtGui.QDoubleValidator()
        validator.setNotation(QtGui.QDoubleValidator.Notation.ScientificNotation)
        if bottom_value:
            validator.setBottom(bottom_value)
        self.setValidator(validator)
        self.setPlaceholderText(placeholder)
        self.setTextMargins(GRAPH_EDITOR_CONFIG.input_field_margin)
        self.setStyleSheet(
            f"""
                background-color: {GRAPH_EDITOR_CONFIG.input_field_bg_color};
                border-radius: {GRAPH_EDITOR_CONFIG.input_field_border_radius}px;
            """
        )

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QField(SparkQWidget):
    """
        QWidget used to represent different types of fields in the SparkGraphEditor's Inspector.
    """

    def __init__(
            self, 
            field_type: str,
            label: str, 
            initial_value: str = '', 
            placeholder: str = '', 
            min_label_width: int | None  = None,
            parent: QtWidgets.QWidget | None = None,
            **kwargs
        ) -> None:
        super().__init__(parent=parent)
        # Get field type.
        self.field_type = field_type.lower()
        field_cls = self.get_field_type(self.field_type)
        # Add layout
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        # Add label
        self._label = QtWidgets.QLabel(
            label, 
            minimumWidth=min_label_width if min_label_width else GRAPH_EDITOR_CONFIG.min_attr_label_size, 
            parent=self,
        )
        self._label.setContentsMargins(GRAPH_EDITOR_CONFIG.label_field_margin)
        layout.addWidget(self._label)
        # Add QLineEdit
        self._line_edit = field_cls(
            initial_value, 
            placeholder, 
            parent=self,
            **kwargs,
        )
        # Setup callback
        self._line_edit.editingFinished.connect(self._on_update)
        # Finalize
        layout.addWidget(self._line_edit)
        self.setLayout(layout)

    def get_value(self):
        if self.field_type == 'str': 
            value = self._line_edit.text()
        elif self.field_type == 'int': 
            value = int(self._line_edit.text())
        elif self.field_type == 'float': 
            value = float(self._line_edit.text())
        else: 
            raise ValueError(
                f'Undefined \"field_type\". Expected \"field_type\" to be in {{\"str\", \"int\", \"float\"}}, got \"{self.field_type}\".'
            )
        return value

    def get_field_type(self, field_type: str) -> type[QtWidgets.QLineEdit]:
        field_type = field_type.lower()
        if field_type == 'str': 
            field_cls = QStrLineEdit
        elif field_type == 'int': 
            field_cls = QIntLineEdit
        elif field_type == 'float': 
            field_cls = QFloatLineEdit
        else: 
            raise ValueError(
                f'Undefined \"field_type\". Expected \"field_type\" to be in {{\"str\", \"int\", \"float\"}}, got \"{field_type}\".'
            )
        return field_cls

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################