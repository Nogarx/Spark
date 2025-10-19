#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

from PySide6 import QtWidgets, QtGui, QtCore
from spark.graph_editor.widgets.base import SparkQWidget
from spark.graph_editor.editor_config import GRAPH_EDITOR_CONFIG
import spark.graph_editor.style.resources_rc

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
        initial_value = str(initial_value) if initial_value is not None else ''
        super().__init__(initial_value, parent=parent)
        self.setPlaceholderText(placeholder)
        self.setTextMargins(GRAPH_EDITOR_CONFIG.input_field_margin)
        self.setStyleSheet(
            f"""
                font-size: {GRAPH_EDITOR_CONFIG.medium_font_size}px;
                background-color: {GRAPH_EDITOR_CONFIG.input_field_bg_color};
                border-radius: {GRAPH_EDITOR_CONFIG.input_field_border_radius}px;
                padding: 0px;
                margin: 0px;
            """
        )

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QString(SparkQWidget):
    """
        Custom QWidget used for string fields in the SparkGraphEditor's Inspector.
    """

    def __init__(
            self, 
            initial_value: str , 
            placeholder: str = '', 
            parent: QtWidgets.QWidget = None
        ) -> None:
        super().__init__(parent=parent)
        # Add layout
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(QtCore.QMargins(0, 0, 0, 0))
        # Add QLineEdit
        self._line_edit = QStrLineEdit(initial_value, placeholder, parent=self)
        self._line_edit.setFixedHeight(20)
        self._line_edit.editingFinished.connect(self._on_update)
        layout.addWidget(self._line_edit)
        # Finalize
        self.setLayout(layout)

    def get_value(self) -> str:
        return self._line_edit.text()

    def set_value(self, value: str) -> None:
        return self._line_edit.setText(value)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QIntLineEdit(QtWidgets.QLineEdit):
    """
        A QLineEdit that only accepts integers.
    """
    def __init__(
            self, 
            initial_value: int,
            bottom_value: int | None = None,
            placeholder: int | None = None,
            parent: QtWidgets.QWidget = None,
            **kwargs,
        ) -> None:
        initial_value = str(initial_value) if initial_value is not None else ''
        super().__init__(initial_value, parent=parent)
        validator = QtGui.QIntValidator()
        if bottom_value:
            validator.setBottom(bottom_value)
        self.setValidator(validator)
        self.setPlaceholderText(placeholder)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.setTextMargins(GRAPH_EDITOR_CONFIG.input_field_margin)
        self.setStyleSheet(
            f"""
                font-size: {GRAPH_EDITOR_CONFIG.medium_font_size}px;
                background-color: {GRAPH_EDITOR_CONFIG.input_field_bg_color};
                border-radius: {GRAPH_EDITOR_CONFIG.input_field_border_radius}px;
            """
        )

    def get_value(self) -> int:
        return int(self.text())

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QInt(SparkQWidget):
    """
        Custom QWidget used for int fields in the SparkGraphEditor's Inspector.
    """

    def __init__(
            self, 
            initial_value: int = 0, 
            bottom_value: int | None = None,
            placeholder: int | None = None, 
            parent: QtWidgets.QWidget = None
        ) -> None:
        super().__init__(parent=parent)
        # Add layout
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(QtCore.QMargins(0, 0, 0, 0))
        # Add QLineEdit
        self._line_edit = QIntLineEdit(initial_value, bottom_value=bottom_value, placeholder=placeholder, parent=self)
        self._line_edit.setFixedHeight(20)
        self._line_edit.editingFinished.connect(self._on_update)
        layout.addWidget(self._line_edit)
        # Finalize
        self.setLayout(layout)

    def get_value(self) -> int:
        return int(self._line_edit.text())

    def set_value(self, value: int) -> None:
        return self._line_edit.setText(str(value))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QFloatLineEdit(QtWidgets.QLineEdit):
    """
        A QLineEdit that only accepts floating-point numbers with flexible precision.
    """
    def __init__(
            self, 
            initial_value: float, 
            bottom_value: float | None = None, 
            placeholder: float | None = None, 
            parent: QtWidgets.QWidget = None,
            **kwargs,
        ) -> None:
        initial_value = str(initial_value) if initial_value is not None else ''
        super().__init__(initial_value, parent=parent)
        validator = QtGui.QDoubleValidator()
        validator.setNotation(QtGui.QDoubleValidator.Notation.ScientificNotation)
        if bottom_value:
            validator.setBottom(bottom_value)
        self.setValidator(validator)
        self.setPlaceholderText(placeholder)
        self.setTextMargins(GRAPH_EDITOR_CONFIG.input_field_margin)
        self.setStyleSheet(
            f"""
                font-size: {GRAPH_EDITOR_CONFIG.medium_font_size}px;
                background-color: {GRAPH_EDITOR_CONFIG.input_field_bg_color};
                border-radius: {GRAPH_EDITOR_CONFIG.input_field_border_radius}px;
            """
        )

    def get_value(self) -> float:
        return float(self.text())

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QFloat(SparkQWidget):
    """
        Custom QWidget used for float fields in the SparkGraphEditor's Inspector.
    """

    def __init__(
            self, 
            initial_value: float = 0, 
            bottom_value: float | None = None,
            placeholder: float | None = None, 
            parent: QtWidgets.QWidget = None
        ) -> None:
        super().__init__(parent=parent)
        # Add layout
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(QtCore.QMargins(0, 0, 0, 0))
        # Add QLineEdit
        self._line_edit = QFloatLineEdit(initial_value, bottom_value=bottom_value, placeholder=placeholder, parent=self)
        self._line_edit.setFixedHeight(20)
        self._line_edit.editingFinished.connect(self._on_update)
        layout.addWidget(self._line_edit)
        # Finalize
        self.setLayout(layout)

    def get_value(self) -> float:
        return float(self._line_edit.text())

    def set_value(self, value: float) -> None:
        return self._line_edit.setText(str(value))

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################