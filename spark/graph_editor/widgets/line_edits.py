#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

from Qt import QtWidgets, QtGui, QtCore
from spark.graph_editor.widgets.base import SparkQWidget
from spark.graph_editor.editor_config import GRAPH_EDITOR_CONFIG

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class QStrLineEdit(QtWidgets.QLineEdit):
    """
        A simple QLineEdit with a placeholder.
    """
    def __init__(self, initial_value: str, placeholder: str, parent: QtWidgets.QWidget = None):
        super().__init__(str(initial_value), parent)
        self.setPlaceholderText(placeholder)
        self.setTextMargins(GRAPH_EDITOR_CONFIG.input_field_margin)
        self.setStyleSheet(f'background-color: {GRAPH_EDITOR_CONFIG.input_field_bg_color};\
                             border-radius: {GRAPH_EDITOR_CONFIG.input_field_border_radius}px;')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QString(SparkQWidget):
    """
        Custom QWidget used for string fields in the SparkGraphEditor's Inspector.
    """

    def __init__(self, 
                 label: str, 
                 initial_value: str = '', 
                 placeholder: str = '', 
                 parent: QtWidgets.QWidget = None):
        super().__init__(parent=parent)

        # Add layout
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        # Add label
        self._label = QtWidgets.QLabel(label, minimumWidth=GRAPH_EDITOR_CONFIG.min_attr_label_size, parent=self)
        self._label.setContentsMargins(GRAPH_EDITOR_CONFIG.label_field_margin)
        layout.addWidget(self._label)
        # Add QLineEdit
        self._line_edit = QStrLineEdit(initial_value, placeholder, parent=self)
        # Update callback
        self._line_edit.editingFinished.connect(self._on_update)
        # Finalize
        layout.addWidget(self._line_edit)
        self.setLayout(layout)

    def get_value(self):
        return self._line_edit.text()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QIntLineEdit(QtWidgets.QLineEdit):
    """
        A QLineEdit that only accepts integers.
    """
    def __init__(self, initial_value: int, bottom_value: int, placeholder: str, parent: QtWidgets.QWidget = None):
        super().__init__(str(initial_value), parent)
        validator = QtGui.QIntValidator()
        if bottom_value:
            validator.setBottom(bottom_value)
        self.setValidator(validator)
        self.setPlaceholderText(placeholder)
        self.setTextMargins(GRAPH_EDITOR_CONFIG.input_field_margin)
        self.setStyleSheet(f'background-color: {GRAPH_EDITOR_CONFIG.input_field_bg_color};\
                             border-radius: {GRAPH_EDITOR_CONFIG.input_field_border_radius}px;')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QInt(SparkQWidget):
    """
        Custom QWidget used for integer fields in the SparkGraphEditor's Inspector.
    """

    def __init__(self, 
                 label: str, 
                 initial_value: int = 0, 
                 bottom_value: int = None, 
                 placeholder: str = '', 
                 parent: QtWidgets.QWidget = None):
        super().__init__(parent=parent)

        # Add layout
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        # Add label
        self._label = QtWidgets.QLabel(label, minimumWidth=GRAPH_EDITOR_CONFIG.min_attr_label_size, parent=self)
        self._label.setContentsMargins(GRAPH_EDITOR_CONFIG.label_field_margin)
        layout.addWidget(self._label)
        # Add QLineEdit
        self._line_edit = QIntLineEdit(initial_value, bottom_value, placeholder, parent=self)
        # Update callback
        self._line_edit.editingFinished.connect(self._on_update)
        # Finalize
        layout.addWidget(self._line_edit)
        self.setLayout(layout)

    def get_value(self):
        return self._line_edit.text()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QFloatLineEdit(QtWidgets.QLineEdit):
    """
        A QLineEdit that only accepts floating-point numbers with flexible precision.
    """
    def __init__(self, initial_value: int, bottom_value: int, placeholder: str, parent: QtWidgets.QWidget = None):
        super().__init__(str(initial_value), parent)
        validator = QtGui.QDoubleValidator()
        validator.setNotation(QtGui.QDoubleValidator.Notation.ScientificNotation)
        if bottom_value:
            validator.setBottom(bottom_value)
        self.setValidator(validator)
        self.setPlaceholderText(placeholder)
        self.setTextMargins(GRAPH_EDITOR_CONFIG.input_field_margin)
        self.setStyleSheet(f'background-color: {GRAPH_EDITOR_CONFIG.input_field_bg_color};\
                             border-radius: {GRAPH_EDITOR_CONFIG.input_field_border_radius}px;')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QFloat(SparkQWidget):
    """
        Custom QWidget used for float fields in the SparkGraphEditor's Inspector.
    """

    def __init__(self, 
                 label: str, 
                 initial_value: int = 0, 
                 bottom_value: int = None, 
                 placeholder: str = '', 
                 parent: QtWidgets.QWidget = None):
        super().__init__(parent=parent)

        # Add layout
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        # Add label
        self._label = QtWidgets.QLabel(label, minimumWidth=GRAPH_EDITOR_CONFIG.min_attr_label_size, parent=self)
        self._label.setContentsMargins(GRAPH_EDITOR_CONFIG.label_field_margin)
        layout.addWidget(self._label)
        # Add QLineEdit
        self._line_edit = QFloatLineEdit(initial_value, bottom_value, placeholder, parent=self)
        # Update callback
        self._line_edit.editingFinished.connect(self._on_update)
        # Finalize
        layout.addWidget(self._line_edit)
        self.setLayout(layout)

    def get_value(self):
        return self._line_edit.text()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################