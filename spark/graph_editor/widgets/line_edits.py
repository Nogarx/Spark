#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

from Qt import QtWidgets, QtGui
from spark.graph_editor.widgets.base import SparkQWidget

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

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QString(SparkQWidget):
    """
        Custom QWidget used for string fields in the SparkGraphEditor's Inspector.
    """

    def __init__(self, 
                 label: str, 
                 initial_value: str = '', 
                 placeholder: str = 'Value', 
                 parent: QtWidgets.QWidget = None):
        super.__init__(parent=parent)

        # Add layout
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        # Add label
        self._label = QtWidgets.QLabel(label, parent=self)
        layout.addWidget(self._label)
        # Add QLineEdit
        self._line_edit = QStrLineEdit(initial_value, placeholder, parent=self)
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

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QInt(SparkQWidget):
    """
        Custom QWidget used for integer fields in the SparkGraphEditor's Inspector.
    """



    def __init__(self, 
                 label: str, 
                 initial_value: int = 0, 
                 bottom_value: int = None, 
                 placeholder: str = 'Value', 
                 parent: QtWidgets.QWidget = None):
        super.__init__(parent=parent)

        # Add layout
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        # Add label
        self._label = QtWidgets.QLabel(label, parent=self)
        layout.addWidget(self._label)
        # Add QLineEdit
        self._line_edit = QIntLineEdit(initial_value, bottom_value, placeholder, parent=self)
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

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QFloat(SparkQWidget):
    """
        Custom QWidget used for float fields in the SparkGraphEditor's Inspector.
    """

    def __init__(self, 
                 label: str, 
                 initial_value: int = 0, 
                 bottom_value: int = None, 
                 placeholder: str = 'Value', 
                 parent: QtWidgets.QWidget = None):
        super.__init__(parent=parent)

        # Add layout
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        # Add label
        self._label = QtWidgets.QLabel(label, parent=self)
        layout.addWidget(self._label)
        # Add QLineEdit
        self._line_edit = QFloatLineEdit(initial_value, bottom_value, placeholder, parent=self)
        layout.addWidget(self._line_edit)
        self.setLayout(layout)

    def get_value(self):
        return self._line_edit.text()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################