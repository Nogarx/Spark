#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import jax.numpy as jnp
from typing import Tuple, List, Any, Callable
from Qt import QtCore, QtWidgets, QtGui

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class NotImplementedWidget(QtWidgets.QLabel):
    """
        A QLineEdit that only accepts integers.
    """
    def __init__(self, parent: QtWidgets.QWidget = None, **kwargs):
        super().__init__(parent=parent)
        self.setText('🏗️ Widget not implemented.')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class StrLineEdit(QtWidgets.QLineEdit):
    """
        A QLineEdit that only accepts integers.
    """
    def __init__(self, initial_value: str = '', parent: QtWidgets.QWidget = None):
        super().__init__(str(initial_value), parent)
        self.setPlaceholderText('Value')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class IntLineEdit(QtWidgets.QLineEdit):
    """
        A QLineEdit that only accepts integers.
    """
    def __init__(self, initial_value: int = 0, bottom_value: int = None, parent: QtWidgets.QWidget = None):
        super().__init__(str(initial_value), parent)
        validator = QtGui.QIntValidator()
        if bottom_value:
            validator.setBottom(bottom_value)
        self.setValidator(validator)
        self.setPlaceholderText('Value')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class FloatLineEdit(QtWidgets.QLineEdit):
    """
        A QLineEdit that only accepts floating-point numbers with flexible precision.
    """
    def __init__(self, initial_value: float = 0.0, bottom_value: float = None,  parent: QtWidgets.QWidget = None):
        super().__init__(str(initial_value), parent)
        validator = QtGui.QDoubleValidator()
        validator.setNotation(QtGui.QDoubleValidator.Notation.ScientificNotation)
        if bottom_value:
            validator.setBottom(bottom_value)
        self.setValidator(validator)
        self.setPlaceholderText('Value')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class KeyValueRow(QtWidgets.QWidget):
    """
        A row widget for a key-value pair.
    """
    editingFinished = QtCore.Signal()

    def __init__(self, key: Any, value: Any, on_delete: Callable, parent: QtWidgets.QWidget = None):
        super().__init__(parent)

        self.key_edit = QtWidgets.QLineEdit()
        self.key_edit.setPlaceholderText('Key')
        self.key_edit.setMaximumWidth(150)
        self.key_edit.setText(str(key))

        self.value_edit = QtWidgets.QLineEdit()
        self.value_edit.setPlaceholderText('Value')
        self.value_edit.setMaximumWidth(1000)
        self.value_edit.setText(str(value))

        self.delete_button = QtWidgets.QPushButton("-")
        self.delete_button.setFixedSize(20,20)
        self.delete_button.setToolTip("Delete this row")

        # --- Layout Changes ---
        # Main horizontal layout for the entire row
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Add the vertical layouts and the delete button to the main layout
        main_layout.addWidget(self.delete_button)
        main_layout.addWidget(self.key_edit)
        main_layout.addWidget(self.value_edit)
        
        self.delete_button.clicked.connect(lambda: on_delete(self))

        self.key_edit.editingFinished.connect(self.on_update)
        self.value_edit.editingFinished.connect(self.on_update)

    def on_update(self,):
        self.editingFinished.emit()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class KeyValueEditor(QtWidgets.QWidget):
    """
        A dynamical list of key-value pairs.
    """
    editingFinished = QtCore.Signal()

    def __init__(self, initial_dict: dict = None, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.rows = []

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(2, 2, 2, 2)

        self.rows_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(self.rows_layout)

        self.add_button = QtWidgets.QPushButton("+")
        self.add_button.setFixedSize(20,20)
        self.add_button.clicked.connect(self.add_row)
        
        main_layout.addWidget(self.add_button, 0, QtCore.Qt.AlignmentFlag.AlignLeft)

        if initial_dict:
            for key, value in initial_dict:
                self.add_row(key, value)
        else:
            self.add_row(None, None)



    def add_row(self, key: Any, value: Any):
        """Creates a new KeyValueRow and adds it to the layout."""
        row = KeyValueRow(key, value, self.remove_row)
        row.editingFinished.connect(self.on_update)
        self.rows.append(row)
        self.rows_layout.addWidget(row)

    def remove_row(self, row_to_delete: KeyValueRow):
        """Removes a specific row from the list and the layout."""
        if row_to_delete in self.rows:
            self.rows.remove(row_to_delete)
            row_to_delete.deleteLater()

    def to_dict(self) -> dict:
        """Converts the current state of the editor to a dictionary."""
        result = {}
        for row in self.rows:
            key = row.key_edit.text().strip()
            value = row.value_edit.text().strip()
            if key:
                result[key] = value
        return result

    def on_update(self,):
        self.editingFinished.emit()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class DtypeWidget(QtWidgets.QComboBox):
    """
    A QComboBox widget for selecting a dtype from a predefined list.
    """
    editingFinished = QtCore.Signal()

    SUPPORTED_DTYPES = [
        jnp.float16, jnp.float32, jnp.float64,
    ]

    def __init__(self, initial_value: jnp.dtype = jnp.float16, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        for dtype in self.SUPPORTED_DTYPES:
            self.addItem(jnp.dtype(dtype).name, userData=dtype)
        self.setDType(initial_value)

    def currentDType(self) -> jnp.dtype:
        """
            Returns the currently selected dtype object.
        """
        return self.currentData()

    def setDType(self, dtype_to_set: jnp.dtype):
        """
            Sets the current selection based on a dtype object.
        """
        index = self.findData(dtype_to_set)
        if index != -1:
            self.setCurrentIndex(index)
        self.editingFinished.emit()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ShapeWidget(QtWidgets.QWidget):
    """
        A widget for inputting or displaying a shape with a dynamic number of dimensions.
    """
    editingFinished = QtCore.Signal()

    def __init__(
        self,
        initial_shape: Tuple[int, ...] = (1,),
        min_dims: int = 1,
        max_dims: int = 8,
        max_shape: int = 1E9,
        is_static: bool = False,
        parent: QtWidgets.QWidget = None
    ):
        super().__init__(parent)
        self.min_dims = min_dims
        self.max_dims = max_dims
        self.max_shape = int(max_shape)
        self.is_static = is_static

        self._layout = QtWidgets.QHBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(5)

        self._dimension_edits: List[QtWidgets.QLineEdit] = []

        # --- Create Control Buttons ---
        if not self.is_static:
            self.removeButton = QtWidgets.QPushButton("-")
            self.removeButton.setFixedSize(20, 20)
            self.removeButton.clicked.connect(self._remove_last_dimension)
            self._layout.addWidget(self.removeButton)

            self.addButton = QtWidgets.QPushButton("+")
            self.addButton.setFixedSize(20, 20)
            self.addButton.clicked.connect(lambda: self._add_dimension())
            self._layout.addWidget(self.addButton)

        # Initialize with the provided shape
        self.setShape(initial_shape if initial_shape is not None else (1,) * self.min_dims)

        self._layout.addStretch()
        self.setLayout(self._layout)

        if not self.is_static:
            self._update_buttons()

    def _add_dimension(self, value: int = 1):
        """
            Adds a new line edit widget.
        """
        if len(self._dimension_edits) >= self.max_dims:
            return

        line_edit = IntLineEdit(initial_value=value, bottom_value=1)
        line_edit.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        line_edit.setAlignment(QtCore.Qt.AlignCenter)
        if self.is_static:
            line_edit.setReadOnly(True)
            # Insert before the stretch
            index = self._layout.count() - 1
            self._layout.insertWidget(index, line_edit)
        else:
            line_edit.setValidator(QtGui.QIntValidator(1, self.max_shape))
            line_edit.textChanged.connect(self._on_shape_changed)
            line_edit.textChanged.connect(lambda: self._adjust_editor_width(line_edit))
            # Insert before the remove button
            index = self._layout.indexOf(self.removeButton)
            self._layout.insertWidget(index, line_edit)

        self._adjust_editor_width(line_edit)
        self._dimension_edits.append(line_edit)

        if not self.is_static:
            self._update_buttons()
        self._on_shape_changed()

    def _remove_last_dimension(self):
        """
            Removes the last dimension widget.
        """
        if len(self._dimension_edits) <= self.min_dims:
            return

        # Get the last editor, remove it from tracking, and delete it
        last_editor = self._dimension_edits.pop()
        last_editor.deleteLater()

        self._update_buttons()
        self._on_shape_changed()

    def _clear_dimensions(self):
        """
            Clear all dimension widgets.
        """
        while self._dimension_edits:
            editor = self._dimension_edits.pop()
            editor.deleteLater()

    def _update_buttons(self):
        """
            Enable/disable the single add/remove buttons based on dimension count.
        """
        if self.is_static:
            return
        num_dims = len(self._dimension_edits)
        self.addButton.setEnabled(num_dims < self.max_dims)
        self.removeButton.setEnabled(num_dims > self.min_dims)

    def _on_shape_changed(self):
        self.editingFinished.emit()

    def shape(self) -> Tuple[int, ...]:
        return tuple(int(edit.text()) for edit in self._dimension_edits if edit.text())

    def setShape(self, new_shape: Tuple[int, ...]):
        """
            Clears existing dimensions and sets a new shape.
        """
        self._clear_dimensions()
        shape_to_set = new_shape if new_shape else (1,) * self.min_dims
        for value in shape_to_set:
            self._add_dimension(value)

    def _adjust_editor_width(self, editor: QtWidgets.QLineEdit):
        """
            Adjusts the width of a line edit to fit its content.
        """
        font_metrics = editor.fontMetrics()
        text_width = font_metrics.horizontalAdvance(editor.text()) + 15
        min_width = 30
        editor.setFixedWidth(max(text_width, min_width))

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################