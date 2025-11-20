#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

from PySide6 import QtCore, QtWidgets, QtGui
import spark.core.utils as utils
from spark.graph_editor.widgets.line_edits import QIntLineEdit
from spark.graph_editor.widgets.base import SparkQWidget
from spark.graph_editor.editor_config import GRAPH_EDITOR_CONFIG

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class QPushButtonSquare(QtWidgets.QPushButton):
    """
        A square QPushButton
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, 
            QtWidgets.QSizePolicy.Policy.Expanding
        )

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return width

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QShapeEdit(QtWidgets.QWidget):
    """
        A widget for inputting or displaying a shape with a dynamic number of dimensions.
    """
    editingFinished = QtCore.Signal()

    def __init__(
        self,
        initial_shape: tuple[int] = None,
        min_dims: int = 1,
        max_dims: int = 8,
        max_shape: int = 1E9,
        parent: QtWidgets.QWidget = None
    ):
        super().__init__(parent)
        self.min_dims = min_dims
        self.max_dims = max_dims
        self.max_shape = int(max_shape)

        self._layout = QtWidgets.QHBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(5)

        self._dimension_edits: list[QtWidgets.QLineEdit] = []

        # --- Create Control Buttons ---
        self.removeButton = QPushButtonSquare("-")
        self.removeButton.setFixedSize(20, 20)
        self.removeButton.clicked.connect(self._remove_last_dimension)
        self.removeButton.setStyleSheet(
            f"""
                background-color: {GRAPH_EDITOR_CONFIG.button_bg_color};
                border-radius: {GRAPH_EDITOR_CONFIG.button_border_radius}px;
            """
        )
        self._layout.addWidget(self.removeButton)

        self.addButton = QPushButtonSquare("+")
        self.addButton.setFixedSize(20, 20)
        self.addButton.clicked.connect(lambda: self._add_dimension())
        self.addButton.setStyleSheet(
            f"""
                background-color: {GRAPH_EDITOR_CONFIG.button_bg_color};
                border-radius: {GRAPH_EDITOR_CONFIG.button_border_radius}px;
            """
        )
        self._layout.addWidget(self.addButton)

        # Initialize with the provided shape
        self.set_shape(initial_shape if initial_shape else (1,) * self.min_dims)

        self._layout.addStretch()
        self.setLayout(self._layout)

        self._update_buttons()

    def _add_dimension(self, value: int = 1):
        """
            Adds a new line edit widget.
        """
        if len(self._dimension_edits) >= self.max_dims:
            return

        line_edit = QIntLineEdit(initial_value=value, bottom_value=1, placeholder='')
        # Style
        line_edit.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        line_edit.setAlignment(QtCore.Qt.AlignCenter)

        line_edit.setValidator(QtGui.QIntValidator(1, self.max_shape))
        line_edit.textChanged.connect(self._on_shape_changed)
        line_edit.textChanged.connect(lambda: self._adjust_editor_width(line_edit))
        # Insert before the remove button
        index = self._layout.indexOf(self.removeButton)
        self._layout.insertWidget(index, line_edit)

        self._adjust_editor_width(line_edit)
        self._dimension_edits.append(line_edit)

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
        if self.isEnabled():
            num_dims = len(self._dimension_edits)
            self.addButton.setEnabled(num_dims < self.max_dims)
            self.removeButton.setEnabled(num_dims > self.min_dims)

    def _on_shape_changed(self):
        self.editingFinished.emit()

    def get_shape(self) -> tuple[int]:
        shape_text = tuple(int(edit.text()) for edit in self._dimension_edits if edit.text())
        shape = utils.validate_shape(shape_text)
        return shape

    def set_shape(self, new_shape: tuple[int]):
        """
            Clears existing dimensions and sets a new shape.
        """
        self._clear_dimensions()
        for value in new_shape:
            self._add_dimension(value)

    def _adjust_editor_width(self, editor: QtWidgets.QLineEdit):
        """
            Adjusts the width of a line edit to fit its content.
        """
        font_metrics = editor.fontMetrics()
        text_width = font_metrics.horizontalAdvance(editor.text()) + 15
        min_width = 30
        editor.setFixedWidth(max(text_width, min_width))

    def setEnabled(self, value: bool) -> None:
        # Enable/disable buttons
        self.addButton.setVisible(value)
        self.removeButton.setVisible(value)
        return super().setEnabled(value)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QShape(SparkQWidget):
    """
        Custom QWidget used for integer fields in the SparkGraphEditor's Inspector.
    """

    def __init__(
            self,
            initial_shape: tuple[int] = (1,),
            min_dims: int = 1,
            max_dims: int = 8,
            max_shape: int = 1E9,
            parent: QtWidgets.QWidget = None
        ) -> None:
        super().__init__(parent=parent)

        # Add layout
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        # Add QShapeEdit
        self._shape_edit = QShapeEdit(initial_shape, min_dims, max_dims, max_shape, self)
        # Update callback
        self._shape_edit.editingFinished.connect(self._on_update)
        # Finalize
        layout.addWidget(self._shape_edit)
        self.setLayout(layout)

    def get_value(self) -> tuple[int]:
        return self._shape_edit.get_shape()

    def set_value(self, value: tuple[int]):
        return self._shape_edit.set_shape(value)
    
    def setEnabled(self, value: bool) -> None:
        self._shape_edit.setEnabled(value)
        return super().setEnabled(value)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################