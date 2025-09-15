#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

from Qt import QtCore, QtWidgets, QtGui
from spark.graph_editor.widgets.line_edits import QIntLineEdit
from spark.graph_editor.widgets.base import SparkQWidget
from spark.graph_editor.editor_config import GRAPH_EDITOR_CONFIG

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class QShapeEdit(QtWidgets.QWidget):
    """
        A widget for inputting or displaying a shape with a dynamic number of dimensions.
    """
    editingFinished = QtCore.Signal()

    def __init__(
        self,
        initial_shape: tuple[int, ...] = (1,),
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

        self._dimension_edits: list[QtWidgets.QLineEdit] = []

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

        line_edit = QIntLineEdit(initial_value=value, bottom_value=1, placeholder='')
        # Style
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

    def get_shape(self) -> tuple[int, ...]:
        return tuple(int(edit.text()) for edit in self._dimension_edits if edit.text())

    def setShape(self, new_shape: tuple[int, ...]):
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

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QShape(SparkQWidget):
    """
        Custom QWidget used for integer fields in the SparkGraphEditor's Inspector.
    """

    def __init__(self,
                 label: str,
                 initial_shape: tuple[int, ...] = (1,),
                 min_dims: int = 1,
                 max_dims: int = 8,
                 max_shape: int = 1E9,
                 is_static: bool = False,
                 parent: QtWidgets.QWidget = None):
        super().__init__(parent=parent)

        # Add layout
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        # Add label
        self._label = QtWidgets.QLabel(label, minimumWidth=GRAPH_EDITOR_CONFIG.min_attr_label_size, parent=self)
        self._label.setContentsMargins(GRAPH_EDITOR_CONFIG.label_field_margin)
        layout.addWidget(self._label)
        # Add QShapeEdit
        self._shape_edit = QShapeEdit(initial_shape, min_dims, max_dims, max_shape, is_static, self)
        # Update callback
        self._shape_edit.editingFinished.connect(self._on_update)
        # Finalize
        layout.addWidget(self._shape_edit)
        self.setLayout(layout)

    def get_value(self):
        return self._shape_edit.get_shape()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################