#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import jax.numpy as jnp
import typing as tp
from Qt import QtCore, QtWidgets, QtGui
from spark.graph_editor.widgets.base import SparkQWidget
from spark.graph_editor.editor_config import GRAPH_EDITOR_CONFIG

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class QDtypeEdit(QtWidgets.QComboBox):
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
        self.setContentsMargins(GRAPH_EDITOR_CONFIG.input_field_margin)
        self.setStyleSheet(f'background-color: {GRAPH_EDITOR_CONFIG.input_field_bg_color};\
                             border-radius: {GRAPH_EDITOR_CONFIG.input_field_border_radius}px;')

    def setDType(self, dtype_to_set: jnp.dtype):
        """
            Sets the current selection based on a dtype object.
        """
        index = self.findData(dtype_to_set)
        if index != -1:
            self.setCurrentIndex(index)
        self.editingFinished.emit()

    def get_dtype(self) -> jnp.dtype:
        """
            Returns the currently selected dtype object.
        """
        return self.currentData()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QDtype(SparkQWidget):
    """
        Custom QWidget used for integer fields in the SparkGraphEditor's Inspector.
    """

    def __init__(self,
                 label: str,
                 initial_value: jnp.dtype = jnp.float16, 
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
        self._dtype_edit = QDtypeEdit(initial_value, self)
        # Update callback
        self._dtype_edit.editingFinished.connect(self._on_update)
        # Finalize
        layout.addWidget(self._dtype_edit)
        self.setLayout(layout)

    def get_value(self):
        return self._dtype_edit.get_dtype()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################