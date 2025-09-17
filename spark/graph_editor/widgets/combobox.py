#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import jax.numpy as jnp
import typing as tp
from Qt import QtCore, QtWidgets, QtGui
from spark.graph_editor.widgets.base import SparkQWidget
from spark.graph_editor.editor_config import GRAPH_EDITOR_CONFIG
from spark.graph_editor.utils import _to_human_readable

# TODO: QComboBox is functional but a QLineEdit + QListView could be better for a more standardize look.

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class QDtypeEdit(QtWidgets.QComboBox):
    """
    A QComboBox widget for selecting a dtype from a predefined list.
    """
    editingFinished = QtCore.Signal()

    def __init__(self, initial_value: jnp.dtype, supported_dtypes: list[jnp.dtype], parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        if not len(supported_dtypes) > 0:
            raise ValueError(f'supported_dtypes cannot be an empty list.')
        self.supported_dtypes = supported_dtypes
        for dtype in self.supported_dtypes:
            self.addItem(_to_human_readable(jnp.dtype(dtype).name), userData=dtype)
        self.set_dtype(initial_value)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        # NOTE: Left padding seems to be missing 4px to match standard QLineEdit style.
        self.setStyleSheet(f'background-color: {GRAPH_EDITOR_CONFIG.input_field_bg_color};\
                             border-radius: {GRAPH_EDITOR_CONFIG.input_field_border_radius}px;\
                             padding-left: {GRAPH_EDITOR_CONFIG.input_field_margin.left()+4}px;\
                             padding-top: {GRAPH_EDITOR_CONFIG.input_field_margin.top()}px;\
                             padding-right: {GRAPH_EDITOR_CONFIG.input_field_margin.right()}px;\
                             padding-bottom: {GRAPH_EDITOR_CONFIG.input_field_margin.bottom()}px;')

    def set_dtype(self, dtype_to_set: jnp.dtype) -> None:
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
                 initial_value: jnp.dtype,
                 values_options: list[jnp.dtype],
                 parent: QtWidgets.QWidget = None):
        super().__init__(parent=parent)
        if not len(values_options) > 0:
            raise ValueError(f'supported_dtypes cannot be an empty list.')
        # Add layout
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        # Add label
        self._label = QtWidgets.QLabel(label, minimumWidth=GRAPH_EDITOR_CONFIG.min_attr_label_size, parent=self)
        self._label.setContentsMargins(GRAPH_EDITOR_CONFIG.label_field_margin)
        layout.addWidget(self._label)
        # Add QShapeEdit
        self._dtype_edit = QDtypeEdit(initial_value, values_options, self)
        # Update callback
        self._dtype_edit.editingFinished.connect(self._on_update)
        # Finalize
        layout.addWidget(self._dtype_edit)
        self.setLayout(layout)

    def get_value(self):
        return self._dtype_edit.get_dtype()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QBoolEdit(QtWidgets.QComboBox):
    """
    A QComboBox widget for selecting a dtype from a predefined list.
    """
    editingFinished = QtCore.Signal()

    def __init__(self, initial_value: bool, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.addItem('True', userData=True)
        self.addItem('False', userData=False)
        self.set_value(initial_value)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        # NOTE: Left padding seems to be missing 4px to match standard QLineEdit style.
        self.setStyleSheet(f'background-color: {GRAPH_EDITOR_CONFIG.input_field_bg_color};\
                             border-radius: {GRAPH_EDITOR_CONFIG.input_field_border_radius}px;\
                             padding-left: {GRAPH_EDITOR_CONFIG.input_field_margin.left()+4}px;\
                             padding-top: {GRAPH_EDITOR_CONFIG.input_field_margin.top()}px;\
                             padding-right: {GRAPH_EDITOR_CONFIG.input_field_margin.right()}px;\
                             padding-bottom: {GRAPH_EDITOR_CONFIG.input_field_margin.bottom()}px;')

    def set_value(self, value: bool) -> None:
        """
            Sets the current selection based on a dtype object.
        """
        index = self.findData(value)
        if index != -1:
            self.setCurrentIndex(index)
        self.editingFinished.emit()

    def get_value(self) -> bool:
        """
            Returns the currently selected dtype object.
        """
        return self.currentData()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QBool(SparkQWidget):
    """
        Custom QWidget used for integer fields in the SparkGraphEditor's Inspector.
    """

    def __init__(self,
                 label: str,
                 initial_value: bool = True,
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
        self._dtype_edit = QBoolEdit(initial_value, self)
        # Update callback
        self._dtype_edit.editingFinished.connect(self._on_update)
        # Finalize
        layout.addWidget(self._dtype_edit)
        self.setLayout(layout)

    def get_value(self):
        return self._dtype_edit.get_value()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################