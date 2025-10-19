#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

# GPU is not required in the editor
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''     
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'

import jax.numpy as jnp
import typing as tp
from PySide6 import QtCore, QtWidgets, QtGui
import spark.core.utils as utils
from spark.graph_editor.widgets.base import SparkQWidget
from spark.graph_editor.editor_config import GRAPH_EDITOR_CONFIG

# TODO: QComboBox is functional but a QLineEdit + QListView could be better for a more standardize look.

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class QComboBoxEdit(QtWidgets.QComboBox):
    """
        A QComboBox widget for selecting a dtype from a predefined list.
    """

    def __init__(
            self, 
            initial_value: jnp.dtype, 
            options_list: list[tuple[str, tp.Any]], 
            parent: QtWidgets.QWidget = None
        ) -> None:
        super().__init__(parent)

        # Populate ComboBox
        if not len(options_list) > 0:
            raise ValueError(
                f'QComboBoxEdit options_list cannot be an empty list.'
            )
        self.options_list = options_list
        for (option_name, option_data) in self.options_list:
            self.addItem(option_name, userData=option_data)
        self.set_value(initial_value)

        # Set style
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        # NOTE: Left padding seems to be missing 4px to match standard QLineEdit style.
        self.setStyleSheet(
            f"""
                QComboBox {{
                    background-color: {GRAPH_EDITOR_CONFIG.input_field_bg_color};
                    border-radius: {GRAPH_EDITOR_CONFIG.input_field_border_radius}px;
                    padding-left: {GRAPH_EDITOR_CONFIG.input_field_margin.left()+4}px;
                    padding-top: {GRAPH_EDITOR_CONFIG.input_field_margin.top()}px;
                    padding-right: {GRAPH_EDITOR_CONFIG.input_field_margin.right()}px;
                    padding-bottom: {GRAPH_EDITOR_CONFIG.input_field_margin.bottom()}px;
                }}
                QComboBox::drop-down {{
                    border: 0px;
                    width: 0px;
                }}
                QComboBox::down-arrow {{
                    image: none;
                    width: 0px;
                    height: 0px;
                }}
            """
        )

    def set_value(self, value: tp.Any) -> None:
        """
            Sets the current object selection.
        """
        index = self.findData(value)
        if index != -1:
            self.setCurrentIndex(index)

    def get_value(self) -> tp.Any:
        """
            Returns the currently selected object.
        """
        return self.currentData()

    def setEnabled(self, value: bool) -> None:
        return super().setEnabled(value)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QDtype(SparkQWidget):
    """
        Custom QWidget used for dtypes fields in the SparkGraphEditor's Inspector.
    """

    def __init__(
            self,
            initial_value: jnp.dtype,
            values_options: list[jnp.dtype],
            parent: QtWidgets.QWidget = None
        ) -> None:
        super().__init__(parent=parent)
        # Add layout
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        # Add QShapeEdit
        options_list = [(utils.to_human_readable(dtype.__name__), dtype) for dtype in values_options]
        self._dtype_edit = QComboBoxEdit(initial_value, options_list, self)
        # Update callback
        self._dtype_edit.currentIndexChanged.connect(self._on_update)
        # Finalize
        layout.addWidget(self._dtype_edit)
        self.setLayout(layout)

    def get_value(self) -> jnp.dtype:
        return self._dtype_edit.get_value()
    
    def set_value(self, value: jnp.dtype) -> None:
        return self._dtype_edit.set_value(value)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QBool(SparkQWidget):
    """
        Custom QWidget used for bool fields in the SparkGraphEditor's Inspector.
    """

    def __init__(
            self,
            initial_value: bool = True,
            parent: QtWidgets.QWidget = None
        ) -> None:
        super().__init__(parent=parent)
        # Add layout
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        # Add QShapeEdit
        self._bool_edit = QComboBoxEdit(initial_value, [('True', True), ('False', False)], self)
        # Update callback
        self._bool_edit.currentIndexChanged.connect(self._on_update)
        # Finalize
        layout.addWidget(self._bool_edit)
        self.setLayout(layout)

    def get_value(self) -> bool:
        return self._bool_edit.get_value()
    
    def set_value(self, value: bool) -> None:
        return self._bool_edit.set_value(value)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QGenericComboBox(SparkQWidget):
    """
        Custom QWidget used for arbitrary selectable fields fields in the SparkGraphEditor's Inspector.
    """

    def __init__(
            self,
            initial_value: str,
            values_options: dict[str, tp.Any],
            parent: QtWidgets.QWidget = None
        ) -> None:
        super().__init__(parent=parent)
        # Add layout
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        # Add QShapeEdit
        options_list = [(key, value) for key, value in values_options.items()]
        self._value_edit = QComboBoxEdit(initial_value, options_list, self)
        # Update callback
        self._value_edit.currentIndexChanged.connect(self._on_update)
        # Finalize
        layout.addWidget(self._value_edit)
        self.setLayout(layout)

    def get_value(self) -> tp.Any:
        return self._value_edit.get_value()
    
    def set_value(self, value: tp.Any) -> None:
        return self._value_edit.set_value(value)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################