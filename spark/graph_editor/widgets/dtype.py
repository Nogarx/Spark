#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import jax.numpy as jnp
import typing as tp
from Qt import QtCore, QtWidgets, QtGui

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class QDtype(QtWidgets.QComboBox):
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

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################