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

class QKeyValueRow(QtWidgets.QWidget):
    """
        A row widget for a key-value pair.
    """
    editingFinished = QtCore.Signal()

    def __init__(self, key: tp.Any, value: tp.Any, on_delete: tp.Callable, parent: QtWidgets.QWidget = None):
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

class QDict(QtWidgets.QWidget):
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



    def add_row(self, key: tp.Any, value: tp.Any):
        """Creates a new QKeyValueRow and adds it to the layout."""
        row = QKeyValueRow(key, value, self.remove_row)
        row.editingFinished.connect(self.on_update)
        self.rows.append(row)
        self.rows_layout.addWidget(row)

    def remove_row(self, row_to_delete: QKeyValueRow):
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

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################