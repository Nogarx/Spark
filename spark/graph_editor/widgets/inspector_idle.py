#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
    
import typing as tp
from PySide6 import QtWidgets, QtCore, QtGui

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class InspectorIdleWidget(QtWidgets.QWidget):
    """
        Constructs the UI shown when no valid node is selected.
    """

    def __init__(self, parent: QtWidgets.QWidget = None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        # Horizontal layout for message
        _message_widget = self._message()
        # Vertical layout for Widget
        layout = QtWidgets.QVBoxLayout()
        layout.addStretch()
        layout.addWidget(_message_widget)
        layout.addStretch()
        self.setLayout(layout)

    def _message(self,) -> QtWidgets.QWidget:
        # Create new widget for message
        _message_widget = QtWidgets.QWidget()
        # Horizontal layout for the icon and label
        h_layout = QtWidgets.QHBoxLayout(self)
        icon_label = QtWidgets.QLabel()
        icon = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileDialogInfoView)
        icon_label.setPixmap(icon.pixmap(15, 15))
        label = QtWidgets.QLabel('Select a node to Inspect')
        # Horizontally center the content
        h_layout.addStretch()
        h_layout.addWidget(icon_label)
        h_layout.addWidget(label)
        h_layout.addStretch()
        # Set widget layout
        _message_widget.setLayout(h_layout)
        return _message_widget

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################