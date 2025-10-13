#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
    
import typing as tp
import dataclasses as dc
import Qt
from Qt import QtCore, QtWidgets, QtGui

from spark.graph_editor.widgets.title_bar import DockTitleBar
from spark.graph_editor.editor_config import GRAPH_EDITOR_CONFIG

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class InspectorPanel(QtWidgets.QDockWidget):
    """
        Container for the Inspector (Node description).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )
        # Header
        self.setTitleBarWidget(DockTitleBar('Inspector'))
        # Main layout
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(6)
        layout.addStretch()
        self.setLayout(layout)

        label = QtWidgets.QLabel('TEST')
        self.layout().addWidget(label)

        self.setObjectName('QDockWidget')
        self.setStyleSheet(
            f"""
                QWidget#QDockWidget>QWidget {{
                    background-color: {GRAPH_EDITOR_CONFIG.primary_bg_color};
                    font-size: {GRAPH_EDITOR_CONFIG.medium_font_size}px;
                    padding: 2px;
                    border-width: 1px;
                    border-style: solid;
                    border-color: {GRAPH_EDITOR_CONFIG.border_color};
                }}
            """
        )

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################