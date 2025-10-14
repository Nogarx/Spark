#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
    
import typing as tp
from PySide6 import QtWidgets, QtCore, QtGui

import dataclasses as dc
from spark.core.config import BaseSparkConfig
from spark.graph_editor.models.nodes import AbstractNode
from spark.graph_editor.widgets.line_edits import QField

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class QNodeConfig(QtWidgets.QWidget):
    """
        Constructs the UI shown when no valid node is selected.
    """

    def __init__(self, node: AbstractNode, parent: QtWidgets.QWidget = None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        # Node reference
        self._target_node = node
        self._cfg_widget_map: dict[str, QtWidgets.QWidget] = {}
        # Widget layout
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(QtCore.QMargins(0, 0, 0, 0))
        layout.setSpacing(2)
        layout.addStretch()
        self.setLayout(layout)
        # Setup layout
        self._setup_layout()

    def addWidget(self, widget: QtWidgets.QWidget) -> None:
        """
            Add a widget to the central content widget's layout.
        """
        widget.installEventFilter(self)
        index = self.layout().count() - 1
        self.layout().insertWidget(index, widget)

    def _setup_layout(self,) -> QtWidgets.QWidget:
        # Add name widget
        name_widget = QField(
            field_type='str',
            label='Name', 
            initial_value=self._target_node.NODE_NAME, 
            min_label_width=50,
            parent=self
        )
        name_widget.on_update.connect(lambda name: self._target_node.set_name(name))
        self.addWidget(name_widget)
        # 
        node_config: BaseSparkConfig = getattr(self._target_node, 'node_config', None)
        #if node_config:
        #    for f in dc.fields():

    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################