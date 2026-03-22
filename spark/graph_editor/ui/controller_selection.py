#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
    
import typing as tp
from PySide6 import QtCore, QtWidgets, QtGui
import PySide6QtAds as ads

from spark.core.registry import REGISTRY
from spark.graph_editor.models.graph import SparkNodeGraph, ControllerType
from spark.graph_editor.models.graph_menu_tree import HierarchicalMenuTree
from spark.graph_editor.models.nodes import SourceNode, SinkNode, AbstractNode, module_to_nodegraph
from spark.graph_editor.ui.console_panel import MessageLevel
from spark.graph_editor.models.graph import ControllerType

# NOTE: Small workaround to at least have base autocompletion.
if tp.TYPE_CHECKING:
    CDockWidget = QtWidgets.QWidget
else:
    CDockWidget = ads.CDockWidget

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class ControllerSelectorDialog(QtWidgets.QDialog):

    def __init__(self, is_init_call: bool, parent=None) -> None:

        super().__init__(parent)
        self.setWindowTitle('Controller type selection')
        self.setMinimumWidth(350)
        self.selected_choice = None

        # Layout
        main_layout = QtWidgets.QVBoxLayout(self)
        buttons_layout = QtWidgets.QHBoxLayout()

        # Option A Button
        self.btn_brain = QtWidgets.QPushButton('Brain')
        brain_icon = QtGui.QPixmap(':/icons/brain_icon.png')
        self.btn_brain.setIcon(brain_icon)
        self.btn_brain.setIconSize(QtCore.QSize(32, 32))
        self.btn_brain.setMinimumHeight(60)

        # Option B Button
        self.btn_neuron = QtWidgets.QPushButton('Neuron')
        neuron_icon = QtGui.QPixmap(':/icons/neuron_icon.png')
        self.btn_neuron.setIcon(neuron_icon)
        self.btn_neuron.setIconSize(QtCore.QSize(32, 32))
        self.btn_neuron.setMinimumHeight(60)

        buttons_layout.addWidget(self.btn_brain)
        buttons_layout.addWidget(self.btn_neuron)
        main_layout.addLayout(buttons_layout)

        # Info text
        self.info_label = QtWidgets.QLabel('Please select one of the configuration options above.')
        self.info_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.info_label.setWordWrap(True)
        # Add a little margin to separate it from the buttons
        self.info_label.setContentsMargins(0, 10, 0, 15) 
        main_layout.addWidget(self.info_label)

        # Events
        if not is_init_call:
            self.btn_brain.clicked.connect(lambda: self.confirm_selection('Brain', ControllerType.BRAIN))
            self.btn_neuron.clicked.connect(lambda: self.confirm_selection('Neuron', ControllerType.NEURON))
            # Add cancel button
            self.btn_cancel = QtWidgets.QPushButton('Cancel')
            main_layout.addWidget(self.btn_cancel)
            self.btn_cancel.clicked.connect(self.reject)
        else:
            self.btn_brain.clicked.connect(lambda: self._confirm_selection(ControllerType.BRAIN))
            self.btn_neuron.clicked.connect(lambda: self._confirm_selection(ControllerType.NEURON))

    def _confirm_selection(self, controller_type: ControllerType):
        self.selected_choice = controller_type
        self.accept()

    def confirm_selection(self, controller_name: str, controller_type: ControllerType):
        """
            Shows the confirmation popup before accepting the dialog.
        """
        ret = QtWidgets.QMessageBox.question(
            self,
            'Confirm Selection',
            f'You have selected <b>{controller_name}</b>.<br>Are you sure you want to proceed?',
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No
        )
        
        # If they click Yes, record the choice and close the dialog with a positive result
        if ret == QtWidgets.QMessageBox.StandardButton.Yes:
            self.selected_choice = controller_type
            self.accept()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################