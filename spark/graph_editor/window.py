#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import os
import enum
import warnings
import typing as tp
import PySide6QtAds as ads
from PySide6 import QtWidgets, QtCore, QtGui

# NOTE: Small workaround to at least have base autocompletion.
if tp.TYPE_CHECKING:
    CDockWidget = QtWidgets.QWidget
else:
    CDockWidget = ads.CDockWidget

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class DockPanels(enum.Enum):
    GRAPH = enum.auto()
    INSPECTOR = enum.auto()
    NODES = enum.auto()
    CONSOLE = enum.auto()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class EditorWindow(QtWidgets.QMainWindow):

    __layout_file__: str = 'layout.xml'
    windowClosed = QtCore.Signal() 

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Header
        self.setWindowTitle('Spark Graph Editor')
        # Create the docking manager and set it as central container
        self.dock_manager = ads.CDockManager(self)
        self.setCentralWidget(self.dock_manager)
        
    def add_dock_widget(self, area: ads.DockWidgetArea, dock_widget: CDockWidget) -> ads.CDockAreaWidget:
        out_area = self.dock_manager.addDockWidget(area, dock_widget)
        return out_area

    # TODO: Figure out how to safely restore the layout. Currently trying to restore the layout state crashes the app.
    def closeEvent(self, event):
        #self.save_layout()
        super().closeEvent(event)
        self.windowClosed.emit()

    def save_layout(self):
        """
            Save current layout.
        """
        try:
            settings = QtCore.QSettings('userPrefs.ini', QtCore.QSettings.IniFormat)
            settings.setValue('dock_manager', self.dock_manager.saveState())
        except Exception as e:
            warnings.warn(f'Error saving layout: {e}')

    def restore_layout(self):
        """
            Try restoring the previous layout.
        """
        if os.path.exists(self.__layout_file__):
            try:
                settings = QtCore.QSettings('userPrefs.ini', QtCore.QSettings.IniFormat)
                self.dock_manager.restoreState(settings.value('dock_manager').toByteArray())
                print("Layout restored.")
            except Exception as e:
                warnings.warn(f'Error restoring layout: {e}')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################