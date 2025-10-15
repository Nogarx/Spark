#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import os
import sys
import warnings
import typing as tp
from PySide6 import QtWidgets, QtCore, QtGui
import PySide6QtAds as ads

from spark.graph_editor.editor_config import GRAPH_EDITOR_CONFIG
from spark.graph_editor.ui.menu_bar import MenuBar
from spark.graph_editor.ui.status_bar import StatusBar
from spark.graph_editor.ui.graph_panel import GraphPanel
from spark.graph_editor.widgets.dock_panel import QDockPanel

from spark.graph_editor.ui.inspector_panel import InspectorPanel

# NOTE: Small workaround to at least have base autocompletion.
if tp.TYPE_CHECKING:
    CDockWidget = QtWidgets.QWidget
else:
    CDockWidget = ads.CDockWidget

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

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

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SparkGraphEditor:

    def __init__(self):
        # QApplication instance.
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication(sys.argv)
        # Check if the editor was launched in the CLI or is using IPykernel
        self._is_interactive = 'ipykernel' in sys.modules
        if self._is_interactive:
            # Integrate Qt event loop so we don't call exec()
            try:
                from IPython import get_ipython
                #get_ipython().enable_gui('qt')
            except Exception:
                # some rare IPython shells may not support enable_gui; ignore
                pass

    def launch(self) -> None:
        """
            Creates and shows the editor window without blocking.
            This method is safe to call multiple times.
        """
        # If a previous window exists, explicitly delete it (safe)
        if getattr(self, 'window', None):
            self.window.close()
            self.window.deleteLater()
            del self.window

        # Create base window.
        self.window = EditorWindow()
        self.window.windowClosed.connect(self._on_window_closed)
        # Default layout
        self._setup_layout()
        # General style
        self.window.setStyleSheet(
            f"""
                color: {GRAPH_EDITOR_CONFIG.default_font_color};
            """
        )
        self.window.showMaximized()

        # Start loop if instance is not being run in a notebook
        if not self._is_interactive:
            # TODO: This should not terminate python shells.
            sys.exit(self.app.exec_())
        else:
            # TODO: This is a workaround to prevent bugs when using launch twice on the same editor instance. 
            # This, however, is not a nice approach and throws some errors in a IPykernel. Moreover, it leaves
            # an ugly 'X' symbol on the terminal suggesting that the app failed: unacceptable (╯`Д´)╯︵ ┻━┻.
            sys.exit(self.app.exec_())
            pass
            #code = self.app.exec_()
            #import warnings
            #from IPython import get_ipython
            #warnings.filterwarnings("ignore", message="To exit: use 'exit', 'quit', or Ctrl-D.")
            #ip = get_ipython()
            #if ip is not None:
            #    ip._showtraceback = lambda *a, **k: None
            #raise SystemExit(code)


    def _on_window_closed(self,) -> None:
        if not self._is_interactive:
            self.app.quit()

    def _setup_layout(self,) -> None:
        """
            Initialize the default window layout.
        """
        # Main panel
        self.graph_panel = GraphPanel(parent=self.window)
        self.window.dock_manager.setCentralWidget(self.graph_panel)
        self.graph_panel._debug_model()
        # Menu bar
        self.menu_bar = MenuBar()
        self.window.setMenuBar(self.menu_bar)
        # Console panel
        console_panel = QDockPanel('Console', parent=self.window)
        self.window.add_dock_widget(GRAPH_EDITOR_CONFIG.console_panel_pos, console_panel)
        # Nodes
        self.nodes_panel = QDockPanel('Nodes', parent=self.window)
        left_panel = self.window.add_dock_widget(GRAPH_EDITOR_CONFIG.nodes_panel_pos, self.nodes_panel)
        # Parameters
        self.parameters_panel = QDockPanel('Parameters', parent=self.window)
        left_panel.addDockWidget(self.parameters_panel)
        left_panel.setCurrentIndex(0)
        # Inspector
        self.inspector_panel = InspectorPanel(parent=self.window)
        self.window.add_dock_widget(GRAPH_EDITOR_CONFIG.inspector_panel_pos, self.inspector_panel)
        # Status bar
        self.status_bar = StatusBar()
        self.window.setStatusBar(self.status_bar)
        # Setup events
        self._setup_events()

    def _setup_events(self,):
        self.graph_panel.graph.node_selection_changed.connect(self.inspector_panel.on_graph_selection_update)


    def _get_styles(self,):
        import os
        import pathlib as pl
        import glob
        path = pl.Path(os.path.abspath(__file__))
        path = pl.Path(path.parent, 'styles/*.qss')
        style = ''
        for p in glob.glob(str(path)):
            with open(p, 'r') as f:
                style = style + f.read()
        return style

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################