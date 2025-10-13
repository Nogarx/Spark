#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import os
import sys
import typing as tp
from Qt import QtWidgets, QtCore, QtGui

from spark.graph_editor.editor_config import GRAPH_EDITOR_CONFIG
from spark.graph_editor.ui.menu_bar import MenuBar
from spark.graph_editor.ui.status_bar import StatusBar
from spark.graph_editor.ui.graph_panel import GraphPanel
from spark.graph_editor.ui.console_panel import ConsolePanel
from spark.graph_editor.ui.inspector_panel import InspectorPanel
from spark.graph_editor.ui.parameters_panel import ParametersPanel


#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class SparkGraphEditor:

    # Singletons.
    app = None
    window = None
    graph = None
    inspector = None
    _current_session_path = None
    _current_model_path = None

    def __init__(self):
        # QApplication instance.
        if SparkGraphEditor.app is None:
            SparkGraphEditor.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

        # Create base window.
        window = QtWidgets.QMainWindow()
        window.setWindowTitle('Spark Graph Editor')
        window.resize(1366, 768)

        window.setMenuBar(MenuBar())

        parameters_panel = ParametersPanel(parent=window)
        window.addDockWidget(GRAPH_EDITOR_CONFIG.parameters_panel_pos, parameters_panel)

        inspector_panel = InspectorPanel(parent=window)
        window.addDockWidget(GRAPH_EDITOR_CONFIG.inspector_panel_pos, inspector_panel)

        console_panel = ConsolePanel(parent=window)
        window.addDockWidget(GRAPH_EDITOR_CONFIG.console_panel_pos, console_panel)

        window.setStatusBar(StatusBar())

        graph_panel = GraphPanel(parent=window)
        window.setCentralWidget(graph_panel)

        window.setCorner(QtCore.Qt.Corner.BottomLeftCorner, QtCore.Qt.DockWidgetArea.LeftDockWidgetArea)
        window.setCorner(QtCore.Qt.Corner.BottomRightCorner, QtCore.Qt.DockWidgetArea.RightDockWidgetArea)

        SparkGraphEditor.window = window

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

    def launch(self) -> None:
        """
            Creates and shows the editor window without blocking.
            This method is safe to call multiple times.
        """
        if self.window.isVisible():
            self.window.activateWindow()
            self.window.raise_()
        else:
            self.window.show()
        if not self.is_interactive():
            sys.exit(SparkGraphEditor.app.exec_())

    def is_interactive(self,):
        """
            Check if the script is running in an interactive environment
            (like a Jupyter notebook or IPython console).
        """
        import __main__ as main
        return not hasattr(main, '__file__')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################