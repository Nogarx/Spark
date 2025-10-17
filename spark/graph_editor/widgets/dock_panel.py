#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
    
import typing as tp
from PySide6 import QtWidgets, QtCore, QtGui
import PySide6QtAds as ads

from spark.graph_editor.editor_config import GRAPH_EDITOR_CONFIG

# NOTE: Small workaround to at least have base autocompletion.
if tp.TYPE_CHECKING:
    CDockWidget = QtWidgets.QWidget
else:
    CDockWidget = ads.CDockWidget

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class QDockPanel(CDockWidget):
    """
        Generic dockable panel.
    """

    def __init__(self, name: str, parent: QtWidgets.QWidget = None, **kwargs):
        super().__init__(name, parent=parent, **kwargs)
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        # Content widget
        _layout = QtWidgets.QVBoxLayout()
        _layout.setContentsMargins(GRAPH_EDITOR_CONFIG.dock_layout_margins)
        _layout.setSpacing(GRAPH_EDITOR_CONFIG.dock_layout_spacing)
        _layout.addStretch()
        _content = QtWidgets.QWidget()
        _content.setLayout(_layout)
        self.setContent(_content)
        self.setMinimumWidth(GRAPH_EDITOR_CONFIG.dock_min_width)
        self.setMinimumHeight(GRAPH_EDITOR_CONFIG.dock_min_height)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding, QtWidgets.QSizePolicy.Policy.MinimumExpanding)

    def minimumSizeHint(self):
        return QtCore.QSize(self.minimumWidth(), self.minimumHeight())


    def setContent(self, content: QtWidgets.QWidget) -> None:
        """
            Replace central widget.
        """
        # Queue current content for deletion
        if getattr(self, '_content', None):
            self._content.deleteLater()
        self._content = content
        self.layout().addWidget(self._content)

    def content(self) -> QtWidgets.QWidget | None:
        """
            Return the current content widget.
        """
        return getattr(self, '_content', None)
    
    def addWidget(self, widget: QtWidgets.QWidget) -> None:
        """
            Add a widget to the central content widget's layout.
        """
        widget.installEventFilter(self)
        index = self._content.layout().count() - 1
        self._content.layout().insertWidget(index, widget)

    def removeWidget(self, widget: QtWidgets.QWidget) -> None:
        """
            Remove widget from the central content widget's layout.
        """
        self._content.layout().removeWidget(widget)
        widget.removeEventFilter(self)

    def clearWidgets(self,):
        """
            Removes all widget from the central content widget's layout.
        """
        _layout = self._content.layout()
        for i in reversed(range(_layout.count())): 
            widget = _layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        _layout.addStretch()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################