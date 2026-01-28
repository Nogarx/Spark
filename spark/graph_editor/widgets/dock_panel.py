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
    #visibilityChanged = QtCore.Signal(bool)

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        # Content widget
        _layout = QtWidgets.QVBoxLayout()
        _layout.setContentsMargins(GRAPH_EDITOR_CONFIG.dock_layout_margins)
        _layout.setSpacing(GRAPH_EDITOR_CONFIG.dock_layout_spacing)
        _layout.addStretch()
        _content = QtWidgets.QWidget()
        _content.setLayout(_layout)
        self._in_hierarchy = True
        self.setContent(_content)
        self.setMinimumWidth(GRAPH_EDITOR_CONFIG.dock_min_width)
        self.setMinimumHeight(GRAPH_EDITOR_CONFIG.dock_min_height)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding, QtWidgets.QSizePolicy.Policy.MinimumExpanding)

    def minimumSizeHint(self) -> QtCore.QSize:
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

    def clearWidgets(self,) -> None:
        """
            Removes all widget from the central content widget's layout.
        """
        _layout = self._content.layout()
        for i in reversed(range(_layout.count())): 
            widget = _layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        _layout.addStretch()

    # TODO: Fix this method. It kinda works except when it doesnt, most of the time just looks like a gigantic bug.
    def toggleView(self, dock_manager: ads.CDockManager) -> None:
        if self._in_hierarchy:
            if len(self.parent().dockWidgets()) > 1:
                super().toggleView(False)
            else:
                self.parent().toggleView(False)
            self.setVisible(False)
            self._in_hierarchy = False
        else:
            if not self.parent():
                dock_manager.addDockWidget(ads.NoDockWidgetArea, self)
                self.parent().toggleView(True)
            else:
                self.parent().toggleView(True)
            self.setVisible(True)
            self._in_hierarchy = True
        self.visibilityChanged.emit(self._in_hierarchy)
        
    def showEvent(self, event) -> None:
        self._in_hierarchy = True
        self.visibilityChanged.emit(self._in_hierarchy)
        super().showEvent(event)

    def hideEvent(self, event) -> None:
        self._in_hierarchy = True if len(self.parent().dockWidgets()) > 1 else False
        self.visibilityChanged.emit(self._in_hierarchy)
        super().hideEvent(event)



#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################