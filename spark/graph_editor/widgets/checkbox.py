#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
    
import typing as tp
from PySide6 import QtWidgets, QtCore, QtGui
from spark.graph_editor.editor_config import GRAPH_EDITOR_CONFIG
import spark.graph_editor.style.resources_rc

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class WarningFlag(QtWidgets.QPushButton):
    """
        Toggleable QPushButton used to indicate that an attribute in a SparkConfig may have conflicts.
    """
    def __init__(
            self, 
            value: bool = False,
            parent: QtWidgets.QWidget = None,
        ) -> None:
        super().__init__(None, parent)
        # Warning icon
        warning_pixmap = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning)
        self.warning_icon = warning_pixmap.pixmap(12, 12)
        self.tooltip_message = None
        # Empty icon
        empty_pixmap = QtGui.QPixmap(QtCore.QSize(12, 12))
        empty_pixmap.fill(QtGui.QColor(0, 0, 0, 0))
        self.empty_icon = empty_pixmap
        # State-aware QIcon
        self.toggle_icon = QtGui.QIcon()
        self.toggle_icon.addPixmap(self.warning_icon, QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.On)
        self.toggle_icon.addPixmap(self.empty_icon, QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        # Format ToggleButton
        self.setFixedSize(QtCore.QSize(16, 16))
        self.setContentsMargins(QtCore.QMargins(0, 0, 0, 0))
        self.setIconSize(QtCore.QSize(12, 12))
        # Initialize button
        self.setCheckable(True) 
        self.setIcon(self.toggle_icon)
        self.setChecked(value)
        self.toolTipDuration()
        self.setStyleSheet(
            f"""
                QPushButton {{
                    background-color: transparent; 
                    border: none; 
                }}
                QToolTip {{
                    background-color: {GRAPH_EDITOR_CONFIG.primary_bg_color};
                    color: {GRAPH_EDITOR_CONFIG.default_font_color};
                    border-radius: 4px;
                }}
            """
        )

    def set_error_status(self, messages: list[str]):
        if len(messages) > 0:
            self.setChecked(True)
            self.tooltip_message = '\n'.join(messages)
        else:
            self.setChecked(False)
            self.tooltip_message = None

    # Ignore mouse events.
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        return

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        return

    def enterEvent(self, event: QtCore.QEvent) -> None:
        # Show the tooltip immediately at the current cursor position
        if self.tooltip_message:
            QtWidgets.QToolTip.showText(self.mapToGlobal(self.pos()), self.tooltip_message, self) 

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        # Hide the tooltip
        QtWidgets.QToolTip.hideText()
#-----------------------------------------------------------------------------------------------------------------------------------------------#

class InheritToggleButton(QtWidgets.QPushButton):
    """
        Toggleable QPushButton used to indicate that an attribute in a SparkConfig is inheriting it's value to child attributes.
    """
    def __init__(
            self, 
            value: bool = False,
            interactable: bool = True,
            parent: QtWidgets.QWidget = None,
        ) -> None:
        super().__init__(None, parent)
        # Icons
        self.link_icon = QtGui.QPixmap(':/icons/link_icon.png')
        self.lock_icon = QtGui.QPixmap(':/icons/lock_icon.png')
        empty_pixmap = QtGui.QPixmap(QtCore.QSize(12, 12))
        empty_pixmap.fill(QtGui.QColor(0, 0, 0, 0))
        self.unlink_icon = empty_pixmap

        # State-aware QIcon
        self.toggle_icon = QtGui.QIcon()
        self.toggle_icon.addPixmap(self.link_icon, QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.On)
        self.toggle_icon.addPixmap(self.unlink_icon, QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        # Format ToggleButton
        self.setFixedSize(QtCore.QSize(16, 16))
        self.setContentsMargins(QtCore.QMargins(0, 0, 0, 0))
        self.setIconSize(QtCore.QSize(12, 12))
        # Initialize button
        self.setCheckable(interactable) 
        self.setIcon(self.toggle_icon)
        self.setChecked(value)

        if interactable:
            self.setStyleSheet(
                f"""
                    QPushButton {{
                        background-color: {GRAPH_EDITOR_CONFIG.input_field_bg_color};
                        border-radius: {GRAPH_EDITOR_CONFIG.input_field_border_radius}px;
                    }}
                """
            )
        else:
            self.setStyleSheet(
                f"""
                    QPushButton {{
                        background-color: {GRAPH_EDITOR_CONFIG.primary_bg_color};
                        border-radius: {GRAPH_EDITOR_CONFIG.input_field_border_radius}px;
                    }}
                """
            )

    def _set_virtual_icon_state(self, state: bool = True) -> None:
        """
            Set the current icon to the icon associated with state.

            Internal method for visual feedback that avoids triggering events.
        """
        self.setIcon(QtGui.QIcon(self.lock_icon if state else self.unlink_icon))

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################