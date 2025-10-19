#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

from functools import partial
import typing as tp
from PySide6 import QtWidgets, QtCore, QtGui
from enum import Enum, auto
from datetime import datetime
import spark.core.utils as utils
from spark.graph_editor.widgets.dock_panel import QDockPanel
from spark.graph_editor.editor_config import GRAPH_EDITOR_CONFIG


#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class MessageLevel(Enum):
    INFO = auto()
    SUCCESS = auto()
    WARNING = auto()
    ERROR = auto()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ConsolePanel(QDockPanel):
    """
        Console panel to show information / errors.
    """

    def __init__(
            self, 
            name: str = 'Console', 
            parent: QtWidgets.QWidget = None, 
            **kwargs
        ) -> None:
        super().__init__(name, parent=parent, **kwargs)
        self._target_node = None
        self._node_config_widget = None
        self.console = Console()
        self.setContent(self.console)

    def clear(self,) -> None:
        self.console.clear()

    def publish_message(self, level: MessageLevel, message: str) -> None:
        self.console.add_message(level, message)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class Console(QtWidgets.QWidget):
    """
        Non-interactable console widget.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        # Filter state: initially all enabled
        self._filters = {
            MessageLevel.INFO: True,
            MessageLevel.SUCCESS: True,
            MessageLevel.WARNING: True,
            MessageLevel.ERROR: True,
        }
        # Keep references to message widgets for filtering/clearing
        self._messages: list[_MessageWidget] = []

        # Widget layout
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(QtCore.QMargins(8, 4, 8, 8))
        layout.setSpacing(0)
        self.setLayout(layout)
        # Scroll area
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(
            f"""
                QScrollArea {{ 
                    background: {GRAPH_EDITOR_CONFIG.input_field_bg_color}; 
                    border: 1px solid {GRAPH_EDITOR_CONFIG.border_color};
                }}
            """
        )
        # Content widget
        self.content = QtWidgets.QWidget()
        self.content.setStyleSheet(
            f"""
                QWidget {{ 
                    background: transparent; 
                }}
            """
        )
        scroll_area.setWidget(self.content)
        content_layout = QtWidgets.QVBoxLayout(self.content)
        content_layout.setContentsMargins(QtCore.QMargins(0, 0, 0, 0))
        content_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        content_layout.setSpacing(0)
        # Top button bar
        self._btn_info = self._make_filter_button("Info", MessageLevel.INFO)
        self._btn_success = self._make_filter_button("Success", MessageLevel.SUCCESS)
        self._btn_warning = self._make_filter_button("Warning", MessageLevel.WARNING)
        self._btn_error = self._make_filter_button("Error", MessageLevel.ERROR)
        self._btn_clear = QtWidgets.QPushButton("Clear")
        self._btn_clear.clicked.connect(self.clear)
        # Layout for top buttons (left-aligned)
        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.setContentsMargins(2, 2, 2, 2)
        buttons_layout.setSpacing(6)
        buttons_layout.addWidget(self._btn_info)
        buttons_layout.addWidget(self._btn_success)
        buttons_layout.addWidget(self._btn_warning)
        buttons_layout.addWidget(self._btn_error)
        buttons_layout.addStretch(1)
        buttons_layout.addWidget(self._btn_clear)
        top_bar = QtWidgets.QWidget()
        top_bar.setLayout(buttons_layout)
        self.vscrollbar = scroll_area.verticalScrollBar()
        self.vscrollbar.rangeChanged.connect(self.scrollToBottom)
        # Main layout: top bar above scroll area
        layout.addWidget(top_bar)
        layout.addWidget(scroll_area)

    def add_message(self, level: MessageLevel, text: str) -> None:
        """
            Add a message to the console.
        """
        # Append timestamp to message
        timestamp = datetime.now().strftime('%H:%M:%S')
        text = f'[{timestamp}] {text}'
        msg = _MessageWidget(level, text, parent=self.content)
        # Insert message
        self.content.layout().addWidget(msg)
        self._messages.append(msg)
        # Show/hide according to current filter
        msg.setVisible(self._filters.get(level, True))

    def info(self, text: str) -> None:
        self.add_message(MessageLevel.INFO, text)

    def success(self, text: str) -> None:
        self.add_message(MessageLevel.SUCCESS, text)

    def warning(self, text: str) -> None:
        self.add_message(MessageLevel.WARNING, text)

    def error(self, text: str) -> None:
        self.add_message(MessageLevel.ERROR, text)

    def clear(self) -> None:
        """
            Remove all messages from the console.
        """
        for w in self._messages:
            w.setParent(None)
            w.deleteLater()
        self._messages.clear()
        while self.content.layout().count():
            item = self.content.layout().takeAt(0)
            if not item:
                break
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        #self.content.layout().addStretch(1)


    def _make_filter_button(self, label: str, level: MessageLevel) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton(label)
        btn.setCheckable(True)
        btn.setChecked(True)
        btn.clicked.connect(partial(lambda level_var: self._on_filter_toggled(level_var), level))
        return btn

    def _on_filter_toggled(self, level: MessageLevel) -> None:
        # Read button state
        enabled = False
        if level == MessageLevel.INFO:
            enabled = self._btn_info.isChecked()
        elif level == MessageLevel.SUCCESS:
            enabled = self._btn_success.isChecked()
        elif level == MessageLevel.WARNING:
            enabled = self._btn_warning.isChecked()
        elif level == MessageLevel.ERROR:
            enabled = self._btn_error.isChecked()
        self._filters[level] = enabled
        # Update visibility of existing messages
        for w in self._messages:
            if w.level == level:
                w.setVisible(enabled)

    def scrollToBottom(self, minimum, maximum):
        self.vscrollbar.setValue(maximum)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class _MessageWidget(QtWidgets.QWidget):
    """
    Visual representation of a single message inside the console.
    Not interactable (no focus); contains small level icon and a wrapping text label.
    """
    def __init__(self, level: MessageLevel, text: str, parent=None):
        super().__init__(parent)
        self.level = level
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        icon_label = QtWidgets.QLabel()
        icon_label.setFixedWidth(18)
        icon_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignHCenter)
        # Visual queues for the messages
        pix = QtGui.QPixmap(18, 18)
        pix.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(pix)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        color_map = {
            MessageLevel.INFO: QtGui.QColor(0, 120, 215),     # blue
            MessageLevel.SUCCESS: QtGui.QColor(40, 160, 80),  # green
            MessageLevel.WARNING: QtGui.QColor(235, 138, 0),  # orange
            MessageLevel.ERROR: QtGui.QColor(205, 40, 40),    # red
        }
        color = color_map.get(level, QtGui.QColor(150, 150, 150))
        p.setBrush(color)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawEllipse(6, 3, 6, 6)
        p.end()
        icon_label.setPixmap(pix)
        # Text content
        text_label = QtWidgets.QLabel(text)
        text_label.setWordWrap(True)
        text_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.NoTextInteraction)
        text_label.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        text_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        # Layout
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(2)
        layout.addWidget(icon_label)
        layout.addWidget(text_label)
        # Style
        self.setStyleSheet(
            f"""
                font-size: {GRAPH_EDITOR_CONFIG.small_font_size}px;
            """
        )

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################