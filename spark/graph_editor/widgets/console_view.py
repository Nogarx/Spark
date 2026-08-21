#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import logging
from shiboken6 import isValid
from datetime import datetime
from enum import Enum
from functools import partial
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QScrollArea, QLabel
from PySide6.QtCore import Qt, QMargins
from PySide6.QtGui import QColor, QPixmap, QPainter
from spark.graph_editor.styles.manager import STYLES
from spark.graph_editor.widgets.scroll_utils import ScrollMarginBalancer

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class MessageLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    SUCCESS = 25
    WARNING = logging.WARNING
    ERROR = logging.ERROR

logging.addLevelName(MessageLevel.SUCCESS.value, 'SUCCESS')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class _MessageWidget(QWidget):
    """
        Visual representation of a single message inside the console.
    """

    def __init__(self, level: MessageLevel, text: str):
        super().__init__()
        self.level = level
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        icon_label = QLabel()
        icon_label.setFixedWidth(18)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        # Visual queues for the messages
        pix = QPixmap(18, 18)
        pix.fill(Qt.GlobalColor.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        color_map = {
            MessageLevel.INFO: QColor(STYLES.get_val('console', 'info_color')),
            MessageLevel.SUCCESS: QColor(STYLES.get_val('console', 'success_color')),
            MessageLevel.WARNING: QColor(STYLES.get_val('console', 'warning_color')),
            MessageLevel.ERROR: QColor(STYLES.get_val('console', 'error_color')),
        }
        color = color_map.get(level, QColor(150, 150, 150))
        p.setBrush(color)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(6, 3, 6, 6)
        p.end()
        icon_label.setPixmap(pix)
        # Text content
        text_label = QLabel(text)
        text_label.setObjectName('consoleMessageText')
        text_label.setWordWrap(True)
        text_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        text_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        text_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        # Layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(icon_label)
        layout.addWidget(text_label)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ConsoleHandler(logging.Handler):
    """
        Logging handler that routes messages to the ConsoleView.
    """
    def __init__(self, console_view: ConsoleView) -> None:
        super().__init__()
        self.console_view = console_view

    def emit(self, record) -> None:
        from shiboken6 import isValid
        if not isValid(self.console_view):
            return
        try:
            msg = self.format(record)
            level = MessageLevel.INFO
            if record.levelno >= logging.ERROR:
                level = MessageLevel.ERROR
            elif record.levelno >= logging.WARNING:
                level = MessageLevel.WARNING
            elif record.levelno == MessageLevel.SUCCESS.value:
                level = MessageLevel.SUCCESS
            self.console_view.add_message(level, msg)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.handleError(record)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ConsoleView(QWidget):
    """
        General console widget connected to the 'spark' logger.
    """
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        # Filter state
        self._filters = {
            MessageLevel.DEBUG: False,
            MessageLevel.INFO: True,
            MessageLevel.SUCCESS: True,
            MessageLevel.WARNING: True,
            MessageLevel.ERROR: True,
        }
        self._messages: list[_MessageWidget] = []
        min_width = STYLES.get_val('console', 'min_width')
        self.setMinimumWidth(min_width)
        self.setMinimumHeight(STYLES.get_val('console', 'min_height', default=185))
        # Widget layout
        layout = QVBoxLayout()
        layout.setContentsMargins(QMargins(0, 0, 0, 0))
        layout.setSpacing(0)
        self.setLayout(layout)
        # Scroll area
        scroll_area = QScrollArea()
        scroll_area.setObjectName('consoleScroll')
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Content widget
        self.content = QWidget()
        self.content.setObjectName('consoleContent')
        scroll_area.setWidget(self.content)
        cm = STYLES.get_val('console', 'content_margins')
        content_layout = QVBoxLayout(self.content)
        content_layout.setContentsMargins(QMargins(*cm))
        content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        # Both gutters stay equal, with or without the vertical scroll bar.
        self._margin_balancer = ScrollMarginBalancer(scroll_area, content_layout, cm)
        content_layout.setSpacing(0)
        # Top button bar
        self._btn_info = self._make_filter_button('Info', MessageLevel.INFO)
        self._btn_success = self._make_filter_button('Success', MessageLevel.SUCCESS)
        self._btn_warning = self._make_filter_button('Warning', MessageLevel.WARNING)
        self._btn_error = self._make_filter_button('Error', MessageLevel.ERROR)
        self._btn_clear = QPushButton('Clear')
        self._btn_clear.clicked.connect(self.clear)
        # Layout for top buttons (left-aligned)
        buttons_layout = QHBoxLayout()
        bm = STYLES.get_val('console', 'btn_bar_margins')
        buttons_layout.setContentsMargins(*bm)
        buttons_layout.setSpacing(STYLES.get_val('console', 'btn_bar_spacing'))
        buttons_layout.addWidget(self._btn_info)
        buttons_layout.addWidget(self._btn_success)
        buttons_layout.addWidget(self._btn_warning)
        buttons_layout.addWidget(self._btn_error)
        buttons_layout.addStretch(1)
        buttons_layout.addWidget(self._btn_clear)
        top_bar = QWidget()
        top_bar.setObjectName('consoleTopBar')
        top_bar.setLayout(buttons_layout)
        self.vscrollbar = scroll_area.verticalScrollBar()
        self.vscrollbar.rangeChanged.connect(self.scrollToBottom)
        # Main layout: top bar above scroll area
        layout.addWidget(top_bar)
        layout.addWidget(scroll_area)
        self._setup_logger()

    def _setup_logger(self) -> None:
        # Set up the 'spark' logger
        self.logger = logging.getLogger('spark')
        self.logger.setLevel(logging.DEBUG)
        # Remove old handlers to avoid memory leaks and publishing to destroyed widgets
        for h in list(self.logger.handlers):
            if isinstance(h, ConsoleHandler):
                self.logger.removeHandler(h)
        handler = ConsoleHandler(self)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        # Stop propagating to root logger to avoid duplicate console output if not desired
        self.logger.propagate = False

    def _make_filter_button(self, label: str, level: MessageLevel) -> QPushButton:
        btn = QPushButton(label)
        btn.setCheckable(True)
        btn.setChecked(True)
        btn.clicked.connect(lambda checked, lvl=level: self._on_filter_toggled(lvl))
        return btn

    def add_message(self, level: MessageLevel, text: str) -> None:
        """
            Add a message to the console.
        """
        # Append timestamp to message
        timestamp = datetime.now().strftime('%H:%M:%S')
        text = f'[{timestamp}] {text}'
        msg = _MessageWidget(level, text)
        # Insert message
        self.content.layout().addWidget(msg)
        self._messages.append(msg)
        # Show/hide according to current filter
        msg.setVisible(self._filters.get(level, True))

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

    def scrollToBottom(self, minimum: int, maximum: int) -> None:
        self.vscrollbar.setValue(maximum)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################