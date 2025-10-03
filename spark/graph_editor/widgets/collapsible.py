#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# Code taken and adapted from https://github.com/pyapp-kit/superqt

from __future__ import annotations

from Qt import QtCore, QtWidgets, QtGui
from spark.graph_editor.editor_config import GRAPH_EDITOR_CONFIG

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# TODO: This code requires a revision, particularly for the case of nested QCollapsible.
# Current implementation works by the grace of god.

class QCollapsible(QtWidgets.QFrame):
    """
        A collapsible widget to hide and unhide child widgets.
        A signal is emitted when the widget is expanded (True) or collapsed (False).
    """

    toggled = QtCore.Signal(bool)
    animationDone = QtCore.Signal()

    def __init__(
        self,
        title: str = '',
        parent: QtWidgets.QWidget | None = None,
        expandedIcon: QtGui.QIcon | str | None = QtWidgets.QStyle.StandardPixmap.SP_ArrowDown,
        collapsedIcon: QtGui.QIcon | str | None = QtWidgets.QStyle.StandardPixmap.SP_ArrowRight,
    ):
        super().__init__(parent)
        self._locked = False
        self._expanded = False
        self._is_animating = False
        self._title = title

        self._toggle_btn = QtWidgets.QPushButton(title)
        self._toggle_btn.setCheckable(True)
        font = self._toggle_btn.font()
        font.setPointSize(13)
        font.setBold(True)
        self._toggle_btn.setFont(font)

        expandedIcon = self.style().standardIcon(expandedIcon)
        collapsedIcon = self.style().standardIcon(collapsedIcon)
        self.setCollapsedIcon(icon=collapsedIcon)
        self.setExpandedIcon(icon=expandedIcon)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)

        self._toggle_btn.setStyleSheet("text-align: left; border: none; outline: none;")
        self._toggle_btn.toggled.connect(self._toggle)

        # frame layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._toggle_btn)
        
        # Create animators
        self._animation = QtCore.QPropertyAnimation(self)
        self._animation.setPropertyName(b"maximumHeight")
        self._animation.setStartValue(0)
        self._animation.finished.connect(self._on_animation_done)
        self.setDuration(300)
        self.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)

        # default content widget
        _content = QtWidgets.QWidget()
        _content.setLayout(QtWidgets.QVBoxLayout())
        _content.setMaximumHeight(0)
        self.setContent(_content)

        # Style
        self.setContentsMargins(GRAPH_EDITOR_CONFIG.section_margin)
        self.setStyleSheet(f'background-color: {GRAPH_EDITOR_CONFIG.section_bg_color};\
                             border-radius: {GRAPH_EDITOR_CONFIG.section_border_radius}px;')
        #self.expand()

    def toggleButton(self) -> QtWidgets.QPushButton:
        """
            Return the toggle button.
        """
        return self._toggle_btn

    def setText(self, text: str) -> None:
        """
            Set the text of the toggle button.
        """
        self._toggle_btn.setText(text)

    def text(self) -> str:
        """
            Return the text of the toggle button.
        """
        return self._toggle_btn.text()

    def setContent(self, content: QtWidgets.QWidget) -> None:
        """
            Replace central widget (the widget that gets expanded/collapsed).
        """
        self._content = content
        self.layout().addWidget(self._content)
        self._animation.setTargetObject(content)

    def content(self) -> QtWidgets.QWidget:
        """
            Return the current content widget.
        """
        return self._content

    def _convert_string_to_icon(self, symbol: str) -> QtGui.QIcon:
        """
            Create a QtGui.QIcon from a string.
        """
        size = self._toggle_btn.font().pointSize()
        pixmap = QtGui.QPixmap(size, size)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        color = self._toggle_btn.palette().color(QtGui.QPalette.ColorRole.WindowText)
        painter.setPen(color)
        painter.drawText(QtCore.QRect(0, 0, size, size), QtCore.Qt.AlignmentFlag.AlignCenter, symbol)
        painter.end()
        return QtGui.QIcon(pixmap)

    def expandedIcon(self) -> QtGui.QIcon:
        """
            Returns the icon used when the widget is expanded.
        """
        return self._expanded_icon

    def setExpandedIcon(self, icon: QtGui.QIcon | str | None = None) -> None:
        """
            Set the icon on the toggle button when the widget is expanded.
        """
        if icon and isinstance(icon, QtGui.QIcon):
            self._expanded_icon = icon
        elif icon and isinstance(icon, str):
            self._expanded_icon = self._convert_string_to_icon(icon)

        if self._expanded:
            self._toggle_btn.setIcon(self._expanded_icon)

    def collapsedIcon(self) -> QtGui.QIcon:
        """
            Returns the icon used when the widget is collapsed.
            """
        return self._collapsed_icon

    def setCollapsedIcon(self, icon: QtGui.QIcon | str | None = None) -> None:
        """
            Set the icon on the toggle button when the widget is collapsed.
        """
        if icon and isinstance(icon, QtGui.QIcon):
            self._collapsed_icon = icon
        elif icon and isinstance(icon, str):
            self._collapsed_icon = self._convert_string_to_icon(icon)

        if not self._expanded:
            self._toggle_btn.setIcon(self._collapsed_icon)

    def setDuration(self, msecs: int) -> None:
        """
            Set duration of the collapse/expand animation.
        """
        self._animation.setDuration(msecs)

    def setEasingCurve(self, easing: QtCore.QEasingCurve | QtCore.QEasingCurve.Type) -> None:
        """
            Set the easing curve for the collapse/expand animation.
        """
        self._animation.setEasingCurve(easing)

    def addWidget(self, widget: QtWidgets.QWidget) -> None:
        """
            Add a widget to the central content widget's layout.
        """
        self._content.layout().addWidget(widget)
        if isinstance(widget, QCollapsible):
            widget.animationDone.connect(self._resize_widget)
        else:
            widget.installEventFilter(self)

    def removeWidget(self, widget: QtWidgets.QWidget) -> None:
        """
            Remove widget from the central content widget's layout.
        """
        self._content.layout().removeWidget(widget)
        if isinstance(widget, QCollapsible):
            widget.animationDone.disconnect(self._resize_widget)
        else:
            widget.removeEventFilter(self)

    def expand(self, animate: bool = False) -> None:
        """
            Expand (show) the collapsible section.
        """
        self._expanded = True
        self._expand_collapse(QtCore.QPropertyAnimation.Direction.Forward, animate)

    def collapse(self, animate: bool = False) -> None:
        """
            Collapse (hide) the collapsible section.
        """
        self._expanded = False
        self._expand_collapse(QtCore.QPropertyAnimation.Direction.Backward, animate)

    def isExpanded(self) -> bool:
        """
            Return whether the collapsible section is visible.
        """
        return self._expanded

    def setLocked(self, locked: bool = True) -> None:
        """
            Set whether collapse/expand is disabled.
        """
        self._locked = locked
        self._toggle_btn.setCheckable(not locked)

    def locked(self) -> bool:
        """
            Return True if collapse/expand is disabled.
        """
        return self._locked

    def _resize_widget(
        self,
    ) -> None:
        """
            Set values for the widget based on whether it is expanding or collapsing.

            An emit flag is included so that the toggle signal is only called once (it
            was being emitted a few times via eventFilter when the widget was expanding
            previously).
        """
        _content_height = self._content.sizeHint().height() #+ 10
        self._content.setMaximumHeight(_content_height if self._expanded else 0)
        self._on_animation_done()  

    def _expand_collapse(
        self,
        direction: QtCore.QPropertyAnimation.Direction,
        animate: bool = False,
        emit: bool = True,
    ) -> None:
        """
            Set values for the widget based on whether it is expanding or collapsing.

            An emit flag is included so that the toggle signal is only called once (it
            was being emitted a few times via eventFilter when the widget was expanding
            previously).
        """
        if self._locked:
            return

        forward = direction == QtCore.QPropertyAnimation.Direction.Forward
        icon = self._expanded_icon if forward else self._collapsed_icon
        self._toggle_btn.setIcon(icon)

        _content_height = self._content.sizeHint().height() #+ 10
        if animate:
            self._animation.setDirection(direction)
            self._animation.setEndValue(_content_height)
            self._is_animating = True
            self._animation.start()
        else:
            self._content.setMaximumHeight(_content_height if self._expanded else 0)
            self._on_animation_done()
        if emit:
            self.toggled.emit(direction == QtCore.QPropertyAnimation.Direction.Forward)

    def _toggle(self) -> None:
        self.expand() if not self._expanded else self.collapse()

    def eventFilter(self, a0: QtCore.QObject, a1: QtCore.QEvent) -> bool:
        """
            If a child widget resizes, we need to update our expanded height.
        """
        if (
            a1.type() == QtCore.QEvent.Type.Resize
            and self._expanded
            and not self._is_animating
        ):
            self._expand_collapse(
                QtCore.QPropertyAnimation.Direction.Forward, animate=False, emit=False
            )
        return False

    def _on_animation_done(self) -> None:
        self._is_animating = False
        _content_height = self._content.sizeHint().height()
        self._content.setMaximumHeight(_content_height if self._expanded else 0)
        self.updateGeometry()
        self.animationDone.emit()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################