#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

from PySide6.QtWidgets import QScrollArea, QLayout
from PySide6.QtCore import QObject, QEvent

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class ScrollMarginBalancer(QObject):
    """
        Keeps the content of a scroll area optically centered.

        A QScrollArea reserves the vertical scroll bar outside of its viewport, so the gap on the right of
        the content is the right margin plus the width of the bar. The right margin is reduced by the width
        of the bar while it is visible, leaving both gutters equal.
    """

    def __init__(self, scroll_area: QScrollArea, layout: QLayout, margins: tuple[int, int, int, int]) -> None:
        super().__init__(scroll_area)
        self._scroll_area = scroll_area
        self._layout = layout
        self._margins = tuple(margins)
        scroll_area.verticalScrollBar().rangeChanged.connect(lambda *_: self.apply())
        scroll_area.installEventFilter(self)
        self.apply()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() in (QEvent.Type.Show, QEvent.Type.Resize):
            self.apply()
        return super().eventFilter(watched, event)

    def apply(self) -> None:
        """
            Recomputes the right margin from the current scroll bar state.
        """
        bar = self._scroll_area.verticalScrollBar()
        # NOTE: The visibility flag still reports the previous state here, Qt updates it later in the same
        # layout pass. The scroll range is the criterion Qt uses to decide whether the bar is needed.
        is_needed = bar.maximum() > bar.minimum()
        reserved = bar.sizeHint().width() if is_needed else 0
        left, top, right, bottom = self._margins
        right = max(0, right - reserved)
        if self._layout.contentsMargins().right() != right:
            self._layout.setContentsMargins(left, top, right, bottom)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
