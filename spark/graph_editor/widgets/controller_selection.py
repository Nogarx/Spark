#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import typing as tp
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QDialog, QSizePolicy, QFrame
)
from PySide6.QtCore import Qt, Signal, QSize

from spark.graph_editor.styles.manager import STYLES
from spark.graph_editor.styles import resources as icons
from spark.graph_editor.models.controller_profile import ControllerProfile, CONTROLLER_PROFILES

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class ControllerCard(QPushButton):
    """
        Selectable card describing a controller profile.
    """

    def __init__(self, profile: ControllerProfile, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.profile = profile
        self.setObjectName('controllerCard')
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # NOTE: QPushButton computes its size hint from its own text/icon and ignores a child layout, so the
        # card has to report the height of its content itself (see sizeHint/heightForWidth below).
        policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

        layout = QHBoxLayout(self)
        margins = STYLES.get_val('start', 'card_margins', default=[16, 14, 16, 14])
        layout.setContentsMargins(*margins)
        layout.setSpacing(STYLES.get_val('start', 'card_spacing', default=12))

        icon_size = STYLES.get_val('start', 'card_icon_size', default=32)
        icon = QLabel()
        icon.setObjectName('controllerCardIcon')
        icon.setPixmap(icons.get_pixmap(profile.icon, icon_size))
        icon.setFixedSize(icon_size, icon_size)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon, 0, Qt.AlignmentFlag.AlignTop)

        text_widget = QWidget()
        text_layout = QVBoxLayout(text_widget)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(STYLES.get_val('start', 'card_text_spacing', default=3))
        title = QLabel(profile.label)
        title.setObjectName('controllerCardTitle')
        text_layout.addWidget(title)
        self._summary = QLabel(profile.summary)
        self._summary.setObjectName('controllerCardSummary')
        self._summary.setWordWrap(True)
        self._summary.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        text_layout.addWidget(self._summary)
        layout.addWidget(text_widget, 1)

        self._min_height = STYLES.get_val('start', 'card_min_height', default=76)

    #-------------------------------------------------------------------------------------------------------#

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        # A word wrapped summary only knows its height once the width is known.
        return max(self._min_height, self.layout().heightForWidth(width))

    def sizeHint(self) -> QSize:
        hint = self.layout().sizeHint()
        width = self.width() if self.width() > 0 else hint.width()
        return QSize(hint.width(), self.heightForWidth(width))

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ControllerChooser(QWidget):
    """
        Card list of every registered controller profile.
    """

    profile_selected = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(STYLES.get_val('start', 'list_spacing', default=8))
        self._cards: list[ControllerCard] = []
        # NOTE: Driven by the profile registry, so a new controller shows up here without touching the UI.
        for profile in CONTROLLER_PROFILES.values():
            card = ControllerCard(profile)
            card.clicked.connect(lambda _=False, p=profile: self.profile_selected.emit(p))
            layout.addWidget(card)
            self._cards.append(card)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._equalize_heights()

    def _equalize_heights(self) -> None:
        """
            Gives every card the height of the tallest one, so summaries of different lengths still align.
        """
        width = max(1, self.width())
        tallest = max((card.heightForWidth(width) for card in self._cards), default=0)
        for card in self._cards:
            if card.minimumHeight() != tallest:
                card.setMinimumHeight(tallest)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class StartView(QWidget):
    """
        Placeholder shown on the canvas while no model is open.
    """

    model_requested = Signal(object)
    open_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName('startView')

        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignmentFlag.AlignCenter)

        panel = QWidget()
        panel.setObjectName('startPanel')
        panel.setMaximumWidth(STYLES.get_val('start', 'panel_max_width', default=420))
        layout = QVBoxLayout(panel)
        margins = STYLES.get_val('start', 'panel_margins', default=[24, 24, 24, 24])
        layout.setContentsMargins(*margins)
        layout.setSpacing(STYLES.get_val('start', 'panel_spacing', default=12))

        title = QLabel('New model')
        title.setObjectName('startTitle')
        layout.addWidget(title)
        subtitle = QLabel('Every graph is the template of one controller. Pick the one to build.')
        subtitle.setObjectName('startSubtitle')
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        chooser = ControllerChooser()
        chooser.profile_selected.connect(self.model_requested.emit)
        layout.addWidget(chooser)

        separator = QFrame()
        separator.setObjectName('startSeparator')
        separator.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(separator)

        open_btn = QPushButton('Open a session...')
        open_btn.setObjectName('startOpenButton')
        open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        open_btn.setMinimumHeight(STYLES.get_val('start', 'open_min_height', default=32))
        open_btn.clicked.connect(self.open_requested.emit)
        layout.addWidget(open_btn)

        outer.addWidget(panel)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class NewModelDialog(QDialog):
    """
        Controller selection dialog, used by "File > New".
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle('New model')
        self.setMinimumWidth(STYLES.get_val('start', 'dialog_min_width', default=360))
        self.selected_profile: ControllerProfile | None = None

        layout = QVBoxLayout(self)
        margins = STYLES.get_val('start', 'panel_margins', default=[24, 24, 24, 24])
        layout.setContentsMargins(*margins)
        layout.setSpacing(STYLES.get_val('start', 'panel_spacing', default=12))

        subtitle = QLabel('Select the controller to build.')
        subtitle.setObjectName('startSubtitle')
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        chooser = ControllerChooser()
        chooser.profile_selected.connect(self._on_profile_selected)
        layout.addWidget(chooser)

        cancel_btn = QPushButton('Cancel')
        cancel_btn.setMinimumHeight(STYLES.get_val('start', 'open_min_height', default=32))
        cancel_btn.clicked.connect(self.reject)
        layout.addWidget(cancel_btn)

    def _on_profile_selected(self, profile: ControllerProfile) -> None:
        self.selected_profile = profile
        self.accept()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
