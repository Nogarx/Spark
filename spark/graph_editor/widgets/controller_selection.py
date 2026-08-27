#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import typing as tp
import pathlib as pl
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
        # card reports the height of its content itself (see sizeHint/heightForWidth below).
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
        # Driven by the profile registry.
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
            Gives every card the height of the tallest one, aligning summaries of different lengths.
        """
        width = max(1, self.width())
        tallest = max((card.heightForWidth(width) for card in self._cards), default=0)
        for card in self._cards:
            if card.minimumHeight() != tallest:
                card.setMinimumHeight(tallest)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class RecentCard(QPushButton):
    """
        One remembered file, shown on the start screen.
    """

    def __init__(self, path: pl.Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.path = path
        self.setObjectName('recentCard')
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(str(path))
        self.setMinimumHeight(STYLES.get_val('start', 'open_min_height', default=32))

        layout = QHBoxLayout(self)
        margins = STYLES.get_val('start', 'recent_margins', default=[10, 4, 10, 4])
        layout.setContentsMargins(*margins)
        layout.setSpacing(STYLES.get_val('start', 'card_spacing', default=12))
        name = QLabel(path.stem)
        name.setObjectName('recentCardName')
        layout.addWidget(name)
        layout.addStretch(1)
        # The suffix is what tells a session from a model, so it is what is shown.
        kind = QLabel(path.suffix.lstrip('.'))
        kind.setObjectName('recentCardKind')
        layout.addWidget(kind)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class StartView(QWidget):
    """
        Start screen, shown on the canvas while no model is open.
    """

    model_requested = Signal(object)
    open_requested = Signal()
    recent_requested = Signal(object)

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

        # Recent files. The section is only shown while it holds something.
        self._recent_title = QLabel('Recent')
        self._recent_title.setObjectName('startRecentTitle')
        layout.addWidget(self._recent_title)
        self._recent_box = QWidget()
        self._recent_layout = QVBoxLayout(self._recent_box)
        self._recent_layout.setContentsMargins(0, 0, 0, 0)
        self._recent_layout.setSpacing(STYLES.get_val('start', 'recent_spacing', default=4))
        layout.addWidget(self._recent_box)
        self.set_recent_files([])

        outer.addWidget(panel)

    #-------------------------------------------------------------------------------------------------------#

    def set_recent_files(self, paths: tp.Sequence[pl.Path]) -> None:
        """
            Shows the remembered files, most recent first.
        """
        while self._recent_layout.count():
            item = self._recent_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # NOTE: Detached before being destroyed. Deletion is deferred, and the old cards are children of
                # the panel until it happens.
                widget.setParent(None)
                widget.deleteLater()
        shown = list(paths)[:STYLES.get_val('start', 'recent_max', default=5)]
        for path in shown:
            card = RecentCard(path)
            card.clicked.connect(lambda _checked=False, target=path: self.recent_requested.emit(target))
            self._recent_layout.addWidget(card)
        self._recent_title.setVisible(bool(shown))
        self._recent_box.setVisible(bool(shown))

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
