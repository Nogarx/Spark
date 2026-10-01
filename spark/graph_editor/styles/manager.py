#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.payloads import SparkPayload

import os
import json
import logging
import typing as tp
import pathlib as pl
from string import Template
from PySide6.QtGui import QColor
from PySide6.QtCore import QSettings, Signal, QObject, QCoreApplication
from PySide6.QtWidgets import QApplication
logger = logging.getLogger('spark')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def _merged(defaults: dict, chosen: dict) -> dict:
    """
        Returns ``chosen`` completed with the values of ``defaults`` it lacks, group by group.
    """
    merged = dict(defaults)
    for key, value in chosen.items():
        merged[key] = _merged(defaults[key], value) if isinstance(value, dict) and isinstance(defaults.get(key), dict) else value
    return merged

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class StyleManager(QObject):
    reloaded = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._config: dict = {}
        self._defaults: dict | None = None
        self._qss = None
        self._default_path = pl.Path(__file__).parent / 'config.json'
        self._active_path = self._default_path

    def init(self) -> None:
        self._ensure_app_identity()
        self._load_config()

    @staticmethod
    def _ensure_app_identity() -> None:
        """
            Gives QSettings somewhere to write.

            Notes
            -----
            Without an organisation and an application name QSettings falls back to
            "Unknown Organization/PySideApp" and reports an access error. The name is set rather than
            defaulted: Qt derives it from the program that was started, which would file the settings under
            whatever launched the editor ("test.ipynb", "-c", a script name).
        """
        QCoreApplication.setOrganizationName('Spark')
        QCoreApplication.setApplicationName('SparkGraphEditor')

    def _flatten_tokens(self) -> dict[str, str]:

        def _to_css(value) -> str:
            if isinstance(value, (list, tuple)):
                if len(value) == 4:
                    r, g, b, a = value
                    return f"rgba({r}, {g}, {b}, {a / 255:.3f})"
                if len(value) == 3:
                    return "rgb({}, {}, {})".format(*value)
            return str(value)

        tokens: dict[str, str] = {}
        for category, entries in self._config.items():
            if not isinstance(entries, dict):
                continue
            for key, value in entries.items():
                tokens[f"{category}_{key}"] = _to_css(value)
        return tokens

    def _resolve_path(self) -> pl.Path:
        custom = QSettings().value('style_config_path', '')
        if custom and pl.Path(custom).exists():
            return pl.Path(custom)
        return self._default_path

    def defaults(self) -> dict:
        """
            Returns the default style, read from ``config.json``.
        """
        if self._defaults is None:
            try:
                self._defaults = json.loads(self._default_path.read_text())
            except Exception:
                logger.error('Default style config unreadable.', exc_info=True)
                self._defaults = {}
        return self._defaults

    def _load_config(self) -> None:
        path = self._resolve_path()
        try:
            chosen = json.loads(path.read_text())
            self._active_path = path
        except Exception:
            logger.warning(f'Failed to load style config from {path}. Loading default config.', exc_info=True)
            chosen = {}
            self._active_path = self._default_path
        # A style saved before a setting existed takes that setting from the defaults.
        self._config = _merged(self.defaults(), chosen)

    def stylesheet(self) -> str:
        if self._qss is None:
            # The stylesheet ships with the editor, only its values are configurable.
            path = self._default_path.parent / 'app.qss'
            self._qss = Template(path.read_text()).safe_substitute(self._flatten_tokens())
        return self._qss

    def apply(self, app: QApplication) -> None:
        app.setStyleSheet(self.stylesheet())

    def reload(self, app: QApplication | None = None) -> None:
        self._qss = None
        self._defaults = None
        self._load_config()
        if app is not None:
            self.apply(app)
        self.reloaded.emit()



    def get_color(self, category, key) -> QColor:
        rgba = self.get_val(category, key, default=[255, 255, 255, 255])
        return QColor(*rgba) if isinstance(rgba, list) else QColor(rgba)

    def get_port_style(self, port_type: type[SparkPayload]) -> dict:
        styles = self._config.get('port', {}).get('type_styles', {})
        return styles.get(str(port_type.__name__), styles.get('default', {'color': '#ffffff', 'shape': 'circle'}))

    def get_val(self, *path: str, default: tp.Any = None) -> dict:
        # Before `init`, the default style.
        v = self._config or self.defaults()
        for p in path[:-1]:
            v = v.get(p, {})
        return v.get(path[-1], default)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

STYLES = StyleManager()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################