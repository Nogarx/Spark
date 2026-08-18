#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import logging
import pathlib as pl
from PySide6.QtCore import QSettings

from spark.graph_editor.styles.manager import STYLES

logger = logging.getLogger('spark')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# NOTE: Sessions and models share one list, in the order they were last touched. Which of the two a path is
# follows from its suffix, so nothing else has to be remembered about it.

SETTINGS_KEY = 'recent_files'

MAX_RECENT = 8
"""
    Number of files kept. Old enough entries are of no use, and a menu that scrolls is worse than a short one.
"""

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _settings() -> QSettings:
    """
        Settings to read the list from.

        NOTE: The identity is ensured here rather than assumed. Without an organisation and an application
        name QSettings writes somewhere else entirely, so a list built before the styles were initialised
        would be written to one place and read back from another.
    """
    STYLES._ensure_app_identity()
    return QSettings()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _stored() -> list[str]:
    """
        Raw list as it sits in the settings.
    """
    value = _settings().value(SETTINGS_KEY, [])
    # NOTE: A single entry comes back as a plain string on some platforms.
    if isinstance(value, str):
        return [value] if value else []
    if value is None:
        return []
    return [str(entry) for entry in value]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _store(paths: list[str]) -> None:
    _settings().setValue(SETTINGS_KEY, paths[:MAX_RECENT])

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def recent_files(existing_only: bool = True) -> list[pl.Path]:
    """
        Files opened or written recently, most recent first.

        Args:
            existing_only: bool, drops the entries that are no longer on disk.

        Returns:
            list[pl.Path], the remembered files.
    """
    paths = [pl.Path(entry) for entry in _stored()]
    if not existing_only:
        return paths
    return [path for path in paths if path.exists()]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def remember(path: str | pl.Path) -> None:
    """
        Puts a file at the top of the list, moving it there if it was already known.
    """
    try:
        resolved = str(pl.Path(path).resolve())
    except OSError as error:
        logger.debug(f'"{path}" was not remembered: {error}')
        return
    entries = [entry for entry in _stored() if entry != resolved]
    _store([resolved] + entries)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def forget(path: str | pl.Path) -> None:
    """
        Drops a file from the list. Used when it turns out not to be there anymore.
    """
    resolved = str(pl.Path(path).resolve())
    _store([entry for entry in _stored() if entry != resolved])

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def clear() -> None:
    """
        Forgets every file.
    """
    _settings().remove(SETTINGS_KEY)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
