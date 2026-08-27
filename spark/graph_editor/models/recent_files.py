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

SETTINGS_KEY = 'recent_files'

MAX_RECENT = 8
"""
    Number of files kept.
"""

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _settings() -> QSettings:
    """
        Settings to read the list from.
    """
    STYLES._ensure_app_identity()
    return QSettings()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _stored() -> list[str]:
    """
        Raw list as it sits in the settings, without the empty and repeated entries.
    """
    value = _settings().value(SETTINGS_KEY, [])
    # NOTE: A single entry comes back as a plain string on some platforms.
    if isinstance(value, str):
        value = [value] if value else []
    elif value is None:
        value = []
    entries = []
    for entry in value:
        entry = str(entry).strip()
        if entry and entry not in entries:
            entries.append(entry)
    return entries

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _store(paths: list[str]) -> None:
    _settings().setValue(SETTINGS_KEY, paths[:MAX_RECENT])

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _entry(path: str | pl.Path) -> str | None:
    """
        The string a path is remembered as, or None when it cannot be one.
    """
    try:
        return str(pl.Path(path).expanduser().resolve())
    except (OSError, ValueError, RuntimeError) as error:
        logger.debug(f'"{path}" is not a path that can be remembered: {error}')
        return None

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _is_file(entry: str) -> bool:
    """
        Whether an entry still points at a file that can be opened.
    """
    try:
        return pl.Path(entry).is_file()
    except OSError as error:
        logger.debug(f'"{entry}" could not be looked up: {error}')
        return False

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def recent_files(existing_only: bool = True) -> list[pl.Path]:
    """
        Files opened or written recently, most recent first.

        Args:
            existing_only: bool, drops the entries that are no longer on disk.

        Returns:
            list[pl.Path], the remembered files.
    """
    entries = _stored()
    if not existing_only:
        return [pl.Path(entry) for entry in entries]
    kept = [entry for entry in entries if _is_file(entry)]
    if len(kept) != len(entries):
        _store(kept)
    return [pl.Path(entry) for entry in kept]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def remember(path: str | pl.Path) -> None:
    """
        Puts a file at the top of the list, moving it there if it was already known.
    """
    resolved = _entry(path)
    if resolved is None:
        return
    entries = [entry for entry in _stored() if entry != resolved]
    _store([resolved] + entries)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def forget(path: str | pl.Path) -> None:
    """
        Drops a file from the list. The path is dropped both as given and as resolved.
    """
    dropped = {str(path), str(pl.Path(path))}
    resolved = _entry(path)
    if resolved is not None:
        dropped.add(resolved)
    _store([entry for entry in _stored() if entry not in dropped])

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def clear() -> None:
    """
        Forgets every file.
    """
    _settings().remove(SETTINGS_KEY)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
