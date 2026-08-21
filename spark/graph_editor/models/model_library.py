#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import shutil
import logging
import pathlib as pl
import typing as tp
from PySide6.QtCore import QSettings, QStandardPaths

from spark.core.registry import REGISTRY, register_neuron_from_config_file
from spark.graph_editor.styles.manager import STYLES

logger = logging.getLogger('spark')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

SETTINGS_KEY = 'model_library_path'

SUFFIX = '.scfg'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _settings() -> QSettings:
    """
        Settings to read the location from.
    """
    STYLES._ensure_app_identity()
    return QSettings()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def default_path() -> pl.Path:
    """
        Returns the location of the default editor's model library.
    """
    STYLES._ensure_app_identity()
    root = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    return pl.Path(root or pl.Path.home() / '.spark') / 'models'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def library_path() -> pl.Path:
    """
        Returns the location of the editor's model library.
    """
    stored = _settings().value(SETTINGS_KEY, '')
    return pl.Path(str(stored)).expanduser() if stored else default_path()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def set_library_path(path: str | pl.Path | None) -> None:
    """
        Sets the location of the editor's model library.

        Args:
            path: str | pl.Path | None, the location.
    """
    if not path:
        _settings().remove(SETTINGS_KEY)
        return
    _settings().setValue(SETTINGS_KEY, str(pl.Path(path).expanduser()))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def model_files(path: str | pl.Path | None = None) -> list[pl.Path]:
    """
        Returns a list of model files in the editor's model library.

        Args:
            path: str | pl.Path | None, a location to read instead of the chosen one.

        Returns:
            list[pl.Path], the files.
    """
    root = pl.Path(path).expanduser() if path else library_path()
    if not root.is_dir():
        return []
    return sorted(entry for entry in root.iterdir() if entry.is_file() and entry.suffix == SUFFIX)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def model_name(path: str | pl.Path) -> str:
    """
        Returns the the name of the file.
    """
    return pl.Path(path).stem

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def read_model(path: str | pl.Path) -> tp.Any:
    """
        Reads a file as a model

        Args:
            path: str | pl.Path, the file to read.

        Returns:
            NeuronConfig, the configuration the file holds.
    """
    from spark.nn.controllers.neuron import NeuronConfig
    config = NeuronConfig.from_file(pl.Path(path))
    if not isinstance(config, NeuronConfig):
        raise TypeError(
            f'A model of the library must be a Neuron, but "{pl.Path(path).name}" holds a '
            f'"{type(config).__name__}".'
        )
    return config

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def register_file(path: str | pl.Path) -> str | None:
    """
        Registers a model from a file

        Args:
            path: str | pl.Path, the file to read.

        Returns:
            str | None, the name the model answers to, or nothing if it was already taken.
    """
    path = pl.Path(path)
    name = model_name(path)
    if REGISTRY.Neurons.get(name):
        logger.debug(f'"{path.name}" was not read: a model named "{name}" is already available.')
        return None
    register_neuron_from_config_file(name, path)
    return name

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def register_library(path: str | pl.Path | None = None) -> tuple[list[str], list[tuple[pl.Path, str]]]:
    """
        Registers all models in the editor's model library.

        Args:
            path: str | pl.Path | None, a location to read instead of the chosen one.

        Returns:
            tuple[list[str], list[tuple[pl.Path, str]]], the names now available and the files that failed.
    """
    registered: list[str] = []
    failed: list[tuple[pl.Path, str]] = []
    for file_path in model_files(path):
        try:
            name = register_file(file_path)
        except Exception as error:
            failed.append((file_path, str(error)))
            logger.warning(f'Unable to read the model "{file_path.name}". Error: {error}.')
            continue
        if name:
            registered.append(name)
    return registered, failed

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def import_model(source: str | pl.Path, overwrite: bool = False) -> pl.Path:
    """
        Appends a model file to the editor's model library.

        Args:
            source: str | pl.Path, the file to take in.
            overwrite: bool, whether a file of that name already in the library may be replaced.

        Returns:
            pl.Path, where the copy was left.
    """
    source = pl.Path(source).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f'No file found at the specified path: "{source}".')
    read_model(source)
    root = library_path()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f'{model_name(source)}{SUFFIX}'
    if destination.exists() and not overwrite:
        raise FileExistsError(f'The library already holds a model named "{model_name(source)}".')
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)
    try:
        register_file(destination)
    except Exception as error:
        logger.warning(f'"{destination.name}" was copied but could not be read back. Error: {error}.')
    return destination

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
