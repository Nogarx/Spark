#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.graph_editor.models.graph_model import GraphModel
    from spark.graph_editor.models.controller_profile import ControllerProfile

import json
import lzma
import logging
import typing as tp
import pathlib as pl
import dataclasses as dc

from spark.core.config import SparkConfig
from spark.core.serializer import SparkJSONEncoder, SparkJSONDecoder
from spark.graph_editor.models.controller_profile import get_controller_profile, profile_for_config
from spark.graph_editor.models.graph_export import build_controller_config

logger = logging.getLogger('spark')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# NOTE: Two file kinds, on purpose.
#   .scfg  a model: a finished controller configuration the framework can instantiate. Written by the core
#          serializer, so it is exactly what SparkConfig.from_file expects and carries nothing else.
#   .sge   a session: work in progress. Same encoding, but the editor owns the layout, and the configuration
#          inside is allowed to be incomplete. Instantiating one is expected to fail.
# The editor state (which controller, where the nodes are) lives in the session only. It cannot travel inside
# the configuration itself: SparkConfig serializes its dataclass fields, so an attribute such as
# __graph_editor_metadata__ is silently dropped by to_file/from_file.

SESSION_SUFFIX = '.sge'
MODEL_SUFFIX = '.scfg'
SESSION_FORMAT = 1

SESSION_FILTER = 'Spark Graph Editor (*.sge);;All Files (*)'
MODEL_FILTER = 'Spark Configuration (*.scfg);;All Files (*)'

_LZMA_MAGIC = b'\xfd7zXZ\x00'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@dc.dataclass
class LoadedSession:
    """
        Contents of a session file.
    """
    profile: tp.Any = None
    config: tp.Any = None
    layout: dict[str, tuple[float, float]] = dc.field(default_factory=dict)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def _read_json(path: pl.Path) -> tp.Any:
    """
        Reads a Spark file, compressed or not.
    """
    with open(path, 'rb') as raw:
        compressed = raw.read(6) == _LZMA_MAGIC
    opener = lzma.open if compressed else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle, cls=SparkJSONDecoder)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _write_json(path: pl.Path, payload: tp.Any, compress: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = lzma.open if compress else open
    with opener(path, 'wt', encoding='utf-8') as handle:
        json.dump(payload, handle, cls=SparkJSONEncoder, indent=4)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def save_session(graph_model: GraphModel, path: str | pl.Path) -> pl.Path:
    """
        Writes the graph as a session, however incomplete it is.

        Returns:
            pl.Path, the path actually written.
    """
    profile = graph_model.profile
    if profile is None:
        raise ValueError('There is no open model to save.')
    path = pl.Path(path).with_suffix(SESSION_SUFFIX)
    exported = build_controller_config(graph_model, strict=False)
    if exported.config is None:
        raise ValueError('; '.join(exported.problems) or 'The graph could not be described.')
    payload = {
        '__spark_session__': SESSION_FORMAT,
        'profile': profile.key,
        'layout': {name: [pos[0], pos[1]] for name, pos in exported.layout.items()},
        'config': exported.config,
    }
    _write_json(path, payload)
    return path

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def load_session(path: str | pl.Path) -> LoadedSession:
    """
        Reads a session file.
    """
    path = pl.Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'No file found at "{path}".')
    payload = _read_json(path)
    if not isinstance(payload, dict) or '__spark_session__' not in payload:
        raise ValueError(f'"{path.name}" is not a Spark Graph Editor session.')
    version = payload.get('__spark_session__')
    if version != SESSION_FORMAT:
        raise ValueError(f'Unsupported session format "{version}", this editor writes version {SESSION_FORMAT}.')
    config = payload.get('config', None)
    if not isinstance(config, SparkConfig):
        raise ValueError(f'"{path.name}" does not contain a controller configuration.')
    # The controller comes from the file, the user is never asked when opening a session.
    profile = get_controller_profile(payload.get('profile', None)) or profile_for_config(config)
    layout = {}
    for name, pos in (payload.get('layout', None) or {}).items():
        if isinstance(pos, (list, tuple)) and len(pos) == 2:
            layout[name] = (float(pos[0]), float(pos[1]))
    return LoadedSession(profile=profile, config=config, layout=layout)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def export_model(graph_model: GraphModel, path: str | pl.Path) -> pl.Path:
    """
        Writes the graph as a model the framework can instantiate.

        Raises:
            ValueError, listing everything that keeps the graph from being a valid model.
    """
    path = pl.Path(path).with_suffix(MODEL_SUFFIX)
    exported = build_controller_config(graph_model, strict=True)
    if not exported.is_complete:
        raise ValueError('\n'.join(exported.problems))
    exported.config.to_file(path, verbose=False)
    return path

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def load_model(path: str | pl.Path) -> tp.Any:
    """
        Reads a model file.
    """
    path = pl.Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'No file found at "{path}".')
    config = SparkConfig.from_file(path)
    if not isinstance(config, SparkConfig):
        raise ValueError(f'"{path.name}" does not contain a Spark configuration.')
    return config

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
