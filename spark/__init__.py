#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# The version of the installed distribution.
import importlib.metadata as _metadata
try:
    __version__ = _metadata.version('spark_snn')
except _metadata.PackageNotFoundError:
    # Imported from a source tree that was never installed.
    __version__ = 'unknown'

# Core
from spark.core.backend import Constant, Variable
from spark.core.payloads import SparkPayload, SpikeArray, CurrentArray, PotentialArray, FloatArray, IntegerArray, BooleanMask
from spark.core.specs import PortSpecs, PortMap, ModuleSpecs
from spark.core.decorators import spark_property as property
from spark.core import tracers
from spark.core import config_validation as validation
from spark.core.backend import jit, scan, eval_shape, split, merge
from spark.core.registry import (
    register_module, register_neuron, register_initializer, register_payload, register_config, register_cfg_validator, register_interface,
    register_neuron_from_config, register_neuron_from_config_file
)

# NN submodule
from spark import nn

# Recording
from spark import recording

# Initialize registry.
from spark.core.registry import REGISTRY
REGISTRY._build()

# Editor
# Imported on first use by __getattr__. The imports below are read by type checkers and editors only.
import typing as _typing
if _typing.TYPE_CHECKING:
    from spark.graph_editor.editor import SparkGraphEditor as GraphEditor
    from spark.graph_editor.runs.viewer import SparkRunViewer as RunViewer

def __getattr__(name: str):
    if name == 'GraphEditor':
        try:
            from spark.graph_editor.editor import SparkGraphEditor
        except ImportError as error:
            raise ImportError(
                'The graph editor is built on PySide6, which this installation does not have. It is asked '
                'for by name: pip install "spark_snn[editor]".'
            ) from error
        globals()['GraphEditor'] = SparkGraphEditor
        return SparkGraphEditor
    if name == 'RunViewer':
        try:
            from spark.graph_editor.runs.viewer import SparkRunViewer
        except ImportError as error:
            raise ImportError(
                'The run viewer is built on PySide6, which this installation does not have. It is asked '
                'for by name: pip install "spark_snn[editor]".'
            ) from error
        globals()['RunViewer'] = SparkRunViewer
        return SparkRunViewer
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

def __dir__() -> list[str]:
    return sorted(set(globals()) | {'GraphEditor', 'RunViewer'})

__all__ = [
    'nn', 
    'recording',
    'tracers', 
    'Constant', 'Variable',
    'SparkPayload', 'SpikeArray', 'CurrentArray', 'PotentialArray', 'FloatArray', 'IntegerArray', 'BooleanMask',
    'PortSpecs', 'PortMap', 'ModuleSpecs',
    'property',
    'validation',
    'jit', 'scan', 'eval_shape', 'split', 'merge',
    'GraphEditor', 'RunViewer',
    'register_module', 'register_neuron', 'register_initializer', 'register_payload', 'register_config', 'register_cfg_validator', 'register_interface',
    'register_neuron_from_config', 'register_neuron_from_config_file',
    'REGISTRY',
]

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################