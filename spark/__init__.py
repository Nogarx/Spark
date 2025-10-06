version = '0.1'

# Core
from spark.core.variables import Constant, Variable
from spark.core.payloads import SparkPayload, SpikeArray, CurrentArray, PotentialArray, FloatArray, IntegerArray, BooleanMask
from spark.core.specs import PortSpecs, InputSpec, OutputSpec, PortMap, ModuleSpecs
from spark.core.shape import Shape, ShapeCollection
from spark.core import tracers
from spark.core import config_validation as validation
from spark.core.flax_imports import split, merge
from spark.core.registry import register_module, register_initializer, register_payload, register_config, register_cfg_validator

# NN submodule
from spark import nn

# Initialize registry.
from spark.core.registry import REGISTRY
REGISTRY._build()

# Editor
from spark.graph_editor.editor import SparkGraphEditor

__all__ = [
    'nn', 
    'tracers', 
    'Constant', 'Variable',
    'SparkPayload', 'SpikeArray', 'CurrentArray', 'PotentialArray', 'FloatArray', 'IntegerArray', 'BooleanMask',
    'PortSpecs', 'InputSpec', 'OutputSpec', 'PortMap', 'ModuleSpecs',
    'validation',
    'split', 'merge',
    'SparkGraphEditor',
    'register_module', 'register_initializer', 'register_payload', 'register_config', 'register_cfg_validator',
    'REGISTRY',
]