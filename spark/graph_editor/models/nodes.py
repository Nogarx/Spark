#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.module import SparkModule
    from spark.core.config import BaseSparkConfig
    from spark.graph_editor.models.graph import SparkNodeGraph
    from spark.core.payloads import SparkPayload

from PySide6 import QtWidgets, QtCore, QtGui
from NodeGraphQt import Port
import abc
import logging
import numpy as np
from jax.typing import DTypeLike
import typing as tp
import spark.core.utils as utils
from NodeGraphQt import BaseNode
from spark.core.registry import REGISTRY, RegistryEntry
from spark.core.payloads import FloatArray
from spark.core.specs import InputSpec, OutputSpec
import spark.core.validation as validation 
from spark.graph_editor.style.painter import DEFAULT_PALLETE
from spark.graph_editor.ui.console_panel import MessageLevel

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class AbstractNode(BaseNode, abc.ABC):
    """
        Abstract node model use to represent different components of a Spark model.
    """

    __identifier__ = 'spark'
    NODE_NAME = 'Abstract Node'
    input_specs: dict[str, InputSpec] = {}
    output_specs: dict[str, OutputSpec] = {}
    graph: SparkNodeGraph

    def __init__(self,) -> None:
        super().__init__()
        # Name edition is handle through the inspector
        self._view._text_item.set_locked(True)

    def update_io_spec(
            self, 
            spec: str,
            port_name: str, 
            payload_type: type[SparkPayload] | None = None,
            shape: tuple[int, ...] | None = None,
            dtype: DTypeLike | None = None,
            description: str | None = None
        ):
        """
            Method to update node IO specs. Valid updates are broadcasted.

            Args:
                spec: str, target spec {input, output}
                port_name: str, the name of the port to update
                payload_type: type[SparkPayload] | None, the new payload_type
                shape: tuple[int, ...] | None,  the new shape
                dtype: DTypeLike | None,  the new dtype
                description: str | None,  the new description
        """
        if spec.lower() not in ['input', 'output']:
            raise KeyError(
                f'Expected \"spec\" to be \"input\" or \"output\", but got {spec}.'
            )
        target_spec = 'input_specs' if spec == 'input' else 'output_specs'
        getattr(self, target_spec)
        # Sanity checks
        if not port_name in getattr(self, target_spec):
            raise ValueError(f'{spec.capitalize()} specs does not define an input port named "{port_name}"')
        # Update port
        errors, info = [], []
        if payload_type:
            try:
                if not validation._is_payload_type(payload_type):
                    raise TypeError(
                        f'Expected \"payload_type\" to be of type \"SparkPayload\", but got {payload_type}.'
                    )
                getattr(self, target_spec)[port_name].payload_type = payload_type
                info.append(f'Updated {spec} port \"{port_name}\" payload_type to \"{payload_type}\".')
            except Exception as e:
                errors.append(f'Failed to update {spec} port \"{port_name}.payload_type\". Error: {e}')
        if shape:
            try:
                shape = utils.validate_shape(shape)
                getattr(self, target_spec)[port_name].shape = shape
                info.append(f'Updated {spec} port \"{port_name}\" shape to \"{shape}\".')
            except Exception as e:
                errors.append(f'Failed to update {spec} port \"{port_name}.shape\". Error: {e}')
        if dtype:
            try:
                if not utils.is_dtype(dtype):
                    raise TypeError(
                        f'Expected \"dtype\" to be of type \"DTypeLike\", but got {dtype}.'
                    )
                getattr(self, target_spec)[port_name].dtype = dtype
                info.append(f'Updated {spec} port \"{port_name}\" dtype to \"{dtype}\".')
            except Exception as e:
                errors.append(f'Failed to update {spec} port \"{port_name}.dtype\". Error: {e}')
        if description:
            try:
                getattr(self, target_spec)[port_name].description = description
                info.append(f'Updated {spec} port \"{port_name}\" description.')
            except Exception as e:
                errors.append(f'Failed to update {spec} port \"{port_name}.description\". Error: {e}')
        # Broadcast
        for e in errors:
            self.graph.broadcast_message.emit(
                MessageLevel.ERROR, f'{self.NODE_NAME} - {e}.'
            )
        for i in info:
            self.graph.broadcast_message.emit(
                MessageLevel.INFO, f'{self.NODE_NAME} - {i}.'
            )
        # Update connected sink nodes.
        if spec == 'output':
            output_port: Port = self.get_output(port_name)
            for target_ports in output_port.connected_ports():
                connected_node: AbstractNode = target_ports.node()
                if isinstance(connected_node, SinkNode):
                    connected_node.update_io_spec(
                        'input', 
                        'value', 
                        payload_type=payload_type, 
                        shape=shape, 
                        dtype=dtype, 
                        description=description
                    )

    @property
    @abc.abstractmethod
    def node_config_metadata(self,) -> dict:
        pass
    
    def _update_graph_metadata(self,) -> None:
        # Update metadata for model graph editor model reconstruction.
        self.node_config_metadata['pos'] = self.pos()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# TODO: Make source node general
class SourceNode(AbstractNode):
    """
        Node representing the input to the system.
    """
    NODE_NAME = 'Source Node'

    def __init__(self,) -> None:
        super().__init__()
        # Delay specs definition for better control.
        self.output_specs = {
            'value': OutputSpec(
                payload_type=FloatArray,
                shape=(1,),
                dtype=np.float16,
                description='Model input port.'
                )
            }
        # Define output port.
        for key, port_spec in self.output_specs.items():
            self.add_output(
                name=key, 
                multi_output=True, 
                display_name=False, 
                painter_func=DEFAULT_PALLETE(port_spec.payload_type.__name__)
            )
        
    @property
    def node_config_metadata(self,) -> dict:
        return None

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# TODO: Make sink node general
class SinkNode(AbstractNode):
    """
        Node representing the output of the system.
    """
    NODE_NAME = 'Sink Node'

    def __init__(self,) -> None:
        super().__init__()
        # Delay specs definition for better control.
        self.input_specs = {
            'value': InputSpec(
                payload_type=FloatArray,
                shape=(1,),
                dtype=np.float16,
                description='Model output port.'
            )
        }
        # Define input port. Note that Sinks only accept one source.
        for key, port_spec in self.input_specs.items():
            self.add_input(
                name=key, 
                multi_input=False, 
                display_name=False, 
                painter_func=DEFAULT_PALLETE(port_spec.payload_type.__name__)
            )

    @property
    def node_config_metadata(self,) -> dict:
        return None

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SparkModuleNode(AbstractNode, abc.ABC):
    """
        Abstract node representing a SparkModule.
    """

    NODE_NAME = 'SparkModule'
    module_cls: type[SparkModule]
    node_config: BaseSparkConfig

    def __init__(self,) -> None:
        # Init super
        super().__init__()
        # Add input ports.
        self.input_specs = self.module_cls._get_input_specs()
        if isinstance(self.input_specs, dict):
            for key, port_spec in self.input_specs.items():
                self.add_input(
                    name=key, 
                    multi_input=True, 
                    painter_func=DEFAULT_PALLETE(port_spec.payload_type.__name__)
                )
        # Add output ports.
        self.output_specs = self.module_cls._get_output_specs()
        if isinstance(self.output_specs, dict):
            for key, port_spec in self.output_specs.items():
                self.add_output(
                    name=key, 
                    multi_output=True, 
                    painter_func=DEFAULT_PALLETE(port_spec.payload_type.__name__)
                )
        # Create partial configuration
        node_config_type = self.module_cls.get_config_spec()
        self.node_config = node_config_type._create_partial(_s_units=(1,))


    @property
    def node_config_metadata(self,) -> dict:
        return self.node_config.__graph_editor_metadata__

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def module_to_nodegraph(entry: RegistryEntry) -> type[SparkModuleNode]:
    """
        Factory function that creates a new NodeGraphQt node class from an Spark module class.
    """
    # TODO: Manually passing base SparkModule __init__ signature to init_args could lead to errors in the futre.  

    # Create the new class on the fly.
    module_cls: type[SparkModule] = entry.class_ref
    nodegraph_class = type(
        f'{module_cls.__name__}',
        (SparkModuleNode,),
        {
            '__identifier__': f'spark',
            'NODE_NAME': module_cls.__name__,
            'module_cls': module_cls,
        } 
    )
    return nodegraph_class

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################