#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.module import SparkModule
    from spark.core.config import BaseSparkConfig
    from spark.graph_editor.models.graph import SparkNodeGraph

from PySide6 import QtWidgets, QtCore, QtGui

import abc
import logging
import jax.numpy as jnp
import typing as tp
import spark.core.utils as utils
from NodeGraphQt import BaseNode
from spark.core.registry import REGISTRY, RegistryEntry
from spark.core.payloads import FloatArray
from spark.core.specs import InputSpec, OutputSpec
from spark.graph_editor.painter import DEFAULT_PALLETE

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
    _graph: SparkNodeGraph

    def __init__(self,) -> None:
        super().__init__()
        # Name edition is handle through the inspector
        self._view._text_item.set_locked(True)

    def update_input_shape(self, port_name: str, value: tuple[int, ...]):
        """
            Updates the shape of an input port and broadcast the update.

            Args:
                port_name (str): The name of the port.
                value (Any): The new value for the attribute.
        """
        # Sanity checks
        if not port_name in self.input_specs:
            raise ValueError(f'Input specs does not define an input port named "{port_name}"')
        # Update port
        value = utils.validate_shape(value)
        self.input_specs[port_name].shape = value
        # Broadcast
        logging.info(f'Updated input port "{port_name}" of node "{self.id}" to "{value}".')
        self._graph.property_changed.emit(self, f'{self.id}.input_port.{port_name}', value)

    def update_output_shape(self, port_name: str, value: tuple[int, ...]):
        """
            Updates the shape of an input port and broadcast the update.

            Args:
                port_name (str): The name of the port.
                value (Any): The new value for the attribute.
        """
        # Sanity checks
        if not port_name in self.output_specs:
            raise ValueError(f'Output specs does not define an input port named "{port_name}"')
        # Update port
        value = utils.validate_shape(value)
        self.output_specs[port_name].shape = value
        # Broadcast
        logging.info(f'Updated output port "{port_name}" of node "{self.id}" to "{value}".')
        self._graph.property_changed.emit(self, f'{self.id}.output_port.{port_name}', value)

    @property
    @abc.abstractmethod
    def node_config_metadata(self,) -> dict:
        pass
    
    def _update_graph_metadata(self,):
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
                shape=None,
                dtype=jnp.float16,
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
                shape=None,
                dtype=jnp.float16,
                is_optional=False,
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
        #self.node_config = node_config_type._create_partial()
        # NOTE: DUMMY TEST
        self.node_config = node_config_type._create_partial(_s_units=utils.validate_shape(1,), _s_async_spikes=True, _s_num_outputs=1)

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