#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.module import SparkModule
    from spark.graph_editor.graph import SparkNodeGraph
    from spark.core.config import BaseSparkConfig

import abc
import logging
import jax.numpy as jnp
import typing as tp
from NodeGraphQt import BaseNode
from spark.core.registry import REGISTRY
from spark.core.payloads import FloatArray
from spark.graph_editor.painter import DEFAULT_PALLETE
from spark.graph_editor.specs import InputSpecEditor, OutputSpecEditor
from spark.core.registry import RegistryEntry
from spark.core.shape import normalize_shape

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class AbstractNode(BaseNode, abc.ABC):
    """
        Abstract node model use to represent different components of a Spark model.
    """

    __identifier__ = 'spark'
    NODE_NAME = 'Abstract Node'
    input_specs: dict[str, InputSpecEditor] = {}
    output_specs: dict[str, OutputSpecEditor] = {}
    graph: SparkNodeGraph

    def __init__(self,):
        super().__init__()
        # Name edition is handle through the inspector
        self._view._text_item.set_locked(True)

    def update_attribute(self, attr_name: str, value: tp.Any):
        """
        Updates a node attribute based on a key and value.

        Args:
            attr_name (str): The name of the attribute to update (e.g., 'learning_rate').
            value (Any): The new value for the attribute.

        Raises:
            ValueError: If the provided value is invalid (e.g., a duplicate name).
            TypeError: If the value has an incompatible type for an init argument.
        """

        # Handle node name updates
        if attr_name == 'name':
            current_name = self.name()
            if value == current_name:
                return # No change
                
            # Use the graph instance attached to the node to check for duplicates
            if self.graph and self.graph.name_exist(value):
                raise ValueError(f'The name "{value}" is already in use.')
            if not value:
                raise ValueError('The name cannot be empty.')

            self.graph.update_node_name(self.id, value)
            logging.info(f'Updated name of node "{self.id}" to "{value}".')
            self.graph.property_changed.emit(self, attr_name, value)
            return

        # NOTE: Only shape updates are expected. All the other arguments are handle by the graph or are set in stone.
        # Handle input specs updates
        if 'input_port.' in attr_name:
            port_name = attr_name.split('.')[1]
            self.input_specs[port_name].shape = normalize_shape(value)
            logging.info(f'Updated init arg "{attr_name}" of node "{self.id}" to "{validated_value}".')
            self.graph.property_changed.emit(self, attr_name, value)
            return
        # Handle output specs update
        if 'output_port.' in attr_name:
            port_name = attr_name.split('.')[1]
            self.output_specs[port_name].shape = normalize_shape(value)
            # TODO: Ideally this should be an event
            self.graph._on_output_shape_update(self.get_output(port_name))
            logging.info(f'Updated init arg "{attr_name}" of node "{self.id}" to "{validated_value}".')
            self.graph.property_changed.emit(self, attr_name, value)
            return
        
        # Handle init args updates
        if attr_name in self.init_args_specs:
            spec = self.init_args_specs[attr_name]
            try:
                # Attempt to cast the value to the expected type
                validated_value = spec.attr_type(value)
            except (ValueError, TypeError):
                raise TypeError(f"Invalid type for '{attr_name}'. Expected {spec.attr_type.__name__}.")

            self.init_args[attr_name] = validated_value
            logging.info(f'Updated init arg "{attr_name}" of node "{self.id}" to "{validated_value}".')
            self.graph.property_changed.emit(self, attr_name, value)
            return
        
        logging.warning(f"Attempted to update unknown or unhandled attribute '{attr_name}' on node {self.id}.")


#-----------------------------------------------------------------------------------------------------------------------------------------------#

# TODO: Make source node general
class SourceNode(AbstractNode):
    """
        Node representing the input to the system.
    """
    NODE_NAME = 'Source Node'

    def __init__(self,):
        super().__init__()
        # Delay specs definition for better control.
        self.output_specs = {'value': OutputSpecEditor(payload_type=FloatArray,
                                                         shape=None,
                                                         dtype=jnp.float16,
                                                         description='Model input port.')}
        # Define output port.
        for key, port_spec in self.output_specs.items():
            self.add_output(name=key, multi_output=True, display_name=False, painter_func=DEFAULT_PALLETE(port_spec.payload_type.__name__))
        
#-----------------------------------------------------------------------------------------------------------------------------------------------#

# TODO: Make sink node general
class SinkNode(AbstractNode):
    """
        Node representing the output of the system.
    """
    NODE_NAME = 'Sink Node'

    def __init__(self,):
        super().__init__()
        # Delay specs definition for better control.
        self.input_specs = {'value': InputSpecEditor(payload_type=FloatArray,
                                                       shape=None,
                                                       dtype=jnp.float16,
                                                       is_optional=False,
                                                       description='Model output port.')}
        # Define input port. Note that Sinks only accept one source.
        for key, port_spec in self.input_specs.items():
            self.add_input(name=key, multi_input=False, display_name=False, painter_func=DEFAULT_PALLETE(port_spec.payload_type.__name__))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SparkModuleNode(AbstractNode, abc.ABC):
    """
        Abstract node representing a SparkModule.
    """

    NODE_NAME = 'SparkModule'
    cls_name: str
    node_config: BaseSparkConfig

    def __init__(self,):
        # Init super
        super().__init__()
        # Get node_cls
        node_cls: type[SparkModule] = REGISTRY.MODULES.get(self.cls_name).class_ref
        print(node_cls)
        # Add input ports.
        self.input_specs = {
            key: InputSpecEditor.from_input_specs(value, []) for key, value in node_cls._get_input_specs().items()
        }
        if isinstance(self.input_specs, dict):
            for key, port_spec in self.input_specs.items():
                self.add_input(name=key, multi_input=True, painter_func=DEFAULT_PALLETE(port_spec.payload_type.__name__))
        # Add output ports.
        self.output_specs = {
            key: OutputSpecEditor.from_output_specs(value) for key, value in node_cls._get_output_specs().items()
        }
        if isinstance(self.output_specs, dict):
            for key, port_spec in self.output_specs.items():
                self.add_output(name=key, multi_output=True, painter_func=DEFAULT_PALLETE(port_spec.payload_type.__name__))
        # Create partial configuration
        node_config_type = node_cls.get_default_config_class()
        self.node_config = node_config_type._create_partial()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def module_to_nodegraph(entry: RegistryEntry) -> type[SparkModuleNode]:
    """
        Factory function that creates a new NodeGraphQt node class from an Spark module class.
    """
    # TODO: Manually passing base SparkModule __init__ signature to init_args could lead to errors in the futre.  

    # Create the new class on the fly.
    module_cls: type[SparkModule] = entry.class_ref
    nodegraph_class = type(
        f'_NG_{module_cls.__name__}',
        (SparkModuleNode,),
        {
            '__identifier__': f'spark',
            'NODE_NAME': module_cls.__name__,
            'cls_name': entry.name,
        } 
    )
    return nodegraph_class

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################