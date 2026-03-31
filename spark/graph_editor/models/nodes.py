#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp
if tp.TYPE_CHECKING:
    from spark.core.module import SparkModule
    from spark.core.config import SparkConfig
    from spark.graph_editor.models.graph import SparkNodeGraph
    from spark.core.payloads import SparkPayload
    from spark.nn.controllers.neuron import Neuron, NeuronConfig

import abc
import numpy as np
from jax.typing import DTypeLike
from NodeGraphQt import BaseNode, Port

import spark.core.utils as utils
import spark.core.validation as validation 
import spark.core.signature_parser as sig_parser
from spark.core.specs import ModuleSpecs, PortMap, PortSpecs
from spark.core.registry import RegistryEntry
from spark.core.payloads import FloatArray
from spark.graph_editor.style.painter import DEFAULT_PALETTE, PortColorStyle
from spark.graph_editor.ui.console_panel import MessageLevel
from spark.graph_editor.models.utils import flattify_controller_config, unflattify_controller_config

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class AbstractNode(BaseNode, abc.ABC):
    """
        Abstract node model use to represent different components of a Spark model.
    """

    __identifier__ = 'spark'
    NODE_NAME = 'Abstract Node'
    input_specs: dict[str, PortSpecs] = {}
    output_specs: dict[str, PortSpecs] = {}
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
    def metadata(self,) -> dict:
        pass
    
    def _update_graph_metadata(self,) -> None:
        # Update metadata for model graph editor model reconstruction.
        self.metadata['pos'] = self.pos()

    @abc.abstractmethod
    def get_module_spec(self) -> tp.Any:
        pass

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
            'value': PortSpecs(
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
                painter_func=DEFAULT_PALETTE(port_spec.payload_type.__name__, color_style=PortColorStyle.DEFAULT)
            )
        
    @property
    def metadata(self,) -> dict:
        return None
    
    def get_module_spec(self) -> dict[str, tp.Any]:
        # Update graph metadata
        #self._update_graph_metadata()
        return {
            'type': 'source',
            'pos': self.pos(),
            'spec': PortSpecs(
                payload_type=self.output_specs['value'].payload_type,
                shape=self.output_specs['value'].shape,
                dtype=self.output_specs['value'].dtype,
                description=self.output_specs['value'].description,
            )
        }

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class PropertyNode(AbstractNode):
    """
        Node representing a property of the system.
    """
    NODE_NAME = 'Property Node'

    def __init__(self, ) -> None:
        super().__init__()
        # Delay specs definition for better control.
        self.output_specs = {
            'value': PortSpecs(
                payload_type=FloatArray,
                shape=(1,),
                dtype=np.float16,
                description='Model property port.'
                )
            }
        # Define output port.
        for key, port_spec in self.output_specs.items():
            self.add_output(
                name=key, 
                multi_output=True, 
                display_name=False, 
                painter_func=DEFAULT_PALETTE(port_spec.payload_type.__name__, color_style=PortColorStyle.PROPERTY)
            )
        
    @property
    def metadata(self,) -> dict:
        return None
    
    def get_module_spec(self) -> dict[str, tp.Any]:
        # Update graph metadata
        #self._update_graph_metadata()
        return {
            'type': 'property',
            'pos': self.pos(),
            'spec': PortSpecs(
                payload_type=self.output_specs['value'].payload_type,
                shape=self.output_specs['value'].shape,
                dtype=self.output_specs['value'].dtype,
                description=self.output_specs['value'].description,
            )
        }


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
            'value': PortSpecs(
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
                painter_func=DEFAULT_PALETTE(port_spec.payload_type.__name__, color_style=PortColorStyle.DEFAULT)
            )

    @property
    def metadata(self,) -> dict:
        return None

    def get_module_spec(self) -> dict[str, tp.Any]:
        # Update graph metadata
        #self._update_graph_metadata()
        return {
            'type': 'sink',
            'pos': self.pos(),
            'spec': PortSpecs(
                payload_type=self.input_specs['value'].payload_type,
                shape=self.input_specs['value'].shape,
                dtype=self.input_specs['value'].dtype,
                description=self.input_specs['value'].description,
            )
        }

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SparkModuleNode(AbstractNode, abc.ABC):
    """
        Abstract node representing a SparkModule.
    """

    NODE_NAME = 'SparkModule'
    module_cls: type[SparkModule]
    node_config: SparkConfig

    def __init__(self,) -> None:
        # Init super
        super().__init__()
        # Add input ports.
        optional_inputs = sig_parser.get_optional_input_names(self.module_cls)
        self.input_specs = self.module_cls._get_input_specs()
        for key, port_spec in self.input_specs.items():
            self.add_input(
                name=key, 
                multi_input=True, 
                painter_func=DEFAULT_PALETTE(port_spec.payload_type.__name__, color_style=PortColorStyle.OPTIONAL if key in optional_inputs else PortColorStyle.DEFAULT)
            )
        # Add output ports.
        self.output_specs = self.module_cls._get_output_specs()
        for key, port_spec in self.output_specs.items():
            self.add_output(
                name=key, 
                multi_output=True, 
                painter_func=DEFAULT_PALETTE(port_spec.payload_type.__name__, color_style=PortColorStyle.DEFAULT)
            )
        # Add property ports.
        self.property_specs = self.module_cls._get_property_specs()
        for key, port_spec in self.property_specs.items():
            self.add_output(
                name=key, 
                multi_output=True, 
                painter_func=DEFAULT_PALETTE(port_spec.payload_type.__name__, color_style=PortColorStyle.PROPERTY)
            )
            port_property = getattr(self.module_cls, key, None)
            if port_property.fset is not None:
                self.add_input(
                    name=key, 
                    multi_input=True, 
                    painter_func=DEFAULT_PALETTE(port_spec.payload_type.__name__, color_style=PortColorStyle.PROPERTY)
                )
        # Create partial configuration
        node_config_type: SparkConfig = self.module_cls.get_config_spec()
        self.node_config = node_config_type._create_partial(_s_units=(1,))

    @property
    def metadata(self,) -> dict:
        return self.node_config.__graph_editor_metadata__

    def get_module_spec(self) -> ModuleSpecs:
        # Build input map
        _inputs = {}
        for port in self.input_ports():
            port_name = port.name()
            # Skip properties
            if not port_name in self.input_specs.keys():
                continue
            _inputs[port_name] = [
                PortMap(
                    origin=other.node().name(), 
                    port=other.name(), is_property=False
                ) for other in port.connected_ports()
            ]
        # Build output map
        _outputs = {}
        for port in self.output_ports():
            port_name = port.name()
            # Check if the port is connected to a sink node.
            for other in port.connected_ports():
                if isinstance(other.node(), SinkNode):
                    _outputs[other.node().name()] = port_name
        # Build effects map
        _effects = {}
        for port in self.input_ports():
            port_name = port.name()
            # Skip non-properties
            if port_name in self.input_specs.keys():
                continue
            _effects[port_name] = [
                PortMap(
                    origin=other.node().name(), 
                    port=other.name(), is_property=False
                ) for other in port.connected_ports()
            ]
        # Update graph metadata
        self._update_graph_metadata()
        return ModuleSpecs(
            name = self.NODE_NAME,
            module_cls = self.module_cls,
            inputs = _inputs,
            outputs = _outputs,
            effects = _effects,
            config = self.node_config,
        )

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SparkNeuronNode(AbstractNode, abc.ABC):
    """
        Abstract node representing a Neuron model.
    """

    NODE_NAME = 'Neuron'
    module_cls: type[Neuron]
    node_config: NeuronConfig

    def __init__(self,) -> None:
        # Init super
        super().__init__()
        # Create partial configuration
        node_config_type: NeuronConfig = self.module_cls.get_config_spec()
        self.node_config = node_config_type._create_partial(_s_units=(1,))
        self.node_config_flat = flattify_controller_config(self.node_config)
        
        # Add input ports.
        optional_inputs = sig_parser.get_optional_input_names(self.module_cls)
        self.input_specs = self.module_cls._get_controller_input_specs(self.node_config.modules_specs)
        for key, port_spec in self.input_specs.items():
            self.add_input(
                name=key, 
                multi_input=True, 
                painter_func=DEFAULT_PALETTE(port_spec.payload_type.__name__, color_style=PortColorStyle.OPTIONAL if key in optional_inputs else PortColorStyle.DEFAULT)
            )
        # Add output ports.
        self.output_specs = self.module_cls._get_controller_output_specs(self.node_config.modules_specs)
        for key, port_spec in self.output_specs.items():
            port_spec = port_spec['spec']
            self.add_output(
                name=key, 
                multi_output=True, 
                painter_func=DEFAULT_PALETTE(port_spec.payload_type.__name__, color_style=PortColorStyle.DEFAULT)
            )
        # Add property ports.
        self.property_specs = self.module_cls._get_controller_property_specs()
        for key, port_spec in self.property_specs.items():
            self.add_output(
                name=key, 
                multi_output=True, 
                painter_func=DEFAULT_PALETTE(port_spec.payload_type.__name__, color_style=PortColorStyle.PROPERTY)
            )
            port_property = getattr(self.module_cls, key, None)
            if port_property.fset is not None:
                self.add_input(
                    name=key, 
                    multi_input=True, 
                    painter_func=DEFAULT_PALETTE(port_spec.payload_type.__name__, color_style=PortColorStyle.PROPERTY)
                )

    @property
    def metadata(self,) -> dict:
        return self.node_config.__graph_editor_metadata__

    def get_module_spec(self) -> ModuleSpecs:
        # Build input map
        _inputs = {}
        for port in self.input_ports():
            port_name = port.name()
            # Skip properties
            if not port_name in self.input_specs.keys():
                continue
            _inputs[port_name] = [
                PortMap(
                    origin=other.node().name(), 
                    port=other.name(), is_property=False
                ) for other in port.connected_ports()
            ]
        # Build output map
        _outputs = {}
        for port in self.output_ports():
            port_name = port.name()
            # Check if the port is connected to a sink node.
            for other in port.connected_ports():
                if isinstance(other.node(), SinkNode):
                    _outputs[other.node().name()] = port_name
        # Build effects map
        _effects = {}
        for port in self.input_ports():
            port_name = port.name()
            # Skip non-properties
            if port_name in self.input_specs.keys():
                continue
            _effects[port_name] = [
                PortMap(
                    origin=other.node().name(), 
                    port=other.name(), is_property=False
                ) for other in port.connected_ports()
            ]
        # Update graph metadata
        self._update_graph_metadata()
        self.node_config = unflattify_controller_config(type(self.node_config), self.node_config_flat)
        return ModuleSpecs(
            name = self.NODE_NAME,
            module_cls = self.module_cls,
            inputs = _inputs,
            outputs = _outputs,
            effects = _effects,
            config = self.node_config,
        )

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

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def neuron_to_nodegraph(entry: RegistryEntry) -> type[SparkModuleNode]:
    """
        Factory function that creates a new NodeGraphQt node class from an Neuron Controller class.
    """
    # TODO: Manually passing base SparkModule __init__ signature to init_args could lead to errors in the futre.  

    # Create the new class on the fly.
    module_cls: type[Neuron] = entry.class_ref
    nodegraph_class = type(
        f'{module_cls.__name__}',
        (SparkNeuronNode,),
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