#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.config import SparkConfig
    from spark.graph_editor.models.edge_model import EdgeModel
    from spark.nn.components.base import Component
    from spark.nn.interfaces.base import Interface

import uuid
import logging
import typing as tp
from PySide6.QtCore import Signal
import spark.core.utils as utils
import spark.core.signature_parser as sig_parser
from spark.graph_editor.models.compartment_model import CompartmentModel
from spark.graph_editor.models.base_model import BaseModel
from spark.graph_editor.models.port_model import PortModel
from spark.core.payloads import FloatArray
logger = logging.getLogger('spark')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class NodeModel(BaseModel):
    position_changed = Signal(float, float)
    name_changed = Signal(str)
    type_changed = Signal(str)
    selected_changed = Signal(bool)
    deleted = Signal()
    
    def __init__(self, name: str | None = None, type_name: str = 'BaseObject', pos=(0, 0), parent=None) -> None:
        super().__init__(parent)
        self.id = str(uuid.uuid4())
        self._name = name
        self._type_name = type_name
        self._pos = pos
        self._is_selected = False
        self.config: SparkConfig | None = None
        self.compartments: list[CompartmentModel] = []
        self.call_section = CompartmentModel('Call', self)
        self.props_section = CompartmentModel('Properties', self)
        self.add_compartment(self.call_section)
        self.add_compartment(self.props_section)

    @property
    def name(self) -> str: 
        return self._name
    
    @name.setter
    def name(self, value: str) -> None:
        if self._name != value:
            self._name = value
            self.name_changed.emit(value)

    @property
    def type_name(self) -> str: 
        return self._type_name
    
    @type_name.setter
    def type_name(self, value: str) -> None:
        if self._type_name != value:
            self._type_name = value
            self.type_changed.emit(value)

    @property
    def pos(self) -> tuple[float, float]: 
        return self._pos
    
    @pos.setter
    def pos(self, value: tuple) -> None:
        if self._pos != value:
            self._pos = value
            self.position_changed.emit(value[0], value[1])

    @property
    def is_selected(self) -> bool: 
        return self._is_selected
    
    @is_selected.setter
    def is_selected(self, value: bool) -> None:
        if self._is_selected != value:
            self._is_selected = value
            self.selected_changed.emit(value)

    def add_compartment(self, compartment: CompartmentModel) -> None:
        if compartment not in self.compartments:
            compartment.node = self
            for port in compartment.ports:
                port.node = self
            self.compartments.append(compartment)

    def get_port_by_name(self, name: str, is_input: bool = None) -> PortModel:
        for comp in self.compartments:
            for p in comp.ports:
                if p.name == name:
                    if is_input is None or p.is_input == is_input:
                        return p
        return None

    def get_all_ports(self) -> list[PortModel]:
        ports = []
        for comp in self.compartments:
            ports.extend(comp.ports)
        return ports

    def delete(self) -> None:
        self.deleted.emit()

    def to_dict(self) -> dict[str, tp.Any]:
        return {
            'class': self.__class__.__name__,
            'id': self.id,
            'name': self.name,
            'type_name': self.type_name,
            'pos': [self.pos[0], self.pos[1]],
            'call_section': self.call_section.to_dict(),
            'props_section': self.props_section.to_dict()
        }

    @classmethod
    def from_dict(cls, data) -> tp.Self | NodeModel | tp.Any:
        # Dynamic subclass instantiation.
        class_name = data.get('class', 'NodeModel')
        
        target_cls = cls
        if class_name in globals():
            target_cls = globals()[class_name]
        elif class_name == 'NodeModel':
            target_cls = NodeModel
            
        node = target_cls(data['name'], data.get('type_name', 'BaseObject'))
        if 'id' in data:
            node.id = data['id']
        if 'pos' in data:
            node.pos = (data['pos'][0], data['pos'][1])
            
        node.compartments.clear()
        
        if 'call_section' in data:
            node.call_section = CompartmentModel.from_dict(data['call_section'])
            node.add_compartment(node.call_section)
            
        if 'props_section' in data:
            node.props_section = CompartmentModel.from_dict(data['props_section'])
            node.add_compartment(node.props_section)
            
        return node

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SourceNodeModel(NodeModel):
    """
        Node standing for one input of the controller.
    """

    def __init__(self, name: str | None = None, type_name: str = 'Source Node', pos=(0, 0), parent=None) -> None:
        if name is None:
            name = 'Source'
        super().__init__(name=name, type_name=type_name, pos=pos, parent=parent)
        self.value_port = PortModel(
            'value', 
            is_input=False, 
            port_type=FloatArray, 
            is_optional=False, 
            multi_connection=True, 
            parent=self,
        )
        self.call_section.add_port(self.value_port)
        self.value_port.connected.connect(self.on_port_connected)

    def on_port_connected(self, edge: EdgeModel) -> None:
        self.value_port.port_type = edge.target_port.port_type

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SinkNodeModel(NodeModel):
    """
        Node standing for one output of the controller.
    """
    
    def __init__(self, name: str | None = None, type_name: str = 'Sink Node', pos=(0, 0), parent=None) -> None:
        if name is None:
            name = 'Sink'
        super().__init__(name=name, type_name=type_name, pos=pos, parent=parent)
        self.value_port = PortModel(
            name='value', 
            is_input=True, 
            port_type=FloatArray, 
            is_optional=False, 
            multi_connection=False, 
            parent=self,
        )
        self.call_section.add_port(self.value_port)
        self.value_port.connected.connect(self.on_port_connected)

    def on_port_connected(self, edge: EdgeModel) -> None:
        self.value_port.port_type = edge.source_port.port_type

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SelfPropertyNodeModel(NodeModel):
    """
        Node standing for one property the controller exposes to its modules.
    """

    def __init__(self, name: str | None = None, type_name: str = 'Controller Property', pos=(0, 0), parent=None,
                 payload_type: type | None = None) -> None:
        if name is None:
            name = 'property'
        super().__init__(name=name, type_name=type_name, pos=pos, parent=parent)
        self.value_port = PortModel(
            name='value', 
            is_input=False, 
            port_type=payload_type or FloatArray, 
            is_optional=False,
            multi_connection=True, 
            parent=self,
        )
        self.call_section.add_port(self.value_port)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class InterfaceNodeModel(NodeModel):
    """
        Node model of an Interface.
    """
    _cls: type[Component]

    def __init__(self, name: str | None = None, type_name: str | None = None, pos=(0, 0), parent=None) -> None:
        if name is None:
            name = self._cls.__name__
        if type_name is None:
            type_name = utils.to_human_readable(self._cls.__name__)
        super().__init__(name=name, type_name=type_name, pos=pos, parent=parent)
        self._setup_ports()
        config_cls: type[SparkConfig] = self._cls.get_config_spec()
        self.config = config_cls.partial()

    def _setup_ports(self) -> None:
        try:
            input_specs = self._cls._get_input_specs()
            output_specs = self._cls._get_output_specs()
            property_specs = self._cls._get_property_specs()
            readonly_properties = set(self._cls.get_readonly_properties())
            optional_inputs = set(sig_parser.get_optional_input_names(self._cls))
        except Exception as e:
            logger.warning(f'Could not fully introspect {self._cls.__name__}: {e}')
            input_specs = {}
            output_specs = {}
            property_specs = {}
            readonly_properties = set()
            optional_inputs = set()
        # Populate Call
        for port_name, spec in input_specs.items():
            port = PortModel(
                name=port_name,
                is_input=True,
                port_type=spec.payload_type,
                is_optional=port_name in optional_inputs,
                multi_connection=True, 
            )
            self.call_section.add_port(port)
        for port_name, spec in output_specs.items():
            port = PortModel(
                name=port_name,
                is_input=False,
                port_type=spec.payload_type,
                multi_connection=True, 
            )
            self.call_section.add_port(port)
        # Populate Properties. A property without a setter is read only and gets an output port alone.
        for port_name, spec in property_specs.items():
            if port_name not in readonly_properties:
                port = PortModel(
                    name=port_name,
                    is_input=True,
                    port_type=spec.payload_type,
                    multi_connection=False,
                )
                self.props_section.add_port(port)
            port = PortModel(
                name=port_name,
                is_input=False,
                multi_connection=True,
                port_type=spec.payload_type,
            )
            self.props_section.add_port(port)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ControllerNodeModel(NodeModel):
    """
        Node model of a nested controller (a Neuron placed inside a Brain).
    """

    _cls: type

    def __init__(self, name: str | None = None, type_name: str | None = None, pos=(0, 0), parent=None) -> None:
        if name is None:
            name = self._cls.__name__
        if type_name is None:
            type_name = utils.to_human_readable(self._cls.__name__)
        super().__init__(name=name, type_name=type_name, pos=pos, parent=parent)
        # The configuration comes first, the ports of a controller are derived from its modules.
        config_cls: type[SparkConfig] = self._cls.get_config_spec()
        self.config = config_cls.partial()
        self._setup_ports()

    def _setup_ports(self) -> None:
        try:
            modules_specs = getattr(self.config, 'modules_specs', ())
            input_specs = self._cls._get_controller_input_specs(modules_specs)
            output_specs = {k: v['spec'] for k, v in self._cls._get_controller_output_specs(modules_specs).items()}
            property_specs = self._cls._get_controller_property_specs()
            readonly_properties = set(self._cls.get_readonly_properties())
        except Exception as e:
            logger.warning(f'Could not fully introspect {self._cls.__name__}: {e}')
            input_specs = {}
            output_specs = {}
            property_specs = {}
            readonly_properties = set()
        # Populate Call
        for port_name, spec in input_specs.items():
            self.call_section.add_port(
                PortModel(
                    name=port_name, 
                    is_input=True, 
                    multi_connection=True, 
                    port_type=spec.payload_type, 
                    is_optional=False
                )
            )
        for port_name, spec in output_specs.items():
            self.call_section.add_port(
                PortModel(
                    name=port_name, 
                    is_input=False, 
                    multi_connection=True, 
                    port_type=spec.payload_type
                )
            )
        # Populate Properties. A property without a setter is read only and gets an output port alone.
        for port_name, spec in property_specs.items():
            if port_name not in readonly_properties:
                self.props_section.add_port(
                    PortModel(
                        name=port_name, 
                        is_input=True, 
                        multi_connection=False, 
                        port_type=spec.payload_type
                    )
                )
            self.props_section.add_port(
                PortModel(
                    name=port_name, 
                    is_input=False, 
                    multi_connection=True, 
                    port_type=spec.payload_type
                )
            )

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ComponentNodeModel(NodeModel):
    """
        Node model of a Component.
    """

    _cls: type[Component]

    def __init__(self, name: str | None = None, type_name: str | None = None, pos=(0, 0), parent=None) -> None:
        if name is None:
            name = self._cls.__name__
        if type_name is None:
            type_name = utils.to_human_readable(self._cls.__name__)
        super().__init__(name=name, type_name=type_name, pos=pos, parent=parent)
        self._setup_ports()
        config_cls: type[SparkConfig] = self._cls.get_config_spec()
        self.config = config_cls.partial()

    def _setup_ports(self) -> None:
        try:
            input_specs = self._cls._get_input_specs()
            output_specs = self._cls._get_output_specs()
            property_specs = self._cls._get_property_specs()
            readonly_properties = set(self._cls.get_readonly_properties())
            optional_inputs = set(sig_parser.get_optional_input_names(self._cls))
        except Exception as e:
            logger.warning(f'Could not fully introspect {self._cls.__name__}: {e}')
            input_specs = {}
            output_specs = {}
            property_specs = {}
            readonly_properties = set()
            optional_inputs = set()
        # Populate Call
        for port_name, spec in input_specs.items():
            port = PortModel(
                name=port_name,
                is_input=True,
                port_type=spec.payload_type,
                multi_connection=True, 
                is_optional=port_name in optional_inputs
            )
            self.call_section.add_port(port)
        for port_name, spec in output_specs.items():
            port = PortModel(
                name=port_name,
                is_input=False,
                multi_connection=True, 
                port_type=spec.payload_type
            )
            self.call_section.add_port(port)
        # Populate Properties. A property without a setter is read only and gets an output port alone.
        for port_name, spec in property_specs.items():
            if port_name not in readonly_properties:
                port = PortModel(
                    name=port_name,
                    is_input=True,
                    multi_connection=False, 
                    port_type=spec.payload_type
                )
                self.props_section.add_port(port)
            port = PortModel(
                name=port_name,
                is_input=False,
                multi_connection=True,
                port_type=spec.payload_type
            )
            self.props_section.add_port(port)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################