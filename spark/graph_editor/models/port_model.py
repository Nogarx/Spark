#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.graph_editor.models.node_model import NodeModel
    from spark.graph_editor.models.edge_model import EdgeModel
    from spark.graph_editor.models.compartment_model import CompartmentModel
    from spark.core.payloads import SparkPayload

import abc
import uuid
import typing as tp
from PySide6.QtCore import Signal
from spark.graph_editor.models.base_model import BaseModel

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class PortModel(BaseModel):

    from spark.graph_editor.models.edge_model import EdgeModel
    connected = Signal(EdgeModel)
    disconnected = Signal(EdgeModel)
    type_changed = Signal(object)
    
    def __init__(self, name: str, is_input: bool, port_type: type[SparkPayload], is_optional: bool = False, multi_connection: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.id = str(uuid.uuid4())
        self.name = name
        self.is_input = is_input
        self.is_optional = is_optional
        self.multi_connection = multi_connection
        # The type the port is declared with. A generic port carries the type of what it is connected to, set by
        # port_types whenever its connections change.
        self.declared_type = port_type
        self._port_type = port_type
        
        self.node: NodeModel | None = None
        self.compartment: CompartmentModel | None = None
        self.edges: list[EdgeModel] = []

    @property
    def port_type(self) -> type[SparkPayload]:
        return self._port_type

    @port_type.setter
    def port_type(self, value: type[SparkPayload]) -> None:
        if value is not self._port_type:
            self._port_type = value
            self.type_changed.emit(value)

    @property
    def is_generic(self) -> bool:
        """
            True if the port is declared by a base payload type, and carries the type of what it is connected to.
        """
        # A base payload type is declared abstract, and stands for every payload derived from it.
        declared = self.declared_type
        return isinstance(declared, type) and abc.ABC in declared.__bases__

    def add_edge(self, edge: EdgeModel) -> None:
        if edge not in self.edges:
            self.edges.append(edge)
            self.connected.emit(edge)

    def remove_edge(self, edge: EdgeModel) -> None:
        if edge in self.edges:
            self.edges.remove(edge)
            self.disconnected.emit(edge)
            
    def get_connected_nodes(self) -> list:
        nodes = []
        for edge in self.edges:
            other = edge.source_port if self.is_input else edge.target_port
            if other and other.node and other.node not in nodes:
                nodes.append(other.node)
        return nodes

    def to_dict(self) -> dict[str, tp.Any]:
        return {
            'id': self.id,
            'name': self.name,
            'is_input': self.is_input,
            'port_type': self.declared_type,
            'is_optional': self.is_optional,
            'multi_connection': self.multi_connection
        }

    @classmethod
    def from_dict(cls, data) -> tp.Self:
        port = cls(
            name=data['name'],
            is_input=data['is_input'],
            port_type=data.get('port_type', 'DATA'),
            is_optional=data.get('is_optional', False),
            multi_connection=data.get('multi_connection', False)
        )
        if 'id' in data:
            port.id = data['id']
        return port

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################