#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.graph_editor.models.node_model import NodeModel


import uuid
import typing as tp
from PySide6.QtCore import Signal
from spark.graph_editor.models.base_model import BaseModel
from spark.graph_editor.models.port_model import PortModel

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class CompartmentModel(BaseModel):

    port_added = Signal(PortModel)
    port_removed = Signal(PortModel)
    
    def __init__(self, name: str, parent=None) -> None:
        super().__init__(parent)
        self.id = str(uuid.uuid4())
        self.name = name
        self.ports: list[PortModel] = []
        self.node: NodeModel | None = None

    def add_port(self, port: PortModel) -> None:
        if port not in self.ports:
            port.compartment = self
            port.node = self.node
            self.ports.append(port)
            self.port_added.emit(port)

    def remove_port(self, port: PortModel) -> None:
        if port in self.ports:
            self.ports.remove(port)
            port.compartment = None
            self.port_removed.emit(port)

    def get_port(self, name: str) -> PortModel | None:
        for p in self.ports:
            if p.name == name:
                return p
        return None

    def to_dict(self) -> dict[str, tp.Any]:
        return {
            'id': self.id,
            'name': self.name,
            'ports': [port.to_dict() for port in self.ports]
        }

    @classmethod
    def from_dict(cls, data) -> tp.Self:
        comp = cls(data['name'])
        if 'id' in data:
            comp.id = data['id']
        for port_data in data.get('ports', []):
            comp.add_port(PortModel.from_dict(port_data))
        return comp

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################