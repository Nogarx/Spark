#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.graph_editor.models.port_model import PortModel

import uuid
import typing as tp
from PySide6.QtCore import Signal
from spark.graph_editor.models.base_model import BaseModel

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class EdgeModel(BaseModel):

    waypoints_changed = Signal()
    deleted = Signal()

    def __init__(self, source_port: PortModel, target_port: PortModel, parent=None) -> None:
        super().__init__(parent)
        self.id = str(uuid.uuid4())
        self.source_port = source_port
        self.target_port = target_port
        self._waypoints: list[tuple] = []

    @property
    def waypoints(self) -> list[tuple]: 
        return self._waypoints
    
    @waypoints.setter
    def waypoints(self, value: list[tuple]) -> None:
        self._waypoints = value
        self.waypoints_changed.emit()

    @classmethod
    def validate_connection(cls, src_port: PortModel, dst_port: PortModel) -> tuple[bool, str]:
        """
            Validates if a connection between two ports is allowed.
        """
        if not src_port or not dst_port:
            return False, 'Missing port.'
        if src_port.node == dst_port.node:
            return True, 'Self connections allowed.'
        if src_port.is_input == dst_port.is_input:
            return False, 'Cannot connect two inputs or two outputs.'
        if src_port.port_type != dst_port.port_type:
            return False, f'Mismatched port types ({src_port.port_type} -> {dst_port.port_type}).'
        return True, 'Valid.'

    def delete(self) -> None:
        self.deleted.emit()

    def to_dict(self) -> dict[str, tp.Any]:
        return {
            'id': self.id,
            'source_port_id': self.source_port.id if self.source_port else None,
            'target_port_id': self.target_port.id if self.target_port else None,
            'waypoints': [(float(x), float(y)) for x, y in self.waypoints]
        }

    @classmethod
    def from_dict(cls, data, all_ports) -> None | tp.Self:
        # We expect a dictionary of all available PortModels, keyed by ID.
        src_port = all_ports.get(data.get('source_port_id'))
        dst_port = all_ports.get(data.get('target_port_id'))
        # Discard edge if ports are missing.
        if not src_port or not dst_port:
            return None
        edge = cls(src_port, dst_port)
        if 'id' in data:
            edge.id = data['id']
        if 'waypoints' in data:
            edge.waypoints = [(float(p[0]), float(p[1])) for p in data['waypoints']]
        return edge

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################