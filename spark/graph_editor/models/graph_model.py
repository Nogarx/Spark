#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import dataclasses as dc
from PySide6.QtCore import Signal
from PySide6.QtGui import QUndoStack

import typing as tp
from PySide6.QtCore import Signal
from spark.graph_editor.models.base_model import BaseModel
from spark.graph_editor.models.node_model import NodeModel
from spark.graph_editor.models.edge_model import EdgeModel
from spark.graph_editor.models.inheritance_tree import InheritanceTree, InheritanceFlags
from spark.core.specs import ModuleSpecs
from spark.core.config import SparkConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class GraphModel(BaseModel):

    node_added = Signal(NodeModel)
    node_removed = Signal(NodeModel)
    edge_added = Signal(EdgeModel)
    edge_removed = Signal(EdgeModel)
    graph_cleared = Signal()
    inheritance_updated = Signal()
    config_value_changed = Signal(str, list, object) # node_id, path, value
    
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.nodes: list[NodeModel] = []
        self.edges: list[EdgeModel] = []
        self.undo_stack = QUndoStack(self)
        self.inheritance_tree = InheritanceTree()
        # 
        self.node_added.connect(self.rebuild_inheritance_tree)
        self.node_removed.connect(self.rebuild_inheritance_tree)
        self.graph_cleared.connect(self.rebuild_inheritance_tree)

    def rebuild_inheritance_tree(self, *args) -> None:

        # Extract old inheriting states to preserve them
        old_inheriting_paths = []
        def _get_inheriting_paths(tree: InheritanceTree) -> None:
            for l, leaf in tree._leaves.items():
                if leaf.is_inheriting():
                    old_inheriting_paths.append(leaf.path)
            for b in tree._branches.values():
                _get_inheriting_paths(b)
        
        if self.inheritance_tree:
            _get_inheriting_paths(self.inheritance_tree)
            
        self.inheritance_tree = InheritanceTree()
        
        def _add_config_to_tree(path_prefix, obj) -> None:
            if isinstance(obj, ModuleSpecs):
                _add_config_to_tree(path_prefix + ['config'], obj.config)
            elif isinstance(obj, SparkConfig):
                for field in dc.fields(obj):
                    val = getattr(obj, field.name)
                    # Check if field is ModuleSpecs
                    if isinstance(val, ModuleSpecs):
                        _add_config_to_tree(path_prefix + [field.name], val)
                    # Check if field is list[ModuleSpecs]
                    elif isinstance(val, list):
                        if len(val) > 0 and isinstance(val[0], ModuleSpecs):
                            for i, item in enumerate(val):
                                _add_config_to_tree(path_prefix + [field.name, f'[{i}]'], item)
                    # Field is a regular field
                    else:
                        if field.metadata.get('allows_inheritance', False):
                            self.inheritance_tree.add_leaf(path_prefix + [field.name], type_string=field.type)
                            
        for node in self.nodes:
            if hasattr(node, 'config') and node.config:
                _add_config_to_tree([node.id], node.config)

        self.inheritance_tree.validate(old_inheriting_paths)
        self.inheritance_updated.emit()

    def get_node_config_value(self, path: list[str]) -> None | tp.Any:
        if len(path) < 2: 
            return None
        node_id = path[0]
        node = self.get_node_by_id(node_id)
        if not node or not node.config: 
            return None
        
        current = node.config
        for part in path[1:-1]:
            if part == 'init_config': continue
            if part.startswith('[') and part.endswith(']'):
                idx = int(part[1:-1])
                current = current[idx]
            else:
                current = getattr(current, part)
                if isinstance(current, ModuleSpecs):
                    current = current.config
                    
        attr_name = path[-1]
        if hasattr(current, attr_name):
            return getattr(current, attr_name)
        return None

    def set_node_config_value(self, path: list[str], value) -> None:
        if len(path) < 2: 
            return
        node_id = path[0]
        node = self.get_node_by_id(node_id)
        if not node or not node.config: 
            return
        
        # Traverse the config object
        current = node.config
        for part in path[1:-1]:
            if part == 'init_config': continue
            # if part is like '[0]', we parse it as list index
            if part.startswith('[') and part.endswith(']'):
                idx = int(part[1:-1])
                current = current[idx]
            else:
                current = getattr(current, part)
                if isinstance(current, ModuleSpecs):
                    current = current.config
                    
        attr_name = path[-1]
        if hasattr(current, attr_name):
            setattr(current, attr_name, value)
            self.config_value_changed.emit(node_id, path, value)
            
    def update_inherited_value(self, origin_path: list[str], value) -> None:
        try:
            if 'init_config' in origin_path:
                idx = origin_path.index('init_config')
                base_path = origin_path[:idx]
                leaf = self.inheritance_tree.get_leaf(base_path)
                
                if not leaf or not leaf.is_inheriting(): 
                    return
                
                prop_name = origin_path[-1]
                for child_path in leaf.inheritance_childs:
                    target_path = child_path + ['init_config', prop_name]
                    self.set_node_config_value(target_path, value)
            else:
                leaf = self.inheritance_tree.get_leaf(origin_path)
                if not leaf or not leaf.is_inheriting(): 
                    return

                # Propagate to all identified descendants in its scope
                for child_path in leaf.inheritance_childs:
                    self.set_node_config_value(child_path, value)
        except KeyError:
            pass

    def toggle_inheritance(self, path: list[str], is_inheriting: bool) -> None:
        try:
            leaf = self.inheritance_tree.get_leaf(path)
            if leaf:
                if is_inheriting:
                    leaf.flags |= InheritanceFlags.IS_INHERITING
                else:
                    leaf.flags &= ~InheritanceFlags.IS_INHERITING
                # Revalidate tree to propagate the state
                self.inheritance_tree._is_valid = False
                self.inheritance_tree.validate()
                self.inheritance_updated.emit()
        except KeyError:
            pass

    def add_node(self, node: NodeModel) -> None:
        if node not in self.nodes:
            self.nodes.append(node)
            self.node_added.emit(node)

    def remove_node(self, node: NodeModel) -> None:
        if node in self.nodes:
            # Clean up associated edges first
            for port in node.get_all_ports():
                for edge in list(port.edges):
                    self.remove_edge(edge)
            self.nodes.remove(node)
            self.node_removed.emit(node)

    def add_edge(self, edge: EdgeModel) -> None:
        if edge not in self.edges:
            self.edges.append(edge)
            if edge.source_port:
                edge.source_port.add_edge(edge)
            if edge.target_port:
                edge.target_port.add_edge(edge)
            self.edge_added.emit(edge)

    def remove_edge(self, edge: EdgeModel) -> None:
        if edge in self.edges:
            self.edges.remove(edge)
            if edge.source_port:
                edge.source_port.remove_edge(edge)
            if edge.target_port:
                edge.target_port.remove_edge(edge)
            self.edge_removed.emit(edge)

    def get_node_by_id(self, node_id: str) -> NodeModel | None:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def clear(self) -> None:
        self.nodes.clear()
        self.edges.clear()
        self.graph_cleared.emit()

    def to_dict(self) -> dict[str, tp.Any]:
        return {
            'nodes': [node.to_dict() for node in self.nodes],
            'edges': [edge.to_dict() for edge in self.edges]
        }

    def from_dict(self, data) -> None:
        self.clear()
        if hasattr(self, 'undo_stack'):
            self.undo_stack.clear()
        
        all_ports = {}
        for node_data in data.get('nodes', []):
            node = NodeModel.from_dict(node_data)
            self.add_node(node)
            for port in node.get_all_ports():
                all_ports[port.id] = port
                
        for edge_data in data.get('edges', []):
            edge = EdgeModel.from_dict(edge_data, all_ports)
            if edge:
                self.add_edge(edge)

    def is_name_taken(self, name: str) -> bool:
        return name in [n.name for n in self.nodes]

    def get_next_free_name(self, name: str) -> bool:
        names = [n.name for n in self.nodes if n.name.startswith(name)]
        if len(names) == 0:
            return name
        else:
            it = 0
            new_name = f'{name}{it}'
            while new_name in names:
                it += 1
                new_name = f'{name}{it}'
            return new_name

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################