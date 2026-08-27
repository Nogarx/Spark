#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import logging
import dataclasses as dc
from PySide6.QtCore import Signal
from PySide6.QtGui import QUndoStack

import typing as tp
from PySide6.QtCore import Signal
from spark.graph_editor.models.base_model import BaseModel
from spark.graph_editor.models.node_model import NodeModel
from spark.graph_editor.models.edge_model import EdgeModel
from spark.graph_editor.models.inheritance_tree import InheritanceTree, InheritanceLeaf, InheritanceFlags
from spark.graph_editor.models.controller_profile import ControllerProfile, get_controller_profile
from spark.core.specs import ModuleSpecs
from spark.core.config import SparkConfig
from spark.nn.initializers import InitializerConfig

logger = logging.getLogger('spark')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class GraphModel(BaseModel):

    # NOTE: The controller has settings of its own (the size of a neuron pool, dt, the seed) that belong to no
    # node. They are addressed through this reserved id, so the inspector, the undo stack and the inheritance
    # trees treat them like any other configuration.
    CONTROLLER_ID = '__controller__'

    node_added = Signal(NodeModel)
    node_removed = Signal(NodeModel)
    edge_added = Signal(EdgeModel)
    edge_removed = Signal(EdgeModel)
    graph_cleared = Signal()
    inheritance_updated = Signal()
    profile_changed = Signal(object) # ControllerProfile | None
    config_value_changed = Signal(str, list, object) # node_id, path, value

    def __init__(self, profile: ControllerProfile | None = None, parent=None) -> None:
        super().__init__(parent)
        self.nodes: list[NodeModel] = []
        self.edges: list[EdgeModel] = []
        # A graph is always the template of one controller. Without a profile there is no document yet.
        self._profile: ControllerProfile | None = profile
        # The controller is a module too: it has a configuration of its own (units, dt, seed, ...) that belongs
        # to the graph rather than to any node.
        self._controller_config: tp.Any = None
        self.undo_stack = QUndoStack(self)
        # NOTE: Inheritance is scoped to a single node: a value only cascades to the nested configurations of
        # the same node (the editor equivalent of the "_s_" shared kwargs of SparkConfig). Each node owns an
        # independent tree, keyed by node id.
        self.inheritance_trees: dict[str, InheritanceTree] = {}
        self.node_added.connect(self.rebuild_inheritance_tree)
        self.node_removed.connect(self.rebuild_inheritance_tree)
        self.graph_cleared.connect(self.rebuild_inheritance_tree)

    #-------------------------------------------------------------------------------------------------------#
    # Controller profile
    #-------------------------------------------------------------------------------------------------------#

    @property
    def profile(self) -> ControllerProfile | None:
        """
            Controller profile this graph is building.
        """
        return self._profile

    def set_profile(self, profile: ControllerProfile | None, force: bool = False) -> bool:
        """
            Sets the controller profile of the graph.

            The profile determines which modules may be placed and how the graph is exported. It can only
            change while the graph is empty, except when loading a model, where the file dictates it.

            Parameters
            ----------
            profile : ControllerProfile or None
                The new profile.
            force : bool, default False
                Apply the profile regardless of the current content. Reserved for loading.

            Returns
            -------
            bool
                True if the profile was applied.
        """
        if profile is self._profile:
            return True
        if not force and len(self.nodes) > 0:
            logger.warning(
                'The controller type cannot be changed while the graph contains nodes. '
                'Create a new model instead.'
            )
            return False
        self._profile = profile
        self._controller_config = None
        self.rebuild_inheritance_tree()
        self.profile_changed.emit(profile)
        return True

    @property
    def controller_config(self) -> tp.Any:
        """
            Configuration of the controller the graph describes, created on demand from the profile.
        """
        if self._controller_config is None and self._profile is not None:
            config_cls = self._profile.config_cls
            if config_cls is not None:
                # NOTE: A controller configuration declares "modules_specs" without a default, so partial() cannot
                # fill it in. The module list is seeded explicitly.
                self._controller_config = config_cls.partial(modules_specs=())
        return self._controller_config

    @controller_config.setter
    def controller_config(self, config: tp.Any) -> None:
        self._controller_config = config
        self.rebuild_inheritance_tree()

    def adopt_controller_config(self, config: tp.Any) -> bool:
        """
            Takes the controller settings of an existing configuration, and only those.

            Parameters
            ----------
            config : SparkConfig
                Configuration to read the settings from.

            Returns
            -------
            bool
                True if the settings were adopted.

            Notes
            -----
            The modules are dropped. The canvas is the single source of truth for what the controller contains.
        """
        if config is None or self._profile is None:
            return False
        config_cls = self._profile.config_cls
        if config_cls is None or not isinstance(config, config_cls):
            return False
        own_fields = {
            field.name: getattr(config, field.name, None)
            for field in dc.fields(config) if field.name != 'modules_specs'
        }
        self.controller_config = config_cls.partial(modules_specs=(), **own_fields)
        return True

    def can_change_profile(self) -> bool:
        """
            True if the controller profile may still be changed.
        """
        return len(self.nodes) == 0

    #-------------------------------------------------------------------------------------------------------#
    # Inheritance
    #-------------------------------------------------------------------------------------------------------#

    @staticmethod
    def _build_node_tree(config: SparkConfig) -> InheritanceTree:
        """
            Builds the inheritance tree of a single node from its configuration.
        """
        tree = InheritanceTree()

        def _add_config_to_tree(path_prefix: list[str], obj: tp.Any) -> None:
            if isinstance(obj, ModuleSpecs):
                _add_config_to_tree(path_prefix + ['config'], obj.config)
                return
            if not isinstance(obj, SparkConfig):
                return
            for field in dc.fields(obj):
                val = getattr(obj, field.name, None)
                if isinstance(val, ModuleSpecs):
                    _add_config_to_tree(path_prefix + [field.name], val)
                    continue
                # NOTE: SparkConfig crystallizes mutable iterables into tuples, so both forms are accepted.
                if isinstance(val, (list, tuple)) and len(val) > 0 and all(isinstance(v, ModuleSpecs) for v in val):
                    for i, item in enumerate(val):
                        _add_config_to_tree(path_prefix + [field.name, f'[{i}]'], item)
                    continue
                # Initializers cascade as a whole, their inner fields are not independent leaves.
                if isinstance(val, InitializerConfig):
                    tree.add_leaf(path_prefix + [field.name], type_string=field.metadata.get('valid_types') or field.type)
                    continue
                # Nested configurations become branches, so their fields can receive from an ancestor.
                if isinstance(val, SparkConfig):
                    _add_config_to_tree(path_prefix + [field.name], val)
                    continue
                tree.add_leaf(path_prefix + [field.name], type_string=field.metadata.get('valid_types') or field.type)

        _add_config_to_tree([], config)
        return tree

    @staticmethod
    def _collect_inheriting_paths(tree: InheritanceTree) -> list[list[str]]:
        """
            Collects the (node relative) paths of every leaf currently cascading its value.
        """
        paths: list[list[str]] = []

        def _walk(subtree: InheritanceTree) -> None:
            for leaf in subtree._leaves.values():
                if leaf.is_inheriting():
                    paths.append(leaf.path)
            for branch in subtree._branches.values():
                _walk(branch)

        _walk(tree)
        return paths

    def rebuild_inheritance_tree(self, *args) -> None:
        """
            Rebuilds the inheritance tree of every node, preserving the currently cascading leaves.
        """
        old_paths = {node_id: self._collect_inheriting_paths(tree) for node_id, tree in self.inheritance_trees.items()}
        self.inheritance_trees = {}
        owners: list[tuple[str, tp.Any]] = [(self.CONTROLLER_ID, self._controller_config)]
        owners += [(node.id, getattr(node, 'config', None)) for node in self.nodes]
        for node_id, config in owners:
            if config is None:
                continue
            tree = self._build_node_tree(config)
            # Restore the cascading state of the leaves that survived the rebuild.
            for path in old_paths.get(node_id, []):
                try:
                    leaf = tree.get_leaf(path)
                except KeyError:
                    continue
                if leaf is not None:
                    leaf.flags |= InheritanceFlags.IS_INHERITING
            tree.invalidate()
            tree.validate()
            self.inheritance_trees[node_id] = tree
        self.inheritance_updated.emit()

    def get_inheritance_tree(self, node_id: str) -> InheritanceTree | None:
        """
            Returns the inheritance tree of a node.
        """
        return self.inheritance_trees.get(node_id, None)

    def get_inheritance_leaf(self, path: list[str]) -> InheritanceLeaf | None:
        """
            Returns the inheritance leaf addressed by a full config path ([node_id, field, ...]).
        """
        if not path or len(path) < 2:
            return None
        tree = self.inheritance_trees.get(path[0], None)
        if tree is None:
            return None
        try:
            return tree.get_leaf(list(path[1:]))
        except KeyError:
            return None

    def get_inheritance_children(self, path: list[str]) -> list[list[str]]:
        """
            Returns the full config paths of every field driven by the leaf addressed by "path".
        """
        leaf = self.get_inheritance_leaf(path)
        if leaf is None or not leaf.is_inheriting():
            return []
        return [[path[0]] + child_path for child_path in leaf.inheritance_childs]

    def is_driven(self, path: list[str]) -> bool:
        """
            Returns True if the field addressed by "path" currently receives its value from an ancestor.
        """
        leaf = self.get_inheritance_leaf(path)
        return bool(leaf is not None and leaf.is_receiving())

    @staticmethod
    def _resolve_step(current: tp.Any, part: str) -> tp.Any:
        """
            Resolves a single segment of a config path.
        """
        # List/tuple index (e.g. "[0]")
        if part.startswith('[') and part.endswith(']'):
            try:
                return current[int(part[1:-1])]
            except (TypeError, ValueError, IndexError, KeyError):
                return None
        # NOTE: ModuleSpecs is transparent unless the path names its "config" attribute explicitly.
        if isinstance(current, ModuleSpecs) and not hasattr(current, part):
            current = current.config
        return getattr(current, part, None)

    def _resolve_owner(self, path: list[str]) -> tp.Any:
        """
            Returns the object holding the attribute addressed by the last segment of a config path.
        """
        if len(path) < 2:
            return None
        if path[0] == self.CONTROLLER_ID:
            current = self.controller_config
            if current is None:
                return None
        else:
            node = self.get_node_by_id(path[0])
            if not node or not getattr(node, 'config', None):
                return None
            current = node.config
        for part in path[1:-1]:
            # Initializer blocks are a UI grouping, they do not exist in the configuration object.
            if part == 'init_config':
                continue
            current = self._resolve_step(current, part)
            if current is None:
                return None
        return current

    def get_node_config_value(self, path: list[str]) -> None | tp.Any:
        owner = self._resolve_owner(path)
        if owner is None:
            return None
        return getattr(owner, path[-1], None)

    def set_node_config_value(self, path: list[str], value, force: bool = False) -> None:
        """
            Writes a value into the configuration of a node.

            Parameters
            ----------
            path : list of str
                Full config path ([node_id, field, ...]).
            value : Any
                The new value.
            force : bool, default False
                Bypass the inheritance guard. Reserved for cascaded writes.
        """
        if len(path) < 2:
            return
        # A driven field is owned by its ancestor, direct writes are rejected.
        if not force and self.is_driven(self._inheritance_path(path)):
            logger.debug(f'Rejected write to "{"/".join(path)}": the field inherits its value from another field.')
            return
        owner = self._resolve_owner(path)
        if owner is None:
            logger.warning(f'Unable to resolve config path "{"/".join(path)}".')
            return
        attr_name = path[-1]
        if not hasattr(owner, attr_name):
            logger.warning(f'Config path "{"/".join(path)}" does not name a valid field.')
            return
        old_value = getattr(owner, attr_name, None)
        setattr(owner, attr_name, value)
        # Replacing a nested configuration (e.g. toggling an initializer) changes the shape of the tree.
        if isinstance(value, SparkConfig) or isinstance(old_value, SparkConfig):
            self.rebuild_inheritance_tree()
        self.config_value_changed.emit(path[0], path, value)

    @staticmethod
    def _inheritance_path(path: list[str]) -> list[str]:
        """
            Maps a config path to the path of the leaf that governs it.

            Initializer sub-fields are not independent leaves; they are governed by the field holding the
            initializer.
        """
        if 'init_config' in path:
            return list(path[:path.index('init_config')])
        return list(path)

    def update_inherited_value(self, origin_path: list[str], value) -> None:
        """
            Propagates a value to every field driven by the leaf that owns "origin_path".
        """
        if len(origin_path) < 2:
            return
        base_path = self._inheritance_path(origin_path)
        children = self.get_inheritance_children(base_path)
        if not children:
            return
        # Sub-fields of an initializer are propagated to the matching sub-field of the driven initializers.
        suffix = list(origin_path[len(base_path):])
        for child_path in children:
            self.set_node_config_value(child_path + suffix, value, force=True)

    def toggle_inheritance(self, path: list[str], is_inheriting: bool) -> None:
        """
            Enables/disables the cascade of the leaf addressed by "path".
        """
        leaf = self.get_inheritance_leaf(path)
        if leaf is None:
            return
        # Leaves without descendants cannot cascade.
        if is_inheriting and not leaf.can_inherit():
            return
        if is_inheriting:
            leaf.flags |= InheritanceFlags.IS_INHERITING
        else:
            leaf.flags &= ~InheritanceFlags.IS_INHERITING
        tree = self.inheritance_trees.get(path[0], None)
        if tree is not None:
            tree.invalidate()
            tree.validate()
        self.inheritance_updated.emit()

    def add_node(self, node: NodeModel) -> None:
        if node not in self.nodes:
            self.nodes.append(node)
            self.node_added.emit(node)

    def remove_node(self, node: NodeModel) -> None:
        if node in self.nodes:
            # The edges of the node are removed first.
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
            'profile': self._profile.key if self._profile else None,
            'nodes': [node.to_dict() for node in self.nodes],
            'edges': [edge.to_dict() for edge in self.edges]
        }

    def from_dict(self, data) -> None:
        self.clear()
        if hasattr(self, 'undo_stack'):
            self.undo_stack.clear()
        # The controller type is dictated by the session.
        self.set_profile(get_controller_profile(data.get('profile', None)), force=True)


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