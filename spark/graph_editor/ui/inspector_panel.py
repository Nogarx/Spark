#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

from mailbox import Message
import typing as tp
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike
from PySide6 import QtWidgets, QtCore, QtGui

import spark.core.utils as utils
from functools import partial
from spark.core.config import BaseSparkConfig
from spark.nn.initializers.base import InitializerConfig
from spark.graph_editor.models.nodes import AbstractNode, SparkModuleNode, SourceNode, SinkNode
from spark.graph_editor.widgets.dock_panel import QDockPanel
from spark.graph_editor.widgets.inspector_idle import InspectorIdleWidget
from spark.graph_editor.widgets.line_edits import NodeNameWidget, QString, QInt, QFloat
from spark.graph_editor.widgets.separators import QHLine
from spark.graph_editor.widgets.base import SparkQWidget, QField, QMissing
from spark.graph_editor.widgets.collapsible import QCollapsible
from spark.graph_editor.widgets.combobox import QDtype, QBool
from spark.graph_editor.widgets.shape import QShape
from spark.graph_editor.ui.console_panel import MessageLevel

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class InspectorPanel(QDockPanel):
    """
        Dockable panel to show node configurations.
    """

    broadcast_message = QtCore.Signal(MessageLevel, str)

    def __init__(
            self, 
            name: str = 'Inspector', 
            parent: QtWidgets.QWidget = None, 
            **kwargs
        ) -> None:
        super().__init__(name, parent=parent, **kwargs)
        self._target_node = None
        self._node_config_widget = None


    def on_selection_update(self, new_selection: list[AbstractNode], previous_selection: list[AbstractNode]) -> None:
        """
            Event handler for graph selections.
        """ 
        if len(new_selection) == 1:
            self.set_node(new_selection[0])
        elif len(new_selection) == 0:
            # Keep the current interface state if the user didn't selected anything.
            pass
        else: 
            self.set_node(None)

    def set_node(self, node: AbstractNode | None):
        """
            Sets the node to be inspected. If the node is None, it clears the inspector.
            
            Input:
                node: The node to inspect, or None to clear.
        """
        # Update node_config_metadata
        if self._target_node:
            # Update target_node metadata 
            if isinstance(self._target_node, SparkModuleNode):
                self._target_node.node_config_metadata['inheritance_tree'] = self._node_config_widget._inheritance_tree.to_dict()

        # Update node selection
        if self._target_node is node:
            return
        self._target_node = node
        
        # Update interface
        if self._target_node:
            if isinstance(self._target_node, SparkModuleNode):
                self._node_config_widget = QNodeConfig(self._target_node)
                self._node_config_widget.error_detected.connect(self.on_error_message)
            elif isinstance(self._target_node, (SinkNode, SourceNode)):
                self._node_config_widget = InspectorIdleWidget()
            else: 
                self._node_config_widget = InspectorIdleWidget()
            self.setContent(
                self._node_config_widget
            )
        else:
            self.setContent(
                InspectorIdleWidget()
            )

    def on_error_message(self, message: str) -> None:
        self.broadcast_message.emit(MessageLevel.ERROR, f'{self._target_node.NODE_NAME} -> {message}')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class QNodeConfig(QtWidgets.QWidget):
    """
        Constructs the UI to modify the SparkConfig associated with the a node.
    """

    error_detected = QtCore.Signal(str)

    def __init__(self, node: AbstractNode, parent: QtWidgets.QWidget = None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        # Node reference
        self._target_node = node
        self._cfg_widget_map: dict[tuple[str, ...], dict[str, QField]] = {}
        self._inheritance_tree = self._get_inheritance_tree(node) 
        self._post_setup_callbacks = []
        # Widget layout
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(QtCore.QMargins(0, 0, 0, 0))
        self.setLayout(layout)
        # Scroll area
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.layout().addWidget(scroll_area)
        scroll_area.setStyleSheet(
            """
                QScrollArea { 
                    background: transparent; 
                }
            """
        )
        # Content widget
        self.content = QtWidgets.QWidget()
        self.content.setStyleSheet(
            """
                QWidget { 
                    background: transparent; 
                }
            """
        )
        scroll_area.setWidget(self.content)
        content_layout = QtWidgets.QVBoxLayout(self.content)
        content_layout.setContentsMargins(QtCore.QMargins(0, 0, 0, 0))
        content_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        # Setup layout
        self._setup_layout()


    def addWidget(self, widget: QtWidgets.QWidget) -> None:
        """
            Add a widget to the central content widget's layout.
        """
        widget.installEventFilter(self)
        self.content.layout().addWidget(widget)

    def _setup_layout(self,) -> None:

        # Add name widget
        name_widget = NodeNameWidget(
            initial_value=self._target_node.NODE_NAME,
            parent=self,
        )
        name_widget.on_update.connect(
            partial(lambda node_var, name_var: self.name_update(node_var, name_var), self._target_node)
        )
        self.addWidget(name_widget)
        self.addWidget(QHLine())
        # Add configs recurrently
        node_config: BaseSparkConfig = getattr(self._target_node, 'node_config', None)
        if node_config:
            self._add_config_recurrently('Main', node_config, ['main'])
        # Execute callbacks
        for f in self._post_setup_callbacks:
            f()

    def name_update(self, node: AbstractNode, name: str) -> None:
       node.graph.viewer().node_name_changed.emit(node.id, name)
       # NOTE: The above line should be enough but it does not propagate to the node widget properly.
       node.set_name(name)

    def _add_config_recurrently(self, config_name: str, config: BaseSparkConfig, path: list[str] = []) -> None:
        # Create new QCollapsible
        collapsible = QCollapsible(utils.to_human_readable(config_name), parent=self)
        # Get nested config's names
        nested_configs = config._get_nested_configs_names()
        type_hints = config._get_type_hints()
        # Exapnd widget map
        self._cfg_widget_map[tuple(path)] = {}
        # Iterate over fields, add simple attributes
        for name, field, value in config:
            # Skip non-configs
            if name in nested_configs:
                continue

            # TODO: We need a reliable way to map TypeVar to widgets.
            # Use QStrings as a simple workaround for Initializers.
            field_type = field.type
            if isinstance(config, InitializerConfig):
                if isinstance(type_hints[name][0], tp.TypeVar):
                    field_type = 'str'

            # Add widget to collapsible
            update_func = partial(lambda config_var, name_var, value: setattr(config_var, name_var, value), config, field.name)
            attr_widget = self._attr_widget(
                field_type,
                update_func,
                value = value,
                metadata= field.metadata,
            )
            # Construct QField
            leaf = self._inheritance_tree.get_leaf(path+[name])
            field_widget = QField(
                utils.to_human_readable(name),
                attr_widget,
                warning_value = False,
                inheritance_box = leaf.can_inherit() or leaf.can_receive(),
                inheritance_interactable = leaf.can_inherit(),
                inheritance_value = leaf.is_inheriting(),
                parent= self
            )
            # Scan for errors
            self._validate_fields(config, name, field_widget, None)
            attr_widget.on_update.connect(
                partial(
                    lambda config_var, name_var, field_var, value: 
                    self._validate_fields(config_var, name_var, field_var, value), 
                    config, name, field_widget
                )
            )
            # Set inheritance events.
            if leaf.is_inheriting():
                self._post_setup_callbacks.append(
                    partial(lambda leaf_var, path_var: self.on_inheritance_toggle(True, leaf_var, path_var), leaf, path)
                )
            field_widget.inheritance_toggled.connect(
                partial(lambda leaf_var, path_var, value: self.on_inheritance_toggle(value, leaf_var, path_var), leaf, path)
            )
            # Add widget to collapsible
            collapsible.addWidget(field_widget)
            # Add widget to map 
            self._cfg_widget_map[tuple(path)][name] = field_widget
        # Add collapsible to layout
        collapsible.expand()
        self.addWidget(collapsible)

        # Iterate over fields, add other configs recurrently
        for name, field, value in config:
            # Skip non-configs
            if name not in nested_configs:
                continue
            # Add widget to collapsible
            self._add_config_recurrently(name, value, path + [name])

    def _attr_widget(
            self, 
            attr_type: tuple[type, ...], 
            attr_update_method: tp.Callable,
            value: tp.Any | None = None, 
            metadata: dict[str, tp.Any] = None
        ) -> SparkQWidget:
        """
            Creates and adds an appropriate widget for a given argument specification.
        """
        # Get appropiate widget
        widget_class = self._get_widget_class_for_type(attr_type)
        if issubclass(widget_class, QMissing):
            widget = widget_class()
            return widget
        # Initialize widget
        if issubclass(widget_class, QDtype):
            widget = widget_class(value, values_options=metadata['value_options'])
        else:
            widget = widget_class(value)
        widget.on_update.connect(attr_update_method)
        return widget

    # TODO: This method needs to define a widget preference order since more than one widget may be a viable option.
    def _get_widget_class_for_type(self, attr_type: tuple[type, ...]) -> type[QtWidgets.QWidget]:
        """
            Maps an argument type to a corresponding widget class.
        """
        # NOTE: attr_type is almost always a str due to __future__. Split the types. 
        if isinstance(attr_type, str):
            attr_type = [t.strip() for t in attr_type.split('|')]

        WIDGET_TYPE_MAP = {
            # Str-based support
            'bool': QBool,
            'int': QInt,
            'float': QFloat,
            'jax.Array': QFloat,
            'str': QString,
            'list[int]': QShape,
            'tuple[int]': QShape,
            'tuple[int, ...]': QShape,
            'np.dtype': QDtype,
            'jnp.dtype': QDtype,
            'DTypeLike': QDtype,
            # Type-based support
            bool: QBool,
            int: QInt,
            float: QFloat,
            jax.Array: QFloat,
            str: QString,
            tuple[int, ...]: QShape,
            np.dtype: QDtype,
            jnp.dtype: QDtype,
            DTypeLike: QDtype,
        }
        for t in attr_type:
            widget_cls = WIDGET_TYPE_MAP.get(t, None)
            if widget_cls:
                return widget_cls
        return QMissing

    def _get_inheritance_tree(self, node: AbstractNode) -> utils.InheritanceTree:
        # Check if inheritance tree was already computed
        inheritance_metadata = node.node_config_metadata.get('inheritance_tree', None)
        if inheritance_metadata:
            try:
                return utils.InheritanceTree.from_dict(inheritance_metadata)
            except:
                pass
        # Tree doesn't exists or is not valid.
        if isinstance(node, SparkModuleNode):
            tree = utils.InheritanceTree(path=[])
            leaves_paths_types = self._get_config_paths_and_types(node.node_config, path=('main',))
            for d in leaves_paths_types:
                tree.add_leaf(d['path'], type_string=d['types'], break_inheritance=d['break_inheritance'])
            tree.validate()
        else:
            # TODO: Input / Output nodes have no node_config. Moreover it is not clear that they require an inheritance tree.
            pass
        return tree

    def _get_config_paths_and_types(
            self, 
            config: BaseSparkConfig, 
            path: tuple[str] = tuple(), 
            break_inheritance: bool = False
        ) -> list[dict]:
        paths = []
        # Get nested config's names
        nested_configs = config._get_nested_configs_names()
        # Iterate over fields, add simple attributes
        for name, field, _ in config:
            # Skip non-configs
            if name in nested_configs:
                continue
            # Field is an attribute -> leaf
            paths.append({'path': path + (name,), 'types':field.type, 'break_inheritance': break_inheritance})
        # Iterate over fields, add other configs recurrently
        for name, _, value in config:
            # Skip non-configs
            if name not in nested_configs:
                continue
            # Field is a config -> branch. Break inheritance chain if config is an initializer,
            # initializer often have their own rules.
            break_branch_inheritance = isinstance(value, InitializerConfig)
            paths += self._get_config_paths_and_types(value, path + (name,), break_inheritance=break_branch_inheritance)
        return paths

    def on_inheritance_toggle(self, value: bool, leaf: utils.InheritanceLeaf, path: list[str]) -> None:
        if value:
            # Set leaf flags.
            leaf.flags = leaf.flags | utils.InheritanceFlags.IS_INHERITING
            # Link relevant childs updates.
            leaf_widget = self._cfg_widget_map[tuple(path)][leaf.name]
            for child_path in leaf.inheritance_childs:
                # Set child flags.
                child_leaf = leaf.parent.get_leaf(child_path)
                child_leaf.flags = (child_leaf.flags | utils.InheritanceFlags.IS_RECEIVING) & ~utils.InheritanceFlags.IS_INHERITING
                # Get child widget.
                sub_path = child_path[:-1]
                child_name = child_path[-1]
                child_widget = self._cfg_widget_map[tuple(path + sub_path)][child_name]
                # Disable child widget interaction.
                child_widget.inheritance_checkbox._set_virtual_icon_state(True)
                child_widget.inheritance_checkbox.setCheckable(False)
                child_widget.widget.setEnabled(False)
                # Link widgets updates.
                leaf_widget.field_updated.connect(child_widget.widget.set_value)
                leaf_widget.field_updated.connect(child_widget.widget.on_update.emit)
        else:
            # Set leaf flags.
            leaf.flags = leaf.flags & ~utils.InheritanceFlags.IS_INHERITING
            # Unlink relevant childs updates.
            leaf_widget = self._cfg_widget_map[tuple(path)][leaf.name]
            for child_path in leaf.inheritance_childs:
                # Set child flags.
                child_leaf = leaf.parent.get_leaf(child_path)
                child_leaf.flags = child_leaf.flags & ~utils.InheritanceFlags.IS_RECEIVING
                # Get child widget.
                sub_path = child_path[:-1]
                child_name = child_path[-1]
                child_widget = self._cfg_widget_map[tuple(path + sub_path)][child_name]
                # Disable child widget interaction.
                child_widget.inheritance_checkbox._set_virtual_icon_state(False)
                child_widget.inheritance_checkbox.setCheckable(True)
                child_widget.widget.setEnabled(True)
                # Unlink widgets updates.
                leaf_widget.field_updated.disconnect(child_widget.widget.set_value)
                leaf_widget.field_updated.disconnect(child_widget.widget.on_update.emit)

    def _validate_fields(self, config: BaseSparkConfig, field_name:str, q_field: QField, value: tp.Any) -> None:
        # Scan for errors
        errors = config.get_field_errors(field_name)
        # Warn the users if errors
        q_field.warning_flag.set_error_status(errors)
        # Broadcast errors
        for e in errors:
            self.error_detected.emit(e)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################