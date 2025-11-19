#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

from mailbox import Message
import typing as tp
import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from PySide6 import QtWidgets, QtCore, QtGui

import spark.core.utils as utils
from NodeGraphQt import Port
from functools import partial
from spark.core.config import BaseSparkConfig
from spark.core.payloads import SparkPayload, FloatArray
from spark.core.registry import REGISTRY
from spark.nn.initializers.base import InitializerConfig
from spark.graph_editor.editor import GRAPH_EDITOR_CONFIG
from spark.graph_editor.models.nodes import AbstractNode, SparkModuleNode, SourceNode, SinkNode
from spark.graph_editor.widgets.dock_panel import QDockPanel
from spark.graph_editor.widgets.line_edits import QStrLineEdit, QString, QInt, QFloat
from spark.graph_editor.widgets.separators import QHLine
from spark.graph_editor.widgets.base import SparkQWidget, QField, QMissing
from spark.graph_editor.widgets.collapsible import QCollapsible
from spark.graph_editor.widgets.combobox import QDtype, QBool, QGenericComboBox
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
        self.setMinimumWidth(GRAPH_EDITOR_CONFIG.inspector_panel_min_width)
        self._target_node = None
        self._multi_selection = False
        self._node_config_widget = None
        self.set_node(None)


    def on_selection_update(self, new_selection: list[AbstractNode], previous_selection: list[AbstractNode]) -> None:
        """
            Event handler for graph selections.
        """ 
        if len(new_selection) == 1:
            self.set_node(new_selection[0])
            self._multi_selection = False
        elif len(new_selection) == 0:
            # Keep the current interface state if the user didn't selected anything.
            if self._multi_selection:
                self._multi_selection = False
                self.set_node(None)
        else: 
            self._multi_selection = True
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
        if self._target_node is not None and self._target_node is node:
            return
        self._target_node = node
        
        # Update interface
        if self._target_node:
            if isinstance(self._target_node, SparkModuleNode):
                self._node_config_widget = QNodeConfig(self._target_node)
                self._node_config_widget.error_detected.connect(self.on_error_message)
            elif isinstance(self._target_node, (SinkNode, SourceNode)):
                self._node_config_widget = QNodeIO(node, parent=self)
            else: 
                self._node_config_widget = InspectorIdleWidget('Node type not supported.')
            self.setContent(
                self._node_config_widget
            )
        else:
            self.setContent(
                InspectorIdleWidget(
                    'Select a node to Inspect' if not self._multi_selection else 'Multi-selection not supported.'
                )
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
        # Get config
        node_config: BaseSparkConfig = getattr(self._target_node, 'node_config', None)
        # Add name widget
        name_widget = NodeHeaderWidget(
            name=self._target_node.NODE_NAME,
            node_cls=getattr(self._target_node, 'module_cls', None).__name__,
            config_tree=node_config._inspect(),
            parent=self,
        )
        name_widget.on_update.connect(
            partial(lambda node_var, name_var: self.name_update(node_var, name_var), self._target_node)
        )
        self.addWidget(name_widget)
        self.addWidget(QHLine())
        # Add configs recurrently
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
                parent = self,
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
            'ArrayLike': QFloat,
            # Type-based support
            bool: QBool,
            int: QInt,
            float: QFloat,
            ArrayLike: QFloat,
            str: QString,
            tuple[int, ...]: QShape,
            np.dtype: QDtype,
            np.dtype: QDtype,
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
                child_widget.attr_widget.setEnabled(False)
                # Link widgets updates.
                leaf_widget.field_updated.connect(child_widget.attr_widget.set_value)
                leaf_widget.field_updated.connect(child_widget.attr_widget.on_update.emit)
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
                child_widget.attr_widget.setEnabled(True)
                # Unlink widgets updates.
                leaf_widget.field_updated.disconnect(child_widget.attr_widget.set_value)
                leaf_widget.field_updated.disconnect(child_widget.attr_widget.on_update.emit)

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

class QNodeIO(QtWidgets.QWidget):
    """
        Constructs the UI to modify the SparkConfig associated with the a inputs and outpus.
    """

    error_detected = QtCore.Signal(str)

    def __init__(self, node: AbstractNode, parent: QtWidgets.QWidget = None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        # Node reference
        self._target_node = node
        self._is_source = isinstance(node, SourceNode)
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
        name_widget = NodeHeaderWidget(
            name=self._target_node.NODE_NAME,
            node_cls=self._target_node.__class__.__name__,
            config_tree=None,
            parent=self,
        )
        name_widget.on_update.connect(
            partial(lambda node_var, name_var: self.name_update(node_var, name_var), self._target_node)
        )
        self.addWidget(name_widget)
        self.addWidget(QHLine())

        # Container to maintain visual consistency
        collapsible = QCollapsible(utils.to_human_readable('Main'), parent=self)
        spec_model = self._target_node.output_specs['value'] if self._is_source else self._target_node.input_specs['value']
        port_name = 'value'
        target_spec = 'output' if self._is_source else 'input'

        # Add shape widget to collapsible
        shape_widget = QShape(
            spec_model.shape, 
            parent=self
        )
        shape_widget.setEnabled(self._is_source)
        if self._is_source:
            shape_widget.on_update.connect(
                lambda shape_var: self._target_node.update_io_spec(spec=target_spec, port_name=port_name, shape=shape_var)
            )
        # Construct QField
        self.shape_field = QField(
            utils.to_human_readable('Shape'),
            shape_widget,
            warning_value = False,
            inheritance_box = False,
            parent= self
        )

        # Add dtype widget to collapsible
        dtype_widget = QDtype(
            spec_model.dtype, 
            values_options=self._get_dtype_options(),
            parent=self
        )
        dtype_widget.setEnabled(self._is_source)
        if self._is_source:
            dtype_widget.on_update.connect(
                lambda dtype_var: self._target_node.update_io_spec(spec=target_spec, port_name=port_name, dtype=dtype_var)
            )
        # Construct QField
        self.dtype_field = QField(
            utils.to_human_readable('Dtype'),
            dtype_widget,
            warning_value = False,
            inheritance_box = False,
            parent= self
        )
        self.addWidget(self.dtype_field)

        # Add dtype widget to collapsible
        payload_widget = QGenericComboBox(
            spec_model.payload_type, 
            values_options=self._get_payload_options(),
            parent=self
        )
        payload_widget.setEnabled(self._is_source)
        if self._is_source:
            payload_widget.on_update.connect(
                lambda payload_widget_var: self._target_node.update_io_spec(
                    spec=target_spec, port_name=port_name, payload_type=payload_widget_var)
            )
        # Construct QField
        self.payload_field = QField(
            utils.to_human_readable('PayloadType'),
            payload_widget,
            warning_value = False,
            inheritance_box = False,
            parent= self
        )
        # Add widgets
        collapsible.addWidget(self.payload_field)
        collapsible.addWidget(self.shape_field)
        collapsible.addWidget(self.dtype_field)
        self.addWidget(collapsible)
        collapsible.expand()
        # Add live updates for sink nodes.
        if not self._is_source:
            self._target_node.graph.port_connected.connect(self._live_sink_update)

    def _live_sink_update(self, input_port: Port, output_port: Port) -> None:
        # Get node info
        input_node: AbstractNode = input_port.node()
        # Update sink node specs
        if isinstance(input_node, SinkNode):
            output_specs = output_port.node().output_specs[output_port.name()]
            if output_specs.payload_type:
                self.payload_field.attr_widget.set_value(output_specs.payload_type)
            if output_specs.shape:
                self.shape_field.attr_widget.set_value(output_specs.shape)
            if output_specs.dtype:
                self.dtype_field.attr_widget.set_value(output_specs.dtype)
        
    def name_update(self, node: AbstractNode, name: str) -> None:
        node.graph.viewer().node_name_changed.emit(node.id, name)
        # NOTE: The above line should be enough but it does not propagate to the node widget properly.
        node.set_name(name)

    def _get_payload_options(self, ) -> list[tuple[str, SparkPayload]]:
        options = {}
        for name, entry in REGISTRY.PAYLOADS.items():
            options[utils.to_human_readable(name, capitalize_all=True)] = entry.class_ref
        return options

    def _get_dtype_options(self, ) -> list[np.dtype]:
        raw_options = [                
            np.uint8,
            np.uint16,
            np.uint32,
            np.int8,
            np.int16,
            np.int32,
            np.float16,
            np.float32,
            np.bool,
        ]
        options = []
        for d in raw_options:
            options.append(d)
        return options

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class InspectorIdleWidget(QtWidgets.QWidget):
    """
        Constructs the UI shown when no valid node is selected.
    """

    def __init__(self, message: str, parent: QtWidgets.QWidget = None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        # Horizontal layout for message
        _message_widget = self._message(message)
        # Vertical layout for Widget
        layout = QtWidgets.QVBoxLayout()
        layout.addStretch()
        layout.addWidget(_message_widget)
        layout.addStretch()
        self.setLayout(layout)

    def _message(self, message: str) -> QtWidgets.QWidget:
        # Create new widget for message
        _message_widget = QtWidgets.QWidget()
        # Horizontal layout for the icon and label
        h_layout = QtWidgets.QHBoxLayout(self)
        icon_label = QtWidgets.QLabel()
        icon = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileDialogInfoView)
        icon_label.setPixmap(icon.pixmap(15, 15))
        label = QtWidgets.QLabel(message)
        # Horizontally center the content
        h_layout.addStretch()
        h_layout.addWidget(icon_label)
        h_layout.addWidget(label)
        h_layout.addStretch()
        # Set widget layout
        _message_widget.setLayout(h_layout)
        return _message_widget
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

class NodeHeaderWidget(SparkQWidget):
    """
        QWidget used for the name of nodes in the SparkGraphEditor's Inspector.
    """

    def __init__(
            self, 
            name: str, 
            node_cls: str,
            config_tree: str | None = None,
            parent: QtWidgets.QWidget | None = None,
            **kwargs
        ) -> None:
        super().__init__(parent=parent)
        # Add layout
        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(2)
        self._target_height = 0
        # Add name
        self.name_widget = NodeNameWidget(name, parent=self, **kwargs)
        self.name_widget.on_update.connect(self._on_update)
        layout.addWidget(self.name_widget)
        self._target_height += self.name_widget.size().height()
        # Class
        self.class_label = QtWidgets.QLabel(node_cls, parent=self)
        self.class_label.setContentsMargins(QtCore.QMargins(16, 0, 0, 12))
        self.class_label.setFixedHeight(32)
        layout.addWidget(self.class_label)
        self._target_height += self.class_label.size().height()
        # Config tree description
        if config_tree:
            # Config tree label
            self.config_tree_label = QtWidgets.QLabel('Configuration Tree', parent=self)
            self.config_tree_label.setContentsMargins(QtCore.QMargins(16, 8, 0, 4))
            self.config_tree_label.setFixedHeight(24)
            self.config_tree_label.setStyleSheet(
                f"""
                    font-size: {GRAPH_EDITOR_CONFIG.small_font_size}px;
                """
            )
            layout.addWidget(self.config_tree_label)
            self.tree_label = TreeDisplay(config_tree, parent=self)
            layout.addWidget(self.tree_label)
            self._target_height += self.config_tree_label.size().height()
            self._target_height += self.tree_label.size().height()
        # Finalize
        self.setLayout(layout)

    def sizeHint(self):
        return QtCore.QSize(super().sizeHint().width(), self._target_height)

    def get_value(self) -> str:
        return self.name_widget.get_value()

    def set_value(self, value: str) -> None:
        return self.name_widget.get_value(value)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class NodeNameWidget(SparkQWidget):
    """
        QWidget used for the name of nodes in the SparkGraphEditor's Inspector.
    """

    def __init__(
            self, 
            name: str, 
            parent: QtWidgets.QWidget | None = None,
            **kwargs
        ) -> None:
        super().__init__(parent=parent)
        # Add layout
        layout = QtWidgets.QHBoxLayout()
        # Add icon. The icon makes it look more professional c:
        _icon = QtGui.QPixmap(':/icons/node_icon.png')
        self._icon_label = QtWidgets.QLabel(parent=self)
        self._icon_label.setPixmap(_icon)
        self._icon_label.setScaledContents(True)
        self._icon_label.setMaximumWidth(32)
        self._icon_label.setMaximumHeight(32)
        layout.addWidget(self._icon_label)
        # Add QLineEdit
        self._line_edit = QStrLineEdit(name, parent=self, **kwargs)
        # Setup callback
        self._line_edit.editingFinished.connect(self._on_update)
        # Finalize
        layout.addWidget(self._line_edit)
        self.setLayout(layout)
        # CSS-ing is hard ¯\_(ツ)_/¯
        self._target_height = 64
        self.setFixedHeight(self._target_height)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)

    def sizeHint(self):
        return QtCore.QSize(super().sizeHint().width(), self._target_height)

    def get_value(self) -> str:
        return self._line_edit.text()

    def set_value(self, value: str) -> None:
        return self._line_edit.setText(value)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TreeDisplay(QtWidgets.QPlainTextEdit):
    def __init__(self, tree: str, parent: QtWidgets.QWidget = None) -> None:
        super().__init__(parent=parent)
        # QPlainTextEdit preserves formatting and monospacing
        self.setPlainText(tree)
        self.setReadOnly(True)
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.setStyleSheet(
            f"""
                QPlainTextEdit {{
                    border: none;
                    color: {GRAPH_EDITOR_CONFIG.default_font_color};
                    font-family: Courier;
                    font-size: {GRAPH_EDITOR_CONFIG.small_font_size}px;
                    margin: 0px 0px 0px 8px;
                    padding: 0px;
                }}
            """
        )
        # CSS-ing is hard ¯\_(ツ)_/¯
        rows_space = len(tree.split('\n')) * self.fontMetrics().boundingRect('M').height() * 1
        line_space = len(tree.split('\n')) * 2
        self._target_height = rows_space + line_space + 10
        self.setFixedHeight(self._target_height)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.setContentsMargins(QtCore.QMargins(16, 0, 0, 4))

    def sizeHint(self):
        return QtCore.QSize(super().sizeHint().width(), self._target_height)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################