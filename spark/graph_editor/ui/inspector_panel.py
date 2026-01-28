#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

from mailbox import Message
import typing as tp
import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from PySide6 import QtWidgets, QtCore, QtGui

import re
import jax
import jax.numpy as jnp
import spark.core.utils as utils
from NodeGraphQt import Port
from functools import partial
from spark.core.config import BaseSparkConfig
from spark.core.payloads import SparkPayload, FloatArray
from spark.core.registry import REGISTRY
from spark.nn.initializers.base import Initializer, InitializerConfig
from spark.graph_editor.editor import GRAPH_EDITOR_CONFIG
from spark.graph_editor.models.nodes import AbstractNode, SparkModuleNode, SourceNode, SinkNode
from spark.graph_editor.widgets.dock_panel import QDockPanel
from spark.graph_editor.widgets.line_edits import QStrLineEdit
from spark.graph_editor.widgets.separators import QHLine
from spark.graph_editor.widgets.base import QField, QInput
from spark.graph_editor.widgets.attributes import QConfigBody, QAttribute, TYPE_PARSER
from spark.graph_editor.widgets.collapsible import QCollapsible
from spark.graph_editor.widgets.combobox import QDtype, QGenericComboBox
from spark.graph_editor.widgets.shape import QShape
from spark.graph_editor.widgets.checkbox import InheritStatus
from spark.graph_editor.ui.console_panel import MessageLevel

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class InspectorPanel(QDockPanel):
    """
        Dockable panel to show node configurations.
    """

    broadcast_message = QtCore.Signal(MessageLevel, str)
    on_update = QtCore.Signal()

    def __init__(
            self, 
            name: str = 'Inspector', 
            **kwargs
        ) -> None:
        super().__init__(name, **kwargs)
        self.setMinimumWidth(GRAPH_EDITOR_CONFIG.inspector_panel_min_width)
        self._target_node = None
        self._multi_selection = False
        self._node_config_widget = None
        self.set_node(None)

    def clear_selection(self,):
        self._multi_selection = False
        self._target_node = None
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
        # Update metadata
        self.update_node_metadata()

        # Update node selection
        if self._target_node is not None and self._target_node is node:
            return
        self._target_node = node
        
        # Update interface
        if self._target_node:
            # Queue prior object for deletion
            if self._node_config_widget is not None:
                try:
                    self._node_config_widget.deleteLater()
                except:
                    # Object was already deleted by qt
                    pass
                self._node_config_widget = None
            if isinstance(self._target_node, SparkModuleNode):
                self._node_config_widget = QNodeConfig(self._target_node)
                self._node_config_widget.error_detected.connect(self.on_error_message)
                self._node_config_widget.on_update.connect(self._on_update)
            elif isinstance(self._target_node, (SinkNode, SourceNode)):
                self._node_config_widget = QNodeIO(node)
                self._node_config_widget.on_update.connect(self._on_update)
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

    def _on_update(self, *args, **kwargs) -> None:
        self.on_update.emit()

    def set_dirty_flag(self, value: bool) -> None:
        if self._node_config_widget is not None:
            self._node_config_widget.set_dirty_flag(value)

    def update_node_metadata(self,) -> None:
        if self._node_config_widget is not None and self._target_node is not None:
            if isinstance(self._target_node, SparkModuleNode):
                self._target_node.metadata['inheritance_tree'] = self._node_config_widget._inheritance_tree.to_dict()

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
    on_update = QtCore.Signal()

    def __init__(self, node: AbstractNode, **kwargs):
        super().__init__(**kwargs)
        # Node reference
        self._target_node = node
        self._cfg_widget_map: dict[tuple[str, ...], QAttribute] = {}
        self._post_setup_callbacks = []
        self._is_dirty = False
        self._inheritance_tree = self._get_inheritance_tree(node) 
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
        # Execute callbacks
        for f in self._post_setup_callbacks:
            f()



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
        )
        name_widget.on_update.connect(
            partial(lambda node_var, name_var: self.name_update(node_var, name_var), self._target_node)
        )
        self.addWidget(name_widget)
        self.addWidget(QHLine())
        # Add configs recurrently
        if node_config:
            self._add_config_recurrently('Main', node_config, ['main'])



    def name_update(self, node: AbstractNode, name: str) -> None:
        node.graph.viewer().node_name_changed.emit(node.id, name)
        # NOTE: The above line should be enough but it does not propagate to the node widget properly.
        node.set_name(name)
        self._on_update()


    def _add_config_recurrently(self, config_name: str, config: BaseSparkConfig, path: list[str] = []) -> None:
        # Create new QCollapsible
        collapsible = QCollapsible(utils.to_human_readable(config_name))
        # Instantiate config body
        config_body = QConfigBody(
            config=config, 
            is_initializer=False, 
            config_path=path,
            ref_widgets_map=self._cfg_widget_map, 
            ref_post_callback_stack=self._post_setup_callbacks,
            inheritance_tree=self._inheritance_tree,
            on_inheritance_toggle=self.on_inheritance_toggle,
        )
        config_body.on_update.connect(self._on_update)
        
        # Add widget to collapsible
        collapsible.addWidget(config_body)
        config_body.on_size_change.connect(lambda _ : collapsible.expand(False))
        # Add collapsible to layout
        self.addWidget(collapsible)
        collapsible.expand()
        # Iterate over fields, add other configs recurrently
        nested_configs = config._get_nested_configs_names(exclude_initializers=True)
        for name, field, value in config:
            # Skip non-configs
            if name not in nested_configs:
                continue
            # Add widget to collapsible
            self._add_config_recurrently(name, value, path + [name])



    def _get_inheritance_tree(self, node: AbstractNode) -> utils.InheritanceTree:
        # Check if inheritance tree was already computed
        inheritance_metadata = node.metadata.get('inheritance_tree', None)
        if inheritance_metadata is not None:
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
            # Link all components that define units and dt (I cannot think of a case in which this is not desireable c: )
            link_paths = [
                ['main', 'units'], 
                ['main', 'dt']
            ]
            for path in link_paths:
                try:
                    leaf = tree.get_leaf(path)
                    if leaf.can_inherit():
                        self._post_setup_callbacks.append(partial(
                            lambda leaf_var:
                            self.on_inheritance_toggle(True, leaf_var),
                            leaf
                        ))
                except:
                    # We expect get_leaf to fail
                    pass
        else:
            # I/O nodes have no node_config. However, they probably do not require an inheritance tree.
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
            paths.append({'path': path + (name,), 'types':TYPE_PARSER.typify(field.type), 'break_inheritance': break_inheritance})
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



    def on_inheritance_toggle(self, value: bool, leaf: utils.InheritanceLeaf) -> None:
        # Set leaf flags.
        if value:
            leaf.flags = leaf.flags | utils.InheritanceFlags.IS_INHERITING
        else:
            leaf.flags = leaf.flags & ~utils.InheritanceFlags.IS_INHERITING
        # Update leaf widget
        leaf_widget: QAttribute = self._cfg_widget_map[tuple(leaf.path)]
        leaf_widget.set_inheritance_status(InheritStatus.LINK if value else InheritStatus.FREE)
        # Link relevant childs updates.
        for child_path in leaf.inheritance_childs:
            # Set child flags.
            child_leaf = leaf.parent.get_leaf(child_path)
            if value:
                child_leaf.flags = (child_leaf.flags | utils.InheritanceFlags.IS_RECEIVING) & ~utils.InheritanceFlags.IS_INHERITING
            else:
                child_leaf.flags = child_leaf.flags & ~utils.InheritanceFlags.IS_RECEIVING
            # Get child widget.
            child_widget: QAttribute = self._cfg_widget_map[tuple(child_leaf.path)]
            # Disable child widget interaction.
            child_widget.set_inheritance_status(InheritStatus.LOCK if value else InheritStatus.FREE)
            if child_widget.control_widgets.initializer_checkbox is not None:
                child_widget.control_widgets.initializer_checkbox.setEnabled(not value)
            child_widget.default_field.setEnabled(not value)
            # Connect events
            if value:
                leaf_widget.on_field_update.connect(child_widget.set_value)
                leaf_widget.on_field_update.connect(child_widget.on_field_update.emit)
                # Update childs value
                child_widget.set_value(leaf_widget.get_value())
            else:
                leaf_widget.on_field_update.disconnect(child_widget.set_value)
                leaf_widget.on_field_update.disconnect(child_widget.on_field_update.emit)
        # TODO: This feels extremely inneficient
        # Update node's metadata
        #self._target_node.metadata['inheritance_tree'] = self._inheritance_tree.to_dict()

    def _on_update(self,) -> None:
        self._is_dirty = True
        self.on_update.emit()

    def set_dirty_flag(self, value: bool) -> None:
        self._is_dirty = value

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class QNodeIO(QtWidgets.QWidget):
    """
        Constructs the UI to modify the SparkConfig associated with the a inputs and outpus.
    """

    error_detected = QtCore.Signal(str)
    on_update = QtCore.Signal()

    def __init__(self, node: AbstractNode, **kwargs):
        super().__init__(**kwargs)
        # Node reference
        self._target_node = node
        self._is_source = isinstance(node, SourceNode)
        self._is_dirty = False
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
        content_layout = QtWidgets.QVBoxLayout(self.content)
        content_layout.setContentsMargins(QtCore.QMargins(0, 0, 0, 0))
        content_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        scroll_area.setWidget(self.content)
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
        )
        name_widget.on_update.connect(
            partial(lambda node_var, name_var: self.name_update(node_var, name_var), self._target_node)
        )
        self.content.layout().addWidget(name_widget)
        self.content.layout().addWidget(QHLine())

        # Container to maintain visual consistency
        collapsible = QCollapsible(utils.to_human_readable('Main'))
        self.content.layout().addWidget(collapsible)
        spec_model = self._target_node.output_specs['value'] if self._is_source else self._target_node.input_specs['value']
        port_name = 'value'
        target_spec = 'output' if self._is_source else 'input'

        collapsible_content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(collapsible_content)
        layout.setContentsMargins(QtCore.QMargins(8,8,8,8))
        layout.setSpacing(0)


        # Add shape widget to collapsible
        self.shape_field = QField(
            attr_label='Shape',
            attr_value=spec_model.shape,
            attr_widget=QShape, 
        )
        self.shape_field.setEnabled(self._is_source)
        if self._is_source:
            self.shape_field.on_field_update.connect(
                lambda shape_var: self._target_node.update_io_spec(spec=target_spec, port_name=port_name, shape=shape_var)
            )
            self.shape_field.on_field_update.connect(self._on_update)
        layout.addWidget(self.shape_field)

        # Add payload widget to collapsible
        self.payload_field = QField(
            attr_label='Payload Type',
            attr_value=spec_model.payload_type,
            attr_widget=QGenericComboBox, 
            value_options=self._get_payload_options(),
        )
        self.payload_field.setEnabled(self._is_source)
        if self._is_source:
            self.payload_field.on_field_update.connect(
                lambda payload_widget_var: self._target_node.update_io_spec(
                    spec=target_spec, port_name=port_name, payload_type=payload_widget_var)
            )
            self.payload_field.on_field_update.connect(self._on_update)
        layout.addWidget(self.payload_field)

        # Add dtype widget to collapsible
        self.dtype_field = QField(
            attr_label='Dtype',
            attr_value=spec_model.dtype,
            attr_widget=QDtype, 
            value_options=self._get_dtype_options(),
        )
        self.dtype_field.setEnabled(self._is_source)
        if self._is_source:
            self.dtype_field.on_field_update.connect(
                lambda dtype_var: self._target_node.update_io_spec(spec=target_spec, port_name=port_name, dtype=dtype_var)
            )
            self.dtype_field.on_field_update.connect(self._on_update)
        layout.addWidget(self.dtype_field)
        collapsible.addWidget(collapsible_content)
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
                self.payload_field.set_value(output_specs.payload_type)
            if output_specs.shape:
                self.shape_field.set_value(output_specs.shape)
            if output_specs.dtype:
                self.dtype_field.set_value(output_specs.dtype)
        
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

    def _on_update(self,) -> None:
        self._is_dirty = True
        self.on_update.emit()

    def set_dirty_flag(self, value: bool) -> None:
        self._is_dirty = value

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class InspectorIdleWidget(QtWidgets.QWidget):
    """
        Constructs the UI shown when no valid node is selected.
    """

    def __init__(self, message: str, **kwargs):
        super().__init__(**kwargs)
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

class NodeHeaderWidget(QInput):
    """
        QWidget used for the name of nodes in the SparkGraphEditor's Inspector.
    """

    def __init__(
            self, 
            name: str, 
            node_cls: str,
            config_tree: str | None = None,
            **kwargs
        ) -> None:
        super().__init__()
        # Add layout
        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(2)
        self._target_height = 0
        # Add name
        self.name_widget = NodeNameWidget(name, **kwargs)
        self.name_widget.on_update.connect(self._on_update)
        layout.addWidget(self.name_widget)
        self._target_height += self.name_widget.size().height()
        # Class
        self.class_label = QtWidgets.QLabel(node_cls)
        self.class_label.setContentsMargins(QtCore.QMargins(16, 0, 0, 12))
        self.class_label.setFixedHeight(32)
        layout.addWidget(self.class_label)
        self._target_height += self.class_label.size().height()
        # Config tree description
        if config_tree:
            # Config tree label
            self.config_tree_label = QtWidgets.QLabel('Configuration Tree')
            self.config_tree_label.setContentsMargins(QtCore.QMargins(16, 8, 0, 4))
            self.config_tree_label.setFixedHeight(24)
            self.config_tree_label.setStyleSheet(
                f"""
                    font-size: {GRAPH_EDITOR_CONFIG.small_font_size}px;
                """
            )
            layout.addWidget(self.config_tree_label)
            self.tree_label = TreeDisplay(config_tree)
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

class NodeNameWidget(QInput):
    """
        QWidget used for the name of nodes in the SparkGraphEditor's Inspector.
    """

    def __init__(
            self, 
            name: str, 
            **kwargs
        ) -> None:
        super().__init__()
        # Add layout
        layout = QtWidgets.QHBoxLayout()
        # Add icon. The icon makes it look more professional c:
        _icon = QtGui.QPixmap(':/icons/node_icon.png')
        self._icon_label = QtWidgets.QLabel()
        self._icon_label.setPixmap(_icon)
        self._icon_label.setScaledContents(True)
        self._icon_label.setMaximumWidth(32)
        self._icon_label.setMaximumHeight(32)
        layout.addWidget(self._icon_label)
        # Add QLineEdit
        self._line_edit = QStrLineEdit(name, **kwargs)
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
    def __init__(self, tree: str, ) -> None:
        super().__init__()
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