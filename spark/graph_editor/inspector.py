#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.config import BaseSparkConfig
    
import dataclasses as dc
import typing as tp
import jax.numpy as jnp
from functools import partial
from Qt import QtCore, QtWidgets, QtGui
from spark.nn.initializers.delay import DelayInitializerConfig, _DELAY_CONFIG_REGISTRY
from spark.nn.initializers.kernel import KernelInitializerConfig, _KERNEL_CONFIG_REGISTRY
from spark.graph_editor.nodes import AbstractNode, SinkNode, SourceNode, SparkModuleNode
from spark.graph_editor.utils import _normalize_section_header, _to_human_readable
from spark.graph_editor.widgets.dict import QDict
from spark.graph_editor.widgets.combobox import QDtype, QBool
from spark.graph_editor.widgets.shape import QShape
from spark.graph_editor.widgets.missing import QMissing
from spark.graph_editor.widgets.collapsible import QCollapsible
from spark.graph_editor.widgets.line_edits import QInt, QFloat, QString
from spark.graph_editor.widgets.initializer import QInitializer
from spark.graph_editor.widgets.base import SparkQWidget
from spark.graph_editor.editor_config import GRAPH_EDITOR_CONFIG

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class NodeInspectorWidget(QtWidgets.QWidget):
    """A widget to inspect and edit the properties of a node in a graph."""

    onWidgetUpdate = QtCore.Signal(str, tp.Any)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._target_node: AbstractNode | None = None
        self._widgets_map: dict[str, QtWidgets.QWidget] = {}
        self._main_layout: QtWidgets.QVBoxLayout | None = None
        self._sections: dict[str, QCollapsible] = {}
        self._setup_ui()
        self._build_idle_menu()

    def set_node(self, node: AbstractNode | None):
        """
        Sets the node to be inspected. If the node is None, it clears the inspector.
        
        Args:
            node: The node to inspect, or None to clear.
        """
        if self._target_node is node:
            return

        self._target_node = node
        self._clear_layout()

        if self._target_node:
            self._build_node_menu()
        else:
            self._build_idle_menu()

    def _setup_ui(self):
        """Initializes the main UI components and layout."""
        top_level_layout = QtWidgets.QVBoxLayout(self)
        top_level_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        top_level_layout.addWidget(scroll_area)

        scroll_content_widget = QtWidgets.QWidget()
        scroll_area.setWidget(scroll_content_widget)

        self._main_layout = QtWidgets.QVBoxLayout(scroll_content_widget)
        self._main_layout.setContentsMargins(16, 4, 16, 4)
        self._main_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        self.setMinimumWidth(400)
        self.setMaximumWidth(800)

    def _build_node_menu(self):
        """Constructs the inspectype_hints['config']tor UI for the currently selected node."""
        if not self._target_node:
            return
        self._build_header()
        self._build_shape_editors()
        self._build_main_section()

    def _build_idle_menu(self):
        """Constructs the UI shown when no node (multiple nodes) is (are) selected."""
        self._build_header()
        # Horizontal separator
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self._main_layout.addWidget(line)
        
        icon_label = QtWidgets.QLabel()
        icon = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileDialogInfoView)
        icon_label.setPixmap(icon.pixmap(15, 15))

        label = QtWidgets.QLabel('Select a node to Inspect')
        label.setFont(label.font())
        label.setAlignment(QtCore.Qt.AlignCenter)

        row_layout = QtWidgets.QHBoxLayout()
        row_layout.addStretch()
        row_layout.addWidget(icon_label)
        row_layout.addWidget(label)
        row_layout.addStretch()

        self._main_layout.addStretch()
        self._main_layout.addLayout(row_layout)
        self._main_layout.addStretch()

    def _build_header(self):
        """Builds the header section of the inspector."""
        icon_label = QtWidgets.QLabel('🔎')
        font = icon_label.font()
        font.setBold(True)
        font.setPointSize(20)
        icon_label.setFont(font)
        
        header_label = QtWidgets.QLabel('Node Inspector')
        font = header_label.font()
        font.setBold(True)
        font.setPointSize(17)
        header_label.setFont(font)
        
        row_layout = QtWidgets.QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addStretch()
        row_layout.addWidget(icon_label)
        row_layout.addWidget(header_label)
        row_layout.addStretch()

        self._main_layout.addLayout(row_layout)
        self._main_layout.addSpacing(10)

    def _build_shape_editors(self):
        """Builds editors for the node's input and output shapes."""
        if not self._target_node:
            return

        shapes_section = self._add_section('Shapes')
        for name, spec in self._target_node.input_specs.items():
            is_static = True #not isinstance(self._target_node, SinkNode)
            widget = QShape(
                _to_human_readable(name), 
                initial_shape=spec.shape, 
                is_static=is_static
            )
            widget.on_update.connect(
                partial(lambda var_name, value: self._target_node.update_input_shape(var_name, value), name)
            )
            shapes_section.addWidget(widget)
            self._widgets_map[f'input_port.{name}'] = widget

        for name, spec in self._target_node.output_specs.items():
            is_static = not isinstance(self._target_node, SourceNode)
            widget = QShape(
                _to_human_readable(name), 
                initial_shape=spec.shape, 
                is_static=is_static
            )
            widget.on_update.connect(
                partial(lambda var_name, value: self._target_node.update_output_shape(var_name, value), name)
            )
            shapes_section.addWidget(widget)
            self._widgets_map[f'output_port.{name}'] = widget
        # Open and lock section
        shapes_section.expand()
        shapes_section.setLocked(True)

    def _build_main_section(self):
        """Builds the main section with node name and simple arguments."""
        if not self._target_node:
            return
        
        main_section = self._add_section('Main')
        # Add name
        self._add_attr_widget_to_layout(
            main_section, 
            'name', 
            'str', 
            lambda name: self._target_node.set_name(name), 
            self._target_node.NODE_NAME
        )
        # Add shape widgets to source/sink nodes.
        if isinstance(self._target_node, (SinkNode, SourceNode)):
            pass
        # Main init_args
        elif isinstance(self._target_node, SparkModuleNode):
            self._add_widgets_from_spark_config(main_section, self._target_node.node_config)
        else:
            # Unknown node type raise error.
            raise TypeError(f'Support for node of type \"{self._target_node.__class__.__name__}\" is not implemented.')
        
        # Open and lock section
        main_section.expand()
        main_section.setLocked(True)

    def _add_widgets_from_spark_config(self, section: QCollapsible, config: BaseSparkConfig):

        # Get nested config's names
        nested_configs = config._get_nested_configs_names()

        # Iterate over fields, add simple attributes
        for field in dc.fields(config):
            # Skip non-configs
            if field.name in nested_configs:
                continue
            # Skip config-metadata
            if field.name in ['__config_delimiter__', '__shared_config_delimiter__']:
                continue
            # Add widget to collapsible
            #if field.name == 'kernel_initializer':
            self._add_attr_widget_to_layout(
                section, 
                field.name, 
                field.type, 
                partial(lambda config_var, name_var, value: setattr(config_var, name_var, value), config, field.name),
                getattr(config, field.name), 
                metadata=field.metadata,
            )

        # Iterate over fields, add nested configs as new collapsibles
        for field in dc.fields(config):
            # Skip configs
            if field.name not in nested_configs:
                continue
            # Skip config-metadata
            if field.name in ['__config_delimiter__', '__shared_config_delimiter__']:
                continue
            # Create a collapsable window
            if not field.name in self._sections:
                self._add_section(field.name, section)
            subsection = self._get_section(field.name)
            # Add collapsible recursively
            type_hints = tp.get_type_hints(config.__class__)
            if issubclass(type_hints[field.name], (KernelInitializerConfig, DelayInitializerConfig)):
                # Initialize an initializer config if none is provided
                class_type = getattr(config, field.name).__class__
                if class_type in [KernelInitializerConfig, DelayInitializerConfig, type(None)]:
                    name_field = {f.name: f for f in dc.fields(type_hints[field.name])}['name']
                    default_initializer = name_field.default if name_field.default else name_field.metadata['value_options'][0]
                    if issubclass(type_hints[field.name], KernelInitializerConfig):
                        default_config = _KERNEL_CONFIG_REGISTRY[default_initializer]()
                    else: 
                        default_config = _DELAY_CONFIG_REGISTRY[default_initializer]()
                    setattr(config, field.name, default_config)
                initializer_config = getattr(config, field.name)
                # Add widget to inspector
                widget = QInitializer(initializer_config, subsection)
                attr_update_method = partial(lambda config_var, name_var, value: setattr(config_var, name_var, value), config, field.name)
                widget.on_update.connect(attr_update_method)
                subsection.addWidget(widget)
                self._widgets_map[field.name] = widget
            else:
                self._add_widgets_from_spark_config(subsection, getattr(config, field.name))

    def _add_attr_widget_to_layout(self, 
            section: QCollapsible, 
            attr_name: str, 
            attr_type: type, 
            attr_update_method: tp.Callable,
            value: tp.Any | None = None, 
            metadata: dict[str, tp.Any] = None) -> SparkQWidget:
        """
            Creates and adds an appropriate widget for a given argument specification.
        """
        # Get appropiate widget
        widget_class = self._get_widget_class_for_type(attr_type)
        if widget_class == QMissing:
            widget: QMissing = widget_class(_to_human_readable(attr_name))
            section.addWidget(widget)
            self._widgets_map[attr_name] = None
            return widget
        # Initialize widget
        if widget_class == QDtype:
            widget: QDtype = widget_class(_to_human_readable(attr_name), value, values_options=metadata['value_options'])
        else:
            widget = widget_class(_to_human_readable(attr_name), value)
        widget.on_update.connect(attr_update_method)
        section.addWidget(widget)
        self._widgets_map[attr_name] = widget
        return widget

    def _add_section(self, name: str, parent_section: QCollapsible = None) -> QCollapsible:
        """
            Adds a new collapsible section with a header and a form layout.
        """
        # Avoid section duplicates to avoid data degeneracy.
        if name in self._sections:
            raise ValueError(f'Name \"{name}\" is already in use by another section.')
        # Create new collapsible
        new_section = QCollapsible(_to_human_readable(name))
        if parent_section:
            parent_section.addWidget(new_section)
        else:
            self._main_layout.addWidget(new_section)
        # Add section reference
        self._sections[name] = new_section
        return new_section

    def _get_section(self, name: str) -> QCollapsible:
        return self._sections[name]

    def _get_widget_class_for_type(self, attr_type: type) -> type[QtWidgets.QWidget]:
        """
            Maps an argument type to a corresponding widget class.
        """
        WIDGET_TYPE_MAP = {
            'bool': QBool,
            'int': QInt,
            'float': QFloat,
            'str': QString,
            'tuple': QShape,
            'DTypeLike': QDtype,
            'dict': QDict,
        }
        return WIDGET_TYPE_MAP.get(attr_type, QMissing)

    def _clear_layout(self):
        """
            Recursively clears all widgets and layouts from the main layout.
        """
        if not self._main_layout:
            return
            
        self._widgets_map.clear()
        
        while self._main_layout.count():
            item = self._main_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            else:
                layout = item.layout()
                if layout:
                    self._recursive_clear_layout(layout)

        self._sections.clear()

    def _recursive_clear_layout(self, layout: QtWidgets.QLayout):
        """
            Helper to recursively delete all items in a layout.
        """
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            else:
                nested_layout = item.layout()
                if nested_layout:
                    self._recursive_clear_layout(nested_layout)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################