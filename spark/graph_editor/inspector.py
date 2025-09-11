#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.config import BaseSparkConfig

import dataclasses
import typing as tp
import jax.numpy as jnp
from functools import partial
from typing import Dict, Any, Type
from Qt import QtCore, QtWidgets, QtGui
from spark.core.specs import InputArgSpec
from spark.core.shape import Shape
from spark.graph_editor.nodes import AbstractNode, SinkNode, SourceNode, SparkModuleNode
from spark.graph_editor.utils import _normalize_section_header, _to_human_readable
from spark.graph_editor.widgets.dict import QDict
from spark.graph_editor.widgets.dtype import QDtype
from spark.graph_editor.widgets.shape import QShape
from spark.graph_editor.widgets.missing import QMissing
from spark.graph_editor.widgets.collapsible import QCollapsible
from spark.graph_editor.widgets.line_edits import QIntLineEdit, QFloatLineEdit, QString

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class NodeInspectorWidget(QtWidgets.QWidget):
    """A widget to inspect and edit the properties of a node in a graph."""

    onWidgetUpdate = QtCore.Signal(str, Any)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._target_node: AbstractNode | None = None
        self._widgets_map: Dict[str, QtWidgets.QWidget] = {}
        self._main_layout: QtWidgets.QVBoxLayout | None = None

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
        """Constructs the inspector UI for the currently selected node."""
        if not self._target_node:
            return
        
        self._build_header()
        self._build_shape_editors()
        self._build_main_section()

        # Add other sections as needed.
        #self._build_dict_editors()
        
        #self._main_layout.addStretch()

    def _build_idle_menu(self):
        """Constructs the UI shown when no node (multiple nodes) is (are) selected."""
        self._build_header()
        self._add_line()
        
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
        if not self._target_node or not self._target_node.input_specs:
            return

        form_layout = self._add_section('Shapes')
        for name, spec in self._target_node.input_specs.items():
            is_static = not isinstance(self._target_node, SourceNode)
            widget = QShape(initial_shape=spec.shape, is_static=is_static)
            prefixed_name = f'input_port.{name}'
            widget.editingFinished.connect(partial(self._handle_widget_update, prefixed_name))
            form_layout.addRow(QtWidgets.QLabel(_to_human_readable(name)), widget)
            self._widgets_map[prefixed_name] = widget

        for name, spec in self._target_node.output_specs.items():
            is_static = True#not isinstance(self._target_node, SinkNode)
            widget = QShape(initial_shape=spec.shape, is_static=is_static)
            prefixed_name = f'output_port.{name}'
            widget.editingFinished.connect(partial(self._handle_widget_update, prefixed_name))
            form_layout.addRow(QtWidgets.QLabel(_to_human_readable(name)), widget)
            self._widgets_map[prefixed_name] = widget
            
    def _build_main_section(self):
        """Builds the main section with node name and simple arguments."""
        if not self._target_node:
            return
        
        form_layout = self._add_section('Main')
        # Add name
        self._add_arg_widget_to_layout(form_layout, 'name', str, self._target_node.NODE_NAME)
        # Add shape widgets to source/sink nodes.
        if isinstance(self._target_node, (SinkNode, SourceNode)):
            pass
        # Main init_args
        elif isinstance(self._target_node, SparkModuleNode):
            print(self._target_node)
            print(self._target_node.node_config)
            self._add_widgets_from_spark_config(self._main_layout, self._target_node.node_config)
        else:
            # Unknown node type raise error.
            raise TypeError(f'Support for node of type \"{self._target_node.__class__.__name__}\" is not implemented.')

    def _add_widgets_from_spark_config(self, config: BaseSparkConfig, name=None):

        # Get nested config's names
        nested_configs = config._get_nested_configs_names()

        # Create a collapsable window
        section = self._add_section(name)

        # Iterate over fields, add simple attributes
        for field in dataclasses.fields(config):
            # Skip non-configs
            if field.name in nested_configs:
                continue
            # Add widget to collapsible
            self._add_arg_widget_to_layout(section.layout(), field.name, field.type, getattr(config, field.name))

        # Iterate over fields, add nested configs as new collapsibles
        for field in dataclasses.fields(config):
            # Skip configs
            if field.name not in nested_configs:
                continue
            # Add collapsible recursively
            self._add_widgets_from_spark_config(getattr(config, field.name), name=field.name)


    def _add_arg_widget_to_layout(self, layout: QtWidgets.QFormLayout, arg_name: str, arg_type: Type, value: Any | None = None):
        """
            Creates and adds an appropriate widget for a given argument specification.
        """
        # Get appropiate widget
        widget_class = self._get_widget_class_for_type(arg_type)
        if widget_class == QMissing:
            widget = widget_class()
            layout.addRow(QtWidgets.QLabel(_to_human_readable(arg_name)), widget)
            self._widgets_map[arg_name] = None
            return
        # Initialize widget
        widget = widget_class(str(value) if value is not None else "")
        widget.editingFinished.connect(partial(self._handle_widget_update, arg_name))
        layout.addRow(QtWidgets.QLabel(_to_human_readable(arg_name)), widget)
        self._widgets_map[arg_name] = widget

    def _build_dict_editors(self):
        """
            Builds editors for dictionary arguments, each in its own section.
        """
        if not self._target_node:
            return

        for key, arg_spec in self._target_node.init_args_specs.items():
            if arg_spec.arg_type is Dict:
                form_layout = self._add_section(key)
                widget = QDict()
                self._widgets_map[key] = widget
                form_layout.addRow(widget)

    def _add_section(self, name: str) -> QtWidgets.QWidget:
        """
            Adds a new collapsible section with a header and a form layout.
        """
        collapsible = QCollapsible(_to_human_readable(name))
        self._main_layout.addWidget(collapsible)
        return collapsible

    def _add_line(self):
        """
            Adds a horizontal separator line to the main layout.
        """
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self._main_layout.addWidget(line)

    def _get_widget_class_for_type(self, arg_type: Type) -> Type[QtWidgets.QWidget]:
        """
            Maps an argument type to a corresponding widget class.
        """
        WIDGET_TYPE_MAP = {
            int: QIntLineEdit,
            float: QFloatLineEdit,
            str: QString,
            Shape: QShape,
            jnp.dtype: QDtype,
            dict: QDict,
        }
        return WIDGET_TYPE_MAP.get(arg_type, QMissing)

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

    def _handle_widget_update(self, attr_name: str):
        """
            A generic handler that links a widget's update to the node's update method.
        """

        if not self._target_node:
            return

        widget = self._widgets_map.get(attr_name)
        if not widget:
            return

        # Get values of the widget
        if isinstance(widget, QDict):
            new_value = widget.to_dict() # Assuming QDict has this method
        elif isinstance(widget, (QIntLineEdit, QFloatLineEdit, QtWidgets.QLineEdit)):
            new_value = widget.text()
        elif isinstance(widget, QShape):
            new_value = widget.shape()
        else:
            raise RuntimeError(f'Widget associated to {attr_name} does not implement a map back to its node attributes.')

        try:
            # Call the node's universal update method
            self._target_node.update_attribute(attr_name, new_value)

        except (ValueError, TypeError) as e:

            # Ignore next Click. Flag will be consumed when user interacts with the warning box.
            self._target_node.graph.ignore_next_click_event()

            # If the node rejects the change, show an error and revert the widget.
            QtWidgets.QMessageBox.warning(self._target_node.graph.viewer(), 'Error', e)

            # Revert the widget to the correct value from the node
            if attr_name == 'name':
                widget.setText(self._target_node.name())
            elif attr_name in self._target_node.init_args:
                widget.setText(str(self._target_node.init_args[attr_name]))

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################