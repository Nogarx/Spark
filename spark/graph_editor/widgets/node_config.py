#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
    
import typing as tp
import jax
import numpy as np
from PySide6 import QtWidgets, QtCore, QtGui

import dataclasses as dc
import spark.core.utils as utils
from functools import partial
from spark.core.config import BaseSparkConfig
from spark.graph_editor.models.nodes import AbstractNode
from spark.graph_editor.widgets.line_edits import NodeNameWidget, QString, QInt, QFloat
from spark.graph_editor.widgets.separators import QHLine
from spark.graph_editor.widgets.base import SparkQWidget, QField, QMissing
from spark.graph_editor.widgets.collapsible import QCollapsible
from spark.graph_editor.widgets.combobox import QDtype, QBool, QGenericComboBox
from spark.graph_editor.widgets.shape import QShape
from spark.graph_editor.editor_config import GRAPH_EDITOR_CONFIG

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class QNodeConfig(QtWidgets.QWidget):
    """
        Constructs the UI shown when no valid node is selected.
    """

    def __init__(self, node: AbstractNode, parent: QtWidgets.QWidget = None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        # Node reference
        self._target_node = node
        self._cfg_widget_map: dict[tuple[str, ...], dict[str, QtWidgets.QWidget]] = {}
        # Widget layout
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(QtCore.QMargins(0, 0, 0, 0))
        self.setLayout(layout)
        # Scroll area
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.layout().addWidget(scroll_area)
        scroll_area.setStyleSheet("QScrollArea { background: transparent; }")
        # Content widget
        self.content = QtWidgets.QWidget()
        self.content.setStyleSheet("QWidget { background: transparent; }")
        scroll_area.setWidget(self.content)
        content_layout = QtWidgets.QVBoxLayout(self.content)
        content_layout.setContentsMargins(QtCore.QMargins(0, 0, 0, 0))
        content_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        # Setup layout
        self._setup_layout()
        print(self._cfg_widget_map)


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
        name_widget.on_update.connect(lambda name: self._target_node.set_name(name))
        self.addWidget(name_widget)
        self.addWidget(QHLine())
        # Add configs recurrently
        node_config: BaseSparkConfig = getattr(self._target_node, 'node_config', None)
        if node_config:
            self._add_config_recurrently('Main', node_config, ['_'])


    def _add_config_recurrently(self, config_name: str, config: BaseSparkConfig, path: list[str] = []) -> None:
        # Create new QCollapsible
        collapsible = QCollapsible(utils.to_human_readable(config_name), parent=self)
        # Get nested config's names
        nested_configs = config._get_nested_configs_names()
        type_hints = config._get_type_hints()
        # Exapnd widget map
        self._cfg_widget_map[tuple(path)] = {}

        #partial(lambda config_var, name_var, value: setattr(config_var, name_var, value), config, field.name),

        # Iterate over fields, add simple attributes
        show_inheritance_box = len(nested_configs) > 0
        for name, field, value in config:
            # Skip non-configs
            if name in nested_configs:
                continue
            # Add widget to collapsible
            update_func = partial(lambda config_var, name_var, value: setattr(config_var, name_var, value), config, field.name)
            attr_widget = self._attr_widget(
                type_hints[name],
                update_func,
                value = value,
                metadata= field.metadata,
            )
            field_widget = QField(
                utils.to_human_readable(name),
                attr_widget,
                warning_value = False,
                inheritance_box = show_inheritance_box,
                inheritance_value = False,
                parent= self
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
            print(value)
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
        WIDGET_TYPE_MAP = {
            bool: QBool,
            int: QInt,
            float: QFloat,
            jax.Array: QFloat,
            str: QString,
            tuple[int, ...]: QShape,
            np.dtype: QDtype,
        }
        for t in attr_type:
            widget_cls = WIDGET_TYPE_MAP.get(t, None)
            if widget_cls:
                return widget_cls
        return QMissing

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################