#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
    
import typing as tp
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
        self._cfg_widget_map: dict[str, QtWidgets.QWidget] = {}
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
            self._add_config_recurrently('Main', node_config)


    def _add_config_recurrently(self, config_name: str, config: BaseSparkConfig) -> None:
        # Create new QCollapsible
        collapsible = QCollapsible(utils.to_human_readable(config_name), parent=self)
        # Get nested config's names
        nested_configs = config._get_nested_configs_names()

        #partial(lambda config_var, name_var, value: setattr(config_var, name_var, value), config, field.name),

        # Iterate over fields, add simple attributes
        for name, field, value in config:
            # Skip non-configs
            if name in nested_configs:
                continue
            # Add widget to collapsible
            update_func = partial(lambda config_var, name_var, value: setattr(config_var, name_var, value), config, field.name)
            attr_widget = self._attr_widget(
                field.type,
                update_func,
                value = value,
            )
            field_widget = QField(
                utils.to_human_readable(name),
                attr_widget,
                warning_value = False,
                inheritance_box = True,
                inheritance_value = False,
                parent= self
            )
            # Add widget to collapsible
            collapsible.addWidget(field_widget)
        # Add collapsible to layout
        collapsible.expand()
        self.addWidget(collapsible)

        # Iterate over fields, add other configs recurrently
        for name, field, value in config:
            # Skip non-configs
            if name not in nested_configs:
                continue
            # Add widget to collapsible
            self._add_config_recurrently(name, value)



    def _attr_widget(
            self, 
            attr_type: type, 
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

    def _get_widget_class_for_type(self, attr_type: type) -> type[QtWidgets.QWidget]:
        """
            Maps an argument type to a corresponding widget class.
        """
        WIDGET_TYPE_MAP = {
            'bool': QBool,
            'int': QInt,
            'float': QFloat,
            'str': QString,
            'Shape': QShape,
            'DTypeLike': QDtype,
        }
        return WIDGET_TYPE_MAP.get(attr_type, QMissing)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################