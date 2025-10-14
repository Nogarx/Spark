#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import time
import jax.numpy as jnp
import typing as tp
import dataclasses as dc
from functools import partial
from PySide6 import QtCore, QtWidgets, QtGui
from spark.nn.initializers.delay import DelayInitializerConfig, _DELAY_CONFIG_REGISTRY
from spark.nn.initializers.kernel import KernelInitializerConfig, _KERNEL_CONFIG_REGISTRY
from spark.graph_editor.editor_config import GRAPH_EDITOR_CONFIG
from spark.graph_editor.utils import _to_human_readable
from spark.graph_editor.widgets.dict import QDict
from spark.graph_editor.widgets.combobox import QDtype, QBool, QGenericComboBox
from spark.graph_editor.widgets.shape import QShape
from spark.graph_editor.widgets.collapsible import QCollapsible
from spark.graph_editor.widgets.missing import QMissing
from spark.graph_editor.widgets.line_edits import QInt, QFloat, QString
from spark.graph_editor.widgets.base import SparkQWidget

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class QInitializer(SparkQWidget):
    """
        Custom QWidget used for dtypes fields in the SparkGraphEditor's Inspector.
    """

    def __init__(self,
                 initializer_config: KernelInitializerConfig | DelayInitializerConfig,
                 section: QCollapsible,
                 parent: QtWidgets.QWidget = None):
        super().__init__(parent=parent)
        # Config reference 
        self._initializer_config = initializer_config
        if isinstance(initializer_config, KernelInitializerConfig):
            self._config_registry = _KERNEL_CONFIG_REGISTRY
        else:
            self._config_registry = _DELAY_CONFIG_REGISTRY
        self._section = section
        self._widgets_map: dict[str, QtWidgets.QWidget] = {}
        # Add layout
        self._layout = QtWidgets.QVBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        # Add initializer selection
        options_list = {_to_human_readable(name): name for name in self._config_registry.keys()}
        self._initializer_selection = QGenericComboBox(_to_human_readable('name'), self._initializer_config.name, options_list, self)
        # Update callback
        self._initializer_selection.on_update.connect(self._on_initializer_selection_update)
        # Finalize
        self._layout.addWidget(self._initializer_selection)
        self.setLayout(self._layout)
        self._redraw_widget_content()

    def get_value(self):
        return self._initializer_config

    def _on_initializer_selection_update(self, value):
        selection = self._config_registry[value]
        if isinstance(self._initializer_config, selection):
            return
        # Replace config
        self._initializer_config = selection()
        self._redraw_widget_content()
        self._on_update()

    def _redraw_widget_content(self):
        # Clear current layout
        self._clear_layout()
        # Iterate over fields, add simple attributes. 
        # NOTE: We assume there are no more nested configs.
        for field in dc.fields(self._initializer_config):
            # Add widget to layout
            if field.name == 'name':
                continue
            self._add_attr_widget_to_layout(
                field.name, 
                field.type, 
                partial(lambda config_var, name_var, value: setattr(config_var, name_var, value), self._initializer_config, field.name),
                getattr(self._initializer_config, field.name), 
                metadata=field.metadata,
            )
        # Rescale section to adjust for widgets sizes.
        QtCore.QTimer.singleShot(0, self._section._resize_widget)
    
    def _add_attr_widget_to_layout(self, 
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
            self._layout.addWidget(widget)
            self._widgets_map[attr_name] = None
            return widget
        # Initialize widget
        if widget_class == QDtype:
            widget: QDtype = widget_class(_to_human_readable(attr_name), value, values_options=metadata['value_options'])
        else:
            widget = widget_class(_to_human_readable(attr_name), value)
        widget.on_update.connect(attr_update_method)
        self._layout.addWidget(widget)
        self._widgets_map[attr_name] = widget
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
            'dict': QDict,
        }
        return WIDGET_TYPE_MAP.get(attr_type, QMissing)
    
    def _clear_layout(self):
        """
            Recursively clears all widgets and layouts from the main layout.
        """
        keys = list(self._widgets_map.keys())
        for key in keys:
            widget = self._widgets_map.pop(key)
            widget.deleteLater()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################