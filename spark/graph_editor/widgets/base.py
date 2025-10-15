#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import abc
import typing as tp
from PySide6 import QtWidgets, QtCore
from spark.graph_editor.editor_config import GRAPH_EDITOR_CONFIG
from spark.graph_editor.widgets.checkbox import InheritToggleButton, WarningFlag

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class SparkQWidgetMeta(type(QtWidgets.QWidget), abc.ABCMeta):
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SparkQWidget(QtWidgets.QWidget, abc.ABC, metaclass=SparkQWidgetMeta):
    """
        Base QWidget class for the graph editor attributes.
    """

    on_update = QtCore.Signal(tp.Any)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def get_value(self,) -> tp.Any:
        """
            Returns the widget value.
        """
        pass

    def _on_update(self):
        self.on_update.emit(self.get_value())

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QField(QtWidgets.QWidget):
    """
        Base QWidget class for the graph editor attributes.
    """

    def __init__(
            self, 
            attr_label: str,
            attr_widget: SparkQWidget,
            warning_value: bool = False,
            inheritance_box: bool = False,
            inheritance_value: bool = False,
            parent: QtWidgets.QWidget | None = None,
            **kwargs
        ) -> None:
        super().__init__(parent=parent, **kwargs)
        # Widget layout
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(GRAPH_EDITOR_CONFIG.field_margin)
        layout.setSpacing(4)
        # Warning icon
        self.warning_flag = WarningFlag(value=warning_value, parent=self)
        layout.addWidget(self.warning_flag)    
        # Inheritance checkbox.
        if inheritance_box:
            self.inheritance_checkbox = InheritToggleButton(value=inheritance_value, parent=self)
            layout.addWidget(self.inheritance_checkbox)
        else:
            dummy_widget = QtWidgets.QWidget(parent=self)
            dummy_widget.setFixedSize(QtCore.QSize(16, 16))
            self.inheritance_checkbox = dummy_widget
            layout.addWidget(self.inheritance_checkbox)
        # Label.
        self.label = QtWidgets.QLabel(attr_label, parent=self)
        self.label.setMinimumWidth(GRAPH_EDITOR_CONFIG.min_attr_label_size)
        self.label.setStyleSheet(
            f"""
                font-size: {GRAPH_EDITOR_CONFIG.medium_font_size}px;
            """
        )
        layout.addWidget(self.label)  
        # Attribute widget
        if not isinstance(attr_widget, SparkQWidget):
            raise TypeError(
                f'Expected \"attr_widget\" to be of type \"SparkQWidget\", but got \"{attr_widget.__class__}\".'
            )
        self.widget = attr_widget
        self.widget.setParent(self)
        layout.addWidget(self.widget)  
        # Set layout
        self.setLayout(layout)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QMissing(SparkQWidget):
    """
        Custom QWidget used for int fields in the SparkGraphEditor's Inspector.
    """

    def __init__(
            self, 
            parent: QtWidgets.QWidget = None
        ) -> None:
        super().__init__(parent=parent)
        # Add layout
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(QtCore.QMargins(0, 0, 0, 0))
        # Add QLineEdit
        self._label = QtWidgets.QLabel('🏗️ Widget not implemented.', parent=self)
        self._label.setFixedHeight(20)
        self._label.setStyleSheet(
            f"""
                font-size: {GRAPH_EDITOR_CONFIG.medium_font_size}px;
                padding: 0px;
                margin: 0px;
            """
        )
        layout.addWidget(self._label)
        # Finalize
        self.setLayout(layout)

    def get_value(self) -> str:
        raise TypeError(
            f'QMissing widget is a debug widget and cannot return a value.'
        )

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

