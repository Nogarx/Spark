#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import abc
import typing as tp
from PySide6 import QtWidgets, QtCore
from spark.graph_editor.editor_config import GRAPH_EDITOR_CONFIG

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class QInput(QtWidgets.QWidget):
    """
        QWidget interface class for the graph editor inputs.

        QInput streamlines updates by setting a common set/get interface.
    """

    # Common event to streamline updates
    on_update = QtCore.Signal(tp.Any)

    def __init__(self, *args, value_options: tp.Any = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def get_value(self,) -> tp.Any:
        """
            Returns the widget value.
        """
        pass

    @abc.abstractmethod
    def set_value(self, value: tp.Any) -> tp.Any:
        """
            Sets the widget value.
        """
        pass

    def _on_update(self) -> None:
        self.on_update.emit(self.get_value())

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QField(QtWidgets.QWidget):
    """
        Base QWidget class for the graph editor fields.
    """

    on_field_update = QtCore.Signal(tp.Any)

    def __init__(
            self, 
            attr_label: str, 
            attr_value: tp.Any,
            attr_widget: type[QInput], 
            value_options: tp.Any = None,
            **kwargs
        ) -> None:
        super().__init__(**kwargs)

        # Widget layout
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(GRAPH_EDITOR_CONFIG.field_margin)
        
        # Label widget
        self.attr_label = QtWidgets.QLabel(attr_label)
        self.attr_label.setMinimumWidth(GRAPH_EDITOR_CONFIG.min_attr_label_size)
        self.attr_label.setStyleSheet(
            f"""
                font-size: {GRAPH_EDITOR_CONFIG.medium_font_size}px;
            """
        )
        layout.addWidget(self.attr_label)  

        # Input widget
        self.attr_widget = attr_widget(attr_value, value_options=value_options, **kwargs)
        #self.attr_widget.set_value(attr_value)
        self.attr_widget.on_update.connect(self._on_field_update)
        layout.addWidget(self.attr_widget)  

    def _on_field_update(self) -> None:
        self.on_field_update.emit(self.attr_widget.get_value())

    def get_value(self) -> tp.Any:
        """
            Get the input widget value.
        """
        return self.attr_widget.get_value()

    def set_value(self, value: tp.Any) -> tp.Any:
        """
            Sets the input widget value.
        """
        self.attr_widget.set_value(value)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QMissing(QInput):
    """
        Custom QWidget used for int fields in the SparkGraphEditor's Inspector.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # Add layout
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(QtCore.QMargins(0, 0, 0, 0))
        # Add QLineEdit
        self._label = QtWidgets.QLabel('🏗️ Widget not implemented.')
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