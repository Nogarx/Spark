#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import re
import numpy as np
import jax 
import jax.numpy as jnp
import typing as tp
import dataclasses as dc
import enum

from PySide6 import QtWidgets, QtCore
import spark.core.utils as utils
from spark.core.config import BaseSparkConfig
from spark.core.registry import REGISTRY
from spark.nn.initializers import Initializer, InitializerConfig
from spark.graph_editor.editor_config import GRAPH_EDITOR_CONFIG
from spark.graph_editor.widgets.base import QField, QInput, QMissing
from spark.graph_editor.widgets.line_edits import  QString, QInt, QFloat
from spark.graph_editor.widgets.combobox import QDtype, QBool
from spark.graph_editor.widgets.shape import QShape
from spark.graph_editor.widgets.checkbox import InheritToggleButton, WarningFlag, InitializerToggleButton, InheritStatus
from functools import partial 

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class QControlFlags(enum.IntFlag):
    SHOW = 0b1000
    INTERACTIVE = 0b0100
    EMPTY_SPACE = 0b0010
    ACTIVE = 0b0001
    SHOW_INTERACTIVE = 0b1100
    SHOW_INTERACTIVE_ACTIVE = 0b1101
    SHOW_ACTIVE = 0b1001
    ALL = 0b1111

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QAttrControls(QtWidgets.QWidget):

    on_update = QtCore.Signal()

    def __init__(
            self,
            on_initializer_toggled_callback: tp.Callable,
            on_inheritance_toggled_callback: tp.Callable,
            warning_flags: QControlFlags,
            inheritance_flags: QControlFlags,
            initializer_flags: QControlFlags,
            **kwargs,
        ) -> QtWidgets.QWidget:
        super().__init__(**kwargs)

        # Layout
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Warning widget
        if warning_flags & QControlFlags.SHOW:
            self.warning_widget = WarningFlag(
                value=warning_flags & QControlFlags.ACTIVE, 
            )
            layout.addWidget(self.warning_widget)    
        elif warning_flags & QControlFlags.EMPTY_SPACE:
            self.warning_widget = None
            dummy_widget = QtWidgets.QWidget()
            dummy_widget.setFixedSize(QtCore.QSize(16, 16))
            layout.addWidget(dummy_widget)

        # Initializer checkbox.
        if initializer_flags & QControlFlags.SHOW:
            self.initializer_checkbox = InitializerToggleButton(
                value=initializer_flags & QControlFlags.ACTIVE, 
                interactable=initializer_flags & QControlFlags.INTERACTIVE, 
            )
            self.initializer_checkbox.toggled.connect(on_initializer_toggled_callback)
            self.initializer_checkbox.toggled.connect(self._on_update)
            layout.addWidget(self.initializer_checkbox)
        elif initializer_flags & QControlFlags.EMPTY_SPACE:
            self.initializer_checkbox = None
            dummy_widget = QtWidgets.QWidget()
            dummy_widget.setFixedSize(QtCore.QSize(16, 16))
            layout.addWidget(dummy_widget)

        # Inheritance checkbox.
        if inheritance_flags & QControlFlags.SHOW:
            self.inheritance_checkbox = InheritToggleButton(
                value=inheritance_flags & QControlFlags.ACTIVE,
                interactable=inheritance_flags & QControlFlags.INTERACTIVE, 
            )
            self.inheritance_checkbox.toggled.connect(on_inheritance_toggled_callback)
            self.inheritance_checkbox.toggled.connect(self._on_update)
            layout.addWidget(self.inheritance_checkbox)
        elif inheritance_flags & QControlFlags.EMPTY_SPACE:
            self.inheritance_checkbox = None
            dummy_widget = QtWidgets.QWidget()
            dummy_widget.setFixedSize(QtCore.QSize(16, 16))
            layout.addWidget(dummy_widget)

    def _on_update(self, *args, **kwargs) -> None:
        self.on_update.emit()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QAttribute(QtWidgets.QWidget):
    """
        Base QWidget class for the graph editor attributes.
    """

    on_initializer_toggled = QtCore.Signal(bool)
    on_inheritance_toggled = QtCore.Signal(bool)
    on_field_update = QtCore.Signal(tp.Any)
    on_size_change = QtCore.Signal(QtCore.QSize)

    def __init__(
            self, 
            attr_label: str, 
            attr_value: tp.Any,
            attr_widget: type[QInput], 
            attr_init_mandatory: bool = False,
            attr_metadata: dict | None = None,
            warning_flags: QControlFlags = QControlFlags.SHOW_INTERACTIVE,
            inheritance_flags: QControlFlags = QControlFlags.EMPTY_SPACE,
            initializer_flags: QControlFlags = QControlFlags.EMPTY_SPACE,
            ref_post_callback_stack: list | None = None,
            is_initializer_attr: bool = False,
            **kwargs
        ) -> None:
        super().__init__(**kwargs)

        # Widget layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.attr_label = attr_label
        self.config_widget = None
        self.init_config = None
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        self._initializer_active = False

        # Default field
        self.main_row = QtWidgets.QWidget()
        self.main_row.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        main_layout = QtWidgets.QHBoxLayout(self.main_row)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)
        layout.addWidget(self.main_row)

        # Control widgets
        self.control_widgets = QAttrControls(
            on_initializer_toggled_callback=self._on_initializer_toggled,
            on_inheritance_toggled_callback=self._on_inheritance_toggled,
            warning_flags=warning_flags,
            inheritance_flags=inheritance_flags,
            initializer_flags=initializer_flags,
        )
        self.control_widgets.on_update.connect(self._on_field_update)
        main_layout.addWidget(self.control_widgets)  

        # General field
        if not attr_init_mandatory:
            label = utils.to_human_readable(attr_label)
            label = f'▷  {label}'  if is_initializer_attr else label
            self.default_field = QField(
                attr_label=label, 
                attr_value=attr_value if not isinstance(attr_value, InitializerConfig) else None,
                attr_widget=attr_widget, 
                value_options=attr_metadata.get('value_options', None),
            )
            self.default_field.on_field_update.connect(self._on_field_update)
            main_layout.addWidget(self.default_field)  
        else: 
            self.default_field = None

        # Initializer field
        self.secondary_row = QtWidgets.QWidget()
        self.secondary_row.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        secondary_layout = QtWidgets.QHBoxLayout(self.secondary_row)
        secondary_layout.setContentsMargins(0, 0, 0, 0)
        secondary_layout.setSpacing(4)
        layout.addWidget(self.secondary_row)

        if (initializer_flags & QControlFlags.SHOW) or (initializer_flags & QControlFlags.ACTIVE):
            initializer_options = {
                utils.to_human_readable(name, capitalize_all=True): entry.class_ref.get_config_spec() 
                for name, entry in REGISTRY.INITIALIZERS.items()
            }
            if isinstance(attr_value, InitializerConfig):
                starting_init_cls = attr_value.__class__
            else:
                starting_init_cls: type[InitializerConfig] = REGISTRY.INITIALIZERS.get('ConstantInitializer').class_ref.get_config_spec()
            from spark.graph_editor.widgets.combobox import QGenericComboBox
            self.config_selector = QField(
                attr_label=utils.to_human_readable(attr_label), 
                attr_value=starting_init_cls,
                attr_widget=QGenericComboBox, 
                value_options=initializer_options,
                **kwargs,
            )
            self.config_selector.on_field_update.connect(self._on_init_selection_update)
            main_layout.addWidget(self.config_selector)  
            # Instantiate an initializer config
            self.init_config = attr_value if isinstance(attr_value, InitializerConfig) else starting_init_cls._create_partial()
            self._setup_initializer_config(
                ref_post_callback_stack=ref_post_callback_stack if (initializer_flags & QControlFlags.ACTIVE) else None,
            )
        else:
            self.config_selector = None

        # Initial visibility setup
        self._toggle_initializer_visibility(initializer_flags & QControlFlags.ACTIVE)

    def _toggle_initializer_visibility(self, state: bool) -> None:
        # Update field visibility
        if self.default_field is not None:
            self.default_field.setVisible(not state)
        if self.config_selector is not None:
            self.config_selector.setVisible(state)
            self.secondary_row.setVisible(state)
            if self.config_widget is None:
                starting_init_cls: type[InitializerConfig] = self.config_selector.get_value()
                config = starting_init_cls._create_partial()
                self._setup_initializer_config(config)
        self._initializer_active = state
        self.on_size_change.emit(self.size())
        

    def _setup_initializer_config(
            self, 
            ref_post_callback_stack: list | None = None,
            **kwargs,
        ) -> None:
        # Clear previous config
        self._clear_initializer_config()
        # Initializer do not support inheritance
        self.config_widget = QConfigBody(
            config=self.init_config, 
            is_initializer=True, 
            ref_post_callback_stack=ref_post_callback_stack,
            **kwargs,
        )
        self.secondary_row.layout().addWidget(self.config_widget)
        self.secondary_row.adjustSize()
        self.adjustSize()
        self.on_size_change.emit(self.size())

    def _clear_initializer_config(self,):
        if self.config_widget:
            self.config_widget.deleteLater()
            self.config_widget = None

    def _on_initializer_toggled(self, state: bool) -> None:
        # Update field visibility
        self._toggle_initializer_visibility(state)
        # Broadcast value update
        if state:
            self.on_field_update.emit(self.init_config)
        else:
            self.on_field_update.emit(self.default_field.get_value())

    def _on_inheritance_toggled(self, state: bool) -> None:
        self.on_inheritance_toggled.emit(state)

    def _on_field_update(self) -> None:
        if self.default_field and self.default_field.isVisible():
            self.on_field_update.emit(self.default_field.get_value())
        elif self.config_widget and self.config_widget.isVisible():
            self.on_field_update.emit(self.config_widget.get_value())

    def _on_init_selection_update(self) -> None:
        config_cls: type[InitializerConfig] = self.config_selector.get_value()
        self.init_config = config_cls()
        self._setup_initializer_config()
        self.on_field_update.emit(self.init_config)

    def set_initializer_status(self, state: bool) -> None:
        """
            Sets the error status of widget.
        """
        if self.control_widgets.initializer_checkbox:
            self.control_widgets.initializer_checkbox.set_error_status(state)

    def set_inheritance_status(self, state: InheritStatus) -> None:
        """
            Sets the error status of widget.
        """
        if self.control_widgets.inheritance_checkbox:
            self.control_widgets.inheritance_checkbox.set_value(state)

    def set_error_status(self, messages: list[str]) -> None:
        """
            Sets the error status of widget.
        """
        self.control_widgets.warning_widget.set_error_status(messages)

    def set_value(self, value: tp.Any) -> tp.Any:
        """
            Set the widget value.
        """
        if isinstance(value, InitializerConfig):
            self._setup_initializer_config(value)
            self.config_selector.set_value(value.__class__)
            if not self._initializer_active:
                self._toggle_initializer_visibility(True)
        else:
            self.default_field.set_value(value)
            if self._initializer_active:
                self._toggle_initializer_visibility(False)

    def get_value(self,) -> tp.Any:
        """
            Get the widget value.
        """
        if self._initializer_active:
            return self.init_config
        else:
            return self.default_field.get_value()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QConfigBody(QtWidgets.QWidget):
    """
        QWidget for the body of configuration object.
    """

    on_error_detected = QtCore.Signal(str)
    on_size_change = QtCore.Signal(QtCore.QSize)
    on_update = QtCore.Signal()

    def __init__(
            self, 
            config: BaseSparkConfig, 
            is_initializer: bool, 
            config_path: str | None = None,
            ref_widgets_map: dict[tuple[str, ...], QtWidgets.QWidget] | None = None, 
            ref_post_callback_stack: list | None = None,
            inheritance_tree: utils.InheritanceTree | None = None,
            on_inheritance_toggle: tp.Callable | None = None,
            **kwargs
        ) -> None:
        super().__init__(**kwargs)
        # Main widget layout
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        layout = QtWidgets.QVBoxLayout(self)
        # We add extra margin to the left to shift the field when it is an initializer attribute and eliminate margin to the right.
        margins = QtCore.QMargins(4+36,0,0,0) if is_initializer else QtCore.QMargins(4,4,8,4)
        layout.setContentsMargins(margins)
        layout.setSpacing(0)

        # Keep config reference
        self.config = config
        nested_configs = config._get_nested_configs_names(exclude_initializers=True)
        populate_map = ref_widgets_map is not None and config_path is not None
        for name, field, value in config:
            # Skip nested_configs.
            if name in nested_configs:
                continue
            # Get widget
            widget_cls, metadata = WIDGET_PARSER.get_widget_raw(field.type)
            # Widget controls
            warning_flags = (
                QControlFlags.SHOW |
                QControlFlags.INTERACTIVE
            )
            # Inheritance flags
            if inheritance_tree is not None:
                leaf = inheritance_tree.get_leaf(tuple(config_path+[name]))
            if not is_initializer:
                if leaf.can_inherit() and leaf.is_inheriting():
                    inheritance_flags = QControlFlags.SHOW_INTERACTIVE_ACTIVE
                elif leaf.can_inherit():
                    inheritance_flags = QControlFlags.SHOW_INTERACTIVE
                elif leaf.can_receive() and leaf.is_inheriting():
                    inheritance_flags = QControlFlags.SHOW_ACTIVE
                elif leaf.can_receive():
                    inheritance_flags = QControlFlags.SHOW
                else:
                    inheritance_flags = 0
            else: 
                inheritance_flags = 0

            # Initializer flags
            if metadata.requires_init:
                initializer_flags = QControlFlags.ACTIVE
            elif metadata.allows_init and isinstance(value, InitializerConfig):
                initializer_flags = QControlFlags.SHOW_INTERACTIVE_ACTIVE
            elif metadata.allows_init:
                initializer_flags = QControlFlags.SHOW_INTERACTIVE
            else:
                initializer_flags = 0

            if not is_initializer:
                inheritance_flags = inheritance_flags | QControlFlags.EMPTY_SPACE
                initializer_flags = initializer_flags | QControlFlags.EMPTY_SPACE

            # Instantiate widget
            widget = QAttribute(
                attr_label=name, 
                attr_value=value,
                attr_widget=widget_cls if not issubclass(widget_cls, QConfigBody) else metadata.second_option, 
                attr_init_mandatory=metadata.requires_init,
                warning_flags=warning_flags,
                inheritance_flags=inheritance_flags,
                initializer_flags=initializer_flags,
                attr_metadata=field.metadata,
                ref_post_callback_stack=ref_post_callback_stack,
                is_initializer_attr=is_initializer,
            )
            widget.on_size_change.connect(self._on_size_changed)
            widget.on_field_update.connect(self._on_update)
            layout.addWidget(widget)
            # Scan for errors
            self._validate_field(name, widget)
            # Connect callbacks
            widget.on_field_update.connect(partial(
                lambda config_var, name_var, value: 
                setattr(config_var, name_var, value), 
                config, field.name
            ))
            widget.on_field_update.connect(partial(
                lambda field_name_var, field_widget_var, value: 
                self._validate_field(field_name_var, field_widget_var, value), 
                name, widget
            ))
            if inheritance_tree is not None and not is_initializer:
                widget.on_inheritance_toggled.connect(partial(
                        lambda leaf_var, value: 
                        on_inheritance_toggle(value, leaf_var), 
                        leaf
                ))
            # Queue inheritance callback.
            if inheritance_tree is not None and not is_initializer and leaf.is_inheriting():
                ref_post_callback_stack.append(partial(
                    lambda leaf_var: 
                    on_inheritance_toggle(True, leaf_var), 
                    leaf
                ))
            if populate_map:
                ref_widgets_map[tuple(config_path+[name])] = widget

    def _on_update(self, *args, **kwargs) -> None:
        self.on_update.emit()

    def _on_size_changed(self, size: QtCore.QSize) -> None:
        self.adjustSize()
        self.on_size_change.emit(self.size())

    def _validate_field(self, field_name:str, field_widget: QAttribute, *args) -> None:
        # Scan for errors
        errors = self.config.get_field_errors(field_name)
        # Warn the users if errors
        field_widget.set_error_status(errors)
        # Broadcast errors
        for e in errors:
            self.on_error_detected.emit(e)

    def get_value(self,) -> BaseSparkConfig:
        return self.config

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class TypeParser:

    GENERIC_TYPES: dict[str, type] = {
            # Primitives
            'bool': bool,
            'int': int,
            'float': float,
            'str': str,
            # Containers
            'list': list,
            'tuple': tuple,
            'dict': dict,
            # Types
            'dtype': np.dtype,
            'np.dtype': np.dtype,
            'jnp.dtype': np.dtype,
            'DTypeLike': np.dtype,
            'SupportsDType': np.dtype,
            # Arrays
            'ndarray': jax.Array,
            'Array': jax.Array,
            'np.ndarray': jax.Array,
            'jax.Array': jax.Array,
            'ArrayLike': jax.Array,
            # Initializers
            'Initializer': Initializer,
            'InitializerConfig': InitializerConfig,
    }

    SPECIAL_TYPES: dict[str, type] = {
            # Shapes
            'list[int]': list[int],
            list[int]: list[int],
            'tuple[int, ...]': tuple[int, ...],
            tuple[int, ...]: tuple[int, ...],
    }

    CONTAINER_TYPES: re.Pattern = re.compile(r'^(\w+)\[.*\]$')

    def _str_type_to_type(self, str_type: str) -> type:
        if str_type in self.SPECIAL_TYPES:
            return self.SPECIAL_TYPES[str_type]
        if str_type in self.GENERIC_TYPES:
            return self.GENERIC_TYPES[str_type]
        match = self.CONTAINER_TYPES.match(str_type)
        if match:
            container = match.group(1)
            return self.GENERIC_TYPES.get(container, None)
        return None
    
    def _process_type(self, t: type | str) -> type:
        if isinstance(t, type):
            return self.SPECIAL_TYPES[t] if t in self.SPECIAL_TYPES else self._str_type_to_type(t.__name__)
        if isinstance(t, str):
            return self._str_type_to_type(t)
    
    def typify(self, raw_types: type | str | tp.Iterable[type | str]) -> tuple[type, ...]:
        if isinstance(raw_types, type):
            return (self._process_type(raw_types),)
        if isinstance(raw_types, str):
            return tuple([self._process_type(t.strip()) for t in raw_types.split('|')])
        if isinstance(raw_types, tp.Iterable):
            return tuple([self._process_type(t) for t in raw_types])
        if tp.get_origin(raw_types) is tp.Union:
            return tuple([self._process_type(t) for t in tp.get_args(raw_types)])
        raise TypeError(
            f'Invalid input format for raw_types: \"{type(raw_types)}\". ' 
            f'Expected raw_types of type: \"type | str | tp.Iterable[str, type] | tp.UnionType[str, type].\".'
        )

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@dc.dataclass
class WidgetParserMetadata:
    allows_init: bool
    requires_init: bool
    allows_inheritance: bool
    missing: bool
    second_option: type[QtWidgets.QWidget]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class WidgetParser:

    WIDGET_TYPE_MAP = {
        bool: QBool,
        int: QInt,
        float: QFloat,
        str: QString,
        list[int]: QShape,
        tuple[int, ...]: QShape,
        np.dtype: QDtype,
        jax.Array: QConfigBody,
        Initializer: QConfigBody,
        InitializerConfig: QConfigBody,
    }

    WIDGET_PRIORITY_MAP = {
        QDtype: 35,
        QShape: 30,
        QFloat: 25,
        QInt: 20,
        QBool: 15,
        QString: 10,
        QConfigBody: 5,
        QMissing: 0,
    }

    def _compute_metadata(self, widgets_set: set[type[QtWidgets.QWidget]], second_option: QtWidgets.QWidget | None = None) -> WidgetParserMetadata:
        return WidgetParserMetadata(
            allows_init = QConfigBody in widgets_set,
            requires_init = QConfigBody in widgets_set and len(widgets_set) == True,
            allows_inheritance = len(widgets_set.difference(set([
                QDtype, QShape, QFloat, QInt, QBool, QString
            ]))) == 0,
            missing = QMissing in widgets_set,
            second_option = second_option
        )

    def get_possible_widgets(self, attr_type: tuple[type, ...]) -> tuple[list[type[QtWidgets.QWidget]], WidgetParserMetadata]:
        """
            Maps an argument type to a corresponding type of possible widgets classes. 
            Returns an ordered list of widgets and metadata to format the inspector.
        """
        widgets_set = set([self.WIDGET_TYPE_MAP.get(t, QMissing) for t in attr_type if t is not None])
        possible_widgets = {w: self.WIDGET_PRIORITY_MAP.get(w) for w in widgets_set}
        order_widgets = []
        while len(possible_widgets) > 0:
            widget = max(possible_widgets, key=possible_widgets.get)
            order_widgets.append(widget)
            possible_widgets.pop(widget)
        metadata = self._compute_metadata(widgets_set, order_widgets[1] if len(order_widgets) > 1 else None)
        return order_widgets, metadata

    def get_possible_widgets_raw(self, raw_types: type | str | tp.Iterable[type | str]) -> tuple[list[type[QtWidgets.QWidget]], WidgetParserMetadata]:
        """
            Maps an argument raw type to a corresponding type of possible widgets classes. 
            Returns an ordered list of widgets and metadata to format the inspector.
        """
        return self.get_possible_widgets(TYPE_PARSER.typify(raw_types))

    def get_widget(self, attr_type: tuple[type, ...]) -> tuple[type[QtWidgets.QWidget], WidgetParserMetadata]:
        """
            Maps an argument type to a corresponding widget class, according to the priority list. 
            Returns the best widget fit and metadata to format the inspector.
        """
        possible_widgets, metadata = self.get_possible_widgets(attr_type)
        return possible_widgets[0], metadata

    def get_widget_raw(self, raw_types: type | str | tp.Iterable[type | str]) -> tuple[type[QtWidgets.QWidget], WidgetParserMetadata]:
        """
            Maps an argument raw type to a corresponding widget class, according to the priority list.  
            Returns the best widget fit and metadata to format the inspector.
        """
        return self.get_widget(TYPE_PARSER.typify(raw_types))
        
#-----------------------------------------------------------------------------------------------------------------------------------------------#

TYPE_PARSER = TypeParser()
WIDGET_PARSER = WidgetParser()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################