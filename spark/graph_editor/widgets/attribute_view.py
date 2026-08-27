#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.graph_editor.models.graph_model import GraphModel
    from spark.graph_editor.models.inspector_model import ConfigNode

import logging
import typing as tp
import jax.numpy as jnp
from shiboken6 import isValid
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QComboBox, QDoubleSpinBox,
    QLineEdit, QLabel, QFormLayout, QListView, QToolButton, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, Signal, QSize
import spark.core.utils as utils
from spark.graph_editor.styles import resources as icons
from spark.graph_editor.models.inspector_model import ConfigValueNode, parse_object_to_state
from spark.graph_editor.models.config_types import (
    FieldKind, classify, type_tokens, type_label, is_optional, accepts_array, initializer_policy,
    dtype_key, values_equal, coerce
)
from spark.graph_editor.commands.inspector_commands import ChangeConfigValueCommand, ToggleInheritanceCommand
from spark.graph_editor.styles.manager import STYLES
from spark.core.registry import REGISTRY
from spark.nn.initializers import InitializerConfig

logger = logging.getLogger('spark')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

DEFAULT_DTYPES = [
    jnp.uint8,
    jnp.uint16,
    jnp.uint32,
    jnp.int8,
    jnp.int16,
    jnp.int32,
    jnp.float16,
    jnp.float32,
    jnp.bool,
]

# Spin box bounds. Doubles represent integers exactly below 2**53, so a single widget covers both integer
# and float fields (seeds are unsigned 32 bit integers and do not fit in a QSpinBox).
_INT_RANGE = (-1e15, 1e15)
_FLOAT_RANGE = (-1e12, 1e12)
_FLOAT_DECIMALS = 6
_MIN_DIMS = 1
_MAX_DIMS = 8

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def _validator_names(metadata: dict) -> set[str]:
    """
        Names of the validators declared by a configuration field.
    """
    return {getattr(v, '__name__', str(v)) for v in (metadata.get('validators') or [])}

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _numeric_bounds(kind: FieldKind, metadata: dict) -> tuple[float, float]:
    """
        Derives the range of a numeric editor from the validators declared by the field.
    """
    validators = _validator_names(metadata)
    low, high = _INT_RANGE if kind is FieldKind.INT else _FLOAT_RANGE
    if 'ZeroOneValidator' in validators or 'BinaryValidator' in validators:
        return (0.0, 1.0)
    if 'PositiveValidator' in validators:
        # Positive means strictly greater than zero.
        low = 1.0 if kind is FieldKind.INT else 10.0 ** (-_FLOAT_DECIMALS)
    return (low, high)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _array_summary(value: tp.Any) -> str:
    """
        Short human readable description of a value that cannot be edited inline.
    """
    # A missing value is rendered as an empty field.
    if value is None:
        return ''
    shape = getattr(value, 'shape', None)
    dtype = getattr(value, 'dtype', None)
    if shape is not None:
        return f'array(shape={tuple(shape)}, dtype={dtype})'
    text = str(value)
    return text if len(text) <= 64 else f'{text[:61]}...'

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class QDimsEdit(QWidget):
    """
        Editor for variable length numeric tuples (e.g. tuple[int, ...] shapes).

        The number of entries is set through the add/remove buttons. The widget always reports a valid tuple.
        """

    value_changed = Signal(object)

    def __init__(
            self,
            value: tp.Iterable | None = None,
            is_integer: bool = True,
            minimum: float = 1,
            maximum: float = 1e9,
            min_dims: int = _MIN_DIMS,
            max_dims: int = _MAX_DIMS,
            parent: QWidget | None = None,
        ) -> None:
        super().__init__(parent)
        self.is_integer = is_integer
        self.minimum = minimum
        self.maximum = maximum
        self.min_dims = max(1, min_dims)
        self.max_dims = max_dims
        self._spins: list[QDoubleSpinBox] = []
        self._emit_blocked = False

        spacing = STYLES.get_val('inspector', 'dims_spacing', default=4)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(spacing)
        self._layout = layout
        self._dims_widget = QWidget()
        self._dims_layout = QHBoxLayout(self._dims_widget)
        self._dims_layout.setContentsMargins(0, 0, 0, 0)
        self._dims_layout.setSpacing(spacing)
        layout.addWidget(self._dims_widget, 1)
        # Add/remove buttons.
        self._remove_btn = QToolButton()
        self._remove_btn.setObjectName('dimsRemoveBtn')
        self._remove_btn.setText('−')
        self._remove_btn.setToolTip('Remove the last dimension')
        self._remove_btn.clicked.connect(self.remove_dim)
        layout.addWidget(self._remove_btn)
        self._add_btn = QToolButton()
        self._add_btn.setObjectName('dimsAddBtn')
        self._add_btn.setText('+')
        self._add_btn.setToolTip('Add a dimension')
        self._add_btn.clicked.connect(self.add_dim)
        layout.addWidget(self._add_btn)

        self.set_value(value)

    #-------------------------------------------------------------------------------------------------------#

    def _make_spin(self, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(0 if self.is_integer else _FLOAT_DECIMALS)
        spin.setRange(self.minimum, self.maximum)
        spin.setValue(value)
        spin.wheelEvent = lambda event: event.ignore()
        spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # An explicit minimum, otherwise a shape with several dimensions cannot be narrowed.
        spin.setMinimumWidth(STYLES.get_val('inspector', 'dims_min_width', default=40))
        spin.valueChanged.connect(lambda _: self._emit())
        return spin

    def _refresh_buttons(self) -> None:
        self._add_btn.setEnabled(len(self._spins) < self.max_dims)
        # An empty shape is not a valid Spark shape, so the editor never drops below min_dims.
        self._remove_btn.setEnabled(len(self._spins) > self.min_dims)

    def _emit(self) -> None:
        if self._emit_blocked:
            return
        self.value_changed.emit(self.value())

    #-------------------------------------------------------------------------------------------------------#

    def value(self) -> tuple | None:
        """
            Current value as a valid tuple.
        """
        if not self._spins:
            return None
        if self.is_integer:
            return tuple(int(round(spin.value())) for spin in self._spins)
        return tuple(float(spin.value()) for spin in self._spins)

    def set_value(self, value: tp.Iterable | None) -> None:
        """
            Rebuilds the editor from a value, without emitting change notifications.
            """
        entries: list[float] = []
        if value is not None:
            try:
                entries = [float(v) for v in (value if isinstance(value, (list, tuple)) else [value])]
            except (TypeError, ValueError):
                entries = []
        if len(entries) < self.min_dims:
            entries += [self.minimum] * (self.min_dims - len(entries))
        self._emit_blocked = True
        try:
            for spin in self._spins:
                self._dims_layout.removeWidget(spin)
                spin.deleteLater()
            self._spins = []
            for entry in entries:
                spin = self._make_spin(max(self.minimum, min(self.maximum, entry)))
                self._spins.append(spin)
                self._dims_layout.addWidget(spin)
            self._refresh_buttons()
        finally:
            self._emit_blocked = False

    def add_dim(self) -> None:
        """
            Appends a new dimension, seeded with the value of the last one.
        """
        if len(self._spins) >= self.max_dims:
            return
        seed = self._spins[-1].value() if self._spins else max(self.minimum, 1)
        spin = self._make_spin(seed)
        self._spins.append(spin)
        self._dims_layout.addWidget(spin)
        self._refresh_buttons()
        self._emit()

    def remove_dim(self) -> None:
        """
            Removes the last dimension. The editor never drops below min_dims.
        """
        if len(self._spins) <= self.min_dims:
            return
        spin = self._spins.pop()
        self._dims_layout.removeWidget(spin)
        spin.deleteLater()
        self._refresh_buttons()
        self._emit()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class QAttrControls(QWidget):
    """
        Holds the toggle buttons for Warning, Initializer, and Inheritance.
    """

    def __init__(self, node: ConfigValueNode, config_path: list[str] = None, graph_model: GraphModel | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.node = node
        self.config_path = config_path
        self.graph_model = graph_model
        # Sub-fields of an initializer are governed by the field holding the initializer.
        self._is_init_child = 'init_config' in (config_path or [])
        tokens = type_tokens(node.type_hint, node.metadata.get('valid_types'))
        self._allows_init, self._init_required = initializer_policy(tokens, node.metadata)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(STYLES.get_val('inspector', 'ctrl_spacing', default=2))
        icon_size = STYLES.get_val('inspector', 'icon_pixmap_size', default=12)
        # Warning Icon
        # There is no warning asset in the editor resources, so this one is a themeable glyph.
        self.warning_btn = self._make_button(
            'No errors', text='⚠', object_name='warningBtn', checkable=False
        )
        layout.addWidget(self.warning_btn)
        # Initializer Toggle
        self.init_btn = self._make_button(
            'Define this value through an initializer',
            icon=icons.get_toggle_icon(icons.COMPLEX, icons.SIMPLE, icon_size),
            icon_size=icon_size,
        )
        layout.addWidget(self.init_btn)
        # Inheritance Toggle
        # The unchecked state still draws something: an empty icon on a transparent button would make an
        # available cascade invisible.
        self._inherit_icon = icons.get_toggle_icon(
            icons.LINK, icons.LINK, icon_size,
            off_opacity=STYLES.get_val('inspector', 'icon_idle_opacity', default=0.35),
        )
        self._lock_icon = icons.get_icon(icons.LOCK, icon_size)
        self.inherit_btn = self._make_button(
            'Cascade this value to the nested configurations',
            icon=self._inherit_icon,
            icon_size=icon_size,
        )
        layout.addWidget(self.inherit_btn)

        # Initializers are only available for array/initializer fields.
        if self._allows_init and not self._is_init_child:
            self.init_btn.setVisible(True)
            self.init_btn.setChecked(bool(self.node.is_initializer_active or self._init_required))
            if self._init_required:
                # An array cannot be typed in, so these fields stay in initializer mode.
                self.init_btn.setEnabled(False)
                self.init_btn.setToolTip('This field holds an array and can only be defined through an initializer')
            else:
                self.init_btn.toggled.connect(self._on_init_toggled)
        else:
            self.init_btn.setVisible(False)
        # Inheritance visibility is resolved against the inheritance tree of the node.
        if self._is_init_child:
            self.inherit_btn.setVisible(False)
        else:
            self.inherit_btn.toggled.connect(self._on_inherit_toggled)

        self.node.errors_changed.connect(self._on_errors_changed)
        self.node.inheritance_changed.connect(self._on_node_inheritance_changed)
        self.node.initializer_changed.connect(self._on_node_init_changed)
        if self.graph_model and self.config_path:
            self.graph_model.inheritance_updated.connect(self._on_inheritance_tree_updated)
        # Initial state.
        self._on_errors_changed(self.node, self.node.errors)
        self._on_inheritance_tree_updated()

    #-------------------------------------------------------------------------------------------------------#

    @staticmethod
    def _make_button(
            tooltip: str,
            text: str | None = None,
            icon: tp.Any = None,
            icon_size: int = 12,
            object_name: str | None = None,
            checkable: bool = True,
        ) -> QToolButton:
        button = QToolButton()
        if object_name:
            button.setObjectName(object_name)
        if text:
            button.setText(text)
        if icon is not None:
            button.setIcon(icon)
            button.setIconSize(QSize(icon_size, icon_size))
        button.setCheckable(checkable)
        button.setToolTip(tooltip)
        # Keep the icon column aligned when a button does not apply to the field.
        policy = button.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        button.setSizePolicy(policy)
        button.setVisible(False)
        return button

    def _leaf(self):
        if not (self.graph_model and self.config_path):
            return None
        return self.graph_model.get_inheritance_leaf(self._inheritance_path())

    def _node_alive(self) -> bool:
        # NOTE: The inspector deletes its widgets with deleteLater(), so a widget can outlive the state node
        # it observes and still receive graph level signals.
        return isValid(self) and isValid(self.node)

    def _inheritance_path(self) -> list[str]:
        path = list(self.config_path or [])
        if 'init_config' in path:
            return path[:path.index('init_config')]
        return path

    def _update_inherit_icon(self, is_linked: bool) -> None:
        # The state aware icon follows the checked state, this only restores it after a lock.
        self.inherit_btn.setIcon(self._inherit_icon)

    #-------------------------------------------------------------------------------------------------------#

    def _on_inheritance_tree_updated(self) -> None:
        if not self._node_alive():
            return
        leaf = self._leaf()
        is_driven = bool(leaf is not None and leaf.is_receiving())
        # A driven field is owned by an ancestor and cannot be edited nor re-routed.
        if self._allows_init and not self._is_init_child and not self._init_required:
            self.init_btn.setEnabled(not is_driven)
        if self._is_init_child:
            return
        can_inherit = bool(leaf is not None and leaf.can_inherit())
        # A driven field shows a lock, and its editor is read-only.
        if is_driven:
            self.inherit_btn.setVisible(True)
            self.inherit_btn.setEnabled(False)
            self.inherit_btn.blockSignals(True)
            self.inherit_btn.setChecked(True)
            self.inherit_btn.setIcon(self._lock_icon)
            self.inherit_btn.blockSignals(False)
            self.inherit_btn.setToolTip('This value is inherited from a parent field')
            return
        # Cascading is only offered when the value can reach a nested field.
        self.inherit_btn.setVisible(can_inherit)
        if not can_inherit:
            # A leaf without descendants cannot cascade, so any leftover state is dropped.
            self.inherit_btn.blockSignals(True)
            self.inherit_btn.setChecked(False)
            self._update_inherit_icon(False)
            self.inherit_btn.blockSignals(False)
            self.node._is_inherited = False
            return
        self.inherit_btn.setEnabled(True)
        is_inheriting = leaf.is_inheriting()
        childs = len(leaf.inheritance_childs)
        self.inherit_btn.setToolTip(
            f'{"Stop cascading" if is_inheriting else "Cascade"} this value to {childs} nested field(s)'
        )
        # Sync the checked state without emitting signals.
        self.inherit_btn.blockSignals(True)
        self.inherit_btn.setChecked(is_inheriting)
        self._update_inherit_icon(is_inheriting)
        self.inherit_btn.blockSignals(False)
        self.node._is_inherited = is_inheriting

    def _on_init_toggled(self, checked: bool) -> None:
        self.node.is_initializer_active = checked

    def _on_inherit_toggled(self, checked: bool) -> None:
        if self.graph_model and getattr(self.graph_model, 'undo_stack', None) and self.config_path:
            cmd = ToggleInheritanceCommand(self.graph_model, self._inheritance_path(), self.node.is_inherited, checked, self.node)
            self.graph_model.undo_stack.push(cmd)
        else:
            self.node.is_inherited = checked
            self._update_inherit_icon(checked)

    def _on_errors_changed(self, node: ConfigNode, errors: list[str]) -> None:
        if errors:
            self.warning_btn.setVisible(True)
            self.warning_btn.setToolTip('\n'.join(errors))
        else:
            self.warning_btn.setVisible(False)
            self.warning_btn.setToolTip('No errors')

    def _on_node_inheritance_changed(self, node: ConfigNode, is_inherited: bool) -> None:
        def _update() -> None:
            if not isValid(self):
                return
            self.inherit_btn.blockSignals(True)
            self.inherit_btn.setChecked(is_inherited)
            self._update_inherit_icon(is_inherited)
            self.inherit_btn.blockSignals(False)
        QTimer.singleShot(0, _update)

    def _on_node_init_changed(self, node: ConfigNode, is_init: bool) -> None:
        def _update() -> None:
            if not isValid(self):
                return
            self.init_btn.blockSignals(True)
            self.init_btn.setChecked(is_init)
            self.init_btn.blockSignals(False)
        QTimer.singleShot(0, _update)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class QAttribute(QWidget):
    """
        Composite widget containing the input field.
    """

    def __init__(self, node: ConfigValueNode, config_path: list[str] = None, graph_model: GraphModel | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.node = node
        self.config_path = config_path
        self.graph_model = graph_model
        self._input_widget = None
        self._apply_value: tp.Callable[[tp.Any], None] | None = None
        self._read_value: tp.Callable[[], tp.Any] | None = None
        self._init_combo = None
        self._init_block = None
        self._init_state_model = None
        self._is_updating_programmatically = False
        # Field typing. The metadata "valid_types" entry is authoritative, the annotation is a fallback.
        self._tokens = type_tokens(node.type_hint, node.metadata.get('valid_types'))
        self._kind = classify(tokens=self._tokens)
        self._optional = is_optional(self._tokens)
        self._allows_init, self._init_required = initializer_policy(self._tokens, node.metadata)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(STYLES.get_val('inspector', 'attr_spacing', default=4))
        self._layout = layout
        self.setLayout(self._layout)
        # Top Row (Input)
        self.top_row = QWidget()
        top_layout = QHBoxLayout(self.top_row)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(STYLES.get_val('inspector', 'top_row_spacing', default=6))
        self._build_standard_input()
        top_layout.addWidget(self._input_widget)
        self._layout.addWidget(self.top_row)
        # Model/graph bindings
        self.node.initializer_changed.connect(self._on_initializer_changed)
        self.node.value_changed.connect(self._on_node_value_changed)
        if self.graph_model and self.config_path:
            self.graph_model.config_value_changed.connect(self._on_global_config_changed)
            self.graph_model.inheritance_updated.connect(self._on_inheritance_tree_updated)
            self._on_inheritance_tree_updated()
        # A mandatory initializer is activated without touching the configuration: the selector opens on a
        # placeholder entry, so an unset field stays unset until one is picked.
        if self._init_required and not self.node.is_initializer_active:
            self.node._is_initializer_active = True
        if self.node.is_initializer_active:
            self._on_initializer_changed(self.node, True)

    #-------------------------------------------------------------------------------------------------------#
    # Inheritance
    #-------------------------------------------------------------------------------------------------------#

    def _inheritance_path(self) -> list[str]:
        path = list(self.config_path or [])
        if 'init_config' in path:
            return path[:path.index('init_config')]
        return path

    def _node_alive(self) -> bool:
        # NOTE: The inspector deletes its widgets with deleteLater(), so a widget can outlive the state node
        # it observes and still receive graph level signals.
        return isValid(self) and isValid(self.node)

    def _set_driven_state(self, is_driven: bool) -> None:
        def _update() -> None:
            if not isValid(self):
                return
            self.setEnabled(not is_driven)
            self.setToolTip('This value is inherited from a parent field.' if is_driven else '')
        QTimer.singleShot(0, _update)

    def _on_inheritance_tree_updated(self) -> None:
        if not (self.graph_model and self.config_path) or not self._node_alive():
            return
        leaf = self.graph_model.get_inheritance_leaf(self._inheritance_path())
        if leaf is None:
            self._set_driven_state(False)
            return
        self._set_driven_state(leaf.is_receiving())
        if 'init_config' not in (self.config_path or []):
            self.node._is_inherited = leaf.is_inheriting()

    #-------------------------------------------------------------------------------------------------------#
    # Value plumbing
    #-------------------------------------------------------------------------------------------------------#

    def _user_changed_value(self, new_val: tp.Any) -> None:
        if self._is_updating_programmatically:
            return
        new_val = coerce(self._kind, new_val)
        # Nothing is pushed when the value did not change.
        if values_equal(getattr(self.node, 'value', None), new_val):
            return
        # NOTE: A driven field is owned by the field cascading into it. The editor is disabled in that state,
        # and the value is refused here as well, so that no rejected write reaches the state model or the
        # undo stack.
        if self.graph_model and self.config_path and self.graph_model.is_driven(self._inheritance_path()):
            logger.debug(f'Ignored edit of "{"/".join(self.config_path)}": the field inherits its value.')
            self._resync_widget()
            return
        if self.graph_model and getattr(self.graph_model, 'undo_stack', None) and self.config_path:
            cmd = ChangeConfigValueCommand(self.graph_model, self.config_path, self.node.value, new_val, self.node)
            self.graph_model.undo_stack.push(cmd)
        else:
            self.node.value = new_val

    def _on_global_config_changed(self, node_id: str, path: list[str], value: tp.Any) -> None:
        if path != self.config_path or not self._node_alive():
            return
        is_init = isinstance(value, InitializerConfig)
        # Update the value held by the node first.
        self._is_updating_programmatically = True
        try:
            self.node.value = value
        finally:
            self._is_updating_programmatically = False
        # Switch the visual mode when needed.
        if self.node.is_initializer_active != is_init:
            self.node.is_initializer_active = is_init
        elif is_init and self._init_combo:
            # Already an initializer, but of a different type (e.g. Constant -> Uniform).
            if self._init_combo.currentData() is not type(value):
                for i in range(self._init_combo.count()):
                    if self._init_combo.itemData(i) is type(value):
                        self._init_combo.blockSignals(True)
                        self._init_combo.setCurrentIndex(i)
                        self._init_combo.blockSignals(False)
                        break
                self._build_initializer_block(value)

    def _on_node_value_changed(self, node: ConfigNode, value: tp.Any) -> None:
        if isinstance(value, InitializerConfig):
            return
        self._resync_widget()

    def _resync_widget(self) -> None:
        """
            Redraws the editor from the value currently held by the state model.
        """
        if self._apply_value is None:
            return
        self._is_updating_programmatically = True
        try:
            if self._input_widget is not None:
                self._input_widget.blockSignals(True)
            self._apply_value(self.node.value)
        finally:
            if self._input_widget is not None:
                self._input_widget.blockSignals(False)
            self._is_updating_programmatically = False

    def _plain_value(self) -> tp.Any:
        """
            Current value of the plain (non initializer) editor.
        """
        if self._read_value is None:
            return None
        try:
            return coerce(self._kind, self._read_value())
        except Exception:
            return None

    #-------------------------------------------------------------------------------------------------------#
    # Input widgets
    #-------------------------------------------------------------------------------------------------------#

    def _build_standard_input(self) -> None:
        builders = {
            FieldKind.BOOL: self._build_bool_input,
            FieldKind.DTYPE: self._build_dtype_input,
            FieldKind.INT: self._build_number_input,
            FieldKind.FLOAT: self._build_number_input,
            FieldKind.INT_TUPLE: self._build_dims_input,
            FieldKind.FLOAT_TUPLE: self._build_dims_input,
            FieldKind.STR: self._build_text_input,
        }
        builders.get(self._kind, self._build_readonly_input)()
        self._input_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # NOTE: Spin boxes size themselves after their widest representable number and combo boxes after their
        # widest entry. An explicit minimum takes precedence over those content driven hints, so the inspector
        # can be narrowed and the text elides instead of overflowing.
        self._input_widget.setMinimumWidth(STYLES.get_val('inspector', 'input_min_width', default=60))
        # Documentation. Descriptions and units are declared in the field metadata of the configuration.
        description = self.node.metadata.get('description', None)
        units = self.node.metadata.get('units', None)
        tooltip = [description] if description else []
        tooltip.append(f'Type: {type_label(self._tokens, self.node.type_hint)}')
        if units:
            tooltip.append(f'Units: {units}')
        if self._optional:
            tooltip.append('Optional: may be left unset.')
        if accepts_array(self._tokens) and not self._allows_init:
            tooltip.append('Array field: it has no editor form and can only be set programmatically.')
        self._input_widget.setToolTip('\n'.join(tooltip))

    def _build_bool_input(self) -> None:
        combo = self._make_combo()
        combo.addItem('True', True)
        combo.addItem('False', False)
        combo.setCurrentIndex(0 if self.node.value else 1)
        combo.currentIndexChanged.connect(lambda idx: self._user_changed_value(combo.itemData(idx)))
        self._input_widget = combo
        self._read_value = lambda: combo.currentData()
        def _apply(value: tp.Any) -> None:
            combo.setCurrentIndex(0 if value else 1)
        self._apply_value = _apply

    def _build_dtype_input(self) -> None:
        combo = self._make_combo()
        options = list(self.node.metadata.get('value_options', None) or DEFAULT_DTYPES)
        current_key = dtype_key(self.node.value)
        keys = [dtype_key(option) for option in options]
        # NOTE: jnp.float16 and np.float16 are distinct objects describing the same dtype, so every lookup is
        # performed on the canonical dtype name.
        if current_key is None:
            combo.addItem('', None)
        elif current_key not in keys:
            # The current value stays visible when it is not part of the declared options.
            options.append(self.node.value)
            keys.append(current_key)
        for option, key in zip(options, keys):
            combo.addItem(str(key), option)

        def _apply(value: tp.Any) -> None:
            key = dtype_key(value)
            for i in range(combo.count()):
                if dtype_key(combo.itemData(i)) == key:
                    combo.setCurrentIndex(i)
                    return

        _apply(self.node.value)
        combo.currentIndexChanged.connect(lambda idx: self._user_changed_value(combo.itemData(idx)))
        self._input_widget = combo
        self._read_value = lambda: combo.currentData()
        self._apply_value = _apply

    def _build_number_input(self) -> None:
        low, high = _numeric_bounds(self._kind, self.node.metadata)
        spin = QDoubleSpinBox()
        spin.setDecimals(0 if self._kind is FieldKind.INT else _FLOAT_DECIMALS)
        # NOTE: Partial configurations leave unset fields as None. Every numeric editor reserves its first
        # tick for that state, which renders as an empty field and maps back to None.
        step = 1.0 if self._kind is FieldKind.INT else 10.0 ** (-_FLOAT_DECIMALS)
        spin.setRange(low - step, high)
        spin.setSpecialValueText(' ')
        spin.wheelEvent = lambda event: event.ignore()
        units = self.node.metadata.get('units', None)
        if units:
            spin.setSuffix(f' {units}')
        self._set_spin_value(spin, self.node.value)
        spin.valueChanged.connect(lambda _: self._user_changed_value(self._read_spin(spin)))
        self._input_widget = spin
        self._read_value = lambda: self._read_spin(spin)
        self._apply_value = lambda value: self._set_spin_value(spin, value)

    def _build_dims_input(self) -> None:
        editor = QDimsEdit(
            value=self.node.value,
            is_integer=self._kind is FieldKind.INT_TUPLE,
            minimum=1 if self._kind is FieldKind.INT_TUPLE else 0.0,
        )
        editor.value_changed.connect(self._user_changed_value)
        self._input_widget = editor
        self._read_value = editor.value
        self._apply_value = editor.set_value

    def _build_text_input(self) -> None:
        line = QLineEdit('' if self.node.value is None else str(self.node.value))
        # NOTE: editingFinished avoids pushing an undo command for every keystroke.
        line.editingFinished.connect(lambda: self._user_changed_value(line.text()))
        self._input_widget = line
        self._read_value = line.text
        self._apply_value = lambda value: line.setText('' if value is None else str(value))

    def _build_readonly_input(self) -> None:
        line = QLineEdit(_array_summary(self.node.value))
        line.setReadOnly(True)
        line.setObjectName('readonlyInput')
        self._input_widget = line
        # NOTE: An array cannot be typed in, so the plain editor of an array field is always "unset". Turning
        # the initializer off clears the field.
        self._read_value = lambda: None
        self._apply_value = lambda value: line.setText(_array_summary(value))

    def _make_combo(self) -> QComboBox:
        combo = QComboBox()
        combo.setView(QListView())
        combo.wheelEvent = lambda event: event.ignore()
        # The size hint does not track the longest entry. The popup still shows the full text.
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(STYLES.get_val('inspector', 'combo_min_chars', default=6))
        return combo

    def _read_spin(self, spin: QDoubleSpinBox) -> tp.Any:
        # The first tick is the "unset" state.
        if spin.value() == spin.minimum():
            return None
        return spin.value()

    def _set_spin_value(self, spin: QDoubleSpinBox, value: tp.Any) -> None:
        if value is None or isinstance(value, InitializerConfig):
            # Optional fields display "unset", mandatory ones fall back to the closest legal value.
            spin.setValue(spin.minimum())
            return
        try:
            spin.setValue(float(value))
        except (TypeError, ValueError):
            spin.setValue(spin.minimum())

    #-------------------------------------------------------------------------------------------------------#
    # Initializers
    #-------------------------------------------------------------------------------------------------------#

    def _build_initializer_selector(self) -> None:
        if self._init_combo:
            return
        self._init_combo = self._make_combo()
        self._init_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._init_combo.setMinimumWidth(STYLES.get_val('inspector', 'input_min_width', default=60))
        # A mandatory initializer on an unset field opens on a placeholder, so selecting the node writes
        # nothing.
        if self._init_required and not isinstance(self.node.value, InitializerConfig):
            self._init_combo.addItem('Select an initializer...', userData=None)
        for name, entry in REGISTRY.Initializers.items():
            try:
                config_cls = entry.get_cls().get_config_spec()
            except Exception as error:
                logger.warning(f'Skipping initializer "{name}": {error}')
                continue
            self._init_combo.addItem(utils.to_human_readable(name), userData=config_cls)
        self._init_combo.currentIndexChanged.connect(self._on_init_combo_changed)

    def _on_init_combo_changed(self, index: int) -> None:
        config_cls = self._init_combo.itemData(index)
        if config_cls is None:
            return
        new_config = config_cls()
        self._build_initializer_block(new_config)
        self._user_changed_value(new_config)
        # The placeholder entry is dropped once an initializer is chosen.
        if self._init_combo.itemData(0) is None:
            self._init_combo.blockSignals(True)
            self._init_combo.removeItem(0)
            self._init_combo.setCurrentIndex(max(0, index - 1))
            self._init_combo.blockSignals(False)

    def _build_initializer_block(self, config_obj: InitializerConfig) -> None:
        if self._init_block:
            self._layout.removeWidget(self._init_block)
            self._init_block.deleteLater()
            self._init_block = None
        self._init_state_model = parse_object_to_state('init_config', config_obj)
        self._init_block = QWidget()
        self._init_block.setObjectName('initBlock')
        margin = STYLES.get_val('inspector', 'block_margin', default=8)
        block_layout = QFormLayout(self._init_block)
        block_layout.setContentsMargins(margin, margin, margin, margin)
        block_layout.setSpacing(STYLES.get_val('inspector', 'block_spacing', default=6))
        block_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        block_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        block_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        for prim in [c for c in self._init_state_model.children if isinstance(c, ConfigValueNode)]:
            lbl_widget = QWidget()
            lbl_layout = QHBoxLayout(lbl_widget)
            lbl_layout.setContentsMargins(0, 0, 0, 0)
            lbl_layout.setSpacing(STYLES.get_val('inspector', 'label_spacing', default=4))
            prim_path = (self.config_path or []) + ['init_config', prim.name]
            lbl_layout.addWidget(QAttrControls(prim, prim_path, self.graph_model))
            label = QLabel(prim.name.replace('_', ' ').title())
            label.setObjectName('attrLabel')
            label.setWordWrap(True)
            lbl_layout.addWidget(label)
            lbl_layout.addStretch(1)
            block_layout.addRow(lbl_widget, QAttribute(prim, prim_path, self.graph_model))
        self._layout.addWidget(self._init_block)

    def _on_initializer_changed(self, node: ConfigNode, is_active: bool) -> None:
        if is_active:
            self._input_widget.setVisible(False)
            self._build_initializer_selector()
            self.top_row.layout().addWidget(self._init_combo)
            self._init_combo.setVisible(True)
            if not isinstance(self.node.value, InitializerConfig):
                # Nothing is written while the placeholder entry is selected.
                if self._init_combo.currentData() is not None:
                    self._on_init_combo_changed(self._init_combo.currentIndex())
            else:
                self._init_combo.blockSignals(True)
                for i in range(self._init_combo.count()):
                    if self._init_combo.itemData(i) is type(self.node.value):
                        self._init_combo.setCurrentIndex(i)
                        break
                self._init_combo.blockSignals(False)
                self._build_initializer_block(self.node.value)
            return
        # Back to the plain editor.
        if self._init_combo:
            self._init_combo.setVisible(False)
        if self._init_block:
            self._init_block.setVisible(False)
        self._input_widget.setVisible(True)
        if isinstance(self.node.value, InitializerConfig):
            self._user_changed_value(self._plain_value())

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
