#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.graph_editor.models.graph_model import GraphModel
    from spark.graph_editor.models.inspector_model import ConfigNode

import types
import typing as tp
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QComboBox, QSpinBox, QDoubleSpinBox,
    QLineEdit, QLabel, QFormLayout, QListView, QToolButton
)
from PySide6.QtCore import Qt, QTimer
from spark.graph_editor.models.inspector_model import ConfigValueNode, ConfigGroupNode, parse_object_to_state
from spark.graph_editor.models.inheritance_tree import InheritanceFlags
from spark.graph_editor.commands.inspector_commands import ChangeConfigValueCommand, ToggleInheritanceCommand
from spark.core.registry import REGISTRY
from spark.nn.initializers import InitializerConfig

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
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        # Warning Icon
        self.warning_btn = QToolButton()
        self.warning_btn.setObjectName('warningBtn')
        self.warning_btn.setText('⚠')
        self.warning_btn.setToolTip('No errors')
        sp = self.warning_btn.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        self.warning_btn.setSizePolicy(sp)
        self.warning_btn.setVisible(False)
        layout.addWidget(self.warning_btn)
        # Initializer Toggle
        self.init_btn = QToolButton()
        self.init_btn.setText('⚙')
        self.init_btn.setCheckable(True)
        self.init_btn.setToolTip('Toggle Initializer')
        if self.node.metadata.get('allows_init', False):
            self.init_btn.setChecked(self.node.is_initializer_active)
            self.init_btn.toggled.connect(self._on_init_toggled)
        else:
            self.init_btn.setEnabled(False)
            self.init_btn.setText('◇')
        layout.addWidget(self.init_btn)
        # Inheritance Toggle
        self.inherit_btn = QToolButton()
        self.inherit_btn.setCheckable(True)
        self.inherit_btn.setToolTip('Toggle Inheritance Linkage')
        if self.node.metadata.get('allows_inheritance', False):
            self.inherit_btn.setChecked(self.node.is_inherited)
            self._update_inherit_icon(self.node.is_inherited)
            self.inherit_btn.toggled.connect(self._on_inherit_toggled)
        else:
            self.inherit_btn.setEnabled(False)
            self.inherit_btn.setText('⚯')
        layout.addWidget(self.inherit_btn)
        # Connect to node state changes
        self.node.errors_changed.connect(self._on_errors_changed)
        self.node.inheritance_changed.connect(self._on_node_inheritance_changed)
        self.node.initializer_changed.connect(self._on_node_init_changed)
        if self.graph_model and self.config_path:
            self.graph_model.inheritance_updated.connect(self._on_inheritance_tree_updated)
            self._on_inheritance_tree_updated()

    def _on_inheritance_tree_updated(self) -> None:
        if self.graph_model and self.config_path:
            try:
                search_path = self.config_path
                if 'init_config' in search_path:
                    idx = search_path.index('init_config')
                    search_path = search_path[:idx]
                leaf = self.graph_model.inheritance_tree.get_leaf(search_path)
                if leaf:
                    is_driven = bool(leaf.flags & InheritanceFlags.IS_RECEIVING)
                    if self.init_btn and self.node.metadata.get('allows_init', False):
                        self.init_btn.setEnabled(not is_driven)
                    if self.inherit_btn and self.node.metadata.get('allows_inheritance', False):
                        self.inherit_btn.setEnabled(not is_driven)
                        
                        if 'init_config' not in self.config_path:
                            # Sync checked state without emitting signals
                            self.inherit_btn.blockSignals(True)
                            self.inherit_btn.setChecked(leaf.is_inheriting())
                            self._update_inherit_icon(leaf.is_inheriting())
                            self.inherit_btn.blockSignals(False)
                            self.node._is_inherited = leaf.is_inheriting()
            except KeyError: 
                pass

    def _update_inherit_icon(self, is_linked: bool) -> None:
        self.inherit_btn.setText('🔗' if is_linked else '⚯')

    def _on_init_toggled(self, checked: bool) -> None:
        self.node.is_initializer_active = checked

    def _on_inherit_toggled(self, checked: bool) -> None:
        if self.graph_model and getattr(self.graph_model, 'undo_stack', None):
            cmd = ToggleInheritanceCommand(self.graph_model, self.config_path, self.node.is_inherited, checked, self.node)
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

    def _on_node_inheritance_changed(self, node: ConfigNode, is_inherited: bool) -> None:
        def _update() -> None:
            self.inherit_btn.blockSignals(True)
            self.inherit_btn.setChecked(is_inherited)
            self._update_inherit_icon(is_inherited)
            self.inherit_btn.blockSignals(False)
        QTimer.singleShot(0, _update)

    def _on_node_init_changed(self, node: ConfigNode, is_init: bool) -> None:
        def _update() -> None:
            self.init_btn.blockSignals(True)
            self.init_btn.setChecked(is_init)
            self.init_btn.blockSignals(False)
        QTimer.singleShot(0, _update)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class QAttribute(QWidget):
    """
        Composite widget containing the input field.
    """

    def __init__(self, node: ConfigValueNode, config_path: list[str] = None, graph_model: GraphModel | None = None, parent: QWidget | None  =None) -> None:
        super().__init__(parent)
        self.node = node
        self.config_path = config_path
        self.graph_model = graph_model
        self._block_layout = None
        self._input_widget = None
        self._init_combo = None
        self._init_block = None
        self._init_state_model = None
        # Load inheritance state
        if self.graph_model and self.config_path:
            try:
                search_path = self.config_path
                if 'init_config' in search_path:
                    idx = search_path.index('init_config')
                    search_path = search_path[:idx]
                leaf = self.graph_model.inheritance_tree.get_leaf(search_path)
                if leaf:
                    if 'init_config' not in self.config_path:
                        self.node._is_inherited = leaf.is_inheriting()
                    is_driven = bool(leaf.flags & InheritanceFlags.IS_RECEIVING)
                    self._set_driven_state(is_driven)
            except KeyError: pass
            self.graph_model.config_value_changed.connect(self._on_global_config_changed)
            self.graph_model.inheritance_updated.connect(self._on_inheritance_tree_updated)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._layout = layout
        self.setLayout(self._layout)
        # Top Row (Input)
        self.top_row = QWidget()
        top_layout = QHBoxLayout(self.top_row)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(6)
        self._build_standard_input()
        top_layout.addWidget(self._input_widget)
        self._layout.addWidget(self.top_row)
        self.node.initializer_changed.connect(self._on_initializer_changed)
        self.node.value_changed.connect(self._on_node_value_changed)
        if self.node.is_initializer_active:
            self._on_initializer_changed(self.node, True)

    def _set_driven_state(self, is_driven: bool) -> None:
        QTimer.singleShot(0, lambda: self.setEnabled(not is_driven))

    def _on_inheritance_tree_updated(self) -> None:
        if self.graph_model and self.config_path:
            try:
                search_path = self.config_path
                if 'init_config' in search_path:
                    idx = search_path.index('init_config')
                    search_path = search_path[:idx]
                    
                leaf = self.graph_model.inheritance_tree.get_leaf(search_path)
                if leaf:
                    is_driven = bool(leaf.flags & InheritanceFlags.IS_RECEIVING)
                    self._set_driven_state(is_driven)
                    if 'init_config' not in self.config_path:
                        self.node.is_inherited = leaf.is_inheriting()
            except KeyError: 
                pass

    def _user_changed_value(self, new_val: tp.Any) -> None:
        if getattr(self, '_is_updating_programmatically', False): 
            return
        # Don't push if value hasn't actually changed
        if hasattr(self.node, 'value') and self.node.value == new_val:
            return
        if self.graph_model and getattr(self.graph_model, 'undo_stack', None) and self.config_path:
            cmd = ChangeConfigValueCommand(self.graph_model, self.config_path, self.node.value, new_val, self.node)
            self.graph_model.undo_stack.push(cmd)
        else:
            self.node.value = new_val

    def _on_global_config_changed(self, node_id: str, path: list[str], value: tp.Any) -> None:
        if path == self.config_path:
            is_init = isinstance(value, InitializerConfig)
            # Update the underlying node value first
            self._is_updating_programmatically = True
            self.node.value = value
            self._is_updating_programmatically = False
            # Then trigger visual mode switch if necessary
            if self.node.is_initializer_active != is_init:
                self.node.is_initializer_active = is_init
            elif is_init and self._init_combo:
                # If already an initializer but the specific type changed (e.g., Constant -> Uniform)
                if self._init_combo.currentData().get_config_spec() != type(value):
                    for i in range(self._init_combo.count()):
                        if self._init_combo.itemData(i).get_config_spec() == type(value):
                            self._init_combo.blockSignals(True)
                            self._init_combo.setCurrentIndex(i)
                            self._init_combo.blockSignals(False)
                            break
                    self._build_initializer_block(value)

    def _on_node_value_changed(self, node: ConfigNode, value: tp.Any) -> None:
        if isinstance(value, InitializerConfig):
            return
        if self._input_widget:
            self._is_updating_programmatically = True
            self._input_widget.blockSignals(True)
            if isinstance(self._input_widget, QDoubleSpinBox):
                self._input_widget.setValue(float(value))
            elif isinstance(self._input_widget, QSpinBox):
                self._input_widget.setValue(int(value))
            elif isinstance(self._input_widget, QComboBox):
                idx = self._input_widget.findData(value)
                if idx != -1: self._input_widget.setCurrentIndex(idx)
            elif isinstance(self._input_widget, QLineEdit):
                self._input_widget.setText(str(value))
            self._input_widget.blockSignals(False)
            self._is_updating_programmatically = False

    def _build_standard_input(self) -> None:
        t_hint = self.node.type_hint
        types_to_check = []
        if isinstance(t_hint, types.UnionType) or (hasattr(t_hint, '__origin__') and t_hint.__origin__ is tp.Union):
            types_to_check = list(tp.get_args(t_hint))
        else: types_to_check = [t_hint]
        if bool in types_to_check:
            cb = QComboBox()
            cb.setView(QListView())
            cb.wheelEvent = lambda event: event.ignore()
            cb.addItem('True', True)
            cb.addItem('False', False)
            cb.setCurrentIndex(0 if self.node.value else 1)
            cb.currentIndexChanged.connect(lambda idx: self._user_changed_value(cb.itemData(idx)))
            self._input_widget = cb
        elif float in types_to_check:
            sb = QDoubleSpinBox(decimals=4)
            sb.wheelEvent = lambda event: event.ignore()
            try:
                val = self.node.value if not isinstance(self.node.value, InitializerConfig) else 0.0
                sb.setValue(float(val))
            except (ValueError, TypeError): sb.setValue(0.0)
            sb.valueChanged.connect(lambda v: self._user_changed_value(v))
            self._input_widget = sb
        elif int in types_to_check:
            sb = QDoubleSpinBox(decimals=0)
            sb.wheelEvent = lambda event: event.ignore()
            try:
                val = self.node.value if not isinstance(self.node.value, InitializerConfig) else 0
                sb.setValue(int(val))
            except (ValueError, TypeError): sb.setValue(0)
            sb.valueChanged.connect(lambda v: self._user_changed_value(v))
            self._input_widget = sb
        else:
            le = QLineEdit(str(self.node.value))
            le.textChanged.connect(lambda v: self._user_changed_value(v))
            self._input_widget = le
        self._input_widget.setSizePolicy(self._input_widget.sizePolicy().Policy.Expanding, self._input_widget.sizePolicy().Policy.Fixed)

    def _build_initializer_selector(self) -> None:
        if not self._init_combo:
            self._init_combo = QComboBox()
            self._init_combo.setView(QListView())
            self._init_combo.wheelEvent = lambda event: event.ignore()
            self._init_combo.setSizePolicy(self._init_combo.sizePolicy().Policy.Expanding, self._init_combo.sizePolicy().Policy.Fixed)
            for name, init_cls in REGISTRY.INITIALIZERS.items(): self._init_combo.addItem(name, userData=init_cls)
            self._init_combo.currentIndexChanged.connect(self._on_init_combo_changed)

    def _on_init_combo_changed(self, index: int) -> None:
        init_cls = self._init_combo.currentData()
        new_config = init_cls.get_config_spec()()
        self._build_initializer_block(new_config)
        self._user_changed_value(new_config)

    def _build_initializer_block(self, config_obj) -> None:
        if self._init_block:
            self._layout.removeWidget(self._init_block)
            self._init_block.deleteLater()
        self._init_state_model = parse_object_to_state('init_config', config_obj)
        self._init_block = QWidget()
        self._init_block.setObjectName('initBlock')
        block_layout = QFormLayout(self._init_block)
        block_layout.setContentsMargins(8, 8, 8, 8); block_layout.setSpacing(6); block_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        for prim in [c for c in self._init_state_model.children if isinstance(c, ConfigValueNode)]:
            lbl_widget = QWidget(); lbl_layout = QHBoxLayout(lbl_widget); lbl_layout.setContentsMargins(0, 0, 0, 0); lbl_layout.setSpacing(4)
            prim_path = (self.config_path or []) + ['init_config', prim.name]
            lbl_layout.addWidget(QAttrControls(prim, prim_path, self.graph_model))
            lbl = QLabel(prim.name.replace('_', ' ').title()); 
            lbl_layout.addWidget(lbl); 
            lbl_layout.addStretch(1)
            block_layout.addRow(lbl_widget, QAttribute(prim, prim_path, self.graph_model))
        self._layout.addWidget(self._init_block)

    def _on_initializer_changed(self, node: ConfigNode, is_active: bool) -> None:
        if is_active:
            self._input_widget.setVisible(False)
            self._build_initializer_selector()
            self.top_row.layout().addWidget(self._init_combo)
            self._init_combo.setVisible(True)
            if not isinstance(self.node.value, InitializerConfig): self._on_init_combo_changed(self._init_combo.currentIndex())
            else:
                for i in range(self._init_combo.count()):
                    if self._init_combo.itemData(i).get_config_spec() == type(self.node.value):
                        self._init_combo.setCurrentIndex(i); break
                self._build_initializer_block(self.node.value)
        else:
            if self._init_combo: 
                self._init_combo.setVisible(False)
            if self._init_block: 
                self._init_block.setVisible(False)
            self._input_widget.setVisible(True)
            if isinstance(self.node.value, InitializerConfig):
                t_hint = self.node.type_hint
                types_to_check = list(tp.get_args(t_hint)) if (isinstance(t_hint, types.UnionType) or (hasattr(t_hint, '__origin__') and t_hint.__origin__ is tp.Union)) else [t_hint]
                new_val = 0.0
                if bool in types_to_check: 
                    new_val = False
                elif float in types_to_check: 
                    new_val = 0.0
                elif int in types_to_check: 
                    new_val = 0
                else: 
                    new_val = ''
                if isinstance(self._input_widget, (QDoubleSpinBox, QSpinBox)):
                    new_val = type(new_val)(self._input_widget.value())
                self._user_changed_value(new_val)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################