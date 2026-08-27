#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.graph_editor.models.node_model import NodeModel
    from spark.graph_editor.models.graph_model import GraphModel

import logging
import typing as tp
from shiboken6 import isValid
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QFormLayout, 
    QLineEdit, QLabel, QHBoxLayout, QPushButton, QPlainTextEdit, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, QSize, QMargins, QEvent, Signal
from spark.graph_editor.styles.manager import STYLES
from spark.graph_editor.styles import resources as icons
from spark.graph_editor.widgets.attribute_view import QAttribute, QAttrControls
from spark.graph_editor.models.inspector_model import (
    ConfigNode, ConfigValueNode, ConfigGroupNode, ConfigListNode, parse_object_to_state
)
from spark.graph_editor.widgets.scroll_utils import ScrollMarginBalancer
from spark.graph_editor.commands.graph_commands import RenameNodeCommand
from spark.core.specs import ModuleSpecs
logger = logging.getLogger('spark')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class NodeNameWidget(QWidget):

    on_update = Signal(str)

    def __init__(self, name: str, **kwargs) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._icon_label = QLabel()
        self._icon_label.setObjectName('nodeNameIcon')
        icon_max = STYLES.get_val('inspector', 'name_icon_max')
        self._icon_label.setPixmap(icons.get_pixmap(icons.NODE, icon_max))
        self._icon_label.setMaximumWidth(icon_max)
        self._icon_label.setMaximumHeight(icon_max)
        layout.addWidget(self._icon_label)
        self._line_edit = QLineEdit(name)
        self._line_edit.setObjectName('nodeNameInput')
        self._line_edit.textChanged.connect(self.on_update.emit)
        layout.addWidget(self._line_edit)
        self.setLayout(layout)
        self._target_height = STYLES.get_val('inspector', 'name_height')
        self.setFixedHeight(self._target_height)

    def sizeHint(self) -> QSize:
        return QSize(super().sizeHint().width(), self._target_height)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TreeDisplay(QPlainTextEdit):

    def __init__(self, tree: str) -> None:
        super().__init__()
        # Trailing newlines count as extra blocks and show up as empty rows.
        self.setPlainText(tree.rstrip('\n'))
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setObjectName('configTreeDisplay')
        self.setContentsMargins(QMargins(16, 0, 0, 4))
        self._update_height()

    def _content_height(self) -> int:
        """
            Height required to show every line of the tree.
        """
        lines = max(1, self.document().blockCount())
        text_height = lines * self.fontMetrics().lineSpacing() + 2 * self.document().documentMargin()
        chrome = max(0, self.height() - self.viewport().height())
        return int(text_height + chrome)

    def _update_height(self) -> None:
        height = self._content_height()
        # Guarded, so the resize triggered by setFixedHeight settles instead of looping.
        if height != self.maximumHeight():
            self.setFixedHeight(height)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._update_height()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # A narrow inspector brings up the horizontal scroll bar, which needs room of its own.
        self._update_height()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange):
            self._update_height()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class NodeHeaderWidget(QWidget):

    def __init__(self, node_model: NodeModel, graph_model: GraphModel | None = None, config_tree: str | None = None, **kwargs) -> None:
        super().__init__()
        self.node_model = node_model
        self.graph_model = graph_model

        
        layout = QVBoxLayout(self)
        hm = STYLES.get_val('inspector', 'header_margins')
        # The separator belongs to the inspector layout rather than to the header, so a single spacing value
        # governs the space above and below it. The header adds no bottom margin.
        layout.setContentsMargins(hm[0], hm[1], hm[2], 0)
        layout.setSpacing(STYLES.get_val('inspector', 'header_spacing'))
        
        self.name_widget = NodeNameWidget(node_model.name, **kwargs)
        self.name_widget.on_update.connect(self._on_name_changed)
        layout.addWidget(self.name_widget)
        if hasattr(self.name_widget, 'on_text_changed'):
            self.name_widget.on_text_changed.connect(self._on_name_typing)
        self.error_label = QLabel('')
        self.error_label.setObjectName('nodeErrorLabel')
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)

        self.class_label = QLabel(node_model.type_name)
        self.class_label.setObjectName('nodeClassLabel')
        self.class_label.setContentsMargins(QMargins(36, 0, 0, 8))
        layout.addWidget(self.class_label)
        if config_tree:
            self.config_tree_label = QLabel('Configuration Tree')
            self.config_tree_label.setObjectName('configTreeLabel')
            self.config_tree_label.setContentsMargins(QMargins(16, 8, 0, 4))
            layout.addWidget(self.config_tree_label)
            self.tree_label = TreeDisplay(config_tree)
            layout.addWidget(self.tree_label)

    def _is_name_taken(self, new_name: str) -> bool:
        if not new_name.strip():
            return True
        if new_name == self.node_model.name:
            return False
        if self.graph_model:
            return self.graph_model.is_name_taken(new_name)
        return False

    def _on_name_typing(self, text: str) -> None:
        if self._is_name_taken(text):
            self.error_label.setText(f'Name "{text}" is already in use.')
            self.error_label.setVisible(True)
            if hasattr(self.name_widget, 'set_valid_state'):
                self.name_widget.set_valid_state(False)
        else:
            self.error_label.setVisible(False)
            if hasattr(self.name_widget, 'set_valid_state'):
                self.name_widget.set_valid_state(True)

    def _on_name_changed(self, new_name: str) -> None:
        new_name = new_name.strip()
        if self.node_model.name == new_name:
            self.error_label.setVisible(False)
            return
        if self._is_name_taken(new_name):
            logger.warning(f'Cannot rename node "{self.node_model.name}" to "{new_name}", name is already in use.')
            self.error_label.setText(f'Name "{new_name}" is already in use.')
            self.error_label.setVisible(True)
            if hasattr(self.name_widget, 'set_text'):
                self.name_widget.set_text(self.node_model.name)
            elif hasattr(self.name_widget, 'setText'):
                self.name_widget.setText(self.node_model.name)
            return
        self.error_label.setVisible(False)
        if self.graph_model and getattr(self.graph_model, 'undo_stack', None):
            cmd = RenameNodeCommand(self.node_model, self.node_model.name, new_name)
            self.graph_model.undo_stack.push(cmd)
        else:
            self.node_model.name = new_name

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ControllerHeaderWidget(QWidget):
    """
        Header of the controller settings, shown while no node is selected.
    """

    def __init__(self, profile, config, **kwargs) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        hm = STYLES.get_val('inspector', 'header_margins')
        layout.setContentsMargins(*hm)
        layout.setSpacing(STYLES.get_val('inspector', 'header_spacing'))

        title_row = QWidget()
        row = QHBoxLayout(title_row)
        row.setContentsMargins(0, 0, 0, 0)
        icon = QLabel()
        icon.setObjectName('nodeNameIcon')
        icon_max = STYLES.get_val('inspector', 'name_icon_max')
        icon.setPixmap(icons.get_pixmap(profile.icon, icon_max))
        icon.setMaximumWidth(icon_max)
        icon.setMaximumHeight(icon_max)
        row.addWidget(icon)
        name = QLabel(profile.label)
        name.setObjectName('controllerTitle')
        row.addWidget(name)
        row.addStretch(1)
        layout.addWidget(title_row)

        subtitle = QLabel('Settings of the controller this graph describes.')
        subtitle.setObjectName('nodeClassLabel')
        subtitle.setWordWrap(True)
        subtitle.setContentsMargins(QMargins(36, 0, 0, 8))
        layout.addWidget(subtitle)
        if config is not None:
            tree_label = QLabel('Configuration Tree')
            tree_label.setObjectName('configTreeLabel')
            tree_label.setContentsMargins(QMargins(16, 8, 0, 4))
            layout.addWidget(tree_label)
            layout.addWidget(TreeDisplay(config._inspect(simplified=True)))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class InspectorView(QWidget):

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.current_node = None
        self.state_model = None
        self.graph_model = None
        self._is_built = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        bg_color = STYLES.get_val('inspector', 'background_color')
        min_width = STYLES.get_val('inspector', 'min_width')
        self.setMinimumWidth(min_width)
        self._scroll = QScrollArea()
        self._scroll.setObjectName('inspectorScroll')
        self._scroll.viewport().setObjectName('inspectorScrollViewport')
        self._scroll.setWidgetResizable(True)
        # The content shrinks to fit the viewport, down to the floor set by the toggle buttons and a wrapped
        # label. Past that point the panel scrolls.
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.content_widget = QWidget()
        self.content_widget.setObjectName('inspectorContent')
        self.content_layout = QVBoxLayout(self.content_widget)
        cm = STYLES.get_val('inspector', 'content_margins')
        self.content_layout.setContentsMargins(*cm)
        self.content_layout.setSpacing(STYLES.get_val('inspector', 'content_spacing'))
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._scroll.setWidget(self.content_widget)
        layout.addWidget(self._scroll)
        # Both gutters stay equal, with or without the vertical scroll bar.
        self._margin_balancer = ScrollMarginBalancer(self._scroll, self.content_layout, cm)
        self._build_empty_state()

    def invalidate(self) -> None:
        """
            Forces the next set_node() to rebuild, even when it names the same target.
        """
        self._is_built = False

    def set_node(self, node_model: NodeModel, graph_model: GraphModel | None = None) -> None:
        """
            Populates the inspector with the properties and configuration of the selected node.
        """
        # NOTE: Selection changes arrive one item at a time, so the same target is asked for repeatedly. The
        # panel is only rebuilt when the target changes.
        if node_model is self.current_node and graph_model is self.graph_model and self._is_built:
            return
        self.current_node = node_model
        self.graph_model = graph_model
        self._is_built = True
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if not self.current_node:
            # With nothing selected the inspector shows the controller itself. Its settings (the size of the
            # pool, dt, the seed) belong to the graph rather than to any node.
            if self.graph_model is not None and self.graph_model.profile is not None:
                self._build_controller_block(self.graph_model)
            else:
                self._build_empty_state()
            return
        # Base Node Properties Block
        self._build_base_block(self.current_node)
        # Config Properties Blocks (Flattened)
        if self.current_node.config is not None:
            self.state_model = parse_object_to_state('Configuration', self.current_node.config)
            self._flatten_and_build_blocks(self.state_model, '', [self.current_node.id])

    def _build_controller_block(self, graph_model: GraphModel) -> None:
        """
            Shows the settings of the controller the graph describes.
        """
        profile = graph_model.profile
        config = graph_model.controller_config
        header = ControllerHeaderWidget(profile, config)
        header.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.content_layout.addWidget(header)
        separator = QFrame()
        separator.setObjectName('inspectorSeparator')
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.content_layout.addWidget(separator)
        if config is None:
            return
        # Only the settings of the controller itself belong here. Its modules live on the canvas.
        self.state_model = parse_object_to_state(profile.label, config)
        self.state_model.children = [
            child for child in self.state_model.children if not isinstance(child, ConfigListNode)
        ]
        self._flatten_and_build_blocks(self.state_model, '', [graph_model.CONTROLLER_ID])

    def _build_empty_state(self) -> None:
        label = QLabel('No Node Selected')
        label.setObjectName('inspectorEmptyState')
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.content_layout.addWidget(label)

    def _create_block_widget(self, title: str) -> tuple[QWidget, QFormLayout]:
        """
            Creates a block widget with a header and a form layout.
        """
        block = QWidget()
        block.setObjectName('inspectorBlock')
        layout = QVBoxLayout(block)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        # Header
        header_btn = QPushButton(f' ▼ {title.upper()}')
        header_btn.setObjectName('inspectorBlockHeader')
        header_btn.setCheckable(True)
        header_btn.setChecked(True)
        header_btn.setProperty('expanded', True)
        
        def _update_header_style(checked) -> None:
            header_btn.setProperty('expanded', bool(checked))
            header_btn.style().unpolish(header_btn)
            header_btn.style().polish(header_btn)

        _update_header_style(True)
        layout.addWidget(header_btn)
        # Form
        form_container = QWidget()
        form_container.setObjectName('inspectorFormContainer')
        form = QFormLayout(form_container)
        fm = STYLES.get_val('inspector', 'form_margins')
        form.setContentsMargins(*fm)
        form.setSpacing(STYLES.get_val('inspector', 'form_spacing'))
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        # NOTE: Side by side rows cannot be narrower than label + field. The field wraps under its label, so
        # the rows do not overflow at small widths.
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.addWidget(form_container)
        
        def _toggle_block(checked) -> None:
            form_container.setVisible(checked)
            arrow = '▼' if checked else '▶'
            header_btn.setText(f' {arrow} {title.upper()}')
            _update_header_style(checked)
        
        header_btn.toggled.connect(_toggle_block)
        self.content_layout.addWidget(block)
        return block, form

    def _build_base_block(self, node_model: NodeModel) -> None:
        config_tree_str = None
        if node_model.config is not None:
            config_tree_str = node_model.config._inspect(simplified=True)
        header_widget = NodeHeaderWidget(node_model, self.graph_model, config_tree_str)
        header_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.content_layout.addWidget(header_widget)
        # Separator between the header and the configuration blocks. Both sides get the same spacing.
        gap = STYLES.get_val('inspector', 'separator_spacing', default=12)
        content_spacing = self.content_layout.spacing()
        self.content_layout.addSpacing(max(0, gap - content_spacing))
        separator = QFrame()
        separator.setObjectName('inspectorSeparator')
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.content_layout.addWidget(separator)
        self.content_layout.addSpacing(max(0, gap - content_spacing))
        # External name updates reach the name widget.
        node_model.name_changed.connect(
            lambda text, widget=header_widget.name_widget._line_edit: widget.setText(text) if isValid(widget) and widget.text() != text else None
        )

    def _flatten_and_build_blocks(self, node: tp.Any, path_prefix: str, config_path: list[str]) -> None:
        """
            Flattens the ConfigNode tree into the blocks shown by the inspector.
        """
        label_color = STYLES.get_val('inspector', 'label_color')
        if isinstance(node, ConfigGroupNode):
            # The title of a ModuleSpecs block is its name.
            title = node.name
            if issubclass(node.class_ref, ModuleSpecs):
                name_child = next((c for c in node.children if c.name == 'name'), None)
                if name_child:
                    title = str(name_child.value)
            full_title = f'{path_prefix} / {title}' if path_prefix else title
            primitives = [c for c in node.children if isinstance(c, ConfigValueNode)]
            # The "name" primitive of a ModuleSpecs is already in the block header.
            if issubclass(node.class_ref, ModuleSpecs):
                primitives = [p for p in primitives if p.name != 'name']
            if primitives:
                _, form = self._create_block_widget(full_title)
                for prim in primitives:
                    lbl_widget = QWidget()
                    lbl_layout = QHBoxLayout(lbl_widget)
                    lbl_layout.setContentsMargins(0, 0, 0, 0)
                    lbl_layout.setSpacing(STYLES.get_val('inspector', 'label_spacing'))
                    prim_path = config_path + [prim.name]
                    controls = QAttrControls(prim, prim_path, self.graph_model)
                    lbl_layout.addWidget(controls)
                    lbl = QLabel(prim.name.replace('_', ' ').title())
                    lbl.setObjectName('attrLabel')
                    # A label reports its full text as its minimum width, so it is allowed to wrap.
                    lbl.setWordWrap(True)
                    description = prim.metadata.get('description', None)
                    if description:
                        lbl.setToolTip(description)
                    lbl_layout.addWidget(lbl)
                    lbl_layout.addStretch(1)
                    inp = QAttribute(prim, prim_path, self.graph_model)
                    form.addRow(lbl_widget, inp)
            for child in node.children:
                if isinstance(child, ConfigGroupNode) or isinstance(child, ConfigListNode):
                    self._flatten_and_build_blocks(child, full_title, config_path + [child.name]) 
        elif isinstance(node, ConfigListNode):
            full_title = f'{path_prefix} / {node.name}' if path_prefix else node.name
            # One block per item of the list.
            for i, child in enumerate(node.children):
                if isinstance(child, ConfigGroupNode):
                    self._flatten_and_build_blocks(child, full_title, config_path + [child.name])
                elif isinstance(child, ConfigValueNode):
                    pass


#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################