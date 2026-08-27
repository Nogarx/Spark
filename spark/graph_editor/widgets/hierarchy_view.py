#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.graph_editor.models.graph_model import GraphModel

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem, QLineEdit, QLabel, QScrollArea, 
    QApplication, QAbstractItemView, QToolButton, QSizePolicy
)
from spark.graph_editor.styles.manager import STYLES
from spark.graph_editor.widgets.scroll_utils import ScrollMarginBalancer
from spark.graph_editor.models.node_model import NodeModel

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class HierarchyView(QWidget):
    node_double_clicked = Signal(NodeModel)

    def __init__(self, model: GraphModel, parent=None) -> None:
        super().__init__(parent)
        self.model = model
        self.node_to_item: dict[NodeModel, tuple[QTreeWidget, QTreeWidgetItem]] = {}
        self.trees: dict[str, QTreeWidget] = {}
        self.blocks: dict[str, QWidget] = {}
        self._updating_selection = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        bg_color = STYLES.get_val('hierarchy', 'background_color')
        min_width = STYLES.get_val('hierarchy', 'min_width')
        self.setMinimumWidth(min_width)
        search_container = QWidget()
        search_container.setObjectName('hierarchySearchContainer')
        search_layout = QVBoxLayout(search_container)
        sm = STYLES.get_val('hierarchy', 'search_margins')
        search_layout.setContentsMargins(*sm)
        self.search_bar = QLineEdit()
        self.search_bar.setObjectName('hierarchySearch')
        self.search_bar.setPlaceholderText('Search nodes...')
        self.search_bar.textChanged.connect(self.on_search_changed)
        search_layout.addWidget(self.search_bar)
        layout.addWidget(search_container)
        
        # Scroll Area (Outer, handles all scrolling)
        self._scroll = QScrollArea()
        self._scroll.setObjectName('hierarchyScroll')
        self._scroll.viewport().setObjectName('hierarchyScrollViewport')
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.content_widget = QWidget()
        self.content_widget.setObjectName('hierarchyContent')
        self.content_layout = QVBoxLayout(self.content_widget)
        cm = STYLES.get_val('hierarchy', 'content_margins')
        self.content_layout.setContentsMargins(*cm)
        self.content_layout.setSpacing(STYLES.get_val('hierarchy', 'content_spacing'))
        self._scroll.setWidget(self.content_widget)
        layout.addWidget(self._scroll)
        # Both gutters stay equal, with or without the vertical scroll bar.
        self._margin_balancer = ScrollMarginBalancer(self._scroll, self.content_layout, cm)
        # Categories
        cat_names = {
            'SourceNodeModel': 'SOURCES',
            'SinkNodeModel': 'SINKS',
            'InterfaceNodeModel': 'INTERFACES',
            'GeneralNodeModel': 'GENERAL'
        }
        for class_name, title in cat_names.items():
            block, tree = self._create_category_block(title)
            self.trees[class_name] = tree
            self.blocks[class_name] = block
        self.content_layout.addStretch(1)
        self.model.node_added.connect(self.on_node_added)
        self.model.node_removed.connect(self.on_node_removed)
        self.model.graph_cleared.connect(self.on_graph_cleared)
        # Add the nodes the model already holds.
        for node in self.model.nodes:
            self.on_node_added(node)
            
    def _create_category_block(self, title: str) -> tuple[QWidget, QTreeWidget]:
        block = QWidget()
        block = QWidget()
        block.setObjectName('hierarchyBlock')
        block.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        layout = QVBoxLayout(block)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header_btn = QToolButton()
        header_btn.setObjectName('hierarchyCategoryHeader')
        header_btn.setText(f' {title.upper()}')
        header_btn.setCheckable(True)
        header_btn.setChecked(True)
        header_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        header_btn.setArrowType(Qt.ArrowType.DownArrow)
        header_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        def _update_header_style(checked) -> None:
            header_btn.setProperty('expanded', bool(checked))
            header_btn.style().unpolish(header_btn)
            header_btn.style().polish(header_btn)

        _update_header_style(True)
        layout.addWidget(header_btn)
        tree = QTreeWidget()
        tree.setObjectName('hierarchyTree')
        tree.setHeaderHidden(True)
        tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        tree.setIndentation(0)
        tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        tree.setExpandsOnDoubleClick(False)
        # Internal scrolling is disabled, the outer QScrollArea handles it.
        tree.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        tree.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        tree.scrollTo = lambda *args, **kwargs: None
        tree.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(tree)
        self._adjust_tree_height(tree)
        def _toggle_block(checked) -> None:
            tree.setVisible(checked)
            header_btn.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
            _update_header_style(checked)
            # Force a layout update to recompute the scroll area.
            self.content_widget.updateGeometry()
        header_btn.toggled.connect(_toggle_block)
        tree.itemSelectionChanged.connect(lambda t=tree: self.on_tree_selection_changed(t))
        tree.itemDoubleClicked.connect(lambda item, col: self.node_double_clicked.emit(item.data(0, Qt.UserRole)))
        self.content_layout.addWidget(block)
        return block, tree

    def on_node_added(self, node: NodeModel) -> None:
        class_name = node.__class__.__name__
        tree = self.trees.get(class_name, self.trees['GeneralNodeModel'])
        item = QTreeWidgetItem(tree)
        self.node_to_item[node] = (tree, item)
        item.setData(0, Qt.ItemDataRole.UserRole, node)
        self._update_item_widget(tree, item, node)
        self._adjust_tree_height(tree)
        node.name_changed.connect(lambda n, nd=node: self._on_node_data_changed(nd))
        node.type_changed.connect(lambda t, nd=node: self._on_node_data_changed(nd))
        node.selected_changed.connect(lambda s, nd=node: self._on_node_selected_changed(nd, s))
        if node.is_selected:
            self._on_node_selected_changed(node, True)
        self.on_search_changed(self.search_bar.text())

    def _create_node_widget(self, node: NodeModel) -> QWidget:
        widget = QWidget()
        widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout = QVBoxLayout(widget)
        im = STYLES.get_val('hierarchy', 'item_margins')
        layout.setContentsMargins(*im)
        layout.setSpacing(STYLES.get_val('hierarchy', 'item_spacing'))
        title_label = QLabel(node.name)
        title_label.setObjectName('hierarchyItemTitle')
        subtitle_label = QLabel(node.type_name)
        subtitle_label.setObjectName('hierarchyItemSubtitle')
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        return widget

    def _update_item_widget(self, tree: QTreeWidget, item: QTreeWidgetItem, node: NodeModel) -> None:
        widget = self._create_node_widget(node)
        tree.setItemWidget(item, 0, widget)
        item.setSizeHint(0, widget.sizeHint())

    def _adjust_tree_height(self, tree: QTreeWidget) -> None:
        # The height of the QTreeWidget follows its contents.
        height = 0
        for i in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(i)
            if not item.isHidden():
                height += item.sizeHint(0).height()
        # The bottom border of the last item is clipped to avoid a double border.
        if height > 0:
            height -= 1
        else:
            # An empty tree leaves an 8px transparent area, so it reads as open and empty.
            height = 8
        tree.setMinimumHeight(height)
        tree.setMaximumHeight(height)

    def _on_node_data_changed(self, node: NodeModel) -> None:
        if node in self.node_to_item:
            tree, item = self.node_to_item[node]
            self._update_item_widget(tree, item, node)
            self._adjust_tree_height(tree)
            self.on_search_changed(self.search_bar.text())

    def _on_node_selected_changed(self, node: NodeModel, is_selected: bool) -> None:
        if self._updating_selection:
            return
        if node in self.node_to_item:
            tree, item = self.node_to_item[node]
            self._updating_selection = True
            item.setSelected(is_selected)
            self._updating_selection = False

    def on_node_removed(self, node: NodeModel) -> None:
        if node in self.node_to_item:
            tree, item = self.node_to_item.pop(node)
            tree.takeTopLevelItem(tree.indexOfTopLevelItem(item))
            self._adjust_tree_height(tree)

    def on_graph_cleared(self) -> None:
        for node, (tree, item) in self.node_to_item.items():
            tree.takeTopLevelItem(tree.indexOfTopLevelItem(item))
            self._adjust_tree_height(tree)
        self.node_to_item.clear()

    def on_tree_selection_changed(self, changed_tree: bool) -> None:
        if self._updating_selection:
            return
        modifiers = QApplication.keyboardModifiers()
        if not (modifiers & Qt.KeyboardModifier.ControlModifier) and not (modifiers & Qt.KeyboardModifier.ShiftModifier):
            # The selection of the other trees is cleared unless the selection spans them.
            self._updating_selection = True
            for t in self.trees.values():
                if t != changed_tree:
                    t.clearSelection()
            self._updating_selection = False
        self._updating_selection = True
        selected_nodes = []
        for t in self.trees.values():
            selected_items = t.selectedItems()
            selected_nodes.extend([item.data(0, Qt.UserRole) for item in selected_items if item.data(0, Qt.UserRole)])
        for node in self.model.nodes:
            if node.is_selected != (node in selected_nodes):
                node.is_selected = (node in selected_nodes)
        self._updating_selection = False

    def on_search_changed(self, text: str) -> None:
        search_text = text.lower()
        for node, (tree, item) in self.node_to_item.items():
            if search_text in node.name.lower() or search_text in node.type_name.lower():
                item.setHidden(False)
            else:
                item.setHidden(True)
        # Adjust the tree heights to the hidden items.
        for class_name, block in self.blocks.items():
            tree = self.trees[class_name]
            self._adjust_tree_height(tree)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################