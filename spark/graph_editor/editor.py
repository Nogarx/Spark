#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import sys
import json
import pathlib
import logging
import dataclasses as dc

from shiboken6 import isValid
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QStatusBar, QLabel, QFileDialog, QStackedWidget, QMessageBox,
    QTabWidget, QMenu
)
from PySide6.QtGui import QAction, QUndoGroup, QShortcut, QKeySequence
from PySide6.QtCore import Qt, Signal

from spark.graph_editor.view.node_item import NodeItem
from spark.graph_editor.view.graph_view import GraphScene, GraphView
from spark.graph_editor.widgets.hierarchy_view import HierarchyView
from spark.graph_editor.widgets.inspector_view import InspectorView
from spark.graph_editor.widgets.console_view import ConsoleView, MessageLevel
from spark.graph_editor.widgets.preferences_dialog import PreferencesDialog
from spark.graph_editor.widgets.controller_selection import StartView, NewModelDialog
from spark.graph_editor.models.controller_profile import ControllerProfile, profile_for_config
from spark.graph_editor.models import session_io, recent_files, model_library
from spark.graph_editor.styles.manager import STYLES
from spark.graph_editor.styles import resources as icons

logger = logging.getLogger('spark')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@dc.dataclass
class EditorDocument:
    """
        One model open in the editor.
    """

    scene: GraphScene
    view: GraphView
    session_path: pathlib.Path | None = None
    model_path: pathlib.Path | None = None
    # NOTE: Two sessions can carry the same name (the same file opened twice). The window hands each one a
    # copy number, kept here so that a number, once given, does not change when another tab closes.
    copy_index: int = 0
    copy_name: str = ''

    @property
    def model(self):
        return self.scene.model

    @property
    def name(self) -> str:
        if self.session_path is not None:
            return self.session_path.stem
        if self.model_path is not None:
            return self.model_path.stem
        return 'Untitled'

    @property
    def is_modified(self) -> bool:
        # NOTE: A stack still reports its clean state while the document it belongs to is destroyed, so the
        # C++ side is asked first.
        try:
            stack = self.model.undo_stack
            return isValid(stack) and not stack.isClean()
        except RuntimeError:
            return False

    @property
    def label(self) -> str:
        """
            Name shown on the tab, numbered when it is not the first of its name.
        """
        name = self.name
        return name if self.copy_index <= 1 else f'{name} ({self.copy_index})'

    @property
    def title(self) -> str:
        return f'{self.label}*' if self.is_modified else self.label

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SparkGraphEditor:

    def __init__(self) -> None:

        self._is_interactive = 'ipykernel' in sys.modules
        if self._is_interactive:
            # Integrating the Qt event loop avoids the %gui qt magic in the terminal.
            from IPython import get_ipython
            get_ipython().enable_gui('qt')

        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)

        self.window: GraphEditorWindow | None = None

    def launch(self) -> None:
        """
            Creates and shows the editor window without blocking.
        """
        if self.window is not None:
            self.window.close()
            self.window.deleteLater()
            del self.window

        STYLES.init()
        STYLES.reloaded.connect(lambda app=self.app: STYLES.apply(app))
        STYLES.apply(self.app)

        self.window = GraphEditorWindow()
        self.window.windowClosed.connect(self.exit_editor)
        
        # DEBUG: SECOND SCREEN
        screens = self.app.screens()
        if len(screens) > 1:
            target_screen = screens[1]
            target_geo = target_screen.availableGeometry()
            self.window.setScreen(target_screen)
            self.window.setGeometry(target_geo)

        self.window.showMaximized()

        self.app.exec_()
    
    def exit_editor(self,) -> None:
        """
            Exits the editor.
        """
        self.app.quit()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class GraphEditorWindow(QMainWindow):

    windowClosed = Signal() 

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Spark Graph Editor')
        self.setDockNestingEnabled(True)
        # The editor holds several models at once, one per tab. The canvas becomes available once a controller
        # is picked, which dictates the palette and the exported configuration.
        self._documents: list[EditorDocument] = []
        # Stands in for "no document open", so the rest of the window can query the current graph without
        # checking first. It carries no profile, which disables the document actions.
        self._empty_scene = GraphScene()
        self._undo_group = QUndoGroup(self)
        self._tabs = QTabWidget()
        self._tabs.setObjectName('documentTabs')
        self._tabs.setDocumentMode(True)
        self._tabs.setTabsClosable(True)
        self._tabs.setMovable(True)
        self._tabs.currentChanged.connect(self._on_document_changed)
        self._tabs.tabCloseRequested.connect(self.close_document)
        self._tabs.tabBar().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tabs.tabBar().customContextMenuRequested.connect(self._on_tab_menu_requested)
        # NOTE: Cycling is a window shortcut rather than the tab widget's own, which only answers while the
        # canvas has the focus.
        QShortcut(QKeySequence('Ctrl+Tab'), self, activated=lambda: self._cycle_document(1))
        QShortcut(QKeySequence('Ctrl+Shift+Tab'), self, activated=lambda: self._cycle_document(-1))
        self._start_view = StartView()
        self._start_view.model_requested.connect(self.new_graph)
        self._start_view.open_requested.connect(self.load_session)
        self._start_view.recent_requested.connect(self.open_path)
        self._stack = QStackedWidget()
        self._stack.addWidget(self._start_view)
        self._stack.addWidget(self._tabs)
        self.setCentralWidget(self._stack)
        self._statusBar = QStatusBar()
        self._statusBar.setObjectName('statusBar')
        self.setStatusBar(self._statusBar)
        self._statusBar.showMessage('Ready')
        # Permanent indicator of the controller being built.
        self._controller_icon = QLabel()
        self._controller_icon.setObjectName('statusControllerIcon')
        self._statusBar.addPermanentWidget(self._controller_icon)
        self._controller_label = QLabel('')
        self._controller_label.setObjectName('statusControllerLabel')
        self._statusBar.addPermanentWidget(self._controller_label)
        self._setup_docks()
        self.dock_hierarchy.widget().node_double_clicked.connect(self._center_on_node)
        # Menus, set up after the docks so they can be toggled.
        self._setup_menus()
        self._update_document_state()
        self._register_library()
        logger.info('Editor Initialized...')


    #-------------------------------------------------------------------------------------------------------#
    # Documents
    #-------------------------------------------------------------------------------------------------------#

    # The window works on "the current document". Its parts are exposed as properties, so that every action
    # (saving, exporting, importing, the inspector) is written against a single graph.

    @property
    def document(self) -> EditorDocument | None:
        index = self._tabs.currentIndex()
        return self._documents[index] if 0 <= index < len(self._documents) else None

    @property
    def _scene(self) -> GraphScene:
        document = self.document
        return document.scene if document is not None else self._empty_scene

    @property
    def view(self) -> GraphView:
        document = self.document
        return document.view if document is not None else self._empty_scene.views()[0] if self._empty_scene.views() else None

    @property
    def _session_path(self) -> pathlib.Path | None:
        document = self.document
        return document.session_path if document is not None else None

    @_session_path.setter
    def _session_path(self, path: pathlib.Path | None) -> None:
        document = self.document
        if document is not None:
            document.session_path = path

    @property
    def _model_path(self) -> pathlib.Path | None:
        document = self.document
        return document.model_path if document is not None else None

    @_model_path.setter
    def _model_path(self, path: pathlib.Path | None) -> None:
        document = self.document
        if document is not None:
            document.model_path = path

    def add_document(self, model: GraphModel | None = None) -> EditorDocument:
        """
            Opens a new tab and makes it current.
        """
        scene = GraphScene(model)
        document = EditorDocument(scene=scene, view=GraphView(scene))
        self._documents.append(document)
        self._connect_document(document)
        index = self._tabs.addTab(document.view, document.title)
        self._tabs.setCurrentIndex(index)
        self._refresh_tab()
        return document

    def _connect_document(self, document: EditorDocument) -> None:
        document.scene.selectionChanged.connect(self._on_selection_changed)
        document.model.profile_changed.connect(self._on_profile_changed)
        # The tab label carries the modified marker.
        document.model.undo_stack.cleanChanged.connect(lambda _clean: self._refresh_tab(document))
        self._undo_group.addStack(document.model.undo_stack)

    def close_document(self, index: int) -> bool:
        """
            Closes a tab, offering to save it first when it holds unsaved work.
        """
        if not (0 <= index < len(self._documents)):
            return False
        document = self._documents[index]
        if document.is_modified:
            answer = QMessageBox.question(
                self, 'Close Session',
                f'"{document.name}" has unsaved changes.',
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if answer == QMessageBox.StandardButton.Cancel:
                return False
            if answer == QMessageBox.StandardButton.Save:
                self._tabs.setCurrentIndex(index)
                if not self.save_session():
                    return False
        try:
            document.model.undo_stack.cleanChanged.disconnect()
        except (RuntimeError, TypeError):
            pass
        self._undo_group.removeStack(document.model.undo_stack)
        self._documents.pop(index)
        self._tabs.removeTab(index)
        document.view.deleteLater()
        document.scene.deleteLater()
        self._update_document_state()
        self._refresh_tab()
        return True

    def _refresh_tab(self, document: EditorDocument | None = None) -> None:
        """
            Relabels every tab. The label of one depends on the others, so they are done together.
        """
        # NOTE: Tabs and stacks still answer while the window is torn down, so the C++ side is checked.
        if not isValid(self._tabs):
            return
        taken: dict[str, set[int]] = {}
        for index, open_document in enumerate(self._documents):
            if not isValid(open_document.model.undo_stack):
                continue
            name = open_document.name
            numbers = taken.setdefault(name, set())
            # A document keeps the number it was given, unless it was renamed or the number is not free.
            if open_document.copy_name != name or open_document.copy_index < 1 or open_document.copy_index in numbers:
                number = 1
                while number in numbers:
                    number += 1
                open_document.copy_name = name
                open_document.copy_index = number
            numbers.add(open_document.copy_index)
            self._tabs.setTabText(index, open_document.title)
            self._tabs.setTabToolTip(
                index,
                str(open_document.session_path) if open_document.session_path else 'Not saved yet',
            )
        self._update_window_title()

    def _cycle_document(self, step: int) -> None:
        """
            Moves to the next or previous tab, wrapping around.
        """
        count = self._tabs.count()
        if count > 1:
            self._tabs.setCurrentIndex((self._tabs.currentIndex() + step) % count)

    def _on_tab_menu_requested(self, position) -> None:
        """
            Menu of the tab under the cursor.
        """
        index = self._tabs.tabBar().tabAt(position)
        if not (0 <= index < len(self._documents)):
            return
        document = self._documents[index]
        menu = QMenu(self)
        close_action = menu.addAction('Close')
        others_action = menu.addAction('Close Others')
        others_action.setEnabled(len(self._documents) > 1)
        menu.addSeparator()
        copy_action = menu.addAction('Copy Path')
        path = document.session_path or document.model_path
        copy_action.setEnabled(path is not None)
        chosen = menu.exec(self._tabs.tabBar().mapToGlobal(position))
        if chosen is close_action:
            self.close_document(index)
        elif chosen is others_action:
            self._close_other_documents(document)
        elif chosen is copy_action and path is not None:
            QApplication.clipboard().setText(str(path))
            self._statusBar.showMessage(f'{path} copied to the clipboard')

    def _close_other_documents(self, keep: EditorDocument) -> None:
        """
            Closes every session but one, stopping wherever the user cancels.
        """
        for document in [entry for entry in self._documents if entry is not keep]:
            if document not in self._documents:
                continue
            if not self.close_document(self._documents.index(document)):
                return

    def _on_document_changed(self, _index: int) -> None:
        """
            Rebinds the panels to the document that just became current.
        """
        document = self.document
        if document is not None:
            self._undo_group.setActiveStack(document.model.undo_stack)
            self.dock_hierarchy.setWidget(HierarchyView(document.model))
            self.dock_hierarchy.widget().node_double_clicked.connect(self._center_on_node)
        self._update_document_state()
        self._refresh_inspector()

    def closeEvent(self, event) -> None:
        # Every open session is offered a chance to save before the window goes away.
        while self._documents:
            if not self.close_document(self._tabs.currentIndex() if self._tabs.currentIndex() >= 0 else 0):
                event.ignore()
                return
        super().closeEvent(event)
        self.windowClosed.emit()

    def _center_on_node(self, node_model) -> None:
        if not node_model: return
        for item in self._scene.items():
            if isinstance(item, NodeItem) and item.model == node_model:
                self.view.centerOn(item)
                break

    def _refresh_inspector(self) -> None:
        """
            Rebuilds the inspector even when the selection did not move.

            Choosing a controller, or adopting settings from a file, changes what "nothing selected" shows.
            """
        self.dock_inspector.widget().invalidate()
        self._on_selection_changed()

    def _on_selection_changed(self) -> None:
        if not isValid(self._scene): 
            return
        selected_nodes = [item.model for item in self._scene.selectedItems() if isinstance(item, NodeItem)]
        if len(selected_nodes) == 1:
            self.dock_inspector.widget().set_node(selected_nodes[0], self._scene.model)
        else:
            self.dock_inspector.widget().set_node(None, self._scene.model)

    def _create_dock_title(self, title) -> QLabel:
        label = QLabel(f' {title.upper()}')
        label.setObjectName('dockTitle')
        return label

    def _setup_docks(self) -> None:
        # The corners let the left and right docks extend to the bottom, around the console.
        self.setCorner(Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea)
        # Left: Hierarchy
        self.dock_hierarchy = QDockWidget('Hierarchy', self)
        self.dock_hierarchy.setTitleBarWidget(self._create_dock_title('Hierarchy'))
        self.dock_hierarchy.setWidget(HierarchyView(self._scene.model))
        self.dock_hierarchy.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_hierarchy)
        # Right: Inspector
        self.dock_inspector = QDockWidget('Inspector', self)
        self.dock_inspector.setTitleBarWidget(self._create_dock_title('Inspector'))
        self.dock_inspector.setWidget(InspectorView())
        self.dock_inspector.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_inspector)
        # Bottom: Console
        self.dock_console = QDockWidget('Console', self)
        self.dock_console.setTitleBarWidget(self._create_dock_title('Console'))
        self.dock_console.setWidget(ConsoleView())
        self.dock_console.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dock_console)

    def _setup_menus(self) -> None:
        menubar = self.menuBar()
        # NOTE: Menus are kept as attributes to stay addressable from code without going through the menu
        # bar. Reaching them through temporaries invalidates the PySide wrapper.
        # File Menu
        self._file_menu = file_menu = menubar.addMenu('&File')
        new_action = QAction('New Session', self)
        new_action.setShortcut('Ctrl+N')
        # NOTE: QAction.triggered carries the checked state, which must not be mistaken for a profile.
        new_action.triggered.connect(lambda _checked=False: self.new_graph())
        file_menu.addAction(new_action)
        load_action = QAction('Load Session...', self)
        load_action.setShortcut('Ctrl+O')
        load_action.triggered.connect(lambda _checked=False: self.load_session())
        file_menu.addAction(load_action)
        self._recent_menu = file_menu.addMenu('Open Recent')
        # Rebuilt every time it is shown: the list changes from anywhere in the window.
        self._recent_menu.aboutToShow.connect(self._refresh_recent_menu)
        save_action = QAction('Save Session', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(lambda _checked=False: self.save_session())
        file_menu.addAction(save_action)
        save_as_action = QAction('Save Session As...', self)
        save_as_action.setShortcut('Ctrl+Shift+S')
        save_as_action.triggered.connect(lambda _checked=False: self.save_session_as())
        file_menu.addAction(save_as_action)
        file_menu.addSeparator()
        # A session is work in progress and always saves. A model is a finished controller and only exports
        # once the graph describes something the framework can instantiate.
        export_action = QAction('Export Model', self)
        export_action.triggered.connect(lambda _checked=False: self.export_model())
        file_menu.addAction(export_action)
        export_as_action = QAction('Export Model As...', self)
        export_as_action.triggered.connect(lambda _checked=False: self.export_model_as())
        file_menu.addAction(export_as_action)
        import_action = QAction('Import Model...', self)
        import_action.triggered.connect(lambda _checked=False: self.import_model_file())
        file_menu.addAction(import_action)
        library_action = QAction('Add Model to Library...', self)
        library_action.triggered.connect(lambda _checked=False: self.add_model_to_library())
        file_menu.addAction(library_action)
        check_action = QAction('Check Model', self)
        check_action.setShortcut('F7')
        check_action.triggered.connect(lambda _checked=False: self.check_model())
        file_menu.addAction(check_action)
        file_menu.addSeparator()
        close_action = QAction('Close Session', self)
        close_action.setShortcut('Ctrl+W')
        close_action.triggered.connect(lambda _checked=False: self.close_document(self._tabs.currentIndex()))
        file_menu.addAction(close_action)
        file_menu.addSeparator()
        # Actions that require an open document.
        self._document_actions = [
            save_action, save_as_action, export_action, export_as_action, import_action, check_action,
            close_action,
        ]
        quit_action = QAction('Quit', self)
        quit_action.setShortcut('Ctrl+Q')
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        # Edit Menu
        self._edit_menu = edit_menu = menubar.addMenu('&Edit')
        # The actions come from the group, so they drive the stack of the current document.
        undo_action = self._undo_group.createUndoAction(self, '&Undo')
        undo_action.setShortcut('Ctrl+Z')
        edit_menu.addAction(undo_action)
        redo_action = self._undo_group.createRedoAction(self, '&Redo')
        redo_action.setShortcut('Ctrl+Shift+Z')
        edit_menu.addAction(redo_action)
        edit_menu.addSeparator()
        prefs_action = QAction('Preferences...', self)
        prefs_action.triggered.connect(self.open_preferences)
        edit_menu.addAction(prefs_action)
        # Window Menu
        self._window_menu = window_menu = menubar.addMenu('&Window')
        hierarchy_action = self.dock_hierarchy.toggleViewAction()
        hierarchy_action.setText('Hierarchy')
        window_menu.addAction(hierarchy_action)
        inspector_action = self.dock_inspector.toggleViewAction()
        inspector_action.setText('Inspector')
        window_menu.addAction(inspector_action)
        console_action = self.dock_console.toggleViewAction()
        console_action.setText('Console')
        window_menu.addAction(console_action)

    def new_graph(self, profile: ControllerProfile | None = None) -> None:
        """
            Starts a new model. The controller is asked for whenever it was not provided.
        """
        # Guards against a signal carrying an unrelated payload (e.g. QAction.triggered(bool)).
        if not isinstance(profile, ControllerProfile):
            profile = None
        if profile is None:
            dialog = NewModelDialog(self)
            if not dialog.exec():
                return
            profile = dialog.selected_profile
        if not isinstance(profile, ControllerProfile):
            return
        # A new session is always a new tab, the one being edited is never replaced.
        document = self.add_document()
        document.model.clear()
        document.model.set_profile(profile, force=True)
        document.session_path = None
        document.model_path = None
        document.model.undo_stack.clear()
        document.model.undo_stack.setClean()
        self._refresh_tab(document)
        self._update_document_state()
        self._statusBar.showMessage(f'New {profile.label.lower()} session started.')
        logging.getLogger('spark').info(f'Started a new empty {profile.label} session.')

    def _on_profile_changed(self, profile: ControllerProfile | None) -> None:
        self._update_document_state()
        # The controller settings are what the inspector shows while nothing is selected.
        self._refresh_inspector()

    def _reusable_document(self) -> EditorDocument | None:
        """
            The current tab when nothing has been done to it yet.

            Opening a file fills the session being edited while that session is still empty, and opens another
            tab once there is something to preserve. Picking a controller does not count as work, since the file
            brings its own. Anything on the canvas, a file name, or an edit that reached the undo stack does.
            """
        document = self.document
        if document is None:
            return None
        untouched = (
            not document.model.nodes
            and document.session_path is None
            and document.model_path is None
            and not document.is_modified
        )
        return document if untouched else None

    def _update_window_title(self) -> None:
        document = self.document
        profile = document.model.profile if document is not None else None
        if document is not None and profile is not None:
            self.setWindowTitle(f'Spark Graph Editor - {profile.label} - {document.title}')
        else:
            self.setWindowTitle('Spark Graph Editor')

    def _update_document_state(self) -> None:
        """
            Synchronizes the window with the presence (and type) of a document.
        """
        profile = self._scene.model.profile
        has_document = profile is not None and self.document is not None
        if not has_document:
            self._start_view.set_recent_files(recent_files.recent_files())
        self._stack.setCurrentWidget(self._tabs if has_document else self._start_view)
        if has_document:
            self._refresh_tab(self.document)
        self._update_window_title()
        if has_document:
            icon_size = STYLES.get_val('start', 'status_icon_size', default=14)
            self._controller_icon.setPixmap(icons.get_pixmap(profile.icon, icon_size))
            self._controller_label.setText(profile.label)
        else:
            self._controller_icon.clear()
            self._controller_label.setText('')
        for action in getattr(self, '_document_actions', []):
            action.setEnabled(has_document)
        if hasattr(self, '_controller_action'):
            self._controller_action.setEnabled(has_document and self._scene.model.can_change_profile())

    #-------------------------------------------------------------------------------------------------------#
    # Sessions and models
    #-------------------------------------------------------------------------------------------------------#

    # A session (.sge) is the document being edited: it always saves, however incomplete it is, and it
    # remembers where the nodes are. A model (.scfg) is the finished controller handed to the framework,
    # and only exports when the graph describes something that can be instantiated.

    def save_session(self) -> bool:
        if self._scene.model.profile is None:
            return False
        if self._session_path is None:
            return self.save_session_as()
        return self._write_session(self._session_path)

    def save_session_as(self) -> bool:
        if self._scene.model.profile is None:
            return False
        # Passing None as parent detaches the dialog from the Qt style tree of the main window, which
        # forces the native dialog of the OS.
        file_name, _ = QFileDialog.getSaveFileName(None, 'Save Session As', '', session_io.SESSION_FILTER)
        if not file_name:
            return False
        return self._write_session(pathlib.Path(file_name))

    def _write_session(self, path: pathlib.Path) -> bool:
        try:
            written = session_io.save_session(self._scene.model, path)
        except Exception as error:
            self._report_error('Unable to save the session', str(error))
            return False
        self._session_path = written
        recent_files.remember(written)
        self._scene.model.undo_stack.setClean()
        self._update_document_state()
        self._statusBar.showMessage(f'Session saved to {written}')
        logging.getLogger('spark').log(MessageLevel.SUCCESS.value, f'Session saved to "{written}".')
        return True

    def load_session(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(None, 'Load Session', '', session_io.SESSION_FILTER)
        if not file_name:
            return
        self.load_session_file(file_name)

    def load_session_file(self, file_name: str | pathlib.Path) -> bool:
        """
            Opens a session from a path, in the session being edited when it is still empty.
        """
        file_name = str(file_name)
        try:
            session = session_io.load_session(file_name)
        except Exception as error:
            self._report_error('Unable to load the session', str(error))
            return False
        if session.profile is None:
            self._report_error('Unable to load the session', 'The controller of this session is not registered.')
            return False
        # A session is opened next to the ones already being edited, never on top of them.
        document = self._reusable_document() or self.add_document()
        model = document.model
        model.clear()
        # The controller comes from the file, the user is never asked when loading.
        model.set_profile(session.profile, force=True)
        model.adopt_controller_config(session.config)
        document.view.import_config(session.config, label=pathlib.Path(file_name).stem, layout=session.layout)
        model.undo_stack.clear()
        model.undo_stack.setClean()
        document.session_path = pathlib.Path(file_name)
        document.model_path = None
        self._refresh_tab(document)
        self._update_document_state()
        self._refresh_inspector()
        recent_files.remember(file_name)
        self._statusBar.showMessage(f'Session loaded from {file_name}')
        logging.getLogger('spark').log(MessageLevel.SUCCESS.value, f'Session loaded from "{file_name}".')
        return True

    def open_path(self, path: str | pathlib.Path) -> bool:
        """
            Opens a file of either kind, telling them apart by their suffix.

            A path that is gone is dropped from the recent list rather than reported as an error.
            """
        path = pathlib.Path(path)
        if not path.is_file():
            recent_files.forget(path)
            self._refresh_recent()
            message = f'"{path}" is no longer there, it was dropped from the recent files.'
            self._statusBar.showMessage(message)
            logging.getLogger('spark').warning(message)
            return False
        if path.suffix == session_io.MODEL_SUFFIX:
            return self.open_model_file(path)
        return self.load_session_file(path)

    def _refresh_recent(self) -> None:
        """
            Brings the recent list up to date wherever it is shown, the menu and the start screen.
            """
        self._refresh_recent_menu()
        self._start_view.set_recent_files(recent_files.recent_files())

    def _refresh_recent_menu(self) -> None:
        """
            Fills the "Open Recent" menu with the files that are still there.
        """
        self._recent_menu.clear()
        paths = recent_files.recent_files()
        if not paths:
            empty_action = QAction('No Recent Files', self)
            empty_action.setEnabled(False)
            self._recent_menu.addAction(empty_action)
            return
        for path in paths:
            # The kind of file is shown: one resumes work, the other starts a session from a finished model.
            kind = 'model' if path.suffix == session_io.MODEL_SUFFIX else 'session'
            action = QAction(f'{path.stem}  ({kind})', self)
            action.setToolTip(str(path))
            action.triggered.connect(lambda _checked=False, target=path: self.open_path(target))
            self._recent_menu.addAction(action)
        self._recent_menu.addSeparator()
        clear_action = QAction('Clear List', self)
        clear_action.triggered.connect(lambda _checked=False: self._clear_recent())
        self._recent_menu.addAction(clear_action)

    def _clear_recent(self) -> None:
        recent_files.clear()
        self._refresh_recent()

    def check_model(self) -> bool:
        """
            Reports what the graph still needs before it can be exported, writing nothing.
        """
        model = self._scene.model
        if model.profile is None:
            return False
        try:
            problems = session_io.check_model(model)
        except Exception as error:
            self._report_error('Unable to check the model', str(error))
            return False
        if problems:
            for problem in problems:
                logging.getLogger('spark').warning(problem)
            self._report_error(
                'This model cannot be exported yet',
                'The following must be resolved first:\n\n  \u2022  ' + '\n  \u2022  '.join(problems),
            )
            return False
        message = f'This {model.profile.label.lower()} is complete and ready to export.'
        self._statusBar.showMessage(message)
        logging.getLogger('spark').log(MessageLevel.SUCCESS.value, message)
        QMessageBox.information(self, 'Check Model', message)
        return True

    def open_model_file(self, path: str | pathlib.Path) -> bool:
        """
            Opens a model as a session of its own.

            Adding the modules to the session being edited is what "Import Model..." does instead.
            """
        path = pathlib.Path(path)
        try:
            config = session_io.load_model(path)
        except Exception as error:
            self._report_error('Unable to read the model', f'{path.name}: {error}')
            return False
        source_profile = profile_for_config(config)
        if source_profile is None:
            self._report_error('Unable to open the model', f'The controller of "{path.name}" is not registered.')
            return False
        self._open_model_as_session(config, path, source_profile, layout=session_io.model_layout(path))
        return True

    def export_model(self) -> bool:
        if self._scene.model.profile is None:
            return False
        if self._model_path is None:
            return self.export_model_as()
        return self._write_model(self._model_path)

    def export_model_as(self) -> bool:
        if self._scene.model.profile is None:
            return False
        file_name, _ = QFileDialog.getSaveFileName(None, 'Export Model As', '', session_io.MODEL_FILTER)
        if not file_name:
            return False
        return self._write_model(pathlib.Path(file_name))

    def _write_model(self, path: pathlib.Path) -> bool:
        try:
            written = session_io.export_model(self._scene.model, path)
        except ValueError as error:
            # An incomplete graph is reported rather than raised.
            problems = str(error).splitlines()
            self._report_error(
                'This model cannot be exported yet',
                'The following must be resolved first:\n\n  \u2022  ' + '\n  \u2022  '.join(problems),
            )
            return False
        except Exception as error:
            self._report_error('Unable to export the model', str(error))
            return False
        self._model_path = written
        recent_files.remember(written)
        self._update_document_state()
        self._statusBar.showMessage(f'Model exported to {written}')
        logging.getLogger('spark').log(MessageLevel.SUCCESS.value, f'Model exported to "{written}".')
        return True

    def _report_error(self, title: str, message: str) -> None:
        self._statusBar.showMessage(message.splitlines()[0])
        logging.getLogger('spark').error(f'{title}: {message}')
        QMessageBox.warning(self, title, message)

    def import_model_file(self) -> None:
        """
            Imports a model saved as a Spark configuration.

            A model can be opened as a session of its own, or its modules can be added to the session being
            edited. Importing does the second, and never adds a second controller.
            """
        model = self._scene.model
        if model.profile is None:
            return
        file_name, _ = QFileDialog.getOpenFileName(None, 'Import Model', '', session_io.MODEL_FILTER)
        if not file_name:
            return
        path = pathlib.Path(file_name)
        try:
            config = session_io.load_model(path)
        except Exception as error:
            self._report_error('Unable to read the model', f'{path.name}: {error}')
            return
        source_profile = profile_for_config(config)
        if source_profile is None:
            self._report_error('Unable to import the model', f'The controller of "{path.name}" is not registered.')
            return
        layout = session_io.model_layout(path)
        same_controller = source_profile is model.profile
        # A model of another controller can still belong on the canvas: a Brain hosts Neurons, placed as a
        # single node rather than as the modules they are made of.
        hosted = model.profile.hosts(source_profile)
        if same_controller and not model.nodes:
            self._open_model_as_session(config, path, source_profile, layout=layout)
            return
        choice = self._ask_import_mode(
            path, source_profile, allow_merge=same_controller, allow_node=hosted,
        )
        if choice == 'new':
            self._open_model_as_session(config, path, source_profile, layout=layout)
        elif choice == 'merge':
            self.view.import_config(config, label=path.stem, layout=layout)
        elif choice == 'node':
            self._add_model_as_node(config, path, source_profile)

    def _ask_import_mode(
            self,
            path: pathlib.Path,
            source_profile: ControllerProfile,
            allow_merge: bool,
            allow_node: bool = False,
        ) -> str | None:
        """
            Asks how a model should join the session.
        """
        profile = self._scene.model.profile
        box = QMessageBox(self)
        box.setWindowTitle('Import Model')
        box.setText(f'"{path.name}" describes a {source_profile.label}.')
        if allow_node:
            box.setInformativeText(
                f'Open it as a new session, or add it to the {profile.label} being edited as a single node?'
            )
        elif allow_merge:
            box.setInformativeText(
                'Open it as a new session, or add its modules to the one being edited?'
            )
        else:
            box.setInformativeText(
                f'The session being edited builds a {profile.label}, which does not hold a '
                f'{source_profile.label}. It can be opened as a new session instead.'
            )
        new_button = box.addButton('New Session', QMessageBox.ButtonRole.AcceptRole)
        node_button = box.addButton('Add as Node', QMessageBox.ButtonRole.ActionRole) if allow_node else None
        merge_button = box.addButton('Add to Session', QMessageBox.ButtonRole.ActionRole) if allow_merge else None
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is new_button:
            return 'new'
        if node_button is not None and clicked is node_button:
            return 'node'
        if merge_button is not None and clicked is merge_button:
            return 'merge'
        return None

    def _add_model_as_node(self, config, path: pathlib.Path, source_profile: ControllerProfile) -> None:
        """
            Places a model on the canvas as a single node.

            A node needs a registered class, so the model is registered under the name of its file. A model
            already registered under that name is reused rather than replaced.
            """
        from spark.core.registry import REGISTRY, register_neuron_from_config

        namespace = source_profile.model_namespace
        if namespace is None:
            self._report_error(
                'Unable to import the model', f'A {source_profile.label} cannot be placed as a node.',
            )
            return
        subregistry = getattr(REGISTRY, namespace.name)
        name = path.stem
        entry = subregistry.get(name)
        reused = entry is not None
        if not reused:
            try:
                register_neuron_from_config(name, config)
            except Exception as error:
                self._report_error('Unable to import the model', f'{path.name}: {error}')
                return
            entry = subregistry.get(name)
        if entry is None:
            self._report_error('Unable to import the model', f'"{name}" could not be registered.')
            return
        try:
            self.view.add_node_for(entry.get_cls(), label=f'Import {path.stem}')
        except Exception as error:
            self._report_error('Unable to import the model', f'{path.name}: {error}')
            return
        self._update_document_state()
        self._refresh_inspector()
        note = 'already registered, the available model was used' if reused else 'registered'
        self._statusBar.showMessage(f'"{name}" added as a node ({note}).', 4000)
        logging.getLogger('spark').log(
            MessageLevel.SUCCESS.value, f'Model "{name}" added to the session as a node.',
        )

    def _open_model_as_session(
            self,
            config,
            path: pathlib.Path,
            source_profile: ControllerProfile,
            layout: dict[str, tuple[float, float]] | None = None,
        ) -> None:
        """
            Replaces the document with a model, as if the file had been opened.
        """
        document = self._reusable_document() or self.add_document()
        model = document.model
        model.clear()
        # The controller comes from the file, the user is never asked for it.
        model.set_profile(source_profile, force=True)
        model.adopt_controller_config(config)
        document.view.import_config(config, label=path.stem, layout=layout)
        model.undo_stack.clear()
        model.undo_stack.setClean()
        document.session_path = None
        document.model_path = path
        recent_files.remember(path)
        self._refresh_tab(document)
        self._update_document_state()
        self._refresh_inspector()
        self._statusBar.showMessage(f'Model opened from {path}')
        logging.getLogger('spark').log(MessageLevel.SUCCESS.value, f'Model opened from "{path}".')

    def _register_library(self) -> None:
        """
            Makes the models of the library available.
        """
        registered, failed = model_library.register_library()
        if registered:
            logger.info(f'Model library: {", ".join(sorted(registered))}.')
        for path, error in failed:
            logger.warning(f'Model library: unable to read "{path.name}". {error}')

    def add_model_to_library(self) -> None:
        """
            Takes a model file into the library, making it available to every model built from now on.
        """
        file_name, _ = QFileDialog.getOpenFileName(
            None, 'Add Model to Library', '', session_io.MODEL_FILTER,
        )
        if not file_name:
            return
        try:
            destination = model_library.import_model(file_name)
        except FileExistsError:
            answer = QMessageBox.question(
                self,
                'Add Model to Library',
                f'The library already holds a model named "{model_library.model_name(file_name)}".\n'
                f'Replace it?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            try:
                destination = model_library.import_model(file_name, overwrite=True)
            except Exception as error:
                QMessageBox.warning(self, 'Add Model to Library', f'Unable to take the model in.\n\n{error}')
                return
        except Exception as error:
            QMessageBox.warning(self, 'Add Model to Library', f'Unable to take the model in.\n\n{error}')
            return
        logger.info(f'Model "{model_library.model_name(destination)}" added to the library.')
        self._statusBar.showMessage(f'Added "{destination.name}" to the model library.', 4000)

    def open_preferences(self) -> None:
        dialog = PreferencesDialog(self)
        # Applying without closing rebuilds the editor, so the effect is visible while adjusting.
        dialog.applied.connect(self.reload_ui)
        dialog.exec()

    def reload_ui(self) -> None:
        """
            Rebuilds the widgets after a style change, for every open document.
        """
        current = self._tabs.currentIndex()
        for index, document in enumerate(self._documents):
            selected = [item.model for item in document.scene.selectedItems() if isinstance(item, NodeItem)]
            old_scene, old_view = document.scene, document.view
            document.scene = GraphScene(document.model)
            document.view = GraphView(document.scene)
            document.scene.selectionChanged.connect(self._on_selection_changed)
            # The tab holds the view, so it is replaced in place to keep the order and the labels.
            self._tabs.removeTab(index)
            self._tabs.insertTab(index, document.view, document.title)
            old_view.deleteLater()
            old_scene.deleteLater()
            for item in document.scene.items():
                if isinstance(item, NodeItem) and item.model in selected:
                    item.setSelected(True)
        if 0 <= current < self._tabs.count():
            self._tabs.setCurrentIndex(current)
        # Recreate the panels, which also carry style.
        document = self.document
        self.dock_hierarchy.setWidget(HierarchyView(document.model if document else self._empty_scene.model))
        self.dock_inspector.setWidget(InspectorView())
        self.dock_console.setWidget(ConsoleView())
        self.dock_hierarchy.widget().node_double_clicked.connect(self._center_on_node)
        self._on_selection_changed()
        self._update_document_state()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################