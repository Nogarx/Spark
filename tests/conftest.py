#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import os
import sys
import typing as tp

DEVICES = ('cpu', 'gpu', 'any')
"""
    Accepted devices: "cpu", "gpu", "any" (jax picks).
"""

def _requested_device() -> str:
    """
        The device this run was asked for.
    """
    for index, argument in enumerate(sys.argv):
        if argument.startswith('--device='):
            return argument.split('=', 1)[1].strip().lower()
        if argument == '--device' and index + 1 < len(sys.argv):
            return sys.argv[index + 1].strip().lower()
    return os.environ.get('SPARK_TEST_DEVICE', 'cpu').strip().lower()

TEST_DEVICE = _requested_device()
if TEST_DEVICE not in DEVICES:
    raise SystemExit(f'Unknown device "{TEST_DEVICE}". Pick one of: {", ".join(DEVICES)}.')
if TEST_DEVICE == 'cpu':
    os.environ['JAX_PLATFORMS'] = 'cpu'
else:
    os.environ.pop('JAX_PLATFORMS', None)

# A window server that never opens a window.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import spark

if tp.TYPE_CHECKING:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QCoreApplication
    from spark.graph_editor.editor import GraphEditorWindow

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def pytest_addoption(parser) -> None:
    """
        Declares "--device" so that it is a legal argument and appears in --help.
    """
    parser.addoption(
        '--device', action='store', default='cpu', choices=DEVICES,
        help='Device the tests run on. Defaults to the processor, which is what continuous integration uses.',
    )

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def pytest_report_header(config) -> str:
    """
        Says what the run is on, so that a green suite here is not mistaken for a green suite elsewhere.
    """
    platforms = ', '.join(sorted({device.platform for device in jax.devices()}))
    return f'spark: asked for "{TEST_DEVICE}", jax reports: {platforms} ({len(jax.devices())} device(s))'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def pytest_configure(config) -> None:
    """
        Refuses a run that asked for a device it did not get.
    """
    if TEST_DEVICE == 'gpu' and all(device.platform == 'cpu' for device in jax.devices()):
        raise pytest.UsageError('"--device=gpu" was asked for, but no GPU is visible to jax.')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@pytest.fixture(scope='session', autouse=True)
def settings_of_their_own(tmp_path_factory) -> tp.Generator[tp.Any, tp.Any, None]:
    """
        Settings the run writes to, so that it never writes to the ones of whoever started it.

        NOTE: The editor remembers the recent files, the model library and the style in QSettings, under a
        fixed organisation and application name. Without a location of its own the suite writes into the
        settings of the user running it, filling their recent files with paths under the temporary directory
        of the run, which are gone by the time they open the editor again.
    """
    pytest.importorskip('PySide6', reason='the graph editor needs PySide6')
    from PySide6.QtCore import QSettings
    root = tmp_path_factory.mktemp('settings')
    # The default format is what the editor asks for, so it is the one that has to be moved. Ini is used
    # everywhere rather than the native format, which setPath cannot move on macOS or Windows.
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
        QSettings.setPath(QSettings.Format.IniFormat, scope, str(root))
    yield root

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@pytest.fixture(scope='session')
def qapp(settings_of_their_own) -> tp.Generator[QCoreApplication | QApplication, tp.Any, None]:
    """
        The one QApplication of the test session.
    """
    PySide6 = pytest.importorskip('PySide6', reason='the graph editor needs PySide6')
    from PySide6.QtWidgets import QApplication
    from spark.graph_editor.styles.manager import STYLES
    app = QApplication.instance() or QApplication([])
    STYLES.init()
    STYLES.apply(app)
    yield app

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@pytest.fixture
def editor(qapp: QCoreApplication | QApplication, monkeypatch: pytest.MonkeyPatch) -> tp.Generator[GraphEditorWindow, tp.Any, None]:
    """
        An editor window, closed and destroyed when the test ends.
    """
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, 'question',
                        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Discard))
    from spark.graph_editor.editor import GraphEditorWindow
    window = GraphEditorWindow()
    window.resize(1200, 700)
    window.show()
    yield window
    while window._documents:
        window.close_document(0)
    window.close()
    window.deleteLater()
    qapp.processEvents()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@pytest.fixture
def answers(monkeypatch: pytest.MonkeyPatch):
    """
        Answers the dialogs of the editor without showing any.

        Returns:
            Answers, a handle to say which file is picked and to read back what was reported.
    """
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    class Answers:
        def __init__(self) -> None:
            self.reported: list[str] = []

        def picks(self, path) -> None:
            """
                The path every file dialog answers with, whether it opens or saves.
            """
            monkeypatch.setattr(QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(path), '')))
            monkeypatch.setattr(QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(path), '')))

    handle = Answers()
    monkeypatch.setattr(QMessageBox, 'question',
                        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Discard))
    monkeypatch.setattr(QMessageBox, 'information', staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, 'warning',
                        staticmethod(lambda *a, **k: handle.reported.append(a[2] if len(a) > 2 else '')))
    return handle

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@pytest.fixture
def spikes() -> tp.Callable:
    """
        A small deterministic burst of spikes.
    """
    rng = np.random.default_rng(42)
    return lambda *shape: spark.SpikeArray(jnp.array(rng.random(shape) < 0.5))

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
