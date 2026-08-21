#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

from PySide6.QtGui import QPixmap, QIcon, QColor, QPainter
from PySide6.QtCore import Qt, QSize

# NOTE: Importing the compiled resource module registers the ":/icons/*" paths with Qt. Pixmaps themselves are
# only built on demand, since a QGuiApplication must exist before any of them can be created.
import spark.graph_editor.styles.resources_rc  # noqa: F401

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

BRAIN = ':/icons/brain_icon.png'
NEURON = ':/icons/neuron_icon.png'
SIMPLE = ':/icons/simple_icon.png'
COMPLEX = ':/icons/complex_icon.png'
LINK = ':/icons/link_icon.png'
LOCK = ':/icons/lock_icon.png'
NODE = ':/icons/node_icon.png'
DOT = ':/icons/dot_icon.png'

_PIXMAP_CACHE: dict[tuple[str, int], QPixmap] = {}

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def get_pixmap(path: str, size: int | None = None) -> QPixmap:
    """
        Returns a (cached) pixmap from the editor resources.

        Input:
            path: str, resource path (e.g. ":/icons/brain_icon.png").
            size: int, optional square size the pixmap is scaled to.

        Returns:
            QPixmap, the requested pixmap. Empty if the resource does not exist.
    """
    key = (path, size or 0)
    pixmap = _PIXMAP_CACHE.get(key, None)
    if pixmap is not None:
        return pixmap
    pixmap = QPixmap(path)
    if pixmap.isNull():
        pixmap = empty_pixmap(size or 16)
    elif size:
        pixmap = pixmap.scaled(
            QSize(size, size),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    _PIXMAP_CACHE[key] = pixmap
    return pixmap

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def empty_pixmap(size: int = 16) -> QPixmap:
    """
        Fully transparent pixmap, used to keep icon slots aligned while empty.
    """
    key = ('__empty__', size)
    pixmap = _PIXMAP_CACHE.get(key, None)
    if pixmap is None:
        pixmap = QPixmap(QSize(size, size))
        pixmap.fill(QColor(0, 0, 0, 0))
        _PIXMAP_CACHE[key] = pixmap
    return pixmap

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def get_faded_pixmap(path: str, size: int | None = None, opacity: float = 0.35) -> QPixmap:
    """
        Returns a dimmed copy of a resource, used to show an action that is available but not active.
    """
    key = (f'{path}#faded{opacity:.2f}', size or 0)
    pixmap = _PIXMAP_CACHE.get(key, None)
    if pixmap is not None:
        return pixmap
    source = get_pixmap(path, size)
    pixmap = QPixmap(source.size())
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setOpacity(opacity)
    painter.drawPixmap(0, 0, source)
    painter.end()
    _PIXMAP_CACHE[key] = pixmap
    return pixmap

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def get_icon(path: str, size: int | None = None) -> QIcon:
    """
        Returns a QIcon built from the editor resources.
    """
    return QIcon(get_pixmap(path, size))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def get_toggle_icon(
        on_path: str | None,
        off_path: str | None,
        size: int | None = None,
        off_opacity: float = 1.0,
    ) -> QIcon:
    """
        Returns a state aware QIcon that follows the checked state of a button.

        Input:
            on_path: str | None, resource shown while checked. None renders nothing.
            off_path: str | None, resource shown while unchecked. None renders nothing.
            off_opacity: float, dims the unchecked resource. Use it to show that an action is available
                without implying that it is active.
    """
    icon = QIcon()
    on_pixmap = get_pixmap(on_path, size) if on_path else empty_pixmap(size or 16)
    if not off_path:
        off_pixmap = empty_pixmap(size or 16)
    elif off_opacity < 1.0:
        off_pixmap = get_faded_pixmap(off_path, size, off_opacity)
    else:
        off_pixmap = get_pixmap(off_path, size)
    icon.addPixmap(on_pixmap, QIcon.Mode.Normal, QIcon.State.On)
    icon.addPixmap(off_pixmap, QIcon.Mode.Normal, QIcon.State.Off)
    icon.addPixmap(on_pixmap, QIcon.Mode.Disabled, QIcon.State.On)
    icon.addPixmap(off_pixmap, QIcon.Mode.Disabled, QIcon.State.Off)
    return icon

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
