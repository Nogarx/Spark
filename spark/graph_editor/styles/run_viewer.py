#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp

import types

import numpy as np
from PySide6.QtGui import QColor, QFont, QImage

from spark.graph_editor.styles.manager import STYLES

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

GROUPS = {'timeline': 'run_viewer_timeline', 'badge': 'run_viewer_badge', 'layout': 'run_viewer_layout'}
"""
    Categories of the style read as groups of the theme, by the name of the group. The category
    ``run_viewer`` is read as the theme itself, and ``run_viewer_status`` by the stylesheet only.
"""

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _is_color(key: str, value: tp.Any) -> bool:
    """
        True for a colour written as ``'#rrggbb'``, or as 3 or 4 integers under a key ending in
        ``_color``.
    """
    if isinstance(value, str):
        return value.startswith('#')
    return (key.endswith('_color') and isinstance(value, (list, tuple)) and len(value) in (3, 4)
            and all(isinstance(x, int) for x in value))

def _color(value: tp.Any, fallback: tp.Any = '#000000') -> QColor:
    """
        Returns a colour written as ``'#rrggbb'`` or as 3 or 4 integers, or ``fallback`` when it is
        not one.
    """
    try:
        color = QColor(*value) if isinstance(value, (list, tuple)) else QColor(value)
    except (TypeError, ValueError):
        color = QColor()
    return color if color.isValid() else _color(fallback)

def _read(category: str) -> dict[str, tp.Any]:
    """
        Returns the values of a category of the style.

        Colours are QColor, named without their ``_color`` suffix, and a colour that is not one
        takes its default. Groups of colours, such as a palette, are lists of QColor in their order.
        Other lists are tuples.
    """
    values = STYLES.get_val(category, default={}) or {}
    defaults = STYLES.defaults().get(category, {})
    read = {}
    for key, value in values.items():
        default = defaults.get(key)
        if isinstance(value, dict):
            fallbacks = list(default.values()) if isinstance(default, dict) else []
            read[key] = [_color(v, fallbacks[i] if i < len(fallbacks) else '#000000') for i, v in enumerate(value.values())]
        elif _is_color(key, value):
            read[key.removesuffix('_color')] = _color(value, default if default is not None else '#000000')
        elif isinstance(value, list):
            read[key] = tuple(value)
        else:
            read[key] = value
    return read

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class RunViewerTheme:
    """
        The style of the run viewer: colours, fonts and sizes of its plots, timeline, badges and
        panels.

        The values of the category ``run_viewer`` of the style are attributes of the theme; those of
        the categories of `GROUPS` are attributes of its groups, such as ``THEME.timeline.row_height``.
        A colour is a QColor named without its ``_color`` suffix (``area_color`` is ``THEME.area``),
        a palette a list of QColor. Read again when the style is reloaded.

        Attributes
        ----------
        colormap_pixels : ndarray
            The colormap of the style over 256 values, as ARGB32 pixels, the lowest first.
        colorbar_image : QImage
            ``colormap_pixels`` as an image one pixel high.
        timeline, badge, layout : SimpleNamespace
            The values of the categories of `GROUPS`.
    """

    def __init__(self) -> None:
        self.read()
        STYLES.reloaded.connect(self.read)

    def read(self) -> None:
        """
            Reads the theme from the style.
        """
        self.__dict__.update(_read('run_viewer'))
        for name, category in GROUPS.items():
            setattr(self, name, types.SimpleNamespace(**_read(category)))
        # The colormap, interpolated between its colours.
        stops = np.array([c.getRgb()[:3] for c in self.colormap or [QColor('#000000'), QColor('#ffffff')]], np.float64)
        at = np.linspace(0, len(stops) - 1, 256)
        lut = np.stack([np.interp(at, np.arange(len(stops)), stops[:, c]) for c in range(3)], axis=1).astype(np.uint32)
        self.colormap_pixels = (0xFF000000 | (lut[:, 0] << 16) | (lut[:, 1] << 8) | lut[:, 2]).astype(np.uint32)
        # The image reads the buffer, which is kept with it.
        self._colorbar = np.ascontiguousarray(self.colormap_pixels.reshape(1, 256))
        self.colorbar_image = QImage(self._colorbar.data, 256, 1, 256 * 4, QImage.Format.Format_ARGB32)

    def font(self, size: int | None = None, bold: bool = False) -> QFont:
        """
            Returns the font of the run viewer, at ``size`` points or its own size.
        """
        font = QFont(self.font_family, size or self.font_size)
        font.setBold(bold)
        return font

    def series_color(self, index: int) -> QColor:
        """
            Returns the colour of the series ``index``, the series colours repeated past the last.
        """
        return QColor(self.series_colors[index % len(self.series_colors)])

    @staticmethod
    def with_alpha(color: QColor | str, alpha: int) -> QColor:
        """
            Returns ``color`` with the opacity ``alpha``, from 0 to 255.
        """
        color = QColor(color)
        color.setAlpha(alpha)
        return color

THEME = RunViewerTheme()
"""
    The style of the run viewer.
"""

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
