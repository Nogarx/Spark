#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import dataclasses as dc
from Qt import QtCore

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@dc.dataclass
class GraphEditorConfig:
    # Color palette
    primary_bg_color: str = '#4A4A4A'
    secondary_bg_color: str = '#3A3A3A'
    tertiary_bg_color: str = '#2A2A2A'
    border_color: str = '#1A1A1A'
    primary_header_bg_color: str = "#313246"
    # Font
    small_font_size: int = 12
    medium_font_size: int = 16
    large_font_size: int = 20
    header_font_size: int = 16

    min_attr_label_size: int = 150
    input_field_bg_color: str = '#2A2A2A'
    input_field_border_radius: int = 4
    input_field_margin: QtCore.QMargins = dc.field(default_factory = lambda: QtCore.QMargins(4, 4, 4, 4))
    label_field_margin: QtCore.QMargins = dc.field(default_factory = lambda: QtCore.QMargins(4, 0, 0, 0))
    field_bg_color: str = '#3A3A3A'
    field_border_radius: int = 4
    field_margin: QtCore.QMargins = dc.field(default_factory = lambda: QtCore.QMargins(4,4,4,4))
    button_bg_color: str = '#2A2A2A' 
    button_border_radius: int = 4
    section_bg_color: str = '#3A3A3A'
    section_border_radius: int = 4
    section_margin: QtCore.QMargins = dc.field(default_factory = lambda: QtCore.QMargins(4,4,4,4))

    # Docks
    dock_header_bg_color : str = "#313D5E" 
    inspector_panel_pos = QtCore.Qt.DockWidgetArea.RightDockWidgetArea
    parameters_panel_pos = QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
    console_panel_pos = QtCore.Qt.DockWidgetArea.BottomDockWidgetArea

GRAPH_EDITOR_CONFIG = GraphEditorConfig()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################