#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import dataclasses as dc
from PySide6 import QtCore
import PySide6QtAds as ads

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@dc.dataclass
class GraphEditorConfig:
    # Color palette
    primary_bg_color: str = '#4A4A4A'
    secondary_bg_color: str = '#404040'
    tertiary_bg_color: str = '#5A5A5A'
    border_color: str = '#2A2A2A'
    hover_color: str = "#444958"
    selected_color: str = "#515E86"
    # Font
    small_font_size: int = 12
    medium_font_size: int = 14
    large_font_size: int = 20
    header_font_size: int = 16
    default_font_color: str = "#E4E4E4"
    # Docks
    inspector_panel_min_width : int = 350 
    nodes_panel_min_width : int = 200 
    dock_min_width : int = 350 
    dock_min_height : int = 150 
    dock_layout_spacing: int = 2
    dock_layout_margins: QtCore.QMargins = dc.field(default_factory = lambda: QtCore.QMargins(0, 4, 0, 4))
    inspector_panel_pos = ads.RightDockWidgetArea
    parameters_panel_pos = ads.LeftDockWidgetArea
    nodes_panel_pos = ads.LeftDockWidgetArea
    console_panel_pos = ads.BottomDockWidgetArea

    # Field configuration
    min_attr_label_size: int = 100


    # ?
    
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


GRAPH_EDITOR_CONFIG = GraphEditorConfig()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################