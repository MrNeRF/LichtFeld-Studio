# Toolbar panels - Data-driven Python implementation
# GizmoToolbar: Horizontal toolbar for transform tools (data-driven from registry)
# UtilityToolbar: Vertical toolbar on left side of viewport

WINDOW_FLAGS = (
    1 << 0 |    # NoTitleBar
    1 << 1 |    # NoResize
    1 << 2 |    # NoMove
    1 << 3 |    # NoScrollbar
    1 << 5 |    # NoCollapse
    1 << 8 |    # NoSavedSettings
    1 << 6 |    # AlwaysAutoResize
    1 << 12 |   # NoFocusOnAppearing
    1 << 13     # NoBringToFrontOnFocus
)


from .types import Panel
from .tools import ToolRegistry
from .tool_defs.definition import ToolDef


def _tool_to_dict(tool: ToolDef) -> dict:
    """Convert ToolDef to dict format for toolbar compatibility."""
    return {
        "id": tool.id,
        "label": tool.label,
        "icon": tool.icon,
        "group": tool.group,
        "shortcut": tool.shortcut,
        "plugin_name": tool.plugin_name,
        "plugin_path": tool.plugin_path,
        "submodes": [
            {"id": s.id, "label": s.label, "icon": s.icon}
            for s in tool.submodes
        ],
        "pivot_modes": [
            {"id": p.id, "label": p.label, "icon": p.icon}
            for p in tool.pivot_modes
        ],
    }


_TOOLBAR_VISIBLE_STATES = ("idle", "ready")


class GizmoToolbar(Panel):
    """Data-driven horizontal gizmo toolbar - queries tool registry for buttons"""
    label = "Gizmo Toolbar"
    space = "VIEWPORT_OVERLAY"
    order = 1

    @classmethod
    def poll(cls, context) -> bool:
        from .ui.state import AppState
        return AppState.trainer_state.value in _TOOLBAR_VISIBLE_STATES

    def _get_tool_icon(self, tool):
        from . import icon_manager
        icon_name = tool.get("icon")
        if not icon_name:
            return 0
        plugin_name = tool.get("plugin_name")
        plugin_path = tool.get("plugin_path")
        if plugin_name and plugin_path:
            return icon_manager.get_plugin_icon(icon_name, plugin_path, plugin_name)
        return icon_manager.get_ui_icon(f"{icon_name}.png")

    def _activate_tool(self, tool_id):
        ToolRegistry.set_active(tool_id)

    def draw(self, layout):
        import lichtfeld as lf
        from .op_context import get_context

        tool_defs = ToolRegistry.get_all()
        if not tool_defs:
            return

        tools = [_tool_to_dict(t) for t in tool_defs]
        context = get_context()

        theme = lf.ui.theme()
        scale = layout.get_dpi_scale()

        btn_size = theme.sizes.toolbar_button_size * scale
        padding = theme.sizes.toolbar_padding * scale
        spacing = theme.sizes.toolbar_spacing * scale

        # Group tools by their group attribute
        groups = {}
        group_order = []
        for tool in tools:
            g = tool.get("group", "default")
            if g not in groups:
                groups[g] = []
                group_order.append(g)
            groups[g].append(tool)

        # Calculate toolbar size
        num_buttons = len(tools)
        width = (num_buttons * btn_size +
                 (num_buttons - 1) * spacing +
                 2 * padding)
        height = btn_size + 2 * padding

        # Position toolbar at top center of viewport
        vp_pos = layout.get_viewport_pos()
        vp_size = layout.get_viewport_size()
        pos_x = vp_pos[0] + (vp_size[0] - width) / 2
        pos_y = vp_pos[1] + 10 * scale

        layout.set_next_window_pos((pos_x, pos_y))
        layout.set_next_window_size((width, height))

        layout.push_style_var("WindowRounding", theme.sizes.window_rounding)
        layout.push_style_var_vec2("WindowPadding", (padding, padding))
        layout.push_style_var_vec2("ItemSpacing", (spacing, 0))
        layout.push_style_var_vec2("FramePadding", (0, 0))
        layout.push_style_color("WindowBg", theme.palette.toolbar_background)

        if layout.begin_window("##GizmoToolbar", WINDOW_FLAGS):
            btn_sz = (btn_size, btn_size)
            active_tool = lf.ui.get_active_tool()

            first_in_toolbar = True
            for group_name in group_order:
                for tool in groups[group_name]:
                    if not first_in_toolbar:
                        layout.same_line()
                    first_in_toolbar = False
                    tool_id = tool["id"]
                    icon = self._get_tool_icon(tool)
                    selected = (active_tool == tool_id)
                    tool_def = ToolRegistry.get(tool_id)
                    enabled = tool_def.can_activate(context) if tool_def else False

                    tooltip = tool.get("label", tool_id)
                    shortcut = tool.get("shortcut")
                    if shortcut:
                        tooltip = f"{tooltip} ({shortcut})"

                    if layout.toolbar_button(f"##{tool_id}", icon, btn_sz,
                                             selected, not enabled, tooltip):
                        if enabled:
                            if selected:
                                ToolRegistry.clear_active()
                            else:
                                self._activate_tool(tool_id)

        layout.end_window()
        layout.pop_style_color(1)
        layout.pop_style_var(4)

        # Draw submodes toolbar if active tool has submodes
        self._draw_submodes_toolbar(layout, scale)

    def _draw_submodes_toolbar(self, layout, scale):
        import lichtfeld as lf
        from . import icon_manager

        active_tool_id = lf.ui.get_active_tool()
        if not active_tool_id:
            return

        tool_def = ToolRegistry.get(active_tool_id)
        if not tool_def:
            return

        if not tool_def.submodes:
            return

        submodes = [
            {"id": s.id, "label": s.label, "icon": s.icon}
            for s in tool_def.submodes
        ]

        theme = lf.ui.theme()
        btn_size = theme.sizes.toolbar_button_size * scale * 0.85
        padding = theme.sizes.toolbar_padding * scale
        spacing = theme.sizes.toolbar_spacing * scale

        num_buttons = len(submodes)
        width = (num_buttons * btn_size +
                 (num_buttons - 1) * spacing +
                 2 * padding)
        height = btn_size + 2 * padding

        vp_pos = layout.get_viewport_pos()
        vp_size = layout.get_viewport_size()
        pos_x = vp_pos[0] + (vp_size[0] - width) / 2
        pos_y = vp_pos[1] + (10 + theme.sizes.toolbar_button_size + 8) * scale

        layout.set_next_window_pos((pos_x, pos_y))
        layout.set_next_window_size((width, height))

        layout.push_style_var("WindowRounding", theme.sizes.window_rounding)
        layout.push_style_var_vec2("WindowPadding", (padding, padding))
        layout.push_style_var_vec2("ItemSpacing", (spacing, 0))
        layout.push_style_var_vec2("FramePadding", (0, 0))
        layout.push_style_color("WindowBg", theme.palette.toolbar_background)

        if layout.begin_window("##SubmodeToolbar", WINDOW_FLAGS):
            btn_sz = (btn_size, btn_size)
            is_mirror_tool = (active_tool_id == "builtin.mirror")
            is_transform_tool = active_tool_id in ("builtin.translate", "builtin.rotate", "builtin.scale")

            if is_transform_tool:
                current_space = lf.ui.get_transform_space()
                SPACE_IDS = {"local": 0, "world": 1}
            else:
                active_submode = lf.ui.get_active_submode()

            first = True
            for mode in submodes:
                if not first:
                    layout.same_line()
                first = False

                mode_id = mode.get("id", "")
                mode_icon = mode.get("icon")
                icon = icon_manager.get_ui_icon(f"{mode_icon}.png") if mode_icon else 0
                tooltip = mode.get("label", mode_id)

                if is_mirror_tool:
                    # Mirror submodes are action buttons, not mode toggles
                    if layout.toolbar_button(f"##sub_{mode_id}", icon, btn_sz,
                                             False, False, tooltip):
                        lf.ui.execute_mirror(mode_id)
                elif is_transform_tool:
                    # Transform space toggles (local/world)
                    space_id = SPACE_IDS.get(mode_id, -1)
                    selected = (current_space == space_id)
                    if layout.toolbar_button(f"##sub_{mode_id}", icon, btn_sz,
                                             selected, False, tooltip):
                        if space_id >= 0:
                            lf.ui.set_transform_space(space_id)
                else:
                    # Selection submodes are mode toggles
                    selected = (active_submode == mode_id)
                    if layout.toolbar_button(f"##sub_{mode_id}", icon, btn_sz,
                                             selected, False, tooltip):
                        lf.ui.set_selection_mode(mode_id)

        layout.end_window()
        layout.pop_style_color(1)
        layout.pop_style_var(4)

        # Also render pivot modes if tool has them
        if tool_def.pivot_modes:
            pivot_modes = [
                {"id": p.id, "label": p.label, "icon": p.icon}
                for p in tool_def.pivot_modes
            ]
            self._draw_pivot_toolbar(layout, scale, pivot_modes)

    def _draw_pivot_toolbar(self, layout, scale, pivot_modes):
        import lichtfeld as lf
        from . import icon_manager

        theme = lf.ui.theme()
        btn_size = theme.sizes.toolbar_button_size * scale * 0.85
        padding = theme.sizes.toolbar_padding * scale
        spacing = theme.sizes.toolbar_spacing * scale

        num_buttons = len(pivot_modes)
        width = (num_buttons * btn_size +
                 (num_buttons - 1) * spacing +
                 2 * padding)
        height = btn_size + 2 * padding

        vp_pos = layout.get_viewport_pos()
        vp_size = layout.get_viewport_size()
        pos_x = vp_pos[0] + (vp_size[0] - width) / 2
        main_toolbar_height = theme.sizes.toolbar_button_size + 8
        submodes_toolbar_height = theme.sizes.toolbar_button_size * 0.85 + 8
        pos_y = vp_pos[1] + (10 + main_toolbar_height + submodes_toolbar_height) * scale

        layout.set_next_window_pos((pos_x, pos_y))
        layout.set_next_window_size((width, height))

        layout.push_style_var("WindowRounding", theme.sizes.window_rounding)
        layout.push_style_var_vec2("WindowPadding", (padding, padding))
        layout.push_style_var_vec2("ItemSpacing", (spacing, 0))
        layout.push_style_var_vec2("FramePadding", (0, 0))
        layout.push_style_color("WindowBg", theme.palette.toolbar_background)

        if layout.begin_window("##PivotToolbar", WINDOW_FLAGS):
            btn_sz = (btn_size, btn_size)
            current_pivot = lf.ui.get_pivot_mode()
            PIVOT_IDS = {"origin": 0, "bounds": 1}

            first = True
            for mode in pivot_modes:
                if not first:
                    layout.same_line()
                first = False

                mode_id = mode.get("id", "")
                mode_icon = mode.get("icon")
                icon = icon_manager.get_ui_icon(f"{mode_icon}.png") if mode_icon else 0
                tooltip = mode.get("label", mode_id)
                pivot_id = PIVOT_IDS.get(mode_id, -1)
                selected = (current_pivot == pivot_id)

                if layout.toolbar_button(f"##pivot_{mode_id}", icon, btn_sz,
                                         selected, False, tooltip):
                    if pivot_id >= 0:
                        lf.ui.set_pivot_mode(pivot_id)

        layout.end_window()
        layout.pop_style_color(1)
        layout.pop_style_var(4)


class UtilityToolbar(Panel):
    """Vertical utility toolbar on left side of viewport"""
    label = "Utility Toolbar"
    space = "VIEWPORT_OVERLAY"
    order = 0

    def draw(self, layout):
        import lichtfeld as lf
        from . import icon_manager
        theme = lf.ui.theme()
        scale = layout.get_dpi_scale()

        btn_size = theme.sizes.toolbar_button_size * scale
        padding = theme.sizes.toolbar_padding * scale
        spacing = theme.sizes.toolbar_spacing * scale

        has_render_manager = True
        try:
            lf.get_render_mode()
        except Exception:
            has_render_manager = False

        num_buttons = 9 if has_render_manager else 4
        num_separators = 3 if has_render_manager else 1

        width = btn_size + 2 * padding
        height = (num_buttons * btn_size +
                  (num_buttons - 1) * spacing +
                  num_separators * spacing +
                  2 * padding)

        vp_pos = layout.get_viewport_pos()
        margin_left = 10 * scale
        margin_top = 5 * scale
        pos = (vp_pos[0] + margin_left, vp_pos[1] + margin_top)

        layout.set_next_window_pos(pos)
        layout.set_next_window_size((width, height))

        layout.push_style_var("WindowRounding", theme.sizes.window_rounding)
        layout.push_style_var_vec2("WindowPadding", (padding, padding))
        layout.push_style_var_vec2("ItemSpacing", (0, spacing))
        layout.push_style_var_vec2("FramePadding", (0, 0))
        layout.push_style_color("WindowBg", theme.palette.toolbar_background)

        if layout.begin_window("##UtilityToolbar", WINDOW_FLAGS):
            btn_sz = (btn_size, btn_size)

            # Home button
            if self._icon_button(layout, "home", btn_sz, False):
                lf.reset_camera()
            if layout.is_item_hovered():
                layout.set_tooltip("Reset Camera (Home)")

            # Fullscreen
            is_fullscreen = lf.is_fullscreen() if hasattr(lf, 'is_fullscreen') else False
            fs_icon = "arrows-minimize" if is_fullscreen else "arrows-maximize"
            if self._icon_button(layout, fs_icon, btn_sz, is_fullscreen):
                lf.toggle_fullscreen()
            if layout.is_item_hovered():
                layout.set_tooltip("Toggle Fullscreen")

            # Toggle UI
            if self._icon_button(layout, "layout-off", btn_sz, False):
                lf.toggle_ui()
            if layout.is_item_hovered():
                layout.set_tooltip("Toggle UI (Tab)")

            if has_render_manager:
                layout.spacing()

                render_mode = lf.get_render_mode()

                modes = [
                    ("blob", lf.RenderMode.SPLATS, "Splat Rendering"),
                    ("dots-diagonal", lf.RenderMode.POINTS, "Point Cloud"),
                    ("ring", lf.RenderMode.RINGS, "Gaussian Rings"),
                    ("circle-dot", lf.RenderMode.CENTERS, "Center Markers"),
                ]
                for icon_name, mode, tooltip in modes:
                    selected = (render_mode == mode)
                    if self._icon_button(layout, icon_name, btn_sz, selected):
                        lf.set_render_mode(mode)
                    if layout.is_item_hovered():
                        layout.set_tooltip(tooltip)

                layout.spacing()

                is_ortho = lf.is_orthographic()
                proj_icon = "box" if is_ortho else "perspective"
                proj_tooltip = "Orthographic" if is_ortho else "Perspective"
                if self._icon_button(layout, proj_icon, btn_sz, is_ortho):
                    lf.set_orthographic(not is_ortho)
                if layout.is_item_hovered():
                    layout.set_tooltip(proj_tooltip)

                layout.spacing()

                sequencer_active = lf.ui.is_sequencer_visible()
                if self._icon_button(layout, "video", btn_sz, sequencer_active):
                    lf.ui.set_sequencer_visible(not sequencer_active)
                if layout.is_item_hovered():
                    layout.set_tooltip("Sequencer (Q)")

        layout.end_window()
        layout.pop_style_color(1)
        layout.pop_style_var(4)

    def _icon_button(self, layout, icon_name, size, selected):
        import lichtfeld as lf
        from . import icon_manager

        tex_id = icon_manager.get_icon(icon_name)
        theme = lf.ui.theme()

        # Match C++ IconButton behavior: transparent bg when not selected
        if selected:
            bg_normal = theme.palette.primary
            bg_hovered = self._lighten(theme.palette.primary, 0.1)
            bg_active = self._darken(theme.palette.primary, 0.1)
            # Tinted towards primary when selected
            tint = (0.7 + theme.palette.primary[0] * 0.3,
                    0.7 + theme.palette.primary[1] * 0.3,
                    0.7 + theme.palette.primary[2] * 0.3, 1.0)
        else:
            bg_normal = (0, 0, 0, 0)  # Transparent
            bg_hovered = (theme.palette.surface_bright[0],
                          theme.palette.surface_bright[1],
                          theme.palette.surface_bright[2], 0.3)
            bg_active = (theme.palette.surface_bright[0],
                         theme.palette.surface_bright[1],
                         theme.palette.surface_bright[2], 0.5)
            tint = (1.0, 1.0, 1.0, 0.9)

        layout.push_style_color("Button", bg_normal)
        layout.push_style_color("ButtonHovered", bg_hovered)
        layout.push_style_color("ButtonActive", bg_active)

        if tex_id:
            clicked = layout.image_button(f"##{icon_name}", tex_id, size, tint)
        else:
            fallback = icon_name[0].upper() if icon_name else "?"
            clicked = layout.button(fallback, size)

        layout.pop_style_color(3)
        return clicked

    def _lighten(self, color, amount):
        return (min(1.0, color[0] + amount),
                min(1.0, color[1] + amount),
                min(1.0, color[2] + amount),
                color[3])

    def _darken(self, color, amount):
        return (max(0.0, color[0] - amount),
                max(0.0, color[1] - amount),
                max(0.0, color[2] - amount),
                color[3])


