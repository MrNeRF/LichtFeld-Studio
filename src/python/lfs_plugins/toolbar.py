from .types import Panel
from .tools import ToolRegistry


def _icon_src(icon_name):
    """Build icon src path relative to the RML document in assets/rmlui/."""
    return f"../icon/{icon_name}.png"


def _icon_button(w, container, button_id, icon_src, tooltip="", tooltip_key=""):
    kwargs = {"tooltip_key": tooltip_key} if tooltip_key else {"tooltip": tooltip}
    return w.icon_button(container, button_id, icon_src, **kwargs)


def _clear_children(container):
    while container.num_children() > 0:
        container.remove_child(container.children()[0])


def _tool_signature(tool_defs):
    return tuple((tool.id, tool.icon, tool.label, tool.shortcut) for tool in tool_defs)


def _mode_signature(active_tool_id, modes):
    if not modes:
        return None
    return (
        active_tool_id,
        tuple((mode.id, mode.icon, mode.label) for mode in modes),
    )


_TOOLBAR_HIDDEN_STATES = ("running", "paused", "stopping", "completed")


class GizmoToolbar(Panel):
    label = "Gizmo Toolbar"
    space = "VIEWPORT_OVERLAY"
    order = 1

    def __init__(self):
        super().__init__()
        self._buttons = {}
        self._submode_buttons = {}
        self._pivot_buttons = {}
        self._built = False
        self._last_tool_signature = ()
        self._last_active_tool = None
        self._last_enabled = {}
        self._last_submode_key = None
        self._last_pivot_key = None
        self._was_hidden = False

    def draw(self, layout):
        import lichtfeld as lf
        from . import rml_widgets as w
        from .op_context import get_context
        from .ui.state import AppState

        doc = lf.ui.rml.get_document("viewport_overlay")
        if doc is None:
            return

        container = doc.get_element_by_id("gizmo-toolbar")
        if container is None:
            return

        hidden = AppState.trainer_state.value in _TOOLBAR_HIDDEN_STATES
        container.set_class("hidden", hidden)

        if hidden:
            submode = doc.get_element_by_id("submode-toolbar")
            if submode:
                submode.set_class("hidden", True)
            pivot = doc.get_element_by_id("pivot-toolbar")
            if pivot:
                pivot.set_class("hidden", True)
            self._last_submode_key = None
            self._last_pivot_key = None
            if not self._was_hidden:
                ToolRegistry.clear_active()
                self._was_hidden = True
            return

        self._was_hidden = False

        tool_defs = ToolRegistry.get_all()
        if not tool_defs:
            return

        tool_signature = _tool_signature(tool_defs)
        context = get_context()
        active_tool = lf.ui.get_active_tool()

        if not self._built or tool_signature != self._last_tool_signature:
            self._rebuild_tools(container, tool_defs, w)
            self._last_tool_signature = tool_signature

        if active_tool != self._last_active_tool or not self._built:
            self._last_active_tool = active_tool
            for tool_def in tool_defs:
                tid = tool_def.id
                btn = self._buttons.get(tid)
                if btn is None:
                    continue
                btn.set_class("selected", active_tool == tid)

        for tool_def in tool_defs:
            tid = tool_def.id
            btn = self._buttons.get(tid)
            if btn is None:
                continue
            enabled = tool_def.can_activate(context)
            if enabled != self._last_enabled.get(tid):
                self._last_enabled[tid] = enabled
                if enabled:
                    btn.remove_attribute("disabled")
                else:
                    btn.set_attribute("disabled", "disabled")

        self._update_submodes(doc, w)
        self._update_pivots(doc, w)

    _TOOL_LOCALE_KEYS = {
        "builtin.select": "toolbar.selection",
        "builtin.translate": "toolbar.translate",
        "builtin.rotate": "toolbar.rotate",
        "builtin.scale": "toolbar.scale",
        "builtin.mirror": "toolbar.mirror",
        "builtin.brush": "toolbar.painting",
        "builtin.align": "toolbar.align_3point",
    }

    def _rebuild_tools(self, container, tool_defs, w):
        _clear_children(container)
        self._buttons.clear()
        self._last_enabled.clear()
        for tool_def in tool_defs:
            icon_src = _icon_src(tool_def.icon)
            locale_key = self._TOOL_LOCALE_KEYS.get(tool_def.id, "")
            if locale_key:
                btn = _icon_button(w, container, f"tool-{tool_def.id}",
                                   icon_src, tooltip_key=locale_key)
            else:
                tooltip = tool_def.label
                if tool_def.shortcut:
                    tooltip = f"{tooltip} ({tool_def.shortcut})"
                btn = _icon_button(w, container, f"tool-{tool_def.id}",
                                   icon_src, tooltip=tooltip)
            tid = tool_def.id
            btn.add_event_listener("click", lambda ev, t=tid: self._on_tool_click(t))
            self._buttons[tid] = btn
        self._built = True

    def _on_tool_click(self, tool_id):
        import lichtfeld as lf
        from .op_context import get_context
        context = get_context()
        tool_def = ToolRegistry.get(tool_id)
        if tool_def and tool_def.can_activate(context):
            active = lf.ui.get_active_tool()
            if active == tool_id:
                ToolRegistry.clear_active()
            else:
                ToolRegistry.set_active(tool_id)

    _SUBMODE_LOCALE_KEYS = {
        "builtin.select:centers": "toolbar.brush_selection",
        "builtin.select:rectangle": "toolbar.rect_selection",
        "builtin.select:polygon": "toolbar.polygon_selection",
        "builtin.select:lasso": "toolbar.lasso_selection",
        "builtin.select:rings": "toolbar.ring_selection",
        "builtin.translate:local": "toolbar.local_space",
        "builtin.translate:world": "toolbar.world_space",
        "builtin.rotate:local": "toolbar.local_space",
        "builtin.rotate:world": "toolbar.world_space",
        "builtin.scale:local": "toolbar.local_space",
        "builtin.scale:world": "toolbar.world_space",
        "builtin.mirror:x": "toolbar.mirror_x",
        "builtin.mirror:y": "toolbar.mirror_y",
        "builtin.mirror:z": "toolbar.mirror_z",
    }

    _PIVOT_LOCALE_KEYS = {
        "origin": "toolbar.origin_pivot",
        "bounds": "toolbar.bounds_center_pivot",
    }

    def _update_submodes(self, doc, w):
        import lichtfeld as lf
        active_tool_id = lf.ui.get_active_tool()
        tool_def = ToolRegistry.get(active_tool_id) if active_tool_id else None

        container = doc.get_element_by_id("submode-toolbar")
        if container is None:
            return

        submodes = tool_def.submodes if tool_def else []
        submode_key = _mode_signature(active_tool_id, submodes)

        if submode_key != self._last_submode_key:
            self._last_submode_key = submode_key
            _clear_children(container)
            self._submode_buttons.clear()

            if not submodes:
                container.set_class("hidden", True)
                return

            container.set_class("hidden", False)
            for mode in submodes:
                icon_src = _icon_src(mode.icon) if mode.icon else ""
                locale_key = self._SUBMODE_LOCALE_KEYS.get(
                    f"{active_tool_id}:{mode.id}", "")
                btn = _icon_button(
                    w, container, f"sub-{mode.id}", icon_src,
                    tooltip=mode.label, tooltip_key=locale_key)
                mid = mode.id
                btn.add_event_listener("click",
                    lambda ev, m=mid: self._on_submode_click(m))
                self._submode_buttons[mode.id] = btn

        if not submodes:
            return

        container.set_class("hidden", False)

        is_mirror = (active_tool_id == "builtin.mirror")
        is_transform = active_tool_id in ("builtin.translate", "builtin.rotate", "builtin.scale")

        if is_transform:
            current_space = lf.ui.get_transform_space()
            space_map = {"local": 0, "world": 1}
            for mode in submodes:
                btn = self._submode_buttons.get(mode.id)
                if btn:
                    btn.set_class("selected", current_space == space_map.get(mode.id, -1))
        elif not is_mirror:
            active_submode = lf.ui.get_active_submode()
            for mode in submodes:
                btn = self._submode_buttons.get(mode.id)
                if btn:
                    btn.set_class("selected", active_submode == mode.id)

    def _on_submode_click(self, mode_id):
        import lichtfeld as lf
        active_tool_id = lf.ui.get_active_tool()
        if active_tool_id == "builtin.mirror":
            lf.ui.execute_mirror(mode_id)
        elif active_tool_id in ("builtin.translate", "builtin.rotate", "builtin.scale"):
            space_map = {"local": 0, "world": 1}
            sid = space_map.get(mode_id, -1)
            if sid >= 0:
                lf.ui.set_transform_space(sid)
        else:
            lf.ui.set_selection_mode(mode_id)

    def _update_pivots(self, doc, w):
        import lichtfeld as lf
        active_tool_id = lf.ui.get_active_tool()
        tool_def = ToolRegistry.get(active_tool_id) if active_tool_id else None

        container = doc.get_element_by_id("pivot-toolbar")
        if container is None:
            return

        pivots = tool_def.pivot_modes if tool_def else []
        pivot_key = _mode_signature(active_tool_id, pivots)

        if pivot_key != self._last_pivot_key:
            self._last_pivot_key = pivot_key
            _clear_children(container)
            self._pivot_buttons.clear()

            if not pivots:
                container.set_class("hidden", True)
                return

            container.set_class("hidden", False)
            for mode in pivots:
                icon_src = _icon_src(mode.icon) if mode.icon else ""
                locale_key = self._PIVOT_LOCALE_KEYS.get(mode.id, "")
                btn = _icon_button(
                    w, container, f"pivot-{mode.id}", icon_src,
                    tooltip=mode.label, tooltip_key=locale_key)
                mid = mode.id
                btn.add_event_listener("click",
                    lambda ev, m=mid: self._on_pivot_click(m))
                self._pivot_buttons[mode.id] = btn

        if not pivots:
            return

        container.set_class("hidden", False)

        pivot_map = {"origin": 0, "bounds": 1}
        current_pivot = lf.ui.get_pivot_mode()
        for mode in pivots:
            btn = self._pivot_buttons.get(mode.id)
            if btn:
                btn.set_class("selected", current_pivot == pivot_map.get(mode.id, -1))

    def _on_pivot_click(self, mode_id):
        import lichtfeld as lf
        pivot_map = {"origin": 0, "bounds": 1}
        pid = pivot_map.get(mode_id, -1)
        if pid >= 0:
            lf.ui.set_pivot_mode(pid)


class UtilityToolbar(Panel):
    label = "Utility Toolbar"
    space = "VIEWPORT_OVERLAY"
    order = 0

    def __init__(self):
        super().__init__()
        self._buttons = {}
        self._built = False
        self._last_render_manager = None
        self._last_state_key = None

    def draw(self, layout):
        import lichtfeld as lf
        from . import rml_widgets as w

        doc = lf.ui.rml.get_document("viewport_overlay")
        if doc is None:
            return

        container = doc.get_element_by_id("utility-toolbar")
        if container is None:
            return

        has_render_manager = True
        try:
            lf.get_render_mode()
        except Exception:
            has_render_manager = False

        if not self._built or has_render_manager != self._last_render_manager:
            self._rebuild(container, has_render_manager, w)
            self._last_render_manager = has_render_manager

        self._update_state(has_render_manager)

    def _rebuild(self, container, has_render_manager, w):
        _clear_children(container)
        self._buttons.clear()

        def add_btn(name, icon, tooltip_key, callback):
            btn = _icon_button(
                w, container, f"util-{name}", _icon_src(icon), tooltip_key=tooltip_key)
            btn.add_event_listener("click", lambda ev: callback())
            self._buttons[name] = btn
            return btn

        import lichtfeld as lf

        add_btn("home", "home", "toolbar.home", lf.reset_camera)
        add_btn("fullscreen", "arrows-maximize", "toolbar.fullscreen",
                lf.toggle_fullscreen)
        add_btn("toggle-ui", "layout-off", "toolbar.toggle_ui", lf.toggle_ui)

        if has_render_manager:
            sep = container.append_child("div")
            sep.set_class_names("toolbar-separator")

            for icon, mode_val, tooltip_key in [
                ("blob", lf.RenderMode.SPLATS, "toolbar.splat_rendering"),
                ("dots-diagonal", lf.RenderMode.POINTS, "toolbar.point_cloud"),
                ("ring", lf.RenderMode.RINGS, "toolbar.gaussian_rings"),
                ("circle-dot", lf.RenderMode.CENTERS, "toolbar.center_markers"),
            ]:
                mv = mode_val
                add_btn(f"render-{icon}", icon, tooltip_key,
                        lambda m=mv: lf.set_render_mode(m))

            sep2 = container.append_child("div")
            sep2.set_class_names("toolbar-separator")

            add_btn("projection", "perspective", "toolbar.perspective",
                    lambda: lf.set_orthographic(not lf.is_orthographic()))

            sep3 = container.append_child("div")
            sep3.set_class_names("toolbar-separator")

            add_btn("sequencer", "video", "toolbar.sequencer",
                    lambda: lf.ui.set_sequencer_visible(not lf.ui.is_sequencer_visible()))

        self._built = True

    def _update_state(self, has_render_manager):
        import lichtfeld as lf

        is_fullscreen = lf.is_fullscreen() if hasattr(lf, 'is_fullscreen') else False
        render_mode = lf.get_render_mode() if has_render_manager else None
        is_ortho = lf.is_orthographic() if has_render_manager else None
        seq_visible = lf.ui.is_sequencer_visible() if has_render_manager else None

        state_key = (is_fullscreen, render_mode, is_ortho, seq_visible)
        if state_key == self._last_state_key:
            return
        self._last_state_key = state_key

        fs_btn = self._buttons.get("fullscreen")
        if fs_btn:
            fs_btn.set_class("selected", is_fullscreen)
            icon_name = "arrows-minimize" if is_fullscreen else "arrows-maximize"
            img = fs_btn.query_selector("img")
            if img:
                img.set_attribute("src", _icon_src(icon_name))

        if not has_render_manager:
            return

        mode_map = {
            "blob": lf.RenderMode.SPLATS,
            "dots-diagonal": lf.RenderMode.POINTS,
            "ring": lf.RenderMode.RINGS,
            "circle-dot": lf.RenderMode.CENTERS,
        }
        for icon, mode_val in mode_map.items():
            btn = self._buttons.get(f"render-{icon}")
            if btn:
                btn.set_class("selected", render_mode == mode_val)

        proj_btn = self._buttons.get("projection")
        if proj_btn:
            proj_btn.set_class("selected", is_ortho)
            proj_btn.set_attribute("data-tooltip",
                "toolbar.orthographic" if is_ortho else "toolbar.perspective")
            img = proj_btn.query_selector("img")
            if img:
                img.set_attribute("src", _icon_src("box" if is_ortho else "perspective"))

        seq_btn = self._buttons.get("sequencer")
        if seq_btn:
            seq_btn.set_class("selected", seq_visible)
