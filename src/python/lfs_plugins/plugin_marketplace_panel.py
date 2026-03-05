# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unified plugin marketplace floating panel."""

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .marketplace import (
    MarketplacePluginEntry,
    PluginMarketplaceCatalog,
)
from .plugin import PluginInfo, PluginState
from .types import RmlPanel

MAX_OUTPUT_LINES = 100
SUCCESS_DISMISS_SEC = 3.0

_PHASE_MILESTONES: List[Tuple[str, float]] = [
    ("cloning", 0.05),
    ("cloned", 0.30),
    ("downloading", 0.05),
    ("extracting", 0.35),
    ("syncing dependencies", 0.40),
    ("updating", 0.05),
    ("updated", 0.50),
    ("unloading", 0.20),
    ("uninstalling", 0.20),
]
_NUDGE_FRACTION = 0.08
_PROGRESS_CEILING = 0.95


class CardOpPhase(Enum):
    IDLE = "idle"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    ERROR = "error"


@dataclass
class CardOpState:
    phase: CardOpPhase = CardOpPhase.IDLE
    message: str = ""
    progress: float = 0.0
    output_lines: List[str] = field(default_factory=list)
    finished_at: float = 0.0


def _xml_escape(text: str) -> str:
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


class PluginMarketplacePanel(RmlPanel):
    """Floating plugin window for browsing, installing, and managing plugins."""

    idname = "lfs.plugin_marketplace"
    label = "Plugin Marketplace"
    space = "FLOATING"
    order = 91
    rml_template = "rmlui/plugin_marketplace.rml"
    rml_height_mode = "content"
    initial_width = 770

    def __init__(self):
        self._catalog = PluginMarketplaceCatalog()
        self._url_plugin_names: Dict[str, str] = {}
        self._manual_url = ""
        self._install_filter_idx = 0
        self._sort_idx = 0

        self._card_ops: Dict[str, CardOpState] = {}
        self._lock = threading.Lock()
        self._pending_uninstall_name = ""
        self._pending_uninstall_card_id = ""

        self._discover_cache: Optional[List[PluginInfo]] = None

        self._doc = None
        self._handle = None
        self._rendered_card_ids: List[str] = []
        self._last_card_phases: Dict[str, Tuple] = {}
        self._force_rebuild = False
        self._formats_open = False

    # ── Data model ────────────────────────────────────────────

    def on_bind_model(self, ctx):
        import lichtfeld as lf

        model = ctx.create_data_model("plugin_marketplace")
        if model is None:
            return

        tr = lf.ui.tr

        model.bind_func("panel_label", lambda: tr("menu.view.plugin_marketplace"))
        model.bind_func("title_line", lambda: tr("plugin_marketplace.title_line"))
        model.bind_func("warning_body", lambda: tr("plugin_marketplace.warning_body"))
        model.bind_func("filter_label", lambda: tr("plugin_marketplace.filter_label"))
        model.bind_func("sort_label", lambda: tr("plugin_marketplace.sort_label"))
        model.bind_func("url_label", lambda: tr("plugin_manager.github_url_or_shorthand"))
        model.bind_func("install_btn_label", lambda: tr("plugin_manager.button.install_plugin"))
        model.bind_func("no_plugins_text", lambda: tr("plugin_marketplace.no_plugins"))
        model.bind_func("edit_list_hint", lambda: tr("plugin_marketplace.edit_list_hint"))
        model.bind_func("confirm_no_label", lambda: tr("plugin_marketplace.confirm_uninstall_no"))
        model.bind_func("confirm_yes_label", lambda: tr("plugin_marketplace.confirm_uninstall_yes"))
        model.bind_func("formats_label", lambda: tr("plugin_manager.supported_formats"))

        model.bind_func("filter_all", lambda: tr("plugin_marketplace.filter.all"))
        model.bind_func("filter_installed", lambda: tr("plugin_marketplace.filter.installed"))
        model.bind_func("filter_not_installed", lambda: tr("plugin_marketplace.filter.not_installed"))
        model.bind_func("sort_pop_desc", lambda: tr("plugin_marketplace.sort.popularity_desc"))
        model.bind_func("sort_pop_asc", lambda: tr("plugin_marketplace.sort.popularity_asc"))
        model.bind_func("sort_name_asc", lambda: tr("plugin_marketplace.sort.name_asc"))
        model.bind_func("sort_name_desc", lambda: tr("plugin_marketplace.sort.name_desc"))

        model.bind(
            "manual_url",
            lambda: self._manual_url,
            lambda v: setattr(self, "_manual_url", v),
        )
        model.bind(
            "filter_idx",
            lambda: str(self._install_filter_idx),
            self._set_filter_idx,
        )
        model.bind(
            "sort_idx",
            lambda: str(self._sort_idx),
            self._set_sort_idx,
        )

        model.bind_event("install_from_url", self._on_install_from_url)
        model.bind_event("confirm_yes", self._on_confirm_yes)
        model.bind_event("confirm_no", self._on_confirm_no)

        self._handle = model.get_handle()

    def _set_filter_idx(self, v):
        try:
            idx = int(v)
        except (ValueError, TypeError):
            return
        if idx != self._install_filter_idx:
            self._install_filter_idx = idx
            self._force_rebuild = True

    def _set_sort_idx(self, v):
        try:
            idx = int(v)
        except (ValueError, TypeError):
            return
        if idx != self._sort_idx:
            self._sort_idx = idx
            self._force_rebuild = True

    # ── Lifecycle ─────────────────────────────────────────────

    def on_load(self, doc):
        super().on_load(doc)
        self._doc = doc

        formats_header = doc.get_element_by_id("formats-header")
        if formats_header:
            formats_header.add_event_listener("click", self._on_toggle_formats)

        grid_el = doc.get_element_by_id("card-grid")
        if grid_el:
            grid_el.add_event_listener("click", self._on_card_click)

    def on_update(self, doc):
        import lichtfeld as lf
        from .manager import PluginManager

        mgr = PluginManager.instance()
        self._ensure_loaded()

        entries, _is_loading = self._catalog.snapshot()
        entries = self._with_local_plugins(entries, mgr)
        installed_lookup = self._get_installed_plugin_lookup(mgr)
        installed_versions = self._get_installed_plugin_versions(mgr)
        installed_names = set(installed_lookup.values())
        entries = self._filter_and_sort_entries(entries, set(installed_lookup.keys()), installed_names)

        card_ids = [e.registry_id or e.name or str(i) for i, e in enumerate(entries)]

        empty_el = doc.get_element_by_id("empty-state")
        grid_el = doc.get_element_by_id("card-grid")
        if empty_el:
            empty_el.set_class("hidden", len(entries) > 0)
        if not grid_el:
            return

        grid_el.set_class("hidden", len(entries) == 0)

        if card_ids != self._rendered_card_ids or self._force_rebuild:
            self._force_rebuild = False
            self._rendered_card_ids = list(card_ids)
            self._last_card_phases.clear()
            rml_parts = []
            for i, entry in enumerate(entries):
                card_id = card_ids[i]
                rml_parts.append(self._build_card_rml(
                    mgr, i, entry, card_id,
                    installed_lookup, installed_versions, installed_names,
                ))
            grid_el.set_inner_rml("\n".join(rml_parts))

        self._update_card_states(doc, entries, card_ids, mgr, installed_lookup, installed_versions, installed_names)
        self._update_manual_feedback(doc)

    # ── Card RML generation ───────────────────────────────────

    def _build_card_rml(self, mgr, idx, entry, card_id,
                        installed_lookup, installed_versions, installed_names):
        import lichtfeld as lf

        tr = lf.ui.tr
        plugin_name = self._resolve_entry_plugin_name(entry, installed_lookup, installed_names)
        plugin_state = mgr.get_state(plugin_name) if plugin_name else None
        is_installed = plugin_name is not None
        is_local = self._is_local_entry(entry)
        is_local_only = self._is_local_only_entry(entry)
        has_github = bool(entry.github_url)
        card_state = self._get_card_state(card_id)

        short_name = _xml_escape(entry.name or entry.repo or tr("plugin_marketplace.unknown_plugin"))
        repo_label = ""
        if entry.owner and entry.repo:
            repo_label = f"{entry.owner}/{entry.repo}"
        elif entry.repo:
            repo_label = entry.repo

        desc = entry.description
        if not desc and plugin_name and self._discover_cache:
            for p in self._discover_cache:
                if p.name == plugin_name:
                    desc = p.description
                    break
        description = _xml_escape(self._truncate_text(desc or tr("plugin_marketplace.no_description"), 90))

        parts = []
        parts.append(f'<div class="plugin-card" id="card-{_xml_escape(card_id)}" data-card-id="{_xml_escape(card_id)}">')

        info_attrs = ""
        if has_github:
            info_attrs = f' data-action="open-url" data-url="{_xml_escape(entry.github_url)}"'
        parts.append(f'  <div class="card-info"{info_attrs}>')
        parts.append(f'    <span class="card-name">{short_name}</span>')
        if plugin_name and plugin_state == PluginState.ACTIVE:
            version = installed_versions.get(plugin_name, "").strip()
            if version:
                version_label = version if version.lower().startswith("v") else f"v{version}"
                parts.append(f'    <span class="card-version status-info">{_xml_escape(version_label)}</span>')
        if repo_label:
            parts.append(f'    <span class="card-repo text-disabled">{_xml_escape(repo_label)}</span>')
        if not is_local_only:
            metrics = []
            if entry.stars > 0:
                metrics.append(f"{tr('plugin_marketplace.stars')}: {entry.stars}")
            if entry.downloads > 0:
                metrics.append(f"{tr('plugin_marketplace.downloads')}: {entry.downloads}")
            if metrics:
                parts.append(f'    <span class="card-metrics mp-warning-text">{_xml_escape("  |  ".join(metrics))}</span>')

        tags = self._entry_type_tags(entry)
        if tags:
            parts.append(f'    <span class="card-tags text-disabled">{_xml_escape("  |  ".join(tags[:3]))}</span>')
        if is_local:
            parts.append(f'    <span class="card-local status-info">{_xml_escape(tr("plugin_marketplace.local_install"))}</span>')

        if is_installed:
            state_str = plugin_state.value if plugin_state else tr("plugin_manager.status_not_loaded")
            css_cls = "status-success" if plugin_state == PluginState.ACTIVE else "status-muted"
            parts.append(f'    <span class="card-status {css_cls}">'
                         f'{_xml_escape(tr("plugin_manager.status"))}: {_xml_escape(state_str)}</span>')

        if entry.error:
            parts.append(f'    <span class="card-error status-error">{_xml_escape(tr("plugin_marketplace.invalid_link"))}</span>')
        else:
            parts.append(f'    <span class="card-description text-disabled">{description}</span>')

        parts.append('    <div class="separator"></div>')
        parts.append('  </div>')

        parts.append(f'  <div class="card-feedback" id="feedback-{_xml_escape(card_id)}"></div>')

        parts.append(self._build_card_buttons_rml(
            mgr, idx, entry, plugin_name, plugin_state,
            is_installed, is_local, is_local_only, card_id, card_state,
        ))

        parts.append('</div>')
        return "\n".join(parts)

    def _build_card_buttons_rml(self, mgr, idx, entry, plugin_name, plugin_state,
                                is_installed, is_local, is_local_only, card_id, card_state):
        import lichtfeld as lf
        from .settings import SettingsManager

        tr = lf.ui.tr
        esc_id = _xml_escape(card_id)
        busy = card_state.phase == CardOpPhase.IN_PROGRESS

        if card_state.phase != CardOpPhase.IDLE:
            return f'  <div class="card-buttons" id="btns-{esc_id}"></div>'

        parts = []
        if is_installed and plugin_name:
            prefs = SettingsManager.instance().get(plugin_name)
            startup = prefs.get("load_on_startup", False)
            checked = ' checked="checked"' if startup else ''
            parts.append(f'  <div class="card-startup-row">')
            parts.append(f'    <input type="checkbox" id="startup-{esc_id}" data-action="startup"'
                         f' data-card-id="{esc_id}" data-plugin="{_xml_escape(plugin_name)}"{checked} />')
            parts.append(f'    <span class="card-startup-label text-disabled">{_xml_escape(tr("plugin_marketplace.load_on_startup"))}</span>')
            parts.append(f'  </div>')

        disabled = ' disabled="disabled"' if busy else ''
        parts.append(f'  <div class="card-buttons" id="btns-{esc_id}">')

        if not is_installed:
            if is_local_only:
                parts.append('  </div>')
                return "\n".join(parts)
            dis_install = ' disabled="disabled"' if (busy or bool(entry.error)) else ''
            parts.append(f'    <button class="btn btn--success" data-action="install"'
                         f' data-card-id="{esc_id}"{dis_install}>'
                         f'{_xml_escape(tr("plugin_marketplace.button.install"))}</button>')
        elif is_local_only:
            load_label = tr("plugin_manager.button.unload") if plugin_state == PluginState.ACTIVE else tr("plugin_manager.button.load")
            load_action = "unload" if plugin_state == PluginState.ACTIVE else "load"
            load_cls = "btn--warning" if plugin_state == PluginState.ACTIVE else "btn--success"
            parts.append(f'    <button class="btn {load_cls}" data-action="{load_action}"'
                         f' data-card-id="{esc_id}" data-plugin="{_xml_escape(plugin_name)}"{disabled}>'
                         f'{_xml_escape(load_label)}</button>')
            parts.append(f'    <button class="btn btn--error" data-action="uninstall"'
                         f' data-card-id="{esc_id}" data-plugin="{_xml_escape(plugin_name)}"{disabled}>'
                         f'{_xml_escape(tr("plugin_manager.button.uninstall"))}</button>')
        elif is_local and bool(entry.github_url):
            load_label = tr("plugin_manager.button.unload") if plugin_state == PluginState.ACTIVE else tr("plugin_manager.button.load")
            load_action = "unload" if plugin_state == PluginState.ACTIVE else "load"
            load_cls = "btn--warning" if plugin_state == PluginState.ACTIVE else "btn--success"
            parts.append(f'    <button class="btn {load_cls}" data-action="{load_action}"'
                         f' data-card-id="{esc_id}" data-plugin="{_xml_escape(plugin_name)}"{disabled}>'
                         f'{_xml_escape(load_label)}</button>')
            parts.append(f'    <button class="btn btn--primary" data-action="update"'
                         f' data-card-id="{esc_id}" data-plugin="{_xml_escape(plugin_name)}"{disabled}>'
                         f'{_xml_escape(tr("plugin_manager.button.update"))}</button>')
            parts.append(f'    <button class="btn btn--error" data-action="uninstall"'
                         f' data-card-id="{esc_id}" data-plugin="{_xml_escape(plugin_name)}"{disabled}>'
                         f'{_xml_escape(tr("plugin_manager.button.uninstall"))}</button>')
        else:
            if plugin_state == PluginState.ACTIVE:
                parts.append(f'    <button class="btn btn--primary" data-action="reload"'
                             f' data-card-id="{esc_id}" data-plugin="{_xml_escape(plugin_name)}"{disabled}>'
                             f'{_xml_escape(tr("plugin_manager.button.reload"))}</button>')
                parts.append(f'    <button class="btn btn--warning" data-action="unload"'
                             f' data-card-id="{esc_id}" data-plugin="{_xml_escape(plugin_name)}"{disabled}>'
                             f'{_xml_escape(tr("plugin_manager.button.unload"))}</button>')
            else:
                parts.append(f'    <button class="btn btn--success" data-action="load"'
                             f' data-card-id="{esc_id}" data-plugin="{_xml_escape(plugin_name)}"{disabled}>'
                             f'{_xml_escape(tr("plugin_manager.button.load"))}</button>')
                parts.append(f'    <button class="btn btn--primary" data-action="update"'
                             f' data-card-id="{esc_id}" data-plugin="{_xml_escape(plugin_name)}"{disabled}>'
                             f'{_xml_escape(tr("plugin_manager.button.update"))}</button>')
            parts.append(f'    <button class="btn btn--error" data-action="uninstall"'
                         f' data-card-id="{esc_id}" data-plugin="{_xml_escape(plugin_name)}"{disabled}>'
                         f'{_xml_escape(tr("plugin_manager.button.uninstall"))}</button>')

        parts.append('  </div>')
        return "\n".join(parts)

    # ── Card state updates (per-frame, minimal DOM touches) ───

    def _update_card_states(self, doc, entries, card_ids, mgr,
                            installed_lookup, installed_versions, installed_names):
        import lichtfeld as lf

        tr = lf.ui.tr

        for i, card_id in enumerate(card_ids):
            state = self._get_card_state(card_id)
            phase_key = (state.phase, state.message, round(state.progress, 2))
            prev_key = self._last_card_phases.get(card_id)
            if prev_key == phase_key:
                continue
            self._last_card_phases[card_id] = phase_key

            prev_phase = prev_key[0] if prev_key else CardOpPhase.IDLE
            if state.phase == CardOpPhase.IDLE and prev_phase != CardOpPhase.IDLE:
                self._force_rebuild = True
                continue

            card_el = doc.get_element_by_id(f"card-{card_id}")
            if not card_el:
                continue

            card_el.set_class("card--in-progress", state.phase == CardOpPhase.IN_PROGRESS)
            card_el.set_class("card--success", state.phase == CardOpPhase.SUCCESS)
            card_el.set_class("card--error", state.phase == CardOpPhase.ERROR)

            feedback_el = doc.get_element_by_id(f"feedback-{card_id}")
            if feedback_el:
                if state.phase == CardOpPhase.IN_PROGRESS:
                    msg = _xml_escape(state.message or tr("plugin_manager.working"))
                    feedback_el.set_inner_rml(
                        f'<progress class="card-progress" value="{state.progress:.2f}" max="1" />'
                        f'<span class="card-progress-text">{msg}</span>'
                    )
                elif state.phase == CardOpPhase.SUCCESS:
                    feedback_el.set_inner_rml(
                        f'<span class="status-text status-success">{_xml_escape(state.message)}</span>'
                    )
                elif state.phase == CardOpPhase.ERROR:
                    feedback_el.set_inner_rml(
                        f'<span class="status-text status-error">{_xml_escape(state.message)}</span>'
                    )
                else:
                    feedback_el.set_inner_rml("")

    def _update_manual_feedback(self, doc):
        card_id = "__manual_url__"
        state = self._get_card_state(card_id)
        feedback_el = doc.get_element_by_id("manual-feedback")
        if not feedback_el:
            return

        import lichtfeld as lf
        tr = lf.ui.tr

        phase_key = (state.phase, state.message, round(state.progress, 2))
        cache_key = "_manual_feedback_"
        if self._last_card_phases.get(cache_key) == phase_key:
            return
        self._last_card_phases[cache_key] = phase_key

        btn = doc.get_element_by_id("btn-install-url")

        if state.phase == CardOpPhase.IN_PROGRESS:
            msg = _xml_escape(state.message or tr("plugin_manager.working"))
            feedback_el.set_inner_rml(
                f'<progress class="card-progress" value="{state.progress:.2f}" max="1" />'
                f'<span class="card-progress-text">{msg}</span>'
            )
            if btn:
                btn.set_attribute("disabled", "disabled")
        elif state.phase == CardOpPhase.SUCCESS:
            feedback_el.set_inner_rml(
                f'<span class="status-text status-success">{_xml_escape(state.message)}</span>'
            )
            if btn:
                btn.remove_attribute("disabled")
            self._manual_url = ""
            if self._handle:
                self._handle.dirty("manual_url")
        elif state.phase == CardOpPhase.ERROR:
            feedback_el.set_inner_rml(
                f'<span class="status-text status-error">{_xml_escape(state.message)}</span>'
            )
            if btn:
                btn.remove_attribute("disabled")
        else:
            feedback_el.set_inner_rml("")
            if btn:
                btn.remove_attribute("disabled")

    # ── Event handlers ────────────────────────────────────────

    def _on_toggle_formats(self, _ev):
        self._formats_open = not self._formats_open
        doc = self._doc
        content = doc.get_element_by_id("formats-content")
        arrow = doc.get_element_by_id("formats-arrow")
        if content:
            content.set_class("collapsed", not self._formats_open)
        if arrow:
            arrow.set_inner_rml("\u25BC" if self._formats_open else "\u25B6")

    def _on_install_from_url(self, handle, event, args):
        from .manager import PluginManager
        mgr = PluginManager.instance()
        self._install_plugin_from_url(mgr, self._manual_url, "__manual_url__")

    def _on_confirm_yes(self, handle, event, args):
        from .manager import PluginManager
        mgr = PluginManager.instance()
        name = self._pending_uninstall_name
        card_id = self._pending_uninstall_card_id
        self._pending_uninstall_name = ""
        self._pending_uninstall_card_id = ""
        overlay = self._doc.get_element_by_id("confirm-overlay")
        if overlay:
            overlay.set_class("hidden", True)
        if name:
            self._uninstall_plugin(mgr, name, card_id)

    def _on_confirm_no(self, handle, event, args):
        self._pending_uninstall_name = ""
        self._pending_uninstall_card_id = ""
        overlay = self._doc.get_element_by_id("confirm-overlay")
        if overlay:
            overlay.set_class("hidden", True)

    def _on_card_click(self, ev):
        import lichtfeld as lf
        from .manager import PluginManager
        from .settings import SettingsManager

        target = ev.target()
        if target is None:
            return

        action, card_id, plugin_name = self._find_card_action(target)
        if not action:
            return

        mgr = PluginManager.instance()

        if action == "open-url":
            url = self._find_data_attr(target, "data-url")
            if url:
                lf.ui.open_url(url)
            return

        if action == "startup":
            if plugin_name:
                prefs = SettingsManager.instance().get(plugin_name)
                cb_el = self._find_element_with_attr(target, "type", "checkbox")
                checked = cb_el.has_attribute("checked") if cb_el else not prefs.get("load_on_startup", False)
                prefs.set("load_on_startup", checked)
            return

        if not card_id:
            return

        entries, _ = self._catalog.snapshot()
        entries = self._with_local_plugins(entries, mgr)
        installed_lookup = self._get_installed_plugin_lookup(mgr)
        installed_names = set(installed_lookup.values())
        entries = self._filter_and_sort_entries(entries, set(installed_lookup.keys()), installed_names)

        entry = None
        for i, e in enumerate(entries):
            eid = e.registry_id or e.name or str(i)
            if eid == card_id:
                entry = e
                break

        if action == "install" and entry:
            self._install_plugin_from_marketplace(mgr, entry, card_id)
        elif action == "load" and plugin_name:
            self._load_plugin(mgr, plugin_name, card_id)
        elif action == "unload" and plugin_name:
            self._unload_plugin(mgr, plugin_name, card_id)
        elif action == "reload" and plugin_name:
            self._reload_plugin(mgr, plugin_name, card_id)
        elif action == "update" and plugin_name:
            self._update_plugin(mgr, plugin_name, card_id)
        elif action == "uninstall" and plugin_name:
            self._request_uninstall_confirmation(plugin_name, card_id, ev)

    def _find_card_action(self, element):
        for _ in range(6):
            if element is None:
                return None, None, None
            action = element.get_attribute("data-action")
            if action:
                card_id = element.get_attribute("data-card-id", "")
                plugin_name = element.get_attribute("data-plugin", "")
                return action, card_id, plugin_name or None
            p = element.parent()
            if p is None:
                return None, None, None
            element = p
        return None, None, None

    def _find_element_with_attr(self, element, attr, value):
        for _ in range(6):
            if element is None:
                return None
            if element.get_attribute(attr, "") == value:
                return element
            p = element.parent()
            if p is None:
                return None
            element = p
        return None

    def _find_data_attr(self, element, attr):
        for _ in range(6):
            if element is None:
                return None
            val = element.get_attribute(attr, "")
            if val:
                return val
            p = element.parent()
            if p is None:
                return None
            element = p
        return None

    def _request_uninstall_confirmation(self, name, card_id, ev):
        import lichtfeld as lf

        if not name:
            return
        self._pending_uninstall_name = name
        self._pending_uninstall_card_id = card_id

        tr = lf.ui.tr
        doc = self._doc

        msg_el = doc.get_element_by_id("confirm-message")
        if msg_el:
            msg_el.set_inner_rml(
                _xml_escape(tr("plugin_marketplace.confirm_uninstall_message").format(name=name))
            )
        overlay = doc.get_element_by_id("confirm-overlay")
        if overlay:
            overlay.set_class("hidden", False)

    # ── Business logic (unchanged) ────────────────────────────

    def _ensure_loaded(self):
        self._catalog.refresh_async()

    def _invalidate_discover_cache(self):
        self._discover_cache = None

    def _get_discovered_plugins(self, mgr) -> List[PluginInfo]:
        cache = self._discover_cache
        if cache is None:
            cache = mgr.discover()
            self._discover_cache = cache
        return cache

    def _get_card_state(self, card_id: str) -> CardOpState:
        with self._lock:
            state = self._card_ops.get(card_id)
            if state is None:
                return CardOpState()
            if state.phase == CardOpPhase.SUCCESS and state.finished_at > 0:
                if time.monotonic() - state.finished_at >= SUCCESS_DISMISS_SEC:
                    state.phase = CardOpPhase.IDLE
                    state.message = ""
                    state.progress = 0.0
                    state.output_lines.clear()
                    state.finished_at = 0.0
            return CardOpState(
                phase=state.phase,
                message=state.message,
                progress=state.progress,
                output_lines=list(state.output_lines),
                finished_at=state.finished_at,
            )

    def _filter_and_sort_entries(
        self,
        entries: List[MarketplacePluginEntry],
        installed_keys: Set[str],
        installed_names: Set[str],
    ) -> List[MarketplacePluginEntry]:
        filtered = []
        for entry in entries:
            is_installed = self._is_marketplace_entry_installed(entry, installed_keys, installed_names)
            if self._install_filter_idx == 1 and not is_installed:
                continue
            if self._install_filter_idx == 2 and is_installed:
                continue
            filtered.append(entry)

        def popularity(e):
            return (e.stars + e.downloads, e.name.lower())

        if self._sort_idx == 1:
            return sorted(filtered, key=popularity)
        if self._sort_idx == 2:
            return sorted(filtered, key=lambda e: e.name.lower())
        if self._sort_idx == 3:
            return sorted(filtered, key=lambda e: e.name.lower(), reverse=True)
        return sorted(filtered, key=popularity, reverse=True)

    @staticmethod
    def _advance_progress(state: CardOpState, msg: str):
        lower = msg.lower()
        for keyword, milestone in _PHASE_MILESTONES:
            if keyword in lower:
                state.progress = max(state.progress, milestone)
                return
        remaining = _PROGRESS_CEILING - state.progress
        if remaining > 0.01:
            state.progress += remaining * _NUDGE_FRACTION

    def _run_async(self, card_id: str, operation, success_msg: str, error_prefix: str):
        with self._lock:
            existing = self._card_ops.get(card_id)
            if existing and existing.phase == CardOpPhase.IN_PROGRESS:
                return
            state = CardOpState(phase=CardOpPhase.IN_PROGRESS)
            self._card_ops[card_id] = state

        def on_progress(msg: str):
            with self._lock:
                self._advance_progress(state, msg)
                state.message = msg
                state.output_lines.append(msg)
                if len(state.output_lines) > MAX_OUTPUT_LINES:
                    state.output_lines = state.output_lines[-MAX_OUTPUT_LINES:]

        def worker():
            try:
                result = operation(on_progress)
                if result is False:
                    raise RuntimeError(error_prefix)
                with self._lock:
                    state.progress = 1.0
                    if isinstance(result, str):
                        state.message = success_msg.format(result)
                    else:
                        state.message = success_msg
                    state.phase = CardOpPhase.SUCCESS
                    state.finished_at = time.monotonic()
            except Exception as e:
                detail = str(e).strip()
                with self._lock:
                    if detail:
                        state.message = f"{error_prefix}: {detail}"
                    else:
                        state.message = error_prefix
                    state.phase = CardOpPhase.ERROR

        threading.Thread(target=worker, daemon=True).start()

    def _install_plugin_from_marketplace(self, mgr, entry: MarketplacePluginEntry, card_id: str):
        import lichtfeld as lf

        tr = lf.ui.tr

        def do_install(on_progress):
            if entry.registry_id:
                name = mgr.install_from_registry(entry.registry_id, on_progress=on_progress)
            else:
                name = mgr.install(entry.source_url, on_progress=on_progress)
            if mgr.get_state(name) == PluginState.ERROR:
                err = mgr.get_error(name) or tr("plugin_manager.status.load_failed")
                raise RuntimeError(err)
            norm_url = self._normalize_url(entry.source_url)
            if norm_url:
                with self._lock:
                    self._url_plugin_names[norm_url] = name
            self._invalidate_discover_cache()
            return name

        self._run_async(
            card_id,
            do_install,
            tr("plugin_manager.status.installed"),
            tr("plugin_manager.status.install_failed"),
        )

    def _install_plugin_from_url(self, mgr, url: str, card_id: str):
        import lichtfeld as lf

        tr = lf.ui.tr
        clean_url = url.strip()
        if not clean_url:
            with self._lock:
                self._card_ops[card_id] = CardOpState(
                    phase=CardOpPhase.ERROR,
                    message=tr("plugin_manager.error.enter_github_url"),
                )
            return

        def do_install(on_progress):
            name = mgr.install(clean_url, on_progress=on_progress)
            if mgr.get_state(name) == PluginState.ERROR:
                err = mgr.get_error(name) or tr("plugin_manager.status.load_failed")
                raise RuntimeError(err)
            with self._lock:
                self._url_plugin_names[self._normalize_url(clean_url)] = name
            self._invalidate_discover_cache()
            return name

        self._run_async(
            card_id,
            do_install,
            tr("plugin_manager.status.installed"),
            tr("plugin_manager.status.install_failed"),
        )

    def _load_plugin(self, mgr, name: str, card_id: str):
        import lichtfeld as lf

        tr = lf.ui.tr

        def do_load(on_progress):
            ok = mgr.load(name, on_progress=on_progress)
            if not ok:
                err = mgr.get_error(name) or tr("plugin_manager.status.load_failed")
                raise RuntimeError(err)
            self._invalidate_discover_cache()
            return True

        self._run_async(
            card_id,
            do_load,
            tr("plugin_manager.status.loaded").format(name=name),
            tr("plugin_manager.status.load_failed"),
        )

    def _unload_plugin(self, mgr, name: str, card_id: str):
        import lichtfeld as lf

        tr = lf.ui.tr

        def do_unload(on_progress):
            on_progress(tr("plugin_manager.status.unloading").format(name=name))
            if not mgr.unload(name):
                raise RuntimeError(tr("plugin_manager.status.unload_failed"))
            self._invalidate_discover_cache()

        self._run_async(
            card_id,
            do_unload,
            tr("plugin_manager.status.unloaded").format(name=name),
            tr("plugin_manager.status.unload_failed"),
        )

    def _reload_plugin(self, mgr, name: str, card_id: str):
        import lichtfeld as lf

        tr = lf.ui.tr

        def do_reload(on_progress):
            mgr.unload(name)
            ok = mgr.load(name, on_progress=on_progress)
            if not ok:
                err = mgr.get_error(name) or tr("plugin_manager.status.reload_failed")
                raise RuntimeError(err)
            self._invalidate_discover_cache()
            return True

        self._run_async(
            card_id,
            do_reload,
            tr("plugin_manager.status.reloaded").format(name=name),
            tr("plugin_manager.status.reload_failed"),
        )

    def _update_plugin(self, mgr, name: str, card_id: str):
        import lichtfeld as lf

        tr = lf.ui.tr

        def do_update(on_progress):
            mgr.update(name, on_progress=on_progress)
            self._invalidate_discover_cache()
            return True

        self._run_async(
            card_id,
            do_update,
            tr("plugin_manager.status.updated").format(name=name),
            tr("plugin_manager.status.update_failed"),
        )

    def _uninstall_plugin(self, mgr, name: str, card_id: str):
        import lichtfeld as lf

        tr = lf.ui.tr

        def do_uninstall(on_progress):
            on_progress(tr("plugin_manager.status.uninstalling").format(name=name))
            if not mgr.uninstall(name):
                raise RuntimeError(tr("plugin_manager.status.uninstall_failed"))
            self._invalidate_discover_cache()

        self._run_async(
            card_id,
            do_uninstall,
            tr("plugin_manager.status.uninstalled").format(name=name),
            tr("plugin_manager.status.uninstall_failed"),
        )

    def _with_local_plugins(self, entries: List[MarketplacePluginEntry], mgr) -> List[MarketplacePluginEntry]:
        merged = list(entries)
        known_keys: Set[str] = set()
        catalog_urls: Set[str] = set()
        for entry in merged:
            known_keys.update(self._entry_keys(entry))
            norm = self._normalize_url(entry.source_url)
            if norm:
                catalog_urls.add(norm)

        for plugin in self._get_discovered_plugins(mgr):
            plugin_keys = self._plugin_keys(plugin.name, plugin.path.name)
            if any(k in known_keys for k in plugin_keys):
                continue

            remote_url = self._git_remote_url(plugin.path)
            if remote_url:
                norm_remote = self._normalize_url(remote_url)
                if norm_remote in catalog_urls:
                    with self._lock:
                        self._url_plugin_names[norm_remote] = plugin.name
                    known_keys.update(plugin_keys)
                    continue

            source_path = str(plugin.path)
            merged.append(
                MarketplacePluginEntry(
                    source_url=source_path,
                    github_url=remote_url or "",
                    owner="",
                    repo=plugin.path.name,
                    name=plugin.name,
                    description=plugin.description or "",
                )
            )
            with self._lock:
                self._url_plugin_names[self._normalize_url(source_path)] = plugin.name
                if remote_url:
                    self._url_plugin_names[self._normalize_url(remote_url)] = plugin.name
            known_keys.update(plugin_keys)

        return merged

    @staticmethod
    def _git_remote_url(plugin_path: Path) -> str:
        import subprocess
        if not (plugin_path / ".git").exists():
            return ""
        try:
            result = subprocess.run(
                ["git", "-C", str(plugin_path), "remote", "get-url", "origin"],
                capture_output=True, text=True, timeout=3,
            )
            url = result.stdout.strip()
            if url.endswith(".git"):
                url = url[:-4]
            return url
        except Exception:
            return ""

    def _get_installed_plugin_lookup(self, mgr) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        for plugin in self._get_discovered_plugins(mgr):
            for key in self._plugin_keys(plugin.name, plugin.path.name):
                lookup[key] = plugin.name
        return lookup

    def _get_installed_plugin_versions(self, mgr) -> Dict[str, str]:
        return {plugin.name: plugin.version for plugin in self._get_discovered_plugins(mgr)}

    def _resolve_entry_plugin_name(
        self,
        entry: MarketplacePluginEntry,
        installed_lookup: Dict[str, str],
        installed_names: Set[str],
    ):
        norm_url = self._normalize_url(entry.source_url)
        by_url = None
        if norm_url:
            with self._lock:
                by_url = self._url_plugin_names.get(norm_url)
        if by_url and by_url in installed_names:
            return by_url
        for key in self._entry_keys(entry):
            plugin_name = installed_lookup.get(key)
            if plugin_name:
                return plugin_name
        return None

    @staticmethod
    def _normalize_url(url: str) -> str:
        return str(url or "").strip().rstrip("/")

    def _is_marketplace_entry_installed(
        self,
        entry: MarketplacePluginEntry,
        installed_keys: Set[str],
        installed_names: Set[str],
    ) -> bool:
        if any(key in installed_keys for key in self._entry_keys(entry)):
            return True
        norm_url = self._normalize_url(entry.source_url)
        if not norm_url:
            return False
        with self._lock:
            by_url = self._url_plugin_names.get(norm_url)
        return by_url is not None and by_url in installed_names

    @staticmethod
    def _is_local_entry(entry: MarketplacePluginEntry) -> bool:
        source = str(entry.source_url or "").strip()
        if not source:
            return False
        if source.startswith(("http://", "https://", "github:")):
            return False
        return Path(source).is_absolute() or source.startswith("~")

    @staticmethod
    def _is_local_only_entry(entry: MarketplacePluginEntry) -> bool:
        return PluginMarketplacePanel._is_local_entry(entry) and not bool(entry.github_url)

    def _entry_keys(self, entry: MarketplacePluginEntry) -> Set[str]:
        from .installer import normalize_repo_name

        normalized_repo = normalize_repo_name(entry.repo) if entry.repo else ""
        return self._plugin_keys(
            entry.repo,
            entry.name,
            normalized_repo,
            f"{entry.owner}-{entry.repo}" if entry.owner and entry.repo else "",
            f"{entry.owner}_{entry.repo}" if entry.owner and entry.repo else "",
        )

    @staticmethod
    def _plugin_keys(*values: str) -> Set[str]:
        keys = set()
        for value in values:
            raw = str(value or "").strip()
            if not raw:
                continue
            lower = raw.lower()
            keys.add(lower)
            normalized = "".join(ch for ch in lower if ch.isalnum())
            if normalized:
                keys.add(normalized)
        return keys

    @staticmethod
    def _entry_type_tags(entry: MarketplacePluginEntry) -> List[str]:
        tags: List[str] = []
        for topic in entry.topics:
            clean = topic.replace("_", " ").replace("-", " ").strip()
            if not clean:
                continue
            pretty = " ".join(part.capitalize() for part in clean.split())
            if pretty and pretty not in tags:
                tags.append(pretty)
        if entry.language and entry.language not in tags and entry.language.lower() != "python":
            tags.append(entry.language)
        return tags

    @staticmethod
    def _truncate_text(value: str, max_chars: int) -> str:
        if len(value) <= max_chars:
            return value
        return value[: max_chars - 3].rstrip() + "..."
