# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unified plugin marketplace floating panel."""

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import lichtfeld as lf

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


class PluginMarketplacePanel(RmlPanel):
    """Floating plugin window for browsing, installing, and managing plugins."""

    idname = "lfs.plugin_marketplace"
    label = "Plugin Marketplace"
    space = "FLOATING"
    order = 91
    rml_template = "rmlui/plugin_marketplace.rml"
    rml_height_mode = "content"
    initial_width = 770
    update_interval_ms = 100

    def __init__(self):
        self._catalog = PluginMarketplaceCatalog()
        self._url_plugin_names: Dict[str, str] = {}
        self._manual_url = ""
        self._install_filter_idx = 0
        self._sort_idx = 0

        self._card_ops: Dict[str, CardOpState] = {}
        self._lock = threading.RLock()
        self._pending_uninstall_name = ""
        self._pending_uninstall_card_id = ""

        self._discover_cache: Optional[List[PluginInfo]] = None

        self._doc = None
        self._handle = None
        self._last_card_phases: Dict[str, Tuple] = {}
        self._entries_dirty = True
        self._needs_resort = True
        self._prev_snapshot_key: Optional[Tuple] = None
        self._cached_entries: List[MarketplacePluginEntry] = []
        self._cached_card_ids: List[str] = []
        self._cached_installed_lookup: Dict[str, str] = {}
        self._cached_installed_versions: Dict[str, str] = {}
        self._cached_installed_names: Set[str] = set()
        self._formats_open = False
        self._last_lang = ""

    # ── Data model ────────────────────────────────────────────

    def on_bind_model(self, ctx):
        import lichtfeld as lf

        model = ctx.create_data_model("plugin_marketplace")
        if model is None:
            return

        model.bind_func("panel_label", lambda: lf.ui.tr("menu.view.plugin_marketplace"))

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

        model.bind_event("confirm_yes", self._on_confirm_yes)
        model.bind_event("confirm_no", self._on_confirm_no)

        model.bind_record_list("plugins")
        self._handle = model.get_handle()

    def _set_filter_idx(self, v):
        try:
            idx = int(v)
        except (ValueError, TypeError):
            return
        if idx != self._install_filter_idx:
            self._install_filter_idx = idx
            self._entries_dirty = True
            self._needs_resort = True

    def _set_sort_idx(self, v):
        try:
            idx = int(v)
        except (ValueError, TypeError):
            return
        if idx != self._sort_idx:
            self._sort_idx = idx
            self._entries_dirty = True
            self._needs_resort = True

    # ── Lifecycle ─────────────────────────────────────────────

    def on_load(self, doc):
        super().on_load(doc)
        self._doc = doc
        self._last_lang = lf.ui.get_current_language()
        self._entries_dirty = True
        self._last_card_phases.clear()

        formats_header = doc.get_element_by_id("formats-header")
        if formats_header:
            formats_header.add_event_listener("click", self._on_toggle_formats)
            formats_content = doc.get_element_by_id("formats-content")
            formats_arrow = doc.get_element_by_id("formats-arrow")
            if formats_content:
                from . import rml_widgets as w
                w.sync_section_state(formats_content, self._formats_open,
                                     formats_header, formats_arrow)

        grid_el = doc.get_element_by_id("card-grid")
        if grid_el:
            grid_el.add_event_listener("click", self._on_card_click)
            grid_el.add_event_listener("change", self._on_card_change)

        manual_form = doc.get_element_by_id("manual-install-form")
        if manual_form:
            manual_form.add_event_listener("submit", self._on_manual_form_submit)
            manual_form.add_event_listener("change", self._on_manual_form_change)

    def on_update(self, doc):
        from .manager import PluginManager

        mgr = PluginManager.instance()
        self._ensure_loaded()

        current_lang = lf.ui.get_current_language()
        if current_lang != self._last_lang:
            self._last_lang = current_lang
            self._entries_dirty = True
            self._last_card_phases.clear()

        entries_raw, is_loading = self._catalog.snapshot()

        snapshot_key = (tuple(entries_raw), is_loading)
        if snapshot_key != self._prev_snapshot_key:
            self._prev_snapshot_key = snapshot_key
            self._entries_dirty = True

        if self._entries_dirty:
            self._entries_dirty = False
            entries = self._with_local_plugins(entries_raw, mgr)
            installed_lookup = self._get_installed_plugin_lookup(mgr)
            installed_versions = self._get_installed_plugin_versions(mgr)
            installed_names = set(installed_lookup.values())
            needs_resort = self._needs_resort
            self._needs_resort = False
            preserve = not needs_resort and bool(self._cached_card_ids)
            entries = self._filter_and_sort_entries(
                entries, set(installed_lookup.keys()), installed_names,
                preserve_order=preserve,
            )
            card_ids = [
                e.registry_id or e.name or str(i)
                for i, e in enumerate(entries)
            ]

            self._cached_entries = entries
            self._cached_card_ids = card_ids
            self._cached_installed_lookup = installed_lookup
            self._cached_installed_versions = installed_versions
            self._cached_installed_names = installed_names

            records = [
                self._build_card_record(
                    entry, card_ids[i], mgr,
                    installed_lookup, installed_versions, installed_names,
                )
                for i, entry in enumerate(entries)
            ]
            self._last_card_phases.clear()
            self._handle.update_record_list("plugins", records)

            empty_el = doc.get_element_by_id("empty-state")
            grid_el = doc.get_element_by_id("card-grid")
            if empty_el:
                empty_el.set_class("hidden", len(entries) > 0)
            if grid_el:
                grid_el.set_class("hidden", len(entries) == 0)

        self._update_card_states(
            doc, self._cached_entries, self._cached_card_ids, mgr,
            self._cached_installed_lookup, self._cached_installed_versions,
            self._cached_installed_names,
        )
        self._update_manual_feedback(doc)

    # ── Card record building ──────────────────────────────────

    def _build_card_record(self, entry, card_id, mgr,
                           installed_lookup, installed_versions, installed_names):
        import lichtfeld as lf
        from .settings import SettingsManager

        tr = lf.ui.tr
        plugin_name = self._resolve_entry_plugin_name(entry, installed_lookup, installed_names)
        plugin_state = mgr.get_state(plugin_name) if plugin_name else None
        is_installed = plugin_name is not None
        is_local = self._is_local_entry(entry)
        is_local_only = self._is_local_only_entry(entry)
        has_github = bool(entry.github_url)
        card_state = self._get_card_state(card_id)
        buttons_busy = card_state.phase == CardOpPhase.IN_PROGRESS

        name = entry.name or entry.repo or tr("plugin_marketplace.unknown_plugin")

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
        description = self._truncate_text(desc or tr("plugin_marketplace.no_description"), 90)

        version_label = ""
        has_version = False
        if plugin_name and plugin_state == PluginState.ACTIVE:
            version = installed_versions.get(plugin_name, "").strip()
            if version:
                version_label = version if version.lower().startswith("v") else f"v{version}"
                has_version = True

        metrics = []
        if not is_local_only:
            if entry.stars > 0:
                metrics.append(f"{tr('plugin_marketplace.stars')}: {entry.stars}")
            if entry.downloads > 0:
                metrics.append(f"{tr('plugin_marketplace.downloads')}: {entry.downloads}")
        metrics_text = "  |  ".join(metrics)

        tags = self._entry_type_tags(entry)
        tags_text = "  |  ".join(tags[:3])

        status_text = ""
        status_class = "status-muted"
        if is_installed:
            state_str = plugin_state.value if plugin_state else tr("plugin_manager.status_not_loaded")
            status_text = f"{tr('plugin_manager.status')}: {state_str}"
            if plugin_state == PluginState.ACTIVE:
                status_class = "status-success"

        is_remote_installed = is_installed and not is_local
        is_local_with_github = is_installed and is_local and has_github

        show_startup = (not buttons_busy) and is_installed and bool(plugin_name)
        startup_checked = False
        if show_startup:
            startup_checked = SettingsManager.instance().get(plugin_name).get("load_on_startup", False)

        return {
            "card_id": card_id,
            "name": name,
            "has_version": has_version,
            "version_label": version_label,
            "has_repo": bool(repo_label),
            "repo_label": repo_label,
            "has_metrics": bool(metrics_text),
            "metrics_text": metrics_text,
            "has_tags": bool(tags_text),
            "tags_text": tags_text,
            "is_local": is_local,
            "is_installed": is_installed,
            "status_text": status_text,
            "status_class": status_class,
            "has_error": bool(entry.error),
            "description": description,
            "info_action": "open-url" if has_github else "",
            "github_url": entry.github_url or "",
            "plugin_name": plugin_name or "",
            "show_install": (not buttons_busy) and not is_installed and not is_local_only and not entry.error,
            "show_load": (not buttons_busy) and is_installed and plugin_state != PluginState.ACTIVE,
            "show_unload": (not buttons_busy) and is_installed and plugin_state == PluginState.ACTIVE,
            "show_reload": (not buttons_busy) and is_remote_installed and plugin_state == PluginState.ACTIVE,
            "show_update": (not buttons_busy) and (is_local_with_github or (is_remote_installed and plugin_state != PluginState.ACTIVE)),
            "show_uninstall": (not buttons_busy) and is_installed,
            "show_startup": show_startup,
            "startup_checked": startup_checked,
        }

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
            if state.phase != prev_phase and state.phase != CardOpPhase.IN_PROGRESS:
                self._entries_dirty = True

            card_el = doc.get_element_by_id(f"card-{card_id}")
            if not card_el:
                continue

            card_el.set_class("card--in-progress", state.phase == CardOpPhase.IN_PROGRESS)
            card_el.set_class("card--success", state.phase == CardOpPhase.SUCCESS)
            card_el.set_class("card--error", state.phase == CardOpPhase.ERROR)
            self._sync_feedback_state(doc, f"feedback-{card_id}", state, tr("plugin_manager.working"))

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

        self._sync_feedback_state(doc, "manual-feedback", state, tr("plugin_manager.working"))

        if btn:
            if state.phase == CardOpPhase.IN_PROGRESS:
                btn.set_attribute("disabled", "disabled")
            else:
                btn.remove_attribute("disabled")

        if state.phase == CardOpPhase.SUCCESS:
            self._manual_url = ""
            if self._handle:
                self._handle.dirty("manual_url")

    def _sync_feedback_state(self, doc, element_prefix: str, state: CardOpState, working_text: str):
        feedback_el = doc.get_element_by_id(element_prefix)
        if not feedback_el:
            return

        show_progress = state.phase == CardOpPhase.IN_PROGRESS
        show_success = state.phase == CardOpPhase.SUCCESS
        show_error = state.phase == CardOpPhase.ERROR

        progress_el = doc.get_element_by_id(f"{element_prefix}-progress")
        progress_text_el = doc.get_element_by_id(f"{element_prefix}-progress-text")
        success_el = doc.get_element_by_id(f"{element_prefix}-success")
        error_el = doc.get_element_by_id(f"{element_prefix}-error")

        feedback_el.set_class("hidden", not (show_progress or show_success or show_error))

        if progress_el:
            progress_el.set_class("hidden", not show_progress)
            if show_progress:
                progress_el.set_attribute("value", f"{state.progress:.2f}")
        if progress_text_el:
            progress_text_el.set_class("hidden", not show_progress)
            progress_text_el.set_text(state.message or working_text if show_progress else "")
        if success_el:
            success_el.set_class("hidden", not show_success)
            success_el.set_text(state.message if show_success else "")
        if error_el:
            error_el.set_class("hidden", not show_error)
            error_el.set_text(state.message if show_error else "")

    # ── Event handlers ────────────────────────────────────────

    def _on_toggle_formats(self, _ev):
        self._formats_open = not self._formats_open
        doc = self._doc
        header = doc.get_element_by_id("formats-header")
        content = doc.get_element_by_id("formats-content")
        arrow = doc.get_element_by_id("formats-arrow")
        if content:
            from . import rml_widgets as w
            w.animate_section_toggle(content, self._formats_open, arrow,
                                     header_element=header)

    def _on_manual_form_submit(self, ev):
        from .manager import PluginManager
        mgr = PluginManager.instance()
        self._install_plugin_from_url(mgr, self._manual_url, "__manual_url__")
        ev.stop_propagation()

    def _on_manual_form_change(self, ev):
        target = ev.target()
        if target is None or not ev.get_bool_parameter("linebreak", False):
            return
        if target.get_attribute("id", "") != "manual-url-input":
            return

        form = ev.current_target()
        if form is not None:
            form.submit()
            ev.stop_propagation()

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
            return

        if not card_id:
            return

        entry = None
        for i, e in enumerate(self._cached_entries):
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

    def _on_card_change(self, ev):
        target = ev.target()
        if target is None:
            return

        action, _card_id, plugin_name = self._find_card_action(target)
        if action != "startup" or not plugin_name:
            return

        self._set_startup_preference(target, plugin_name)

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

    def _set_startup_preference(self, element, plugin_name: str):
        from .settings import SettingsManager

        cb_el = self._find_element_with_attr(element, "type", "checkbox")
        checked = cb_el.has_attribute("checked") if cb_el else False
        prefs = SettingsManager.instance().get(plugin_name)
        if prefs.get("load_on_startup", False) == checked:
            return

        prefs.set("load_on_startup", checked)
        self._entries_dirty = True

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
            msg_el.set_text(tr("plugin_marketplace.confirm_uninstall_message").format(name=name))
        overlay = doc.get_element_by_id("confirm-overlay")
        if overlay:
            overlay.set_class("hidden", False)

    # ── Business logic (unchanged) ────────────────────────────

    def _ensure_loaded(self):
        self._catalog.refresh_async()

    def _invalidate_discover_cache(self):
        self._discover_cache = None
        self._entries_dirty = True

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
        preserve_order: bool = False,
    ) -> List[MarketplacePluginEntry]:
        filtered = []
        for entry in entries:
            is_installed = self._is_marketplace_entry_installed(entry, installed_keys, installed_names)
            if self._install_filter_idx == 1 and not is_installed:
                continue
            if self._install_filter_idx == 2 and is_installed:
                continue
            filtered.append(entry)

        if preserve_order:
            return self._stable_merge(filtered)
        return self._sort_entries(filtered)

    def _sort_entries(self, entries: List[MarketplacePluginEntry]) -> List[MarketplacePluginEntry]:
        def popularity(e):
            return (e.stars + e.downloads, e.name.lower())

        if self._sort_idx == 1:
            return sorted(entries, key=popularity)
        if self._sort_idx == 2:
            return sorted(entries, key=lambda e: e.name.lower())
        if self._sort_idx == 3:
            return sorted(entries, key=lambda e: e.name.lower(), reverse=True)
        return sorted(entries, key=popularity, reverse=True)

    @staticmethod
    def _entry_key(entry: MarketplacePluginEntry) -> str:
        if entry.owner and entry.repo:
            return entry.owner.lower() + "/" + entry.repo.lower()
        return entry.registry_id or entry.name or entry.source_url

    def _stable_merge(self, entries: List[MarketplacePluginEntry]) -> List[MarketplacePluginEntry]:
        """Keep existing card order, append new entries sorted at the end."""
        prev_order = {self._entry_key(e): i for i, e in enumerate(self._cached_entries)}
        existing = []
        new_entries = []
        for e in entries:
            idx = prev_order.get(self._entry_key(e))
            if idx is not None:
                existing.append((idx, e))
            else:
                new_entries.append(e)

        existing.sort(key=lambda t: t[0])
        return [e for _, e in existing] + self._sort_entries(new_entries)

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

        try:
            if not mgr.unload(name):
                with self._lock:
                    state = self._card_ops.setdefault(card_id, CardOpState())
                    state.phase = CardOpPhase.ERROR
                    state.message = tr("plugin_manager.status.unload_failed")
                return
            self._invalidate_discover_cache()
            with self._lock:
                state = self._card_ops.setdefault(card_id, CardOpState())
                state.phase = CardOpPhase.SUCCESS
                state.message = tr("plugin_manager.status.unloaded").format(name=name)
                state.finished_at = time.monotonic()
        except Exception as e:
            with self._lock:
                state = self._card_ops.setdefault(card_id, CardOpState())
                state.phase = CardOpPhase.ERROR
                state.message = f"{tr('plugin_manager.status.unload_failed')}: {e}"

    def _reload_plugin(self, mgr, name: str, card_id: str):
        import lichtfeld as lf

        tr = lf.ui.tr

        if not mgr.unload(name):
            with self._lock:
                state = self._card_ops.setdefault(card_id, CardOpState())
                state.phase = CardOpPhase.ERROR
                state.message = tr("plugin_manager.status.unload_failed")
            return

        def do_load(on_progress):
            ok = mgr.load(name, on_progress=on_progress)
            if not ok:
                err = mgr.get_error(name) or tr("plugin_manager.status.reload_failed")
                raise RuntimeError(err)
            self._invalidate_discover_cache()
            return True

        self._run_async(
            card_id,
            do_load,
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
