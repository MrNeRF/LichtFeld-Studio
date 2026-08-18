# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for application preferences behavior."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


@pytest.fixture
def preferences_panel_module(monkeypatch):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))

    state = SimpleNamespace(
        language="it",
        set_language_calls=[],
        mcp_preferences={
            "enabled": True,
            "expose_network": False,
            "port": 45677,
            "request_logging": False,
            "safe_mode": False,
        },
        mcp_status={
            "enabled": True,
            "running": True,
            "phase": "running",
            "expose_network": False,
            "port": 45677,
            "request_count": 0,
            "success_count": 0,
            "error_count": 0,
            "request_logging": False,
            "endpoints": [
                "http://127.0.0.1:45677/mcp",
                "http://localhost:45677/mcp",
            ],
            "log_file": "",
            "error": "",
            "error_kind": "none",
            "error_address": "",
            "error_port": 0,
            "safe_mode": False,
        },
        set_mcp_calls=[],
        panel_enabled_calls=[],
        section_request="",
    )

    def set_mcp_preferences(enabled, expose_network, port, request_logging):
        config = {
            "enabled": bool(enabled),
            "expose_network": bool(expose_network),
            "port": int(port),
            "request_logging": bool(request_logging),
        }
        state.set_mcp_calls.append(config)
        state.mcp_preferences = dict(config)
        state.mcp_status.update(config)
        return True

    def take_preferences_section_request():
        section = state.section_request
        state.section_request = ""
        return section

    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        PanelSpace=SimpleNamespace(FLOATING="FLOATING"),
        PanelHeightMode=SimpleNamespace(FILL="fill"),
        PanelOption=SimpleNamespace(DEFAULT_CLOSED="DEFAULT_CLOSED"),
        get_current_language=lambda: state.language,
        set_language=lambda language: state.set_language_calls.append(language),
        get_mcp_preferences=lambda: dict(state.mcp_preferences),
        set_mcp_preferences=set_mcp_preferences,
        get_mcp_status=lambda: dict(state.mcp_status),
        set_panel_enabled=lambda panel_id, enabled: state.panel_enabled_calls.append(
            (panel_id, bool(enabled))
        ),
        take_preferences_section_request=take_preferences_section_request,
        tr=lambda key: key,
    )

    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)
    sys.modules.pop("lfs_plugins.preferences_panel", None)
    sys.modules.pop("lfs_plugins", None)
    module = import_module("lfs_plugins.preferences_panel")
    return module, state


def test_language_selection_does_not_reload_active_language(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._language_catalog = [("en", "English"), ("it", "Italiano")]

    panel._set_language_index("1")

    assert state.set_language_calls == []


def test_language_selection_applies_a_different_language(
    preferences_panel_module, monkeypatch
):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._language_catalog = [("en", "English"), ("it", "Italiano")]
    monkeypatch.setattr(panel, "_refresh_selection", lambda: None)

    panel._set_language_index("0")

    assert state.set_language_calls == ["en"]


def test_mcp_port_is_drafted_until_explicit_confirmation(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._read_mcp_preferences()

    panel._set_mcp_port("50123")
    assert state.set_mcp_calls == []

    assert panel._commit_mcp_port()
    assert state.set_mcp_calls == [
        {
            "enabled": True,
            "expose_network": False,
            "port": 50123,
            "request_logging": False,
        }
    ]


def test_invalid_mcp_port_blocks_application(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._read_mcp_preferences()

    panel._set_mcp_port("70000")

    assert not panel._commit_mcp_port()
    assert state.set_mcp_calls == []
    assert panel._mcp_error_text() == "preferences.mcp_invalid_port"


def test_failed_bind_can_retry_the_same_confirmed_port(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._read_mcp_preferences()
    state.mcp_status.update(
        {
            "running": False,
            "phase": "failed",
            "error": "bind failed",
            "error_kind": "bind_failed",
        }
    )

    assert panel._commit_mcp_port()

    assert state.set_mcp_calls == [
        {
            "enabled": True,
            "expose_network": False,
            "port": 45677,
            "request_logging": False,
        }
    ]


def test_starting_listener_does_not_retry_an_already_confirmed_port(
    preferences_panel_module,
):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._read_mcp_preferences()
    state.mcp_status.update({"running": False, "phase": "starting"})

    assert panel._commit_mcp_port()
    assert state.set_mcp_calls == []


def test_mcp_section_request_is_consumed_once(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    state.section_request = "mcp"

    panel._consume_section_request()

    assert panel._section == "mcp"
    assert state.section_request == ""


def test_mcp_endpoint_fallback_never_exposes_listener_wildcard(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    state.mcp_status.update({"expose_network": True, "endpoints": []})

    endpoints = panel._mcp_endpoint_text()

    assert "0.0.0.0" not in endpoints
    assert "http://127.0.0.1:45677/mcp" in endpoints


def test_mcp_runtime_error_explains_the_failed_endpoint(
    preferences_panel_module, monkeypatch
):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    state.mcp_status.update(
        {
            "phase": "failed",
            "error": "Unable to bind 0.0.0.0:45677",
            "error_kind": "bind_failed",
            "error_address": "0.0.0.0",
            "error_port": 45677,
        }
    )
    monkeypatch.setattr(
        module.lf.ui,
        "tr",
        lambda key: "Cannot listen on {endpoint}" if key == "preferences.mcp_bind_failed" else key,
    )

    assert panel._mcp_error_text() == "Cannot listen on 0.0.0.0:45677"


def test_mcp_starting_phase_is_not_reported_as_an_error(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    state.mcp_status.update({"running": False, "phase": "starting"})

    assert panel._mcp_status_text() == "preferences.mcp_status_starting"


def test_title_bar_close_listener_discards_unconfirmed_mcp_port(
    preferences_panel_module, monkeypatch
):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()

    class CloseButton:
        def __init__(self):
            self.listeners = []

        def add_event_listener(self, event_name, callback):
            assert event_name == "click"
            self.listeners.append(callback)

    close_button = CloseButton()
    document = SimpleNamespace(
        get_element_by_id=lambda element_id: (
            close_button if element_id == "close-btn" else None
        )
    )
    monkeypatch.setattr(panel, "_dirty_expanded_sections", lambda: None)
    monkeypatch.setattr(panel, "_rebuild_records", lambda: None)
    monkeypatch.setattr(panel, "_consume_section_request", lambda: None)
    monkeypatch.setattr(panel, "_state", lambda: ())
    monkeypatch.setattr(panel, "_refresh_selection", lambda: None)

    panel.on_mount(document)
    panel._set_mcp_port("47000")

    assert len(close_button.listeners) == 1
    close_button.listeners[0](None)

    assert panel._mcp_port == "45677"
    assert state.set_mcp_calls == []
    assert state.panel_enabled_calls[-1] == ("lfs.preferences", False)


def test_external_mcp_toggle_preserves_an_unconfirmed_port_draft(
    preferences_panel_module,
):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._read_mcp_preferences()
    panel._set_mcp_port("47000")

    state.mcp_preferences["expose_network"] = True
    state.mcp_status["expose_network"] = True
    panel._sync_mcp_runtime()

    assert panel._mcp_port == "47000"
    assert panel._mcp_applied_port == 45677
    assert panel._mcp_expose_network is True


def test_safe_mode_disables_live_mcp_mutations(preferences_panel_module):
    module, state = preferences_panel_module
    state.mcp_preferences.update({"enabled": False, "safe_mode": True})
    state.mcp_status.update({"enabled": False, "phase": "disabled", "safe_mode": True})
    panel = module.PreferencesPanel()
    panel._read_mcp_preferences()

    panel._set_mcp_enabled(True)
    panel._set_mcp_expose_network(True)
    panel._set_mcp_port("47000")
    panel._set_mcp_request_logging(True)

    assert panel._mcp_safe_mode is True
    assert panel._mcp_enabled is False
    assert panel._mcp_port == "45677"
    assert state.set_mcp_calls == []


def test_safe_mode_reset_does_not_stage_mcp_changes(preferences_panel_module):
    module, state = preferences_panel_module
    state.mcp_preferences.update({"enabled": False, "safe_mode": True})
    state.mcp_status.update({"enabled": False, "phase": "disabled", "safe_mode": True})
    panel = module.PreferencesPanel()
    panel._read_mcp_preferences()
    panel._refresh_selection = lambda: None

    snapshot = (
        panel._mcp_enabled,
        panel._mcp_expose_network,
        panel._mcp_port,
        panel._mcp_applied_port,
        panel._mcp_request_logging,
        panel._mcp_safe_mode,
    )

    panel._section = "mcp"
    panel._reset_section()

    assert state.set_mcp_calls == []
    assert (
        panel._mcp_enabled,
        panel._mcp_expose_network,
        panel._mcp_port,
        panel._mcp_applied_port,
        panel._mcp_request_logging,
        panel._mcp_safe_mode,
    ) == snapshot

    def confirm_dialog(_title, _message, _buttons, callback):
        callback("preferences.reset_all_settings")

    module.lf.ui.confirm_dialog = confirm_dialog
    module.lf.ui.message_dialog = lambda *_a, **_k: None
    module.lf.ui.set_theme = lambda *_a, **_k: None
    module.lf.ui.set_ui_scale = lambda *_a, **_k: None
    module.lf.ui.set_remember_camera_navigation = lambda *_a, **_k: None
    module.lf.ui.set_remember_camera_view_snap = lambda *_a, **_k: None
    module.lf.ui.reset_layout = lambda: None
    module.lf.ui.reset_window_state = lambda: None
    module.lf.set_camera_navigation_mode = lambda *_a, **_k: None
    module.lf.set_camera_view_snap_enabled = lambda *_a, **_k: None

    panel._on_reset_all_settings(None, None, None)

    assert state.set_mcp_calls == []
    assert (
        panel._mcp_enabled,
        panel._mcp_expose_network,
        panel._mcp_port,
        panel._mcp_applied_port,
        panel._mcp_request_logging,
        panel._mcp_safe_mode,
    ) == snapshot


def test_safe_mode_disables_preferences_and_status_bar_mcp_controls():
    project_root = Path(__file__).parent.parent.parent
    resources = project_root / "src" / "visualizer" / "gui" / "rmlui" / "resources"
    preferences = (resources / "preferences.rml").read_text(encoding="utf-8")
    status_bar = (resources / "statusbar.rml").read_text(encoding="utf-8")

    assert preferences.count('data-attrif-disabled="mcp_safe_mode"') == 5
    assert preferences.count('data-class-disabled="mcp_safe_mode"') == 2
    assert (
        '<button id="mcp-toggle" data-class-is-on="mcp_server_enabled" '
        'data-attr-title="mcp_toggle_label" data-attrif-disabled="safe_mode" '
        'data-class-disabled="safe_mode">'
        in status_bar
    )


def test_preferences_content_scrolls_without_overlapping_the_fixed_footer():
    project_root = Path(__file__).parent.parent.parent
    stylesheet = (
        project_root
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "resources"
        / "preferences.rcss"
    ).read_text(encoding="utf-8")
    content_rule = stylesheet.split(".preferences-content {", 1)[1].split("}", 1)[0]

    assert "min-height: 0;" in content_rule
    assert "overflow-y: auto;" in content_rule
    assert "padding-right: 6dp;" in content_rule


def test_mcp_python_bindings_release_the_gil_around_native_work():
    project_root = Path(__file__).parent.parent.parent
    source = (project_root / "src" / "python" / "lfs" / "py_ui.cpp").read_text(
        encoding="utf-8"
    )

    for name in (
        '"get_mcp_preferences"',
        '"set_mcp_preferences"',
        '"get_mcp_status"',
        '"get_mcp_log_directory"',
    ):
        start = source.index(name)
        end = source.find("m.def(", start + len(name))
        block = source[start : end if end != -1 else len(source)]
        assert "nb::gil_scoped_release" in block


def test_footer_ok_commits_mcp_port_before_closing(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._read_mcp_preferences()
    panel._set_mcp_port("47000")

    panel._on_accept_and_close(None, None, None)

    assert state.set_mcp_calls == [
        {
            "enabled": True,
            "expose_network": False,
            "port": 47000,
            "request_logging": False,
        }
    ]


def test_mcp_failed_listener_does_not_advertise_inactive_endpoints(
    preferences_panel_module,
):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    state.mcp_status.update(
        {
            "running": False,
            "error": "Unable to bind 127.0.0.1:45677",
        }
    )

    assert panel._mcp_endpoint_text() == "preferences.mcp_no_active_endpoint"
