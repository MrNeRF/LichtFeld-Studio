# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for plugin marketplace feedback rendering."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


def _install_lf_stub(monkeypatch):
    state = SimpleNamespace(translations={})
    panel_space = SimpleNamespace(
        SIDE_PANEL="SIDE_PANEL",
        FLOATING="FLOATING",
        VIEWPORT_OVERLAY="VIEWPORT_OVERLAY",
        MAIN_PANEL_TAB="MAIN_PANEL_TAB",
        SCENE_HEADER="SCENE_HEADER",
        STATUS_BAR="STATUS_BAR",
    )
    panel_height_mode = SimpleNamespace(FILL="fill", CONTENT="content")
    panel_option = SimpleNamespace(DEFAULT_CLOSED="DEFAULT_CLOSED", HIDE_HEADER="HIDE_HEADER")

    def tr(key):
        return state.translations.get(key, key)

    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        PanelSpace=panel_space,
        PanelHeightMode=panel_height_mode,
        PanelOption=panel_option,
        tr=tr,
        request_redraw=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)
    return state


@pytest.fixture
def plugin_marketplace_module(monkeypatch):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))

    sys.modules.pop("lfs_plugins.plugin_marketplace_panel", None)
    sys.modules.pop("lfs_plugins", None)
    state = _install_lf_stub(monkeypatch)
    module = import_module("lfs_plugins.plugin_marketplace_panel")
    monkeypatch.setattr(
        module,
        "PluginMarketplaceCatalog",
        lambda: SimpleNamespace(
            snapshot=lambda: ([], False, False),
            refresh_async=lambda force=False, **_kwargs: None,
        ),
    )
    return module, state


class _HandleStub:
    def __init__(self):
        self.dirty_fields = []
        self.request_update_count = 0

    def dirty(self, name):
        self.dirty_fields.append(name)

    def request_update(self):
        self.request_update_count += 1


class _ElementStub:
    def __init__(self):
        self.text = ""
        self.classes = {}
        self.attributes = {}
        self.inner_rml = ""
        self._parent = None

    def set_text(self, value):
        self.text = value

    def set_class(self, name, enabled):
        self.classes[name] = enabled

    def set_attribute(self, name, value):
        self.attributes[name] = value

    def remove_attribute(self, name):
        self.attributes.pop(name, None)

    def get_attribute(self, name, default=""):
        return self.attributes.get(name, default)

    def has_attribute(self, name):
        return name in self.attributes

    def set_inner_rml(self, value):
        self.inner_rml = value

    def parent(self):
        return self._parent


class _DocStub:
    def __init__(self, elements):
        self._elements = elements

    def get_element_by_id(self, element_id):
        return self._elements.get(element_id)


def test_plugin_marketplace_syncs_feedback_nodes(plugin_marketplace_module):
    module, _state = plugin_marketplace_module
    panel = module.PluginMarketplacePanel()
    doc = _DocStub({
        "feedback-card": _ElementStub(),
        "feedback-card-progress": _ElementStub(),
        "feedback-card-progress-text": _ElementStub(),
        "feedback-card-success": _ElementStub(),
        "feedback-card-error": _ElementStub(),
    })

    panel._sync_feedback_state(
        doc,
        "feedback-card",
        module.CardOpState(
            phase=module.CardOpPhase.IN_PROGRESS,
            message="Installing plugin",
            progress=0.42,
        ),
        "plugin_manager.working",
    )

    assert doc.get_element_by_id("feedback-card").classes["hidden"] is False
    assert doc.get_element_by_id("feedback-card-progress").classes["hidden"] is False
    assert doc.get_element_by_id("feedback-card-progress").attributes["value"] == "0.42"
    assert doc.get_element_by_id("feedback-card-progress-text").text == "Installing plugin"
    assert doc.get_element_by_id("feedback-card-success").classes["hidden"] is True
    assert doc.get_element_by_id("feedback-card-error").classes["hidden"] is True


def test_plugin_marketplace_uses_dirty_update_policy(plugin_marketplace_module):
    module, _state = plugin_marketplace_module
    assert module.PluginMarketplacePanel.update_policy == "dirty"
    assert "update_interval_ms" not in module.PluginMarketplacePanel.__dict__


def test_plugin_marketplace_requests_update_on_language_generation(plugin_marketplace_module):
    module, _state = plugin_marketplace_module
    panel = module.PluginMarketplacePanel()
    panel._handle = _HandleStub()
    module.RuntimeState.language_generation._fallback = 0

    panel._subscribe_reactive_state()
    module.RuntimeState.language_generation.value = 1

    assert panel._handle.request_update_count == 1

    panel._unsubscribe_reactive_state()


def test_plugin_marketplace_manual_success_clears_url(plugin_marketplace_module):
    module, _state = plugin_marketplace_module
    panel = module.PluginMarketplacePanel()
    panel._handle = _HandleStub()
    panel._manual_url = "owner/repo"
    panel._card_ops["__manual_url__"] = module.CardOpState(
        phase=module.CardOpPhase.SUCCESS,
        message="Installed",
    )

    doc = _DocStub({
        "manual-feedback": _ElementStub(),
        "manual-feedback-progress": _ElementStub(),
        "manual-feedback-progress-text": _ElementStub(),
        "manual-feedback-success": _ElementStub(),
        "manual-feedback-error": _ElementStub(),
        "btn-install-url": _ElementStub(),
    })

    panel._update_manual_feedback(doc)

    assert doc.get_element_by_id("manual-feedback-success").text == "Installed"
    assert "disabled" not in doc.get_element_by_id("btn-install-url").attributes
    assert panel._manual_url == ""
    assert panel._handle.dirty_fields == ["manual_url"]


def test_plugin_marketplace_confirm_message_sets_plain_text(plugin_marketplace_module):
    module, state = plugin_marketplace_module
    state.translations["plugin_marketplace.confirm_uninstall_message"] = "Remove {name}?"

    panel = module.PluginMarketplacePanel()
    message_el = _ElementStub()
    overlay_el = _ElementStub()
    panel._doc = _DocStub({
        "confirm-message": message_el,
        "confirm-overlay": overlay_el,
    })

    panel._request_uninstall_confirmation("Sample Plugin", "sample-card", None)

    assert panel._pending_uninstall_name == "Sample Plugin"
    assert panel._pending_uninstall_card_id == "sample-card"
    assert message_el.text == "Remove Sample Plugin?"
    assert overlay_el.classes["hidden"] is False


def test_plugin_marketplace_renders_git_checkbox_when_available(plugin_marketplace_module):
    module, state = plugin_marketplace_module
    state.translations["plugin_marketplace.install_as_git_checkout"] = "Install as git checkout"

    panel = module.PluginMarketplacePanel()
    panel._git_available = True

    markup = panel._build_card_markup({
        "card_id": "sample-card",
        "name": "Sample Plugin",
        "show_install": True,
        "show_git_checkout": True,
        "git_checkout_selected": True,
    })

    assert 'data-action="git-checkout"' in markup
    assert 'checked="checked"' in markup
    assert "Install as git checkout" in markup


def test_plugin_marketplace_install_uses_git_transport_when_selected(plugin_marketplace_module):
    module, _state = plugin_marketplace_module

    panel = module.PluginMarketplacePanel()
    panel._git_available = True
    panel._git_checkout_selected["sample-card"] = True
    panel._run_async = lambda _card_id, operation, _success, _error: operation(lambda _msg: None)

    calls = {}

    class _ManagerStub:
        def install(self, url, on_progress=None, transport="archive"):
            calls["url"] = url
            calls["transport"] = transport
            return "sample_plugin"

        def get_state(self, name):
            assert name == "sample_plugin"
            return module.PluginState.ACTIVE

        def get_error(self, _name):
            return ""

    entry = module.MarketplacePluginEntry(
        source_url="https://github.com/owner/repo",
        github_url="https://github.com/owner/repo",
        owner="owner",
        repo="repo",
        name="Sample Plugin",
        description="",
    )

    panel._install_plugin_from_marketplace(_ManagerStub(), entry, "sample-card")

    assert calls["url"] == "https://github.com/owner/repo"
    assert calls["transport"] == "git"


def test_plugin_marketplace_list_view_marks_selected_toggle(plugin_marketplace_module):
    module, _state = plugin_marketplace_module

    panel = module.PluginMarketplacePanel()
    doc = _DocStub({
        "view-cards-btn": _ElementStub(),
        "view-list-btn": _ElementStub(),
    })

    panel._doc = doc
    panel._set_view_mode("list")

    assert panel._view_mode == "list"
    assert panel._entries_dirty is True
    assert doc.get_element_by_id("view-cards-btn").classes["selected"] is False
    assert doc.get_element_by_id("view-list-btn").classes["selected"] is True


def test_plugin_marketplace_renders_list_markup(plugin_marketplace_module):
    module, state = plugin_marketplace_module
    state.translations["plugin_marketplace.local_install"] = "Local Installation"

    panel = module.PluginMarketplacePanel()
    markup = panel._build_list_markup({
        "card_id": "sample-card",
        "name": "Sample Plugin",
        "description": "Plugin description",
        "has_error": False,
        "has_version": True,
        "version_label": "v1.0.0",
        "has_repo": True,
        "repo_label": "owner/repo",
        "has_metrics": True,
        "metrics_text": "Stars: 10",
        "has_tags": True,
        "tags_text": "Utility",
        "is_local": True,
        "is_installed": True,
        "status_text": "Status: active",
        "summary_status_text": "Status: active",
        "status_class": "status-success",
        "summary_marker": "●",
        "summary_marker_class": "plugin-list-marker--local",
        "plugin_name": "sample_plugin",
        "show_install": False,
        "show_load": False,
        "show_unload": True,
        "show_reload": True,
        "show_update": False,
        "show_uninstall": True,
        "show_startup": True,
        "startup_checked": True,
        "show_git_checkout": True,
        "git_checkout_selected": True,
    })

    assert 'class="plugin-list-row"' in markup
    assert 'data-action="toggle-expand"' in markup
    assert 'plugin-list-toggle' not in markup
    assert 'type="checkbox"' in markup
    assert 'data-action="startup"' in markup
    assert 'checked="checked"' in markup
    assert 'class="plugin-list-option-row"' in markup
    assert 'class="plugin-list-option-label' in markup
    assert 'data-action="unload"' in markup
    assert 'class="plugin-list-detail-meta"' in markup
    assert 'data-action="unload"' in markup
    assert 'data-action="reload"' in markup
    assert "Local Installation" in markup


def test_plugin_marketplace_collapsed_list_row_marks_details_hidden(plugin_marketplace_module):
    module, _state = plugin_marketplace_module

    panel = module.PluginMarketplacePanel()
    markup = panel._build_list_markup({
        "card_id": "sample-card",
        "name": "Sample Plugin",
        "description": "Plugin description",
        "has_error": False,
        "has_version": False,
        "has_repo": False,
        "has_metrics": False,
        "has_tags": False,
        "is_local": False,
        "summary_status_text": "",
        "status_class": "status-muted",
        "summary_marker": "★",
        "summary_marker_class": "plugin-list-marker--remote",
        "show_install": True,
        "show_load": False,
        "show_unload": False,
        "show_reload": False,
        "show_update": False,
        "show_uninstall": False,
        "show_startup": False,
        "show_git_checkout": False,
    })

    assert 'class="plugin-list-details hidden"' in markup
    assert '>▶<' in markup


def test_plugin_marketplace_unload_failure_sets_dismissable_error(plugin_marketplace_module):
    module, state = plugin_marketplace_module
    state.translations["plugin_manager.status.unload_failed"] = "Unload failed"

    panel = module.PluginMarketplacePanel()

    class _ManagerStub:
        def get_state(self, name):
            assert name == "sample_plugin"
            return module.PluginState.ACTIVE

        def unload(self, name):
            assert name == "sample_plugin"
            return False

    panel._unload_plugin(_ManagerStub(), "sample_plugin", "sample-card")

    op_state = panel._card_ops["sample-card"]
    assert op_state.phase == module.CardOpPhase.ERROR
    assert op_state.message == "Unload failed"
    assert op_state.finished_at > 0.0


def test_plugin_marketplace_error_state_auto_dismisses_and_collapses_row(
    plugin_marketplace_module,
    monkeypatch,
):
    module, _state = plugin_marketplace_module

    panel = module.PluginMarketplacePanel()
    panel._card_ops["sample-card"] = module.CardOpState(
        phase=module.CardOpPhase.ERROR,
        message="Unload failed",
        finished_at=10.0,
    )

    monkeypatch.setattr(
        module.time,
        "monotonic",
        lambda: 10.0 + module.ERROR_DISMISS_SEC + 0.1,
    )

    current_state = panel._get_card_state("sample-card")

    assert current_state.phase == module.CardOpPhase.IDLE
    assert current_state.message == ""
    assert panel._is_list_row_expanded("sample-card") is False


def test_plugin_marketplace_auto_expanded_row_can_be_collapsed_and_reopened(
    plugin_marketplace_module,
    monkeypatch,
):
    module, _state = plugin_marketplace_module

    panel = module.PluginMarketplacePanel()
    panel._card_ops["sample-card"] = module.CardOpState(
        phase=module.CardOpPhase.ERROR,
        message="Unload failed",
        finished_at=10.0,
    )
    monkeypatch.setattr(module.time, "monotonic", lambda: 10.0)

    assert panel._is_list_row_expanded("sample-card") is True

    panel._set_list_row_expanded("sample-card", False, rerender=False)
    assert panel._is_list_row_expanded("sample-card") is False

    panel._set_list_row_expanded("sample-card", True, rerender=False)
    assert panel._is_list_row_expanded("sample-card") is True
