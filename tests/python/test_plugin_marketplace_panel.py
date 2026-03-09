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

    def tr(key):
        return state.translations.get(key, key)

    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(tr=tr)
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
            snapshot=lambda: ([], False),
            refresh_async=lambda force=False: None,
        ),
    )
    return module, state


class _HandleStub:
    def __init__(self):
        self.dirty_fields = []

    def dirty(self, name):
        self.dirty_fields.append(name)


class _ElementStub:
    def __init__(self):
        self.text = ""
        self.classes = {}
        self.attributes = {}

    def set_text(self, value):
        self.text = value

    def set_class(self, name, enabled):
        self.classes[name] = enabled

    def set_attribute(self, name, value):
        self.attributes[name] = value

    def remove_attribute(self, name):
        self.attributes.pop(name, None)


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
