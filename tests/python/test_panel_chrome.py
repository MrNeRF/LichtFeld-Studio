# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the generic per-panel GUIL chrome hook API."""

from lfs_plugins.panels import apply_panel_chrome, capture_panel_chrome


class _ChromePanel:
    def __init__(self):
        self.collapsed = {"lod"}
        self.query = ""
        self.lock = True

    def capture_chrome(self):
        return {
            "collapsed": sorted(self.collapsed),
            "property_search": self.query,
            "steps_scaling_lock": bool(self.lock),
        }

    def apply_chrome(self, payload):
        self.collapsed = {"lod"}
        self.query = ""
        self.lock = True
        if not isinstance(payload, dict):
            return
        collapsed = payload.get("collapsed")
        if isinstance(collapsed, (list, tuple)):
            self.collapsed = {str(name) for name in collapsed}
        if isinstance(payload.get("property_search"), str):
            self.query = payload["property_search"]
        if "steps_scaling_lock" in payload:
            self.lock = bool(payload["steps_scaling_lock"])


def test_capture_and_apply_panel_chrome_helpers():
    panel = _ChromePanel()
    panel.collapsed = {"dataset"}
    panel.query = "lr"
    panel.lock = False
    payload = capture_panel_chrome(panel)
    assert payload == {
        "collapsed": ["dataset"],
        "property_search": "lr",
        "steps_scaling_lock": False,
    }

    other = _ChromePanel()
    apply_panel_chrome(other, payload)
    assert other.collapsed == {"dataset"}
    assert other.query == "lr"
    assert other.lock is False

    apply_panel_chrome(other, None)
    assert other.collapsed == {"lod"}
    assert other.query == ""
    assert other.lock is True


def test_capture_panel_chrome_skips_missing_hook():
    class _Bare:
        pass

    assert capture_panel_chrome(_Bare()) is None
    apply_panel_chrome(_Bare(), {"collapsed": []})
