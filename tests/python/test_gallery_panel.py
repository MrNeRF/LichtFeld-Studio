"""The former Gallery browser only redirects into Asset Manager."""
from importlib import import_module
from types import SimpleNamespace
from test_asset_manager_panel import panel_module


def test_gallery_redirect_focuses_scope_and_selected_project(panel_module, monkeypatch):
    module = import_module("lfs_plugins.gallery_panel")
    focused = []
    monkeypatch.setattr(module.lf.ui, "get_panel_object", lambda name: SimpleNamespace(focus_gallery=focused.append), raising=False)
    panel = module.GalleryPanel()
    panel.focus_project("/test/project.licht")
    assert focused == ["/test/project.licht"]
    assert module.lf._test_state.enabled == [("lfs.asset_manager", True), ("lfs.gallery", False)]
    panel.on_mount(None)
    assert focused[-1] is None


def test_U4_redirect_only_on_mount_not_on_update(panel_module, monkeypatch):
    module = import_module('lfs_plugins.gallery_panel')
    focused = []
    monkeypatch.setattr(module.lf.ui, 'get_panel_object', lambda _: SimpleNamespace(focus_gallery=focused.append), raising=False)
    panel = module.GalleryPanel()
    panel.on_mount(None)
    for _ in range(4):
        assert panel.on_update(None) is False
    assert focused == [None]
    assert module.lf._test_state.enabled == [('lfs.asset_manager',True),('lfs.gallery',False)]
