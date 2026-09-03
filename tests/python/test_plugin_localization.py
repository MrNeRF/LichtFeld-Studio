# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for plugin-owned localization catalogs."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _source_python_path(monkeypatch):
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / "src" / "python"))
    monkeypatch.setitem(sys.modules, "lichtfeld", _runtime_stub(_LocalizationUiStub()))


def _write_catalog(plugin_dir: Path, language: str, document: dict) -> Path:
    locales_dir = plugin_dir / "locales"
    locales_dir.mkdir(parents=True, exist_ok=True)
    path = locales_dir / f"{language}.json"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path


class _LocalizationUiStub:
    def __init__(self):
        self.catalogs = {}
        self.next_token = 1

    def _loc_register_catalog(self, owner_id, language, entries):
        token = self.next_token
        self.next_token += 1
        self.catalogs[token] = (owner_id, language, dict(entries))
        return token

    def _loc_unregister_catalog(self, token):
        return self.catalogs.pop(token, None) is not None

    def tr(self, key):
        for owner_id, language, entries in self.catalogs.values():
            prefix = f"plugins.{owner_id}."
            if language == "en" and key.startswith(prefix):
                value = entries.get(key[len(prefix):])
                if value is not None:
                    return value
        return key

    @staticmethod
    def unregister_panels_for_module(_module_name):
        pass

    @staticmethod
    def free_plugin_icons(_plugin_name):
        pass

    @staticmethod
    def free_plugin_textures(_plugin_name):
        pass


def _runtime_stub(ui):
    logger = SimpleNamespace(info=lambda _message: None, warn=lambda _message: None,
                             error=lambda _message: None)
    return SimpleNamespace(ui=ui, log=logger)


def test_catalogs_are_flattened_and_owner_is_canonical(tmp_path):
    from lfs_plugins.plugin_localization import read_plugin_catalogs

    _write_catalog(
        tmp_path,
        "en",
        {"panel": {"title": "Title", "count": "{count} items"}},
    )
    _write_catalog(tmp_path, "it", {"panel": {"title": "Titolo"}})

    bundle = read_plugin_catalogs(tmp_path, "Example_Plugin.Name")

    assert bundle.owner_id == "example-plugin-name"
    assert bundle.catalogs["en"] == {
        "panel.title": "Title",
        "panel.count": "{count} items",
    }
    assert bundle.catalogs["it"] == {"panel.title": "Titolo"}


def test_plugin_without_locales_remains_valid(tmp_path):
    from lfs_plugins.plugin_localization import read_plugin_catalogs

    assert read_plugin_catalogs(tmp_path, "Legacy Plugin!").catalogs == {}


@pytest.mark.parametrize(
    ("filename", "contents", "message"),
    [
        ("it.json", '{"panel":{"title":"Titolo"}}', "en.json is required"),
        ("EN.json", '{"panel":{"title":"Title"}}', "lowercase language code"),
        ("en.json", '["Title"]', "root must be a JSON object"),
        ("en.json", '{"panel":{"title":7}}', "value must be a string"),
        ("en.json", '{"panel":{"title":"  "}}', "value must not be blank"),
        ("en.json", '{"plugins":{"core":"Title"}}', "invalid relative localization key"),
        ("en.json", '{"panel.title":"One","panel":{"title":"Two"}}', "flattened localization key is duplicated"),
        ("en.json", '{"panel":{"title":"One","title":"Two"}}', "duplicate JSON key"),
    ],
)
def test_invalid_catalog_shapes_are_rejected(tmp_path, filename, contents, message):
    from lfs_plugins.plugin_localization import PluginCatalogError, read_plugin_catalogs

    locales_dir = tmp_path / "locales"
    locales_dir.mkdir()
    (locales_dir / filename).write_text(contents, encoding="utf-8")

    with pytest.raises(PluginCatalogError, match=message):
        read_plugin_catalogs(tmp_path, "example_plugin")


def test_invalid_utf8_and_bom_are_rejected(tmp_path):
    from lfs_plugins.plugin_localization import PluginCatalogError, read_plugin_catalogs

    locales_dir = tmp_path / "locales"
    locales_dir.mkdir()
    english = locales_dir / "en.json"
    english.write_bytes(b"\xff")
    with pytest.raises(PluginCatalogError, match="invalid UTF-8"):
        read_plugin_catalogs(tmp_path, "example_plugin")

    english.write_bytes(b"\xef\xbb\xbf{}")
    with pytest.raises(PluginCatalogError, match="BOM is not allowed"):
        read_plugin_catalogs(tmp_path, "example_plugin")


def test_catalog_resource_limits_are_enforced(tmp_path):
    from lfs_plugins.plugin_localization import (
        MAX_CATALOG_BYTES,
        MAX_CATALOG_ENTRIES,
        MAX_KEY_LENGTH,
        MAX_NESTING_DEPTH,
        MAX_VALUE_BYTES,
        PluginCatalogError,
        read_plugin_catalogs,
    )

    locales_dir = tmp_path / "locales"
    locales_dir.mkdir()
    english = locales_dir / "en.json"

    english.write_bytes(b"{" + b" " * MAX_CATALOG_BYTES + b"}")
    with pytest.raises(PluginCatalogError, match="1 MiB size limit"):
        read_plugin_catalogs(tmp_path, "example_plugin")

    _write_catalog(tmp_path, "en", {"k" * (MAX_KEY_LENGTH + 1): "value"})
    with pytest.raises(PluginCatalogError, match="invalid relative localization key"):
        read_plugin_catalogs(tmp_path, "example_plugin")

    _write_catalog(tmp_path, "en", {"message": "x" * (MAX_VALUE_BYTES + 1)})
    with pytest.raises(PluginCatalogError, match="value exceeds the 16 KiB limit"):
        read_plugin_catalogs(tmp_path, "example_plugin")

    _write_catalog(
        tmp_path,
        "en",
        {f"key_{index}": "value" for index in range(MAX_CATALOG_ENTRIES + 1)},
    )
    with pytest.raises(PluginCatalogError, match="entry limit"):
        read_plugin_catalogs(tmp_path, "example_plugin")

    nested = {"value": "text"}
    for _ in range(MAX_NESTING_DEPTH + 1):
        nested = {"level": nested}
    _write_catalog(tmp_path, "en", nested)
    with pytest.raises(PluginCatalogError, match="JSON nesting is too deep"):
        read_plugin_catalogs(tmp_path, "example_plugin")


def test_translation_keys_and_placeholders_must_match_english(tmp_path):
    from lfs_plugins.plugin_localization import PluginCatalogError, read_plugin_catalogs

    _write_catalog(tmp_path, "en", {"panel": {"title": "Hello {name}"}})
    italian = _write_catalog(
        tmp_path,
        "it",
        {"panel": {"title": "Ciao {user}", "extra": "Extra"}},
    )

    with pytest.raises(PluginCatalogError, match="placeholders do not match"):
        read_plugin_catalogs(tmp_path, "example_plugin")

    italian.write_text(
        json.dumps({"panel": {"title": "Ciao {name}", "extra": "Extra"}}),
        encoding="utf-8",
    )
    with pytest.raises(PluginCatalogError, match="not defined in locales/en.json"):
        read_plugin_catalogs(tmp_path, "example_plugin")


def test_placeholder_order_may_change_but_malformed_english_is_rejected(tmp_path):
    from lfs_plugins.plugin_localization import PluginCatalogError, read_plugin_catalogs

    _write_catalog(tmp_path, "en", {"message": "{first} then {second}"})
    _write_catalog(tmp_path, "it", {"message": "{second}, poi {first}"})
    bundle = read_plugin_catalogs(tmp_path, "example_plugin")
    assert bundle.catalogs["it"]["message"] == "{second}, poi {first}"

    _write_catalog(tmp_path, "en", {"message": "Broken {placeholder"})
    with pytest.raises(PluginCatalogError, match="malformed format placeholder"):
        read_plugin_catalogs(tmp_path, "example_plugin")


def test_local_plugin_checker_reports_catalog_errors(tmp_path):
    from lfs_plugins.validator import validate_plugin

    (tmp_path / "pyproject.toml").write_text(
        """[project]
name = "example_plugin"
version = "0.1.0"
description = "Example"
dependencies = []

[tool.lichtfeld]
hot_reload = true
plugin_api = ">=1,<2"
lichtfeld_version = ">=0.4.2"
required_features = []
""",
        encoding="utf-8",
    )
    (tmp_path / "__init__.py").write_text(
        "def on_load():\n    pass\n\ndef on_unload():\n    pass\n",
        encoding="utf-8",
    )
    _write_catalog(tmp_path, "it", {"panel": {"title": "Titolo"}})

    errors = validate_plugin(tmp_path)

    assert any("plugin localization" in error and "en.json is required" in error for error in errors)


def test_manager_registers_catalogs_before_import_and_unloads_by_token(tmp_path, monkeypatch):
    from lfs_plugins.installer import PluginInstaller
    from lfs_plugins.manager import PluginManager
    from lfs_plugins.plugin import PluginInfo, PluginInstance, PluginState

    ui = _LocalizationUiStub()
    monkeypatch.setitem(sys.modules, "lichtfeld", _runtime_stub(ui))
    _write_catalog(tmp_path, "en", {"panel": {"title": "Plugin title"}})

    plugin = PluginInstance(
        info=PluginInfo(
            name="Example_Plugin",
            version="0.1.0",
            path=tmp_path,
            plugin_api=">=1,<2",
            lichtfeld_version=">=0.4.2",
        )
    )
    manager = PluginManager()
    manager._plugins[plugin.info.name] = plugin

    def load_module(loading_plugin):
        assert ui.tr("plugins.example-plugin.panel.title") == "Plugin title"
        loading_plugin.module = SimpleNamespace(on_load=lambda: None)

    monkeypatch.setattr(manager, "_load_module", load_module)
    monkeypatch.setattr(PluginInstaller, "ensure_venv", lambda _self, *_args: True)
    monkeypatch.setattr(PluginInstaller, "install_dependencies", lambda _self, *_args: True)
    assert manager.load(plugin.info.name)

    assert plugin.state == PluginState.ACTIVE
    assert len(plugin.localization_tokens) == 1
    assert manager.unload(plugin.info.name)
    assert plugin.state == PluginState.UNLOADED
    assert plugin.localization_tokens == []
    assert ui.catalogs == {}


def test_reload_replaces_catalog_before_import(tmp_path, monkeypatch):
    from lfs_plugins.capabilities import CapabilityRegistry
    from lfs_plugins.installer import PluginInstaller
    from lfs_plugins.manager import PluginManager
    from lfs_plugins.plugin import PluginInfo, PluginInstance
    from lfs_plugins import utils

    ui = _LocalizationUiStub()
    monkeypatch.setitem(sys.modules, "lichtfeld", _runtime_stub(ui))
    english = _write_catalog(tmp_path, "en", {"panel": {"title": "Before"}})
    plugin = PluginInstance(
        info=PluginInfo(
            name="example_plugin",
            version="0.1.0",
            path=tmp_path,
            plugin_api=">=1,<2",
            lichtfeld_version=">=0.4.2",
        )
    )
    manager = PluginManager()
    manager._plugins[plugin.info.name] = plugin
    observed_titles = []

    def load_module(loading_plugin):
        observed_titles.append(ui.tr("plugins.example-plugin.panel.title"))
        loading_plugin.module = SimpleNamespace(on_load=lambda: None, on_unload=lambda: None)

    monkeypatch.setattr(manager, "_load_module", load_module)
    monkeypatch.setattr(PluginInstaller, "ensure_venv", lambda _self, *_args: True)
    monkeypatch.setattr(PluginInstaller, "install_dependencies", lambda _self, *_args: True)
    monkeypatch.setattr(
        CapabilityRegistry.instance(), "unregister_all_for_plugin", lambda _name: None
    )
    monkeypatch.setattr(utils, "get_gpu_memory", lambda: 0)

    assert manager.load(plugin.info.name)
    old_tokens = set(ui.catalogs)
    english.write_text('{"panel":{"title":"After"}}', encoding="utf-8")
    assert manager.reload(plugin.info.name)

    assert observed_titles == ["Before", "After"]
    assert old_tokens.isdisjoint(ui.catalogs)
    assert len(plugin.localization_tokens) == 1


def test_failed_activation_rolls_back_localization_catalogs(tmp_path, monkeypatch):
    from lfs_plugins.capabilities import CapabilityRegistry
    from lfs_plugins.installer import PluginInstaller
    from lfs_plugins.manager import PluginManager
    from lfs_plugins.plugin import PluginInfo, PluginInstance, PluginState

    ui = _LocalizationUiStub()
    monkeypatch.setitem(sys.modules, "lichtfeld", _runtime_stub(ui))
    _write_catalog(tmp_path, "en", {"panel": {"title": "Plugin title"}})

    plugin = PluginInstance(
        info=PluginInfo(
            name="example_plugin",
            version="0.1.0",
            path=tmp_path,
            plugin_api=">=1,<2",
            lichtfeld_version=">=0.4.2",
        )
    )
    manager = PluginManager()
    manager._plugins[plugin.info.name] = plugin

    def load_module(loading_plugin):
        def fail_activation():
            raise RuntimeError("activation failed")

        loading_plugin.module = SimpleNamespace(
            on_load=fail_activation,
            on_unload=lambda: None,
        )

    monkeypatch.setattr(manager, "_load_module", load_module)
    monkeypatch.setattr(
        CapabilityRegistry.instance(), "unregister_all_for_plugin", lambda _name: None
    )
    monkeypatch.setattr(PluginInstaller, "ensure_venv", lambda _self, *_args: True)
    monkeypatch.setattr(PluginInstaller, "install_dependencies", lambda _self, *_args: True)
    assert not manager.load(plugin.info.name)

    assert plugin.state == PluginState.ERROR
    assert plugin.localization_tokens == []
    assert ui.catalogs == {}


def test_locale_files_participate_in_hot_reload_tracking(tmp_path):
    from lfs_plugins.plugin import PluginInfo, PluginInstance, iter_plugin_watch_files
    from lfs_plugins.watcher import PluginWatcher

    source = tmp_path / "__init__.py"
    source.write_text("", encoding="utf-8")
    english = _write_catalog(tmp_path, "en", {"panel": {"title": "Title"}})
    plugin = PluginInstance(
        info=PluginInfo(name="example_plugin", version="0.1.0", path=tmp_path)
    )
    plugin.file_mtimes = {
        path: path.stat().st_mtime for path in iter_plugin_watch_files(tmp_path)
    }
    watcher = PluginWatcher(SimpleNamespace())

    assert not watcher._has_changes(plugin)
    english.unlink()
    assert watcher._has_changes(plugin)


def test_native_catalog_binding_round_trip(lf):
    key = "plugins.catalog-binding-test.panel.title"
    token = lf.ui._loc_register_catalog(
        "catalog-binding-test", "en", {"panel.title": "Plugin title"}
    )
    try:
        assert token > 0
        assert lf.ui.tr(key) == "Plugin title"
    finally:
        assert lf.ui._loc_unregister_catalog(token)

    assert lf.ui.tr(key) == key
