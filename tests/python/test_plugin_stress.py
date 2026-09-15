# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic stress coverage for plugin install and marketplace edges.

All plugin sources are generated as local zip archives.  The tests never call
GitHub, PyPI, or the production registry.  Cancellation tests use deterministic
fake transports and subprocess runners.
"""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import zipfile

import pytest


def _plugin_manifest(name: str, version: str) -> str:
    return f"""
[project]
name = "{name}"
version = "{version}"
description = "local stress plugin"
dependencies = []

[tool.lichtfeld]
auto_start = false
hot_reload = true
plugin_api = ">=1,<2"
lichtfeld_version = ">=0.4.2"
required_features = []
"""


def _archive(name: str, version: str, code: str) -> bytes:
    payload = BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{name}/pyproject.toml", _plugin_manifest(name, version))
        archive.writestr(f"{name}/__init__.py", code)
    return payload.getvalue()


@pytest.fixture
def plugin_runtime(tmp_path, monkeypatch, bypass_plugin_installer):
    """Give each test fresh manager/settings singletons and a private plugin dir."""
    from lfs_plugins.capabilities import CapabilityRegistry
    from lfs_plugins.manager import PluginManager
    from lfs_plugins.settings import SettingsManager

    old_manager = PluginManager._instance
    old_settings = SettingsManager._instance
    old_capabilities = CapabilityRegistry._instance
    PluginManager._instance = None
    SettingsManager._instance = None
    CapabilityRegistry._instance = None

    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    manager = PluginManager.instance()
    manager._plugins_dir = plugins_dir
    settings = SettingsManager.instance()
    settings._settings_dir = plugins_dir

    yield manager, plugins_dir, settings

    if manager._watcher:
        manager.stop_watcher()
    for name in list(manager._plugins):
        try:
            manager.unload(name)
        except Exception:
            pass
    for module_name in list(sys.modules):
        if module_name.startswith("lfs_plugins.stress_"):
            sys.modules.pop(module_name, None)
    PluginManager._instance = old_manager
    SettingsManager._instance = old_settings
    CapabilityRegistry._instance = old_capabilities


@pytest.fixture
def local_archive_preparer(plugin_runtime, monkeypatch, tmp_path):
    """Patch GitHub archive preparation to extract generated local bytes."""
    manager, plugins_dir, _settings = plugin_runtime
    from lfs_plugins.installer import PluginSourceInfo, extract_archive
    import lfs_plugins.manager as manager_module

    archives = {}
    archive_counter = 0

    def prepare(url, staging_parent, on_progress=None, should_cancel=None):
        nonlocal archive_counter
        if on_progress:
            on_progress(f"Downloading local archive for {url}")
        raw = archives[url]
        archive_counter += 1
        archive_path = tmp_path / f"archive-{archive_counter}.zip"
        archive_path.write_bytes(raw)
        staging = Path(tempfile.mkdtemp(prefix=".stress-", dir=staging_parent))
        try:
            extract_archive(archive_path, staging, should_cancel)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        finally:
            archive_path.unlink(missing_ok=True)
        owner, repo = "local", url.rsplit("/", 1)[-1]
        return staging, PluginSourceInfo(
            transport="archive",
            origin=url,
            github_url=f"https://github.com/{owner}/{repo}",
            owner=owner,
            repo=repo,
            archive_url=url,
        )

    monkeypatch.setattr(manager_module, "prepare_github_archive", prepare)
    return manager, plugins_dir, archives


def test_install_load_unload_uninstall_cycles(local_archive_preparer):
    manager, plugins_dir, archives = local_archive_preparer
    url = "github:local/stress_cycle"
    archives[url] = _archive(
        "stress_cycle",
        "1.0.0",
        "VERSION = '1.0.0'\n\ndef on_load():\n    pass\n",
    )

    from lfs_plugins import PluginState

    for _ in range(3):
        assert manager.install(url, auto_load=True) == "stress_cycle"
        assert manager.get_state("stress_cycle") == PluginState.ACTIVE
        assert manager.unload("stress_cycle")
        assert manager.get_state("stress_cycle") == PluginState.UNLOADED
        assert manager.uninstall("stress_cycle")
        assert not (plugins_dir / "stress_cycle").exists()
        assert manager.get_state("stress_cycle") is None


@pytest.mark.parametrize("phase", ["environment", "dependencies"])
def test_cancel_during_install_load_leaves_no_new_plugin(local_archive_preparer, monkeypatch, phase):
    manager, plugins_dir, archives = local_archive_preparer
    url = f"github:local/stress_cancel_{phase}"
    name = f"stress_cancel_{phase}"
    archives[url] = _archive(name, "1.0.0", "def on_load():\n    pass\n")
    from lfs_plugins.errors import PluginLoadCancelled
    from lfs_plugins.installer import PluginInstaller

    def cancel(*_args, **_kwargs):
        raise PluginLoadCancelled(f"cancelled during {phase}")

    method = "ensure_venv" if phase == "environment" else "install_dependencies"
    monkeypatch.setattr(PluginInstaller, method, cancel)

    with pytest.raises(PluginLoadCancelled):
        manager.install(url, auto_load=True)

    assert not (plugins_dir / name).exists()
    assert manager.get_state(name) is None


@pytest.mark.parametrize("transport", ["archive", "git"])
def test_install_cancellation_callback_is_supported(local_archive_preparer, monkeypatch, transport):
    manager, _plugins_dir, archives = local_archive_preparer
    url = "github:local/stress_cancel_transport"
    archives[url] = _archive("stress_cancel_transport", "1.0.0", "")

    if transport == "git":
        import lfs_plugins.manager as manager_module
        from lfs_plugins.errors import PluginLoadCancelled

        def fake_clone(_url, _plugins_dir, _on_progress=None, should_cancel=None):
            assert should_cancel is not None and should_cancel()
            raise PluginLoadCancelled("cancelled during git clone")

        monkeypatch.setattr(manager_module, "clone_from_url", fake_clone)

    from lfs_plugins.errors import PluginLoadCancelled

    with pytest.raises(PluginLoadCancelled):
        manager.install(
            url,
            auto_load=False,
            transport=transport,
            should_cancel=lambda: True,
        )


def test_concurrent_install_of_two_local_plugins(local_archive_preparer):
    manager, plugins_dir, archives = local_archive_preparer
    urls = ["github:local/stress_concurrent_a", "github:local/stress_concurrent_b"]
    for url in urls:
        name = url.rsplit("/", 1)[-1]
        archives[url] = _archive(name, "1.0.0", "def on_load():\n    pass\n")

    entered = threading.Barrier(2)
    import lfs_plugins.manager as manager_module

    original_prepare = manager_module.prepare_github_archive

    def overlapping_prepare(url, parent, on_progress=None):
        entered.wait(timeout=2)
        return original_prepare(url, parent, on_progress)

    # Keep the overlap deterministic while preserving the archive extraction
    # implementation used by the rest of this file.
    manager_module.prepare_github_archive = overlapping_prepare
    try:
        results = []
        errors = []

        def install(url):
            try:
                results.append(manager.install(url, auto_load=True))
            except Exception as exc:  # pragma: no cover - assertion reports it
                errors.append(exc)

        threads = [threading.Thread(target=install, args=(url,)) for url in urls]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        assert not errors
        assert sorted(results) == sorted(name.rsplit("/", 1)[-1] for name in urls)
        assert all((plugins_dir / name.rsplit("/", 1)[-1]).exists() for name in urls)
    finally:
        manager_module.prepare_github_archive = original_prepare


def test_update_newer_version_while_loaded(local_archive_preparer):
    manager, _plugins_dir, archives = local_archive_preparer
    url = "github:local/stress_update"
    archives[url] = _archive(
        "stress_update", "1.0.0", "VERSION = '1.0.0'\n\ndef on_load():\n    pass\n"
    )
    assert manager.install(url, auto_load=True) == "stress_update"

    archives[url] = _archive(
        "stress_update", "2.0.0", "VERSION = '2.0.0'\n\ndef on_load():\n    pass\n"
    )
    assert manager.update("stress_update")
    assert manager.get_info("stress_update").version == "2.0.0"
    assert manager.get_state("stress_update").value == "active"
    assert sys.modules["lfs_plugins.stress_update"].VERSION == "2.0.0"


def test_import_failure_has_error_and_no_half_installed_directory(local_archive_preparer):
    manager, plugins_dir, archives = local_archive_preparer
    url = "github:local/stress_import_error"
    archives[url] = _archive(
        "stress_import_error",
        "1.0.0",
        "raise RuntimeError('intentional stress import failure')\n",
    )

    from lfs_plugins.errors import PluginError

    with pytest.raises(PluginError, match="intentional stress import failure"):
        manager.install(url, auto_load=True)

    assert manager.get_state("stress_import_error") is None
    assert not (plugins_dir / "stress_import_error").exists()


def test_autostart_preference_round_trip_uses_panel_storage(plugin_runtime):
    _manager, plugins_dir, settings = plugin_runtime
    prefs = settings.get("stress_autostart")
    prefs.set("load_on_startup", True)

    # Recreate the singleton just as a new panel/application session would.
    from lfs_plugins.settings import SettingsManager

    SettingsManager._instance = None
    restored = SettingsManager.instance()
    restored._settings_dir = plugins_dir
    assert restored.get("stress_autostart").get("load_on_startup") is True
    assert (plugins_dir / "stress_autostart" / "settings.json").exists()


def _write_plugin_tree(plugin_dir: Path, name: str, version: str = "1.0.0") -> None:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "pyproject.toml").write_text(
        _plugin_manifest(name, version), encoding="utf-8"
    )
    (plugin_dir / "__init__.py").write_text("VERSION = 'test'\n", encoding="utf-8")


def _write_install_metadata(plugin_dir: Path) -> None:
    from lfs_plugins.installer import PluginSourceInfo, write_plugin_source_metadata

    write_plugin_source_metadata(
        plugin_dir,
        PluginSourceInfo(
            transport="archive",
            origin="github:local/hardened",
            github_url="https://github.com/local/hardened",
            owner="local",
            repo="hardened",
        ),
    )


def test_manifest_name_must_be_a_safe_path_component(plugin_runtime, tmp_path):
    manager, _plugins_dir, _settings = plugin_runtime
    invalid_names = [
        "",
        ".",
        "..",
        "../escape",
        "nested/name",
        "nested\\name",
        "name with spaces",
    ]

    for index, name in enumerate(invalid_names):
        staging = tmp_path / f"staging-{index}"
        _write_plugin_tree(staging, name)
        with pytest.raises(ValueError, match="Invalid plugin name"):
            manager._parse_manifest(staging)


def test_settings_delete_returns_and_persists(plugin_runtime):
    _manager, plugins_dir, _settings = plugin_runtime
    from lfs_plugins.settings import PluginSettings

    settings = PluginSettings("settings_plugin", plugins_dir)
    settings.set("remove_me", True)
    result = []
    thread = threading.Thread(target=lambda: result.append(settings.delete("remove_me")))
    thread.start()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert result == [True]
    assert settings.get("remove_me") is None


def test_update_restores_tree_and_venv_when_metadata_write_fails(plugin_runtime, monkeypatch):
    manager, plugins_dir, _settings = plugin_runtime
    import lfs_plugins.manager as manager_module

    plugin_dir = plugins_dir / "hardened"
    _write_plugin_tree(plugin_dir, "hardened", "1.0.0")
    (plugin_dir / ".venv").mkdir()
    (plugin_dir / ".venv" / "marker").write_text("old", encoding="utf-8")
    _write_install_metadata(plugin_dir)
    manager.pre_register(manager.discover())

    def prepare(_url, parent, _on_progress=None):
        staging = parent / ".hardened-update"
        _write_plugin_tree(staging, "hardened", "2.0.0")
        return staging, manager_module.PluginSourceInfo(
            transport="archive", origin="github:local/hardened"
        )

    monkeypatch.setattr(manager_module, "prepare_github_archive", prepare)

    def fail_metadata(*_args, **_kwargs):
        raise OSError("metadata disk full")

    monkeypatch.setattr(manager_module, "write_plugin_source_metadata", fail_metadata)
    with pytest.raises(Exception, match="metadata disk full"):
        manager.update("hardened")

    assert manager._parse_manifest(plugin_dir).version == "1.0.0"
    assert (plugin_dir / ".venv" / "marker").read_text(encoding="utf-8") == "old"
    assert not list(plugins_dir.glob(".hardened.backup-*"))


def test_update_aborts_when_unload_fails(plugin_runtime, monkeypatch):
    manager, plugins_dir, _settings = plugin_runtime
    import lfs_plugins.manager as manager_module
    from lfs_plugins import PluginState

    plugin_dir = plugins_dir / "hardened"
    _write_plugin_tree(plugin_dir, "hardened")
    _write_install_metadata(plugin_dir)
    manager.pre_register(manager.discover())
    manager._plugins["hardened"].state = PluginState.ACTIVE
    monkeypatch.setattr(manager, "unload", lambda _name: False)
    called = []
    monkeypatch.setattr(
        manager,
        "_update_archive_plugin_from_github",
        lambda *_args, **_kwargs: called.append(True),
    )

    with pytest.raises(Exception, match="unload failed"):
        manager.update("hardened")
    assert called == []
    assert (plugin_dir / "pyproject.toml").exists()


def test_uninstall_failure_keeps_manager_entry(plugin_runtime, monkeypatch):
    manager, plugins_dir, _settings = plugin_runtime
    import lfs_plugins.manager as manager_module

    plugin_dir = plugins_dir / "hardened"
    _write_plugin_tree(plugin_dir, "hardened")
    manager.pre_register(manager.discover())
    monkeypatch.setattr(
        manager_module,
        "uninstall_plugin",
        lambda _path: (_ for _ in ()).throw(OSError("locked")),
    )

    with pytest.raises(OSError, match="locked"):
        manager.uninstall("hardened")
    assert manager.get_info("hardened") is not None
    assert manager.get_state("hardened") is not None
    assert plugin_dir.exists()


def test_same_name_update_and_uninstall_are_serialized(plugin_runtime, monkeypatch):
    manager, plugins_dir, _settings = plugin_runtime
    import lfs_plugins.manager as manager_module

    plugin_dir = plugins_dir / "hardened"
    _write_plugin_tree(plugin_dir, "hardened", "1.0.0")
    _write_install_metadata(plugin_dir)
    manager.pre_register(manager.discover())
    entered = threading.Event()
    release = threading.Event()
    errors = []

    def prepare(_url, parent, _on_progress=None):
        entered.set()
        assert release.wait(timeout=2.0)
        staging = parent / ".hardened-update"
        _write_plugin_tree(staging, "hardened", "2.0.0")
        return staging, manager_module.PluginSourceInfo(
            transport="archive", origin="github:local/hardened"
        )

    monkeypatch.setattr(manager_module, "prepare_github_archive", prepare)

    update_thread = threading.Thread(target=lambda: manager.update("hardened"))
    update_thread.start()
    assert entered.wait(timeout=2.0)

    uninstall_done = threading.Event()

    def uninstall():
        try:
            manager.uninstall("hardened")
        except Exception as exc:
            errors.append(exc)
        finally:
            uninstall_done.set()

    uninstall_thread = threading.Thread(target=uninstall)
    uninstall_thread.start()
    time.sleep(0.05)
    assert not uninstall_done.is_set()
    release.set()
    update_thread.join(timeout=2.0)
    uninstall_thread.join(timeout=2.0)

    assert not errors
    assert not update_thread.is_alive()
    assert not uninstall_thread.is_alive()
    assert manager.get_state("hardened") is None
    assert not plugin_dir.exists()


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _size=-1):
        return self._body


def test_cancel_during_download_cleans_archive_temp(monkeypatch, tmp_path):
    from lfs_plugins import installer
    from lfs_plugins.errors import PluginLoadCancelled

    class SlowResponse(_FakeResponse):
        def __init__(self):
            super().__init__(b"partial archive")
            self.reads = 0

        def read(self, _size=-1):
            self.reads += 1
            return super().read(_size) if self.reads == 1 else b""

    response = SlowResponse()
    created_paths = []
    real_named_temporary_file = installer.tempfile.NamedTemporaryFile

    def tracking_named_temporary_file(*args, **kwargs):
        kwargs["dir"] = tmp_path
        tmp = real_named_temporary_file(*args, **kwargs)
        created_paths.append(Path(tmp.name))
        return tmp

    monkeypatch.setattr(installer, "urlopen", lambda *_args, **_kwargs: response)
    monkeypatch.setattr(installer.tempfile, "NamedTemporaryFile", tracking_named_temporary_file)

    with pytest.raises(PluginLoadCancelled, match="during download"):
        installer.prepare_archive_from_download_url(
            "https://example.test/plugin.zip",
            tmp_path,
            temp_prefix=".download-",
            should_cancel=lambda: response.reads >= 1,
        )

    assert created_paths and not created_paths[0].exists()
    assert not list(tmp_path.glob(".download-*"))


def test_cancel_during_extraction_removes_partial_staging(monkeypatch, tmp_path):
    from lfs_plugins import installer
    from lfs_plugins.errors import PluginLoadCancelled

    archive_path = tmp_path / "source.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("stress_extract/first.txt", "first")
        archive.writestr("stress_extract/second.txt", "second")

    checks = 0

    def should_cancel():
        nonlocal checks
        checks += 1
        return checks > 2

    monkeypatch.setattr(installer, "_download_url_to_temp", lambda *_args, **_kwargs: archive_path)
    with pytest.raises(PluginLoadCancelled, match="during extraction"):
        installer.prepare_archive_from_download_url(
            "https://example.test/plugin.zip",
            tmp_path,
            temp_prefix=".extract-",
            should_cancel=should_cancel,
        )

    assert not list(tmp_path.glob(".extract-*"))


def test_cancel_during_git_clone_removes_staging(monkeypatch, tmp_path):
    from lfs_plugins import installer
    from lfs_plugins.errors import PluginLoadCancelled

    monkeypatch.setattr(installer.shutil, "which", lambda _name: "/fake/git")
    checks = 0

    def should_cancel():
        nonlocal checks
        checks += 1
        return checks > 1

    def fake_process(_cmd, **kwargs):
        assert kwargs["should_cancel"]()
        raise PluginLoadCancelled("cancelled during git clone")

    monkeypatch.setattr(installer, "_run_cancellable_process", fake_process)
    with pytest.raises(PluginLoadCancelled, match="during git clone"):
        installer.clone_from_url(
            "github:local/stress_git_cancel",
            tmp_path,
            should_cancel=should_cancel,
        )

    assert not list(tmp_path.glob(".stress_git_cancel-*"))


def test_staging_failure_cleans_downloaded_archive(monkeypatch, tmp_path):
    from lfs_plugins import installer

    archive_path = tmp_path / "download.archive"
    archive_path.write_bytes(b"downloaded")
    monkeypatch.setattr(installer, "_download_url_to_temp", lambda *_args, **_kwargs: archive_path)

    def fail_mkdtemp(*_args, **_kwargs):
        raise OSError("staging parent unavailable")

    monkeypatch.setattr(installer.tempfile, "mkdtemp", fail_mkdtemp)
    with pytest.raises(OSError, match="staging parent unavailable"):
        installer.prepare_archive_from_download_url(
            "https://example.test/plugin.zip",
            tmp_path,
            temp_prefix=".failure-",
        )

    assert not archive_path.exists()


def test_marketplace_refresh_returns_during_slow_failing_github_enrichment(
    plugin_runtime, monkeypatch, tmp_path
):
    manager, _plugins_dir, _settings = plugin_runtime
    from lfs_plugins import marketplace
    from lfs_plugins import http as http_module
    from lfs_plugins.registry import RegistryClient

    index = {
        "plugins": [{
            "name": "stress_catalog",
            "namespace": "community",
            "display_name": "Stress Catalog",
            "summary": "local registry entry",
            "author": "tests",
            "latest_version": "1.0.0",
            "keywords": ["stress"],
            "repository": "https://github.com/local/stress_catalog",
        }]
    }
    detail = {
        "name": "stress_catalog",
        "namespace": "community",
        "latest_version": "1.0.0",
        "versions": {"1.0.0": {
            "version": "1.0.0",
            "plugin_api": ">=1,<2",
            "lichtfeld_version": ">=0.4.2",
            "required_features": [],
        }},
    }
    registry_root = tmp_path / "fake-registry"
    (registry_root / "plugins" / "community").mkdir(parents=True)
    (registry_root / "index.json").write_text(json.dumps(index))
    (registry_root / "plugins" / "community" / "stress_catalog.json").write_text(
        json.dumps(detail)
    )
    registry_base = "fake://stress-registry"
    registry = RegistryClient(cache_dir=tmp_path / "registry-cache")
    registry._registry_urls = (registry_base,)
    manager._registry = registry

    github_started = threading.Event()
    github_release = threading.Event()

    def fake_urlopen(request, *, timeout, **_kwargs):
        url = getattr(request, "full_url", str(request))
        if url.endswith("/index.json"):
            return _FakeResponse((registry_root / "index.json").read_bytes())
        if "/plugins/" in url:
            return _FakeResponse(
                (registry_root / "plugins" / "community" / "stress_catalog.json").read_bytes()
            )
        if "api.github.com/repos/" in url:
            github_started.set()
            github_release.wait(timeout=2)
            raise OSError("simulated GitHub rate limit/offline")
        raise AssertionError(f"unexpected test URL: {url}")

    # The production modules import the helper directly; patch both the public
    # helper and those aliases so this remains a no-network test.
    monkeypatch.setattr(http_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(sys.modules["lfs_plugins.registry"], "urlopen", fake_urlopen)
    monkeypatch.setattr(marketplace, "urlopen", fake_urlopen)
    monkeypatch.setattr(marketplace, "_catalog_cache", None)

    catalog = marketplace.PluginMarketplaceCatalog()
    started = time.monotonic()
    catalog.refresh_async(force=True, require_github_enrichment=True)
    returned_in = time.monotonic() - started
    assert returned_in < 0.25
    assert github_started.wait(timeout=2)

    started = time.monotonic()
    _entries, loading, _registry_loaded = catalog.snapshot()
    assert time.monotonic() - started < 0.25
    assert loading

    started = time.monotonic()
    catalog.refresh_async(force=True, require_github_enrichment=True)
    assert time.monotonic() - started < 0.25
    github_release.set()

    deadline = time.monotonic() + 3
    while catalog.snapshot()[1] and time.monotonic() < deadline:
        time.sleep(0.01)
    entries, loading, registry_loaded = catalog.snapshot()
    assert not loading
    assert registry_loaded
    assert any(entry.name == "Stress Catalog" for entry in entries)
