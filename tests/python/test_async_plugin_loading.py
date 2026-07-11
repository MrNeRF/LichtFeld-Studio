# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic coverage for cancellable and transactional plugin loading."""

import builtins
import concurrent.futures
from pathlib import Path
import sys
import threading
import time
from unittest.mock import patch

import pytest


def _write_plugin(plugin_dir: Path, name: str, code: str) -> None:
    plugin_dir.mkdir()
    (plugin_dir / "pyproject.toml").write_text(
        f"""
[project]
name = "{name}"
version = "1.0.0"
description = "Async loading test plugin"
dependencies = []

[tool.lichtfeld]
auto_start = false
hot_reload = false
plugin_api = ">=1,<2"
lichtfeld_version = ">=0.4.2"
required_features = []
"""
    )
    (plugin_dir / "__init__.py").write_text(code)


@pytest.fixture
def async_plugin_manager(tmp_path):
    from lfs_plugins.capabilities import CapabilityRegistry
    from lfs_plugins.manager import PluginManager

    original_manager = PluginManager._instance
    original_capabilities = CapabilityRegistry._instance
    PluginManager._instance = None
    CapabilityRegistry._instance = None

    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    manager = PluginManager.instance()
    manager._plugins_dir = plugins_dir

    yield manager, plugins_dir

    for name in list(manager._plugins):
        manager.unload(name)
    for module_name in [
        name
        for name in sys.modules
        if name.startswith("lfs_plugins.async_test_")
    ]:
        sys.modules.pop(module_name, None)
    PluginManager._instance = original_manager
    CapabilityRegistry._instance = original_capabilities


@pytest.fixture
def bypass_plugin_installer():
    with patch(
        "lfs_plugins.installer.PluginInstaller.ensure_venv", return_value=True
    ), patch(
        "lfs_plugins.installer.PluginInstaller.install_dependencies",
        return_value=True,
    ):
        yield


def test_load_reports_structured_stages(async_plugin_manager, bypass_plugin_installer):
    manager, plugins_dir = async_plugin_manager
    name = "async_test_stages"
    _write_plugin(plugins_dir / name, name, "def on_load():\n    pass\n")
    stages = []

    assert manager.load(name, on_stage=lambda phase, detail: stages.append((phase, detail)))

    assert [phase for phase, _ in stages] == [
        "environment",
        "dependencies",
        "import",
        "activation",
        "complete",
    ]
    assert all(name in detail for _, detail in stages)


def test_cancel_after_import_rolls_back_module(
    async_plugin_manager, bypass_plugin_installer
):
    from lfs_plugins import PluginLoadCancelled, PluginState

    manager, plugins_dir = async_plugin_manager
    name = "async_test_cancel"
    _write_plugin(plugins_dir / name, name, "VALUE = 42\n")
    cancel = threading.Event()

    def on_stage(phase, _detail):
        if phase == "import":
            cancel.set()

    with pytest.raises(PluginLoadCancelled):
        manager.load(name, on_stage=on_stage, should_cancel=cancel.is_set)

    assert manager.get_state(name) == PluginState.UNLOADED
    assert manager.get_error(name) is None
    assert f"lfs_plugins.{name}" not in sys.modules
    assert str(plugins_dir / name) not in sys.path


def test_activation_failure_rolls_back_registrations(
    async_plugin_manager, bypass_plugin_installer, monkeypatch
):
    from lfs_plugins import CapabilityRegistry, PluginState

    manager, plugins_dir = async_plugin_manager
    name = "async_test_rollback"
    marker = "ASYNC_PLUGIN_ROLLBACK_CALLED"
    monkeypatch.delattr(builtins, marker, raising=False)
    _write_plugin(
        plugins_dir / name,
        name,
        f"""
import builtins
from lfs_plugins import CapabilityRegistry

def on_load():
    CapabilityRegistry.instance().register(
        "async.rollback", lambda args, ctx: {{}}, plugin_name="{name}"
    )
    raise RuntimeError("activation failed")

def on_unload():
    builtins.{marker} = True
""",
    )

    assert manager.load(name) is False

    assert manager.get_state(name) == PluginState.ERROR
    assert "activation failed" in manager.get_error(name)
    assert not CapabilityRegistry.instance().has("async.rollback")
    assert f"lfs_plugins.{name}" not in sys.modules
    assert str(plugins_dir / name) not in sys.path
    assert getattr(builtins, marker) is True


def test_concurrent_loads_share_one_activation(
    async_plugin_manager, bypass_plugin_installer
):
    manager, plugins_dir = async_plugin_manager
    name = "async_test_concurrent"
    marker = "ASYNC_PLUGIN_LOAD_COUNT"
    if hasattr(builtins, marker):
        delattr(builtins, marker)
    _write_plugin(
        plugins_dir / name,
        name,
        f"""
import builtins
import time

def on_load():
    time.sleep(0.05)
    builtins.{marker} = getattr(builtins, "{marker}", 0) + 1
""",
    )
    barrier = threading.Barrier(2)
    results = []

    def load_plugin():
        barrier.wait(timeout=2)
        results.append(manager.load(name))

    threads = [threading.Thread(target=load_plugin) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert results == [True, True]
    assert getattr(builtins, marker) == 1
    delattr(builtins, marker)


def test_module_execution_never_replaces_process_import_hook(
    async_plugin_manager, bypass_plugin_installer, monkeypatch
):
    manager, plugins_dir = async_plugin_manager
    name = "async_test_import_owner"
    _write_plugin(
        plugins_dir / name,
        name,
        "import time\ntime.sleep(0.1)\ndef on_load():\n    pass\n",
    )
    original_import = builtins.__import__

    def owner_import(*args, **kwargs):
        return original_import(*args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", owner_import)
    thread = threading.Thread(target=lambda: manager.load(name))
    thread.start()
    observed = []
    while thread.is_alive():
        observed.append(builtins.__import__)
        time.sleep(0.002)
    thread.join(timeout=1)

    assert observed
    assert all(import_hook is owner_import for import_hook in observed)


def test_cancellable_process_streams_output_and_stops_promptly():
    from lfs_plugins.errors import PluginLoadCancelled
    from lfs_plugins.installer import _run_cancellable_process

    cancel = threading.Event()
    output = []

    def on_output(line):
        output.append(line)
        if line == "ready":
            cancel.set()

    started = time.monotonic()
    with pytest.raises(PluginLoadCancelled):
        _run_cancellable_process(
            [
                sys.executable,
                "-c",
                "import time; print('ready', flush=True); time.sleep(30)",
            ],
            on_output=on_output,
            should_cancel=cancel.is_set,
        )

    assert output == ["ready"]
    assert time.monotonic() - started < 2.0


def test_process_group_configuration_is_platform_specific(monkeypatch):
    from lfs_plugins import installer

    monkeypatch.setattr(installer.sys, "platform", "linux")
    assert installer._process_group_popen_kwargs() == {"start_new_session": True}

    monkeypatch.setattr(installer.sys, "platform", "win32")
    kwargs = installer._process_group_popen_kwargs()
    assert kwargs["creationflags"] == getattr(
        installer.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200
    )


def test_startup_coordinator_status_is_python_visible():
    import lichtfeld as lf

    status = lf.plugins.startup_load_status()

    assert set(status) == {
        "state",
        "phase",
        "plugin",
        "detail",
        "attempted",
        "total",
        "failed",
        "progress",
        "active",
    }
    assert status["state"] in {
        "not_started",
        "discovering",
        "loading",
        "completed",
        "cancelled",
    }
    assert status["phase"] in {
        "idle",
        "environment",
        "dependencies",
        "import",
        "activation",
    }
    assert 0.0 <= status["progress"] <= 1.0
    assert status["attempted"] <= status["total"]


def test_icon_texture_creation_rejects_worker_thread():
    import lichtfeld as lf

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            lf.ui.load_icon, "__async_plugin_affinity_guard__.png"
        )
        with pytest.raises(RuntimeError, match="graphics thread"):
            future.result(timeout=5)


def test_cancelled_dependency_sync_never_writes_stamp(tmp_path, monkeypatch):
    from lfs_plugins.errors import PluginLoadCancelled
    from lfs_plugins.installer import PluginInstaller
    from lfs_plugins.plugin import PluginInfo, PluginInstance

    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "pyproject.toml").write_text("[project]\nname='cancelled'\n")
    venv = plugin_dir / ".venv"
    (venv / "bin").mkdir(parents=True)
    venv_python = venv / "bin" / "python"
    venv_python.touch()
    plugin = PluginInstance(
        info=PluginInfo(name="cancelled", version="1", path=plugin_dir)
    )
    plugin.venv_path = venv
    installer = PluginInstaller(plugin)
    uv = tmp_path / "uv"
    uv.touch()

    monkeypatch.setattr(installer, "_require_bundled_python", lambda: Path(sys.executable))
    monkeypatch.setattr(installer, "_find_uv", lambda: uv)

    def cancelled(*_args, **_kwargs):
        raise PluginLoadCancelled("cancelled")

    installer_module = sys.modules[PluginInstaller.__module__]
    monkeypatch.setattr(installer_module, "_run_cancellable_process", cancelled)

    with pytest.raises(PluginLoadCancelled):
        installer.install_dependencies()
    assert not installer._deps_stamp_path().exists()
