# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for plugin system thread safety.

Targets:
- manager.py:62 - _plugins dict accessed without lock in watcher
- watcher.py:72-82 - file_mtimes accessed without synchronization
- State transition races (LOADING → ACTIVE interrupted)
"""

import concurrent.futures
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest


def create_plugin_files(plugin_dir: Path, name: str, delay: float = 0.0):
    """Create a minimal plugin with optional artificial delay in on_load."""
    plugin_dir.mkdir(exist_ok=True)

    (plugin_dir / "pyproject.toml").write_text(
        f"""
[project]
name = "{name}"
version = "1.0.0"
description = ""

[tool.lichtfeld]
auto_start = false
hot_reload = true
plugin_api = ">=1,<2"
lichtfeld_version = ">=0.4.2"
required_features = []
"""
    )

    delay_code = f"import time; time.sleep({delay})" if delay > 0 else ""
    (plugin_dir / "__init__.py").write_text(
        f"""
LOAD_COUNT = 0
UNLOAD_COUNT = 0

def on_load():
    global LOAD_COUNT
    {delay_code}
    LOAD_COUNT += 1

def on_unload():
    global UNLOAD_COUNT
    UNLOAD_COUNT += 1
"""
    )


@pytest.fixture
def thread_plugins_dir(monkeypatch):
    """Create temporary plugins directory for concurrency tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plugins_dir = Path(tmpdir) / "plugins"
        plugins_dir.mkdir()

        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from lfs_plugins.manager import PluginManager

        original_instance = PluginManager._instance
        PluginManager._instance = None

        mgr = PluginManager.instance()
        mgr._plugins_dir = plugins_dir

        yield plugins_dir

        # Cleanup
        for name in list(mgr._plugins.keys()):
            try:
                mgr.unload(name)
            except Exception:
                pass

        PluginManager._instance = original_instance


class TestPluginConcurrency:
    """Tests for plugin system thread safety."""

    def test_concurrent_load_unload(self, thread_plugins_dir):
        """Multiple threads loading and unloading different plugins simultaneously."""
        from lfs_plugins import PluginManager, PluginState

        N_PLUGINS = 4
        N_ITERATIONS = 3

        for i in range(N_PLUGINS):
            create_plugin_files(thread_plugins_dir / f"plugin_{i}", f"plugin_{i}")

        mgr = PluginManager.instance()
        errors = []
        barrier = threading.Barrier(N_PLUGINS)

        def worker(plugin_name):
            try:
                barrier.wait(timeout=5.0)
                for _ in range(N_ITERATIONS):
                    mgr.load(plugin_name)
                    assert mgr.get_state(plugin_name) == PluginState.ACTIVE
                    mgr.unload(plugin_name)
                    assert mgr.get_state(plugin_name) == PluginState.UNLOADED
            except Exception as e:
                errors.append((plugin_name, e))

        with concurrent.futures.ThreadPoolExecutor(max_workers=N_PLUGINS) as ex:
            futures = [ex.submit(worker, f"plugin_{i}") for i in range(N_PLUGINS)]
            concurrent.futures.wait(futures, timeout=30.0)

        assert not errors, f"Errors occurred: {errors}"

    def test_load_during_unload(self, thread_plugins_dir):
        """Load same plugin while it's being unloaded."""
        from lfs_plugins import PluginManager, PluginState

        create_plugin_files(thread_plugins_dir / "race_plugin", "race_plugin", delay=0.1)

        mgr = PluginManager.instance()
        mgr.load("race_plugin")

        errors = []
        load_result = None
        unload_done = threading.Event()

        def unloader():
            try:
                mgr.unload("race_plugin")
                unload_done.set()
            except Exception as e:
                errors.append(("unloader", e))

        def loader():
            try:
                time.sleep(0.02)
                nonlocal load_result
                load_result = mgr.load("race_plugin")
            except Exception as e:
                errors.append(("loader", e))

        threads = [threading.Thread(target=unloader), threading.Thread(target=loader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert not errors, f"Errors occurred: {errors}"
        state = mgr.get_state("race_plugin")
        assert state in (PluginState.ACTIVE, PluginState.UNLOADED)

        if state == PluginState.ACTIVE:
            mgr.unload("race_plugin")

    def test_unload_during_load(self, thread_plugins_dir):
        """Unload plugin while it's loading."""
        from lfs_plugins import PluginManager, PluginState

        create_plugin_files(thread_plugins_dir / "slow_plugin", "slow_plugin", delay=0.2)

        mgr = PluginManager.instance()
        errors = []
        load_started = threading.Event()

        def loader():
            try:
                load_started.set()
                mgr.load("slow_plugin")
            except Exception as e:
                errors.append(("loader", e))

        def unloader():
            try:
                load_started.wait(timeout=2.0)
                time.sleep(0.05)
                mgr.unload("slow_plugin")
            except Exception as e:
                errors.append(("unloader", e))

        threads = [threading.Thread(target=loader), threading.Thread(target=unloader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        state = mgr.get_state("slow_plugin")
        assert state in (PluginState.ACTIVE, PluginState.UNLOADED, PluginState.LOADING, None)

        if state == PluginState.ACTIVE:
            mgr.unload("slow_plugin")

    def test_reload_during_reload(self, thread_plugins_dir):
        """Multiple threads trying to reload same plugin."""
        from lfs_plugins import PluginManager, PluginState

        create_plugin_files(thread_plugins_dir / "reload_plugin", "reload_plugin", delay=0.05)

        mgr = PluginManager.instance()
        mgr.load("reload_plugin")

        N_THREADS = 4
        errors = []
        barrier = threading.Barrier(N_THREADS)

        def reloader():
            try:
                barrier.wait(timeout=5.0)
                mgr.reload("reload_plugin")
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS) as ex:
            futures = [ex.submit(reloader) for _ in range(N_THREADS)]
            concurrent.futures.wait(futures, timeout=30.0)

        assert mgr.get_state("reload_plugin") in (PluginState.ACTIVE, PluginState.ERROR)
        if mgr.get_state("reload_plugin") == PluginState.ACTIVE:
            mgr.unload("reload_plugin")

    def test_watcher_race_with_unload(self, thread_plugins_dir):
        """Watcher checking file while plugin is unloading."""
        from lfs_plugins import PluginManager, PluginState
        from lfs_plugins.watcher import PluginWatcher

        create_plugin_files(thread_plugins_dir / "watch_plugin", "watch_plugin")

        mgr = PluginManager.instance()
        mgr.load("watch_plugin")

        watcher = PluginWatcher(mgr, poll_interval=0.01)
        watcher.start()

        errors = []

        def unload_reload():
            for _ in range(5):
                try:
                    if mgr.get_state("watch_plugin") == PluginState.ACTIVE:
                        mgr.unload("watch_plugin")
                    time.sleep(0.02)
                    mgr.load("watch_plugin")
                    time.sleep(0.02)
                except Exception as e:
                    errors.append(e)

        thread = threading.Thread(target=unload_reload)
        thread.start()
        thread.join(timeout=5.0)

        watcher.stop()

        assert not errors, f"Errors occurred: {errors}"
        if mgr.get_state("watch_plugin") == PluginState.ACTIVE:
            mgr.unload("watch_plugin")

    def test_file_mtime_race_condition(self, thread_plugins_dir):
        """Access file_mtimes dict while watcher is checking."""
        from lfs_plugins import PluginManager, PluginState
        from lfs_plugins.watcher import PluginWatcher

        create_plugin_files(thread_plugins_dir / "mtime_plugin", "mtime_plugin")

        mgr = PluginManager.instance()
        mgr.load("mtime_plugin")

        watcher = PluginWatcher(mgr, poll_interval=0.01)
        watcher.start()

        errors = []

        def modify_files():
            plugin = mgr._plugins.get("mtime_plugin")
            if not plugin:
                return
            for _ in range(10):
                try:
                    # Modify files to trigger mtime checks
                    init_file = thread_plugins_dir / "mtime_plugin" / "__init__.py"
                    content = init_file.read_text()
                    init_file.write_text(content + "\n# comment")
                    time.sleep(0.02)
                except Exception as e:
                    errors.append(e)

        thread = threading.Thread(target=modify_files)
        thread.start()
        thread.join(timeout=5.0)

        watcher.stop()

        assert not errors, f"Errors occurred: {errors}"
        if mgr.get_state("mtime_plugin") == PluginState.ACTIVE:
            mgr.unload("mtime_plugin")

    def test_concurrent_discover(self, thread_plugins_dir):
        """Multiple threads calling discover() simultaneously."""
        from lfs_plugins import PluginManager

        for i in range(5):
            create_plugin_files(thread_plugins_dir / f"disc_plugin_{i}", f"disc_plugin_{i}")

        mgr = PluginManager.instance()
        results = []
        errors = []
        barrier = threading.Barrier(4)

        def discoverer():
            try:
                barrier.wait(timeout=5.0)
                plugins = mgr.discover()
                results.append(len(plugins))
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(discoverer) for _ in range(4)]
            concurrent.futures.wait(futures, timeout=10.0)

        assert not errors, f"Errors occurred: {errors}"
        assert all(r == 5 for r in results), f"Inconsistent discover results: {results}"

    def test_callback_concurrent_modification(self, thread_plugins_dir):
        """Modify callback list while callbacks are being invoked."""
        from lfs_plugins import PluginManager, PluginState

        create_plugin_files(thread_plugins_dir / "cb_plugin", "cb_plugin")

        mgr = PluginManager.instance()
        callback_count = [0]
        errors = []

        def callback(info):
            callback_count[0] += 1
            # Try to add more callbacks during invocation
            mgr.on_plugin_loaded(lambda i: None)

        mgr.on_plugin_loaded(callback)

        def loader():
            try:
                for _ in range(3):
                    if mgr.get_state("cb_plugin") == PluginState.ACTIVE:
                        mgr.unload("cb_plugin")
                    mgr.load("cb_plugin")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=loader) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert not errors, f"Errors occurred: {errors}"
        if mgr.get_state("cb_plugin") == PluginState.ACTIVE:
            mgr.unload("cb_plugin")
