# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for hot reload reliability fixes.

Covers:
- Issue 1: Watch loop exception logging (watcher.py:56-57)
- Issue 2: Thread-safe plugin access via get_active_plugins_snapshot (watcher.py:62)
- Issue 3: Reload failure logging (watcher.py:146-147)
- Issue 4: Mtime granularity - hash-based sub-second change detection (watcher.py:78-83)
- Issue 5: OSError handling with proper logging (watcher.py:84-85)
- Issue 6: Hash cleanup on unload (watcher.py clear_plugin_hashes)
"""

import concurrent.futures
import logging
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def plugin_test_dir(monkeypatch):
    """Create temporary plugins directory for tests."""
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

        mgr.stop_watcher()

        for name in list(mgr._plugins.keys()):
            try:
                mgr.unload(name)
            except Exception:
                pass

        PluginManager._instance = original_instance


def create_plugin(plugin_dir: Path, name: str, code: str):
    """Create plugin with given code."""
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

    (plugin_dir / "__init__.py").write_text(code)


class TestWatchLoopExceptionLogging:
    """Issue 1: Watch loop should log exceptions, not swallow them."""

    def test_watch_loop_logs_exception(self, plugin_test_dir, caplog):
        """Verify exceptions in watch loop are logged with traceback."""
        from lfs_plugins import PluginManager
        from lfs_plugins.watcher import PluginWatcher

        create_plugin(
            plugin_test_dir / "watch_err",
            "watch_err",
            """
def on_load():
    pass
def on_unload():
    pass
""",
        )

        mgr = PluginManager.instance()
        mgr.load("watch_err")

        watcher = PluginWatcher(mgr, poll_interval=0.05)

        with patch.object(watcher, "_check_for_changes", side_effect=RuntimeError("Test error")):
            with caplog.at_level(logging.ERROR, logger="lfs_plugins.watcher"):
                watcher.start()
                time.sleep(0.15)
                watcher.stop()

        assert any("Watcher loop error" in r.message for r in caplog.records)
        assert any("Test error" in r.message for r in caplog.records)

        mgr.unload("watch_err")

    def test_watch_loop_continues_after_exception(self, plugin_test_dir):
        """Verify watch loop continues running after an exception."""
        from lfs_plugins import PluginManager
        from lfs_plugins.watcher import PluginWatcher

        create_plugin(
            plugin_test_dir / "recovery_test",
            "recovery_test",
            """
def on_load():
    pass
def on_unload():
    pass
""",
        )

        mgr = PluginManager.instance()
        mgr.load("recovery_test")

        watcher = PluginWatcher(mgr, poll_interval=0.05)
        call_count = [0]
        original_check = watcher._check_for_changes

        def failing_check():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise RuntimeError("Temporary failure")
            original_check()

        with patch.object(watcher, "_check_for_changes", side_effect=failing_check):
            watcher.start()
            time.sleep(0.3)
            watcher.stop()

        assert call_count[0] >= 3, "Watcher should continue after exceptions"
        mgr.unload("recovery_test")


class TestThreadSafePluginAccess:
    """Issue 2: Thread-safe plugin access via get_active_plugins_snapshot."""

    def test_snapshot_returns_copy(self, plugin_test_dir):
        """Verify snapshot returns a copy that doesn't reflect later changes."""
        from lfs_plugins import PluginManager, PluginState

        create_plugin(
            plugin_test_dir / "snap1",
            "snap1",
            "def on_load(): pass\ndef on_unload(): pass",
        )
        create_plugin(
            plugin_test_dir / "snap2",
            "snap2",
            "def on_load(): pass\ndef on_unload(): pass",
        )

        mgr = PluginManager.instance()
        mgr.load("snap1")
        mgr.load("snap2")

        snapshot = mgr.get_active_plugins_snapshot()
        assert len(snapshot) == 2

        mgr.unload("snap1")

        assert len(snapshot) == 2
        current_snapshot = mgr.get_active_plugins_snapshot()
        assert len(current_snapshot) == 1

        mgr.unload("snap2")

    def test_snapshot_only_active_plugins(self, plugin_test_dir):
        """Verify snapshot only includes ACTIVE plugins."""
        from lfs_plugins import PluginManager, PluginState

        create_plugin(
            plugin_test_dir / "active_test",
            "active_test",
            "def on_load(): pass\ndef on_unload(): pass",
        )

        mgr = PluginManager.instance()

        snapshot_before = mgr.get_active_plugins_snapshot()
        assert len(snapshot_before) == 0

        mgr.load("active_test")

        snapshot_active = mgr.get_active_plugins_snapshot()
        assert len(snapshot_active) == 1
        assert snapshot_active[0][0] == "active_test"

        mgr.unload("active_test")

        snapshot_after = mgr.get_active_plugins_snapshot()
        assert len(snapshot_after) == 0

    def test_concurrent_snapshot_access(self, plugin_test_dir):
        """Verify snapshot is safe under concurrent access."""
        from lfs_plugins import PluginManager

        for i in range(3):
            create_plugin(
                plugin_test_dir / f"conc_{i}",
                f"conc_{i}",
                "def on_load(): pass\ndef on_unload(): pass",
            )

        mgr = PluginManager.instance()
        for i in range(3):
            mgr.load(f"conc_{i}")

        errors = []
        results = []
        barrier = threading.Barrier(5)

        def snapshot_reader():
            try:
                barrier.wait(timeout=5.0)
                for _ in range(20):
                    snap = mgr.get_active_plugins_snapshot()
                    results.append(len(snap))
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            futures = [ex.submit(snapshot_reader) for _ in range(5)]
            concurrent.futures.wait(futures, timeout=10.0)

        assert not errors, f"Errors: {errors}"
        assert all(r <= 3 for r in results)

        for i in range(3):
            mgr.unload(f"conc_{i}")


class TestReloadFailureLogging:
    """Issue 3: Reload failures should be logged, not silently swallowed."""

    def test_failed_reload_logged(self, plugin_test_dir, caplog):
        """Verify reload failure is logged with error message."""
        from lfs_plugins import PluginManager, PluginState
        from lfs_plugins.watcher import PluginWatcher

        create_plugin(
            plugin_test_dir / "fail_reload",
            "fail_reload",
            "def on_load(): pass\ndef on_unload(): pass",
        )

        mgr = PluginManager.instance()
        mgr.load("fail_reload")

        (plugin_test_dir / "fail_reload" / "__init__.py").write_text("syntax error {{{")

        watcher = PluginWatcher(mgr, poll_interval=0.05)

        with caplog.at_level(logging.ERROR, logger="lfs_plugins.watcher"):
            watcher._pending_reloads.add("fail_reload")
            watcher._process_pending_reloads()

        assert any("Hot-reload failed" in r.message and "fail_reload" in r.message for r in caplog.records)

    def test_successful_reload_logged(self, plugin_test_dir, caplog):
        """Verify successful reload is logged."""
        from lfs_plugins import PluginManager
        from lfs_plugins.watcher import PluginWatcher

        create_plugin(
            plugin_test_dir / "good_reload",
            "good_reload",
            "VERSION = 1\ndef on_load(): pass\ndef on_unload(): pass",
        )

        mgr = PluginManager.instance()
        mgr.load("good_reload")

        (plugin_test_dir / "good_reload" / "__init__.py").write_text(
            "VERSION = 2\ndef on_load(): pass\ndef on_unload(): pass"
        )

        watcher = PluginWatcher(mgr, poll_interval=0.05)

        with caplog.at_level(logging.INFO, logger="lfs_plugins.watcher"):
            watcher._pending_reloads.add("good_reload")
            watcher._process_pending_reloads()

        assert any("Hot-reloaded plugin" in r.message and "good_reload" in r.message for r in caplog.records)

        mgr.unload("good_reload")

    def test_reload_exception_logged_with_traceback(self, plugin_test_dir, caplog):
        """Verify reload exceptions include traceback."""
        from lfs_plugins import PluginManager
        from lfs_plugins.watcher import PluginWatcher

        create_plugin(
            plugin_test_dir / "exc_reload",
            "exc_reload",
            "def on_load(): pass\ndef on_unload(): pass",
        )

        mgr = PluginManager.instance()
        mgr.load("exc_reload")

        watcher = PluginWatcher(mgr, poll_interval=0.05)

        with patch.object(mgr, "reload", side_effect=RuntimeError("Unexpected reload error")):
            with caplog.at_level(logging.ERROR, logger="lfs_plugins.watcher"):
                watcher._pending_reloads.add("exc_reload")
                watcher._process_pending_reloads()

        error_records = [r for r in caplog.records if "Hot-reload exception" in r.message]
        assert len(error_records) > 0
        assert "exc_reload" in error_records[0].message

        mgr.unload("exc_reload")


class TestHashBasedChangeDetection:
    """Issue 4: Sub-second changes detected via content hash."""

    def test_same_mtime_different_content_detected(self, plugin_test_dir):
        """Verify content change detected when mtime unchanged."""
        from lfs_plugins import PluginManager
        from lfs_plugins.watcher import PluginWatcher

        create_plugin(
            plugin_test_dir / "hash_test",
            "hash_test",
            "VERSION = 1\ndef on_load(): pass\ndef on_unload(): pass",
        )

        mgr = PluginManager.instance()
        mgr.load("hash_test")

        plugin = mgr._plugins["hash_test"]
        init_file = plugin_test_dir / "hash_test" / "__init__.py"

        watcher = PluginWatcher(mgr, poll_interval=0.05)

        watcher._content_changed("hash_test", init_file)

        init_file.write_text("VERSION = 2\ndef on_load(): pass\ndef on_unload(): pass")

        current_mtime = init_file.stat().st_mtime
        plugin.file_mtimes[init_file] = current_mtime

        changed = watcher._content_changed("hash_test", init_file)
        assert changed is True

        mgr.unload("hash_test")

    def test_same_content_not_flagged_as_changed(self, plugin_test_dir):
        """Verify identical content doesn't trigger false positive."""
        from lfs_plugins import PluginManager
        from lfs_plugins.watcher import PluginWatcher

        code = "VERSION = 1\ndef on_load(): pass\ndef on_unload(): pass"
        create_plugin(plugin_test_dir / "no_change", "no_change", code)

        mgr = PluginManager.instance()
        mgr.load("no_change")

        init_file = plugin_test_dir / "no_change" / "__init__.py"
        watcher = PluginWatcher(mgr, poll_interval=0.05)

        watcher._content_changed("no_change", init_file)

        changed = watcher._content_changed("no_change", init_file)
        assert changed is False

        mgr.unload("no_change")

    def test_first_hash_check_returns_false(self, plugin_test_dir):
        """Verify first hash check doesn't trigger reload (no previous hash)."""
        from lfs_plugins import PluginManager
        from lfs_plugins.watcher import PluginWatcher

        create_plugin(
            plugin_test_dir / "first_hash",
            "first_hash",
            "def on_load(): pass\ndef on_unload(): pass",
        )

        mgr = PluginManager.instance()
        mgr.load("first_hash")

        init_file = plugin_test_dir / "first_hash" / "__init__.py"
        watcher = PluginWatcher(mgr, poll_interval=0.05)

        changed = watcher._content_changed("first_hash", init_file)
        assert changed is False

        mgr.unload("first_hash")


class TestOSErrorHandling:
    """Issue 5: OSError handling with proper logging."""

    def test_file_deleted_race_condition(self, plugin_test_dir):
        """Verify FileNotFoundError during stat is handled (race condition)."""
        from lfs_plugins import PluginManager
        from lfs_plugins.watcher import PluginWatcher

        create_plugin(
            plugin_test_dir / "fnf_test",
            "fnf_test",
            "def on_load(): pass\ndef on_unload(): pass",
        )

        mgr = PluginManager.instance()
        mgr.load("fnf_test")

        plugin = mgr._plugins["fnf_test"]
        init_file = plugin_test_dir / "fnf_test" / "__init__.py"

        assert init_file in plugin.file_mtimes

        watcher = PluginWatcher(mgr, poll_interval=0.05)

        deleted = [False]
        original_stat = Path.stat

        def stat_that_deletes(path_self, *args, **kwargs):
            if not deleted[0] and path_self == init_file:
                deleted[0] = True
                path_self.unlink()
            return original_stat(path_self, *args, **kwargs)

        with patch.object(Path, "stat", stat_that_deletes):
            has_changes = watcher._has_changes(plugin)

        assert has_changes is True

    def test_permission_error_logged(self, plugin_test_dir, caplog):
        """Verify PermissionError is logged as warning."""
        from lfs_plugins import PluginManager
        from lfs_plugins.watcher import PluginWatcher

        create_plugin(
            plugin_test_dir / "perm_test",
            "perm_test",
            "def on_load(): pass\ndef on_unload(): pass",
        )

        mgr = PluginManager.instance()
        mgr.load("perm_test")

        plugin = mgr._plugins["perm_test"]

        watcher = PluginWatcher(mgr, poll_interval=0.05)

        original_stat = Path.stat

        def stat_that_raises_permission(path_self, *args, **kwargs):
            if "__init__.py" in str(path_self) and "perm_test" in str(path_self):
                raise PermissionError("Access denied")
            return original_stat(path_self, *args, **kwargs)

        with caplog.at_level(logging.WARNING, logger="lfs_plugins.watcher"):
            with patch.object(Path, "stat", stat_that_raises_permission):
                watcher._has_changes(plugin)

        assert any("Permission denied" in r.message for r in caplog.records)

        mgr.unload("perm_test")

    def test_generic_oserror_logged_at_debug(self, plugin_test_dir, caplog):
        """Verify generic OSError is logged at debug level."""
        from lfs_plugins import PluginManager
        from lfs_plugins.watcher import PluginWatcher

        create_plugin(
            plugin_test_dir / "os_err",
            "os_err",
            "def on_load(): pass\ndef on_unload(): pass",
        )

        mgr = PluginManager.instance()
        mgr.load("os_err")

        plugin = mgr._plugins["os_err"]

        watcher = PluginWatcher(mgr, poll_interval=0.05)

        original_stat = Path.stat

        def stat_that_raises_oserror(path_self, *args, **kwargs):
            if "__init__.py" in str(path_self) and "os_err" in str(path_self):
                raise OSError(5, "I/O error")
            return original_stat(path_self, *args, **kwargs)

        with caplog.at_level(logging.DEBUG, logger="lfs_plugins.watcher"):
            with patch.object(Path, "stat", stat_that_raises_oserror):
                watcher._has_changes(plugin)

        assert any("OSError" in r.message for r in caplog.records)

        mgr.unload("os_err")


class TestHashCleanupOnUnload:
    """Issue 6: Hash cleanup when plugin is unloaded."""

    def test_hashes_cleared_on_unload(self, plugin_test_dir):
        """Verify plugin hashes are cleared when plugin is unloaded."""
        from lfs_plugins import PluginManager
        from lfs_plugins.watcher import PluginWatcher

        create_plugin(
            plugin_test_dir / "clear_hash",
            "clear_hash",
            "def on_load(): pass\ndef on_unload(): pass",
        )

        mgr = PluginManager.instance()

        watcher = PluginWatcher(mgr, poll_interval=1.0)
        mgr._watcher = watcher

        mgr.load("clear_hash")

        init_file = plugin_test_dir / "clear_hash" / "__init__.py"
        watcher._content_changed("clear_hash", init_file)

        assert "clear_hash" in watcher._file_hashes

        mgr.unload("clear_hash")

        assert "clear_hash" not in watcher._file_hashes

        mgr._watcher = None

    def test_clear_nonexistent_hash_is_safe(self, plugin_test_dir):
        """Verify clearing hash for unknown plugin doesn't crash."""
        from lfs_plugins.watcher import PluginWatcher
        from lfs_plugins import PluginManager

        mgr = PluginManager.instance()
        watcher = PluginWatcher(mgr, poll_interval=1.0)

        watcher.clear_plugin_hashes("nonexistent_plugin")

    def test_hashes_isolated_between_plugins(self, plugin_test_dir):
        """Verify hash clearing only affects target plugin."""
        from lfs_plugins import PluginManager
        from lfs_plugins.watcher import PluginWatcher

        create_plugin(
            plugin_test_dir / "iso1",
            "iso1",
            "def on_load(): pass\ndef on_unload(): pass",
        )
        create_plugin(
            plugin_test_dir / "iso2",
            "iso2",
            "def on_load(): pass\ndef on_unload(): pass",
        )

        mgr = PluginManager.instance()

        watcher = PluginWatcher(mgr, poll_interval=1.0)
        mgr._watcher = watcher

        mgr.load("iso1")
        mgr.load("iso2")

        init1 = plugin_test_dir / "iso1" / "__init__.py"
        init2 = plugin_test_dir / "iso2" / "__init__.py"
        watcher._content_changed("iso1", init1)
        watcher._content_changed("iso2", init2)

        assert "iso1" in watcher._file_hashes
        assert "iso2" in watcher._file_hashes

        mgr.unload("iso1")

        assert "iso1" not in watcher._file_hashes
        assert "iso2" in watcher._file_hashes

        mgr.unload("iso2")
        mgr._watcher = None


class TestRLockForReentrantCalls:
    """Test that RLock allows reentrant locking (uninstall calls unload)."""

    def test_uninstall_calls_unload_without_deadlock(self, plugin_test_dir):
        """Verify uninstall can call unload without deadlocking."""
        from lfs_plugins import PluginManager, PluginState

        create_plugin(
            plugin_test_dir / "uninstall_test",
            "uninstall_test",
            "def on_load(): pass\ndef on_unload(): pass",
        )

        mgr = PluginManager.instance()
        mgr.load("uninstall_test")

        assert mgr.get_state("uninstall_test") == PluginState.ACTIVE

        result = mgr.uninstall("uninstall_test")
        assert result is True
        assert "uninstall_test" not in mgr._plugins

    def test_concurrent_uninstall_safe(self, plugin_test_dir):
        """Verify concurrent uninstall attempts are handled safely."""
        from lfs_plugins import PluginManager

        for i in range(3):
            create_plugin(
                plugin_test_dir / f"uninst_{i}",
                f"uninst_{i}",
                "def on_load(): pass\ndef on_unload(): pass",
            )
            PluginManager.instance().load(f"uninst_{i}")

        mgr = PluginManager.instance()
        errors = []
        barrier = threading.Barrier(3)

        def uninstaller(name):
            try:
                barrier.wait(timeout=5.0)
                mgr.uninstall(name)
            except Exception as e:
                errors.append((name, e))

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            futures = [ex.submit(uninstaller, f"uninst_{i}") for i in range(3)]
            concurrent.futures.wait(futures, timeout=10.0)

        assert not errors, f"Errors: {errors}"
