# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for icon cache.

Targets:
- py_ui.cpp:154-191 - Icon loading without GL context
- py_ui.cpp:297-304 - Concurrent access to icon cache
"""

import concurrent.futures
import os
import sys
import tempfile
import threading
from pathlib import Path

import pytest

_has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


@pytest.mark.skipif(not _has_display, reason="No display available for GL context")
class TestIconCache:
    """Tests for icon cache edge cases."""

    def test_icon_load_without_gl_context(self, lf):
        """Icon loading without OpenGL context should be handled."""
        # Without GUI, GL context is not available
        try:
            # This would require an icon path that exists
            result = lf.ui.load_icon("nonexistent_icon.png")
        except (RuntimeError, FileNotFoundError, OSError):
            pass  # Expected without GL context

    def test_load_invalid_icon_path(self, lf):
        """Loading icon from invalid path."""
        try:
            result = lf.ui.load_icon("/nonexistent/path/icon.png")
        except (FileNotFoundError, RuntimeError, OSError):
            pass  # Expected

    def test_load_icon_empty_name(self, lf):
        """Loading icon with empty name."""
        try:
            result = lf.ui.load_icon("")
        except (ValueError, FileNotFoundError, RuntimeError):
            pass  # Expected

    def test_concurrent_icon_operations(self, lf):
        """Concurrent icon cache operations should be thread-safe."""
        errors = []
        barrier = threading.Barrier(4)

        def load_icons():
            try:
                barrier.wait(timeout=5.0)
                for i in range(10):
                    try:
                        lf.ui.load_icon(f"icon_{i}.png")
                    except (FileNotFoundError, RuntimeError):
                        pass  # Expected - icons don't exist
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(load_icons) for _ in range(4)]
            concurrent.futures.wait(futures, timeout=30.0)

        # Only check for thread safety errors, not file not found
        critical_errors = [e for e in errors if not isinstance(e, FileNotFoundError)]
        assert not critical_errors, f"Thread safety errors: {critical_errors}"


@pytest.fixture
def plugin_icons_dir(monkeypatch):
    """Create temporary plugins directory for icon tests."""
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

        for name in list(mgr._plugins.keys()):
            try:
                mgr.unload(name)
            except Exception:
                pass

        PluginManager._instance = original_instance


@pytest.mark.skipif(not _has_display, reason="No display available for GL context")
class TestPluginIconCleanup:
    """Tests for plugin icon cleanup."""

    def test_plugin_icon_cleanup(self, plugin_icons_dir, lf):
        """Plugin icons should be cleaned up on unload."""
        from lfs_plugins import PluginManager, PluginState

        plugin_dir = plugin_icons_dir / "icon_plugin"
        plugin_dir.mkdir()

        (plugin_dir / "pyproject.toml").write_text(
            """
[project]
name = "icon_plugin"
version = "1.0.0"
description = ""

[tool.lichtfeld]
auto_start = false
hot_reload = true
"""
        )

        (plugin_dir / "__init__.py").write_text(
            """
def on_load():
    pass

def on_unload():
    pass
"""
        )

        mgr = PluginManager.instance()
        mgr.load("icon_plugin")

        # free_plugin_icons is called on unload
        mgr.unload("icon_plugin")

        # Should not crash


@pytest.mark.skipif(not _has_display, reason="No display available for GL context")
class TestIconCacheMemory:
    """Tests for icon cache memory management."""

    def test_repeated_load_same_icon(self, lf):
        """Repeatedly loading same icon should use cache."""
        for _ in range(100):
            try:
                lf.ui.load_icon("some_icon.png")
            except (FileNotFoundError, RuntimeError):
                pass  # Expected - icon doesn't exist

        # Should not leak memory (can't easily verify, but no crash)

    def test_load_many_different_icons(self, lf):
        """Loading many different icons."""
        for i in range(50):
            try:
                lf.ui.load_icon(f"icon_{i}.png")
            except (FileNotFoundError, RuntimeError):
                pass  # Expected

        # Should handle without crash
