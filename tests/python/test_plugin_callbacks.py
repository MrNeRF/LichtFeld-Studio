# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for plugin lifecycle callbacks.

Targets:
- manager.py:145-162 - Callback may unload plugin during load
- manager.py:210-245 - Unload callback may trigger reload
"""

import sys
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def callback_plugins_dir(monkeypatch):
    """Create temporary plugins directory for callback tests."""
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

        # Clear callbacks
        mgr._on_plugin_loaded.clear()
        mgr._on_plugin_unloaded.clear()

        PluginManager._instance = original_instance


def create_simple_plugin(plugin_dir: Path, name: str, extra_code: str = ""):
    """Create a simple plugin."""
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

    (plugin_dir / "__init__.py").write_text(
        f"""
def on_load():
    pass

def on_unload():
    pass

{extra_code}
"""
    )


class TestPluginCallbacks:
    """Tests for plugin load/unload callbacks."""

    def test_unload_in_on_load_callback(self, callback_plugins_dir):
        """Callback that unloads plugin during its load."""
        from lfs_plugins import PluginManager, PluginState

        create_simple_plugin(callback_plugins_dir / "victim", "victim")

        mgr = PluginManager.instance()

        def evil_callback(info):
            if info.name == "victim":
                mgr.unload("victim")

        mgr.on_plugin_loaded(evil_callback)

        # This creates a tricky situation - plugin loaded then immediately unloaded
        mgr.load("victim")

        # State may be ACTIVE or UNLOADED depending on timing
        state = mgr.get_state("victim")
        assert state in (PluginState.ACTIVE, PluginState.UNLOADED)

        if state == PluginState.ACTIVE:
            mgr.unload("victim")

    def test_reload_in_on_unload_callback(self, callback_plugins_dir):
        """Callback that reloads plugin during its unload."""
        from lfs_plugins import PluginManager, PluginState

        create_simple_plugin(callback_plugins_dir / "phoenix", "phoenix")

        mgr = PluginManager.instance()
        reload_count = [0]

        def reload_callback(info):
            if info.name == "phoenix" and reload_count[0] < 2:
                reload_count[0] += 1
                mgr.reload("phoenix")

        mgr.on_plugin_unloaded(reload_callback)
        mgr.load("phoenix")
        mgr.unload("phoenix")

        # May have reloaded, final state depends on implementation
        # Should not crash or infinite loop

        # Cleanup
        if mgr.get_state("phoenix") == PluginState.ACTIVE:
            # Remove callback first to prevent recursion
            mgr._on_plugin_unloaded.clear()
            mgr.unload("phoenix")

    def test_load_same_plugin_in_callback(self, callback_plugins_dir):
        """Callback that loads same plugin again."""
        from lfs_plugins import PluginManager, PluginState

        create_simple_plugin(callback_plugins_dir / "recursive", "recursive")

        mgr = PluginManager.instance()
        load_count = [0]

        def load_again(info):
            if info.name == "recursive" and load_count[0] < 2:
                load_count[0] += 1
                mgr.load("recursive")

        mgr.on_plugin_loaded(load_again)
        mgr.load("recursive")

        # Should handle without infinite recursion
        assert mgr.get_state("recursive") == PluginState.ACTIVE

        mgr._on_plugin_loaded.clear()
        mgr.unload("recursive")

    def test_exception_in_on_loaded_callback(self, callback_plugins_dir):
        """Exception in on_plugin_loaded callback."""
        from lfs_plugins import PluginManager, PluginState

        create_simple_plugin(callback_plugins_dir / "exc_target", "exc_target")

        mgr = PluginManager.instance()
        callback_called = [False]
        other_callback_called = [False]

        def failing_callback(info):
            callback_called[0] = True
            raise RuntimeError("Callback failed")

        def other_callback(info):
            other_callback_called[0] = True

        mgr.on_plugin_loaded(failing_callback)
        mgr.on_plugin_loaded(other_callback)

        mgr.load("exc_target")

        # First callback should have been called
        assert callback_called[0]

        # Plugin should still be loaded despite callback failure
        assert mgr.get_state("exc_target") == PluginState.ACTIVE

        mgr._on_plugin_loaded.clear()
        mgr.unload("exc_target")

    def test_circular_plugin_dependencies(self, callback_plugins_dir):
        """Plugins that try to load each other."""
        from lfs_plugins import PluginManager, PluginState

        # Plugin A loads plugin B in on_load
        plugin_a_dir = callback_plugins_dir / "dep_a"
        plugin_a_dir.mkdir()
        (plugin_a_dir / "pyproject.toml").write_text(
            """
[project]
name = "dep_a"
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
        (plugin_a_dir / "__init__.py").write_text(
            """
def on_load():
    from lfs_plugins import PluginManager
    try:
        PluginManager.instance().load("dep_b")
    except Exception:
        pass

def on_unload():
    pass
"""
        )

        # Plugin B loads plugin A in on_load
        plugin_b_dir = callback_plugins_dir / "dep_b"
        plugin_b_dir.mkdir()
        (plugin_b_dir / "pyproject.toml").write_text(
            """
[project]
name = "dep_b"
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
        (plugin_b_dir / "__init__.py").write_text(
            """
def on_load():
    from lfs_plugins import PluginManager
    try:
        PluginManager.instance().load("dep_a")
    except Exception:
        pass

def on_unload():
    pass
"""
        )

        mgr = PluginManager.instance()

        # Should not infinite loop or deadlock
        mgr.load("dep_a")

        # Cleanup
        if mgr.get_state("dep_a") == PluginState.ACTIVE:
            mgr.unload("dep_a")
        if mgr.get_state("dep_b") == PluginState.ACTIVE:
            mgr.unload("dep_b")

    def test_callback_order(self, callback_plugins_dir):
        """Callbacks should be called in registration order."""
        from lfs_plugins import PluginManager

        create_simple_plugin(callback_plugins_dir / "order_test", "order_test")

        mgr = PluginManager.instance()
        call_order = []

        mgr.on_plugin_loaded(lambda info: call_order.append(1))
        mgr.on_plugin_loaded(lambda info: call_order.append(2))
        mgr.on_plugin_loaded(lambda info: call_order.append(3))

        mgr.load("order_test")

        assert call_order == [1, 2, 3]

        mgr._on_plugin_loaded.clear()
        mgr.unload("order_test")
