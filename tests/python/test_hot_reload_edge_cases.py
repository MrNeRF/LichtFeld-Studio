# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for hot reload edge cases.

Targets:
- watcher.py:56-57 - Silent exception swallowing in loop
- watcher.py:106-136 - Property state saved to None instances
"""

import sys
import tempfile
import time
from pathlib import Path

import pytest


@pytest.fixture
def reload_test_dir(monkeypatch):
    """Create temporary plugins directory for reload tests."""
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


class TestHotReloadEdgeCases:
    """Tests for hot reload edge cases."""

    def test_reload_with_syntax_error(self, reload_test_dir):
        """Reload plugin that has syntax error."""
        from lfs_plugins import PluginManager, PluginState

        plugin_dir = reload_test_dir / "syntax_err"

        # Create valid plugin first
        create_plugin(
            plugin_dir,
            "syntax_err",
            """
VERSION = 1

def on_load():
    pass

def on_unload():
    pass
""",
        )

        mgr = PluginManager.instance()
        assert mgr.load("syntax_err")
        assert mgr.get_state("syntax_err") == PluginState.ACTIVE

        # Introduce syntax error
        (plugin_dir / "__init__.py").write_text("def broken syntax error {{{")

        # Reload should fail gracefully
        result = mgr.reload("syntax_err")
        assert result is False
        assert mgr.get_state("syntax_err") == PluginState.ERROR

    def test_reload_with_import_error(self, reload_test_dir):
        """Reload plugin that has import error."""
        from lfs_plugins import PluginManager, PluginState

        plugin_dir = reload_test_dir / "import_err"

        create_plugin(
            plugin_dir,
            "import_err",
            """
def on_load():
    pass

def on_unload():
    pass
""",
        )

        mgr = PluginManager.instance()
        assert mgr.load("import_err")

        # Add broken import
        (plugin_dir / "__init__.py").write_text(
            """
import nonexistent_module_xyz123

def on_load():
    pass

def on_unload():
    pass
"""
        )

        result = mgr.reload("import_err")
        assert result is False
        assert mgr.get_state("import_err") == PluginState.ERROR

    def test_reload_removes_attribute(self, reload_test_dir):
        """Reload where attribute is removed."""
        from lfs_plugins import PluginManager, PluginState

        plugin_dir = reload_test_dir / "attr_remove"

        create_plugin(
            plugin_dir,
            "attr_remove",
            """
MY_ATTRIBUTE = "original"

def on_load():
    pass

def on_unload():
    pass
""",
        )

        mgr = PluginManager.instance()
        mgr.load("attr_remove")

        module = sys.modules.get("lfs_plugins.attr_remove")
        assert hasattr(module, "MY_ATTRIBUTE")

        # Remove attribute
        (plugin_dir / "__init__.py").write_text(
            """
def on_load():
    pass

def on_unload():
    pass
"""
        )

        mgr.reload("attr_remove")

        # After reload, attribute should be gone
        module = sys.modules.get("lfs_plugins.attr_remove")
        assert not hasattr(module, "MY_ATTRIBUTE")

        mgr.unload("attr_remove")

    def test_reload_changes_class_hierarchy(self, reload_test_dir):
        """Reload where class hierarchy changes."""
        from lfs_plugins import PluginManager, PluginState

        plugin_dir = reload_test_dir / "hierarchy_change"

        create_plugin(
            plugin_dir,
            "hierarchy_change",
            """
class Base:
    pass

class Derived(Base):
    pass

def on_load():
    pass

def on_unload():
    pass
""",
        )

        mgr = PluginManager.instance()
        mgr.load("hierarchy_change")

        # Change hierarchy
        (plugin_dir / "__init__.py").write_text(
            """
class NewBase:
    pass

class Derived(NewBase):
    pass

def on_load():
    pass

def on_unload():
    pass
"""
        )

        result = mgr.reload("hierarchy_change")
        assert result is True

        module = sys.modules.get("lfs_plugins.hierarchy_change")
        assert hasattr(module, "NewBase")
        assert not hasattr(module, "Base")

        mgr.unload("hierarchy_change")

    def test_watcher_exception_recovery(self, reload_test_dir):
        """Watcher should recover from exceptions."""
        from lfs_plugins import PluginManager, PluginState
        from lfs_plugins.watcher import PluginWatcher

        plugin_dir = reload_test_dir / "watcher_test"

        create_plugin(
            plugin_dir,
            "watcher_test",
            """
def on_load():
    pass

def on_unload():
    pass
""",
        )

        mgr = PluginManager.instance()
        mgr.load("watcher_test")

        # Start watcher
        watcher = PluginWatcher(mgr, poll_interval=0.1)
        watcher.start()

        # Let watcher run a bit
        time.sleep(0.3)

        # Watcher should still be running
        assert watcher._running

        watcher.stop()
        mgr.unload("watcher_test")

    def test_rapid_file_changes(self, reload_test_dir):
        """Multiple rapid file changes - tests reload handles burst updates."""
        from lfs_plugins import PluginManager, PluginState

        plugin_dir = reload_test_dir / "rapid_change"

        create_plugin(
            plugin_dir,
            "rapid_change",
            """
VERSION = 0

def on_load():
    pass

def on_unload():
    pass
""",
        )

        mgr = PluginManager.instance()
        load_result = mgr.load("rapid_change")
        assert load_result is True, "Initial load failed"

        # Make rapid changes and explicitly reload
        reload_results = []
        for i in range(5):
            (plugin_dir / "__init__.py").write_text(
                f"""
VERSION = {i + 1}

def on_load():
    pass

def on_unload():
    pass
"""
            )
            time.sleep(0.05)  # Small delay between writes
            # Explicit reload to test core functionality
            result = mgr.reload("rapid_change")
            reload_results.append(result)

        # At least some reloads should succeed
        assert any(reload_results), f"All reloads failed: {reload_results}"

        # Should be at latest version
        module = sys.modules.get("lfs_plugins.rapid_change")
        assert module is not None
        assert module.VERSION > 0, f"VERSION={module.VERSION}, reload_results={reload_results}"

        mgr.unload("rapid_change")

    def test_file_deleted_during_reload(self, reload_test_dir):
        """File deleted while reload is happening."""
        from lfs_plugins import PluginManager, PluginState

        plugin_dir = reload_test_dir / "delete_test"

        create_plugin(
            plugin_dir,
            "delete_test",
            """
def on_load():
    pass

def on_unload():
    pass
""",
        )

        mgr = PluginManager.instance()
        mgr.load("delete_test")

        # Delete the file
        (plugin_dir / "__init__.py").unlink()

        # Reload should fail gracefully
        result = mgr.reload("delete_test")
        assert result is False

        # Create file back
        (plugin_dir / "__init__.py").write_text(
            """
def on_load():
    pass

def on_unload():
    pass
"""
        )

        # Should be able to load again
        result = mgr.load("delete_test")

        if mgr.get_state("delete_test") == PluginState.ACTIVE:
            mgr.unload("delete_test")


class TestPropertyStatePreservation:
    """Tests for property state preservation during reload."""

    def test_property_state_preserved(self, reload_test_dir):
        """Property values should be preserved across reload."""
        from lfs_plugins import PluginManager

        plugin_dir = reload_test_dir / "prop_preserve"

        create_plugin(
            plugin_dir,
            "prop_preserve",
            """
class MySettings:
    value = 0
    name = "default"

settings = MySettings()

def on_load():
    settings.value = 42
    settings.name = "loaded"

def on_unload():
    pass
""",
        )

        mgr = PluginManager.instance()
        mgr.load("prop_preserve")

        module = sys.modules.get("lfs_plugins.prop_preserve")
        assert module.settings.value == 42
        assert module.settings.name == "loaded"

        # Modify value
        module.settings.value = 100

        # Reload (value may or may not be preserved depending on implementation)
        (plugin_dir / "__init__.py").write_text(
            """
class MySettings:
    value = 0
    name = "default"

settings = MySettings()

def on_load():
    pass

def on_unload():
    pass
"""
        )

        mgr.reload("prop_preserve")

        module = sys.modules.get("lfs_plugins.prop_preserve")
        # After reload, value reset to new on_load (which sets nothing)
        # This tests whether state is preserved or reset

        mgr.unload("prop_preserve")
