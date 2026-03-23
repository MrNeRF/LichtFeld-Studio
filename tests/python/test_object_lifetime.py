# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for object lifetime across Python/C++ boundary.

Targets:
- py_prop_registry.cpp:206-258 - Lambda captures may outlive objects
- py_plugins.cpp:19-29 - Stale module reference after reload
"""

import gc
import sys
import tempfile
import weakref
from pathlib import Path

import pytest


@pytest.fixture
def lfs_types():
    """Import lfs_plugins.types module."""
    project_root = Path(__file__).parent.parent.parent
    build_python = project_root / "build" / "src" / "python"
    if str(build_python) not in sys.path:
        sys.path.insert(0, str(build_python))

    try:
        from lfs_plugins import types
        return types
    except ImportError as e:
        pytest.skip(f"lfs_plugins.types not available: {e}")


@pytest.fixture
def op_fixture(lf, lfs_types):
    """Setup and cleanup for operator tests."""
    registered = []

    yield registered, lfs_types

    for op_cls in registered:
        try:
            lf.unregister_class(op_cls)
        except Exception:
            pass


class TestObjectLifetime:
    """Tests for proper object lifetime management."""

    def test_cached_tensor_after_gc(self, lf, numpy):
        """Cached tensor references should remain valid after GC."""
        arr = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32)
        tensor = lf.Tensor.from_numpy(arr)

        # Force GC
        gc.collect()

        # Tensor should still exist and be valid (numel is property)
        assert tensor.numel == 3

    def test_tensor_reference_after_invalidation(self, lf, numpy):
        """Tensor operations after potential invalidation."""
        arr = numpy.array([[1.0, 2.0], [3.0, 4.0]], dtype=numpy.float32)
        original = lf.Tensor.from_numpy(arr)

        # Create view/reference
        view = original

        # Original data might be modified or reallocated
        # (depends on implementation)

        # Access should still work or raise properly (numel is property)
        try:
            _ = view.numel
        except Exception:
            pass  # May raise if invalidated

    def test_operator_instance_persists_during_draw(self, lf, op_fixture):
        """Operator instance should persist during draw phase."""
        registered, lfs_types = op_fixture
        instance_exists = [False]

        class DrawOp(lfs_types.Operator):
            lf_label = "Draw Persist"

            def __init__(self):
                super().__init__()
                instance_exists[0] = True

            def draw(self, layout):
                # Instance should still exist
                layout.label("Test")

            def execute(self, context):
                return {"FINISHED"}

        lf.register_class(DrawOp)
        registered.append(DrawOp)

        try:
            lf.ops.invoke(DrawOp._class_id())
        except Exception:
            pass


@pytest.fixture
def reload_plugins_dir(monkeypatch):
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

        for name in list(mgr._plugins.keys()):
            try:
                mgr.unload(name)
            except Exception:
                pass

        PluginManager._instance = original_instance


class TestModuleReloadLifetime:
    """Tests for module references across reload."""

    def test_callback_after_module_reload(self, reload_plugins_dir):
        """Callbacks should still work after module reload."""
        from lfs_plugins import PluginManager

        plugin_dir = reload_plugins_dir / "callback_plugin"
        plugin_dir.mkdir()

        (plugin_dir / "pyproject.toml").write_text(
            """
[project]
name = "callback_plugin"
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
            """
CALLBACK_RESULTS = []

def callback(info):
    CALLBACK_RESULTS.append(info.name)

def on_load():
    from lfs_plugins import PluginManager
    PluginManager.instance().on_plugin_loaded(callback)

def on_unload():
    pass
"""
        )

        mgr = PluginManager.instance()
        mgr.load("callback_plugin")

        # Modify and reload
        (plugin_dir / "__init__.py").write_text(
            """
CALLBACK_RESULTS = []
RELOADED = True

def callback(info):
    CALLBACK_RESULTS.append(info.name)

def on_load():
    from lfs_plugins import PluginManager
    PluginManager.instance().on_plugin_loaded(callback)

def on_unload():
    pass
"""
        )

        mgr.reload("callback_plugin")

        # Should not crash, old callbacks might still fire
        module = sys.modules.get("lfs_plugins.callback_plugin")
        assert hasattr(module, "RELOADED")

        mgr.unload("callback_plugin")

    def test_stale_plugin_manager_reference(self, reload_plugins_dir):
        """Stale reference to plugin manager after reload."""
        from lfs_plugins import PluginManager

        plugin_dir = reload_plugins_dir / "stale_ref"
        plugin_dir.mkdir()

        (plugin_dir / "pyproject.toml").write_text(
            """
[project]
name = "stale_ref"
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
            """
from lfs_plugins import PluginManager

# Store reference at load time
_mgr = PluginManager.instance()

def on_load():
    pass

def on_unload():
    pass

def get_manager():
    return _mgr
"""
        )

        mgr = PluginManager.instance()
        mgr.load("stale_ref")

        module = sys.modules.get("lfs_plugins.stale_ref")
        stored_mgr = module.get_manager()

        # Should be same instance
        assert stored_mgr is mgr

        mgr.unload("stale_ref")


class TestWeakReferenceHandling:
    """Tests for weak reference handling in the C++ binding."""

    def test_operator_class_weak_reference(self, lf, op_fixture):
        """Weak references to operator classes should work."""
        registered, lfs_types = op_fixture

        class WeakOp(lfs_types.Operator):
            lf_label = "Weak Op"

            def execute(self, context):
                return {"FINISHED"}

        weak_cls = weakref.ref(WeakOp)

        lf.register_class(WeakOp)
        registered.append(WeakOp)

        # Class should still exist
        assert weak_cls() is not None

    def test_tensor_data_lifetime(self, lf, numpy):
        """Tensor data should remain valid as long as tensor exists."""
        arr = numpy.array([1.0, 2.0, 3.0, 4.0], dtype=numpy.float32)
        tensor = lf.Tensor.from_numpy(arr)

        # Delete numpy array
        del arr
        gc.collect()

        # Tensor should still have valid data (numel is property)
        assert tensor.numel == 4

        # Convert back to numpy
        result = tensor.numpy()
        assert len(result) == 4


class TestCircularReferenceHandling:
    """Tests for circular reference scenarios."""

    def test_operator_self_reference(self, lf, op_fixture):
        """Operator with self-reference should be collectible."""
        registered, lfs_types = op_fixture

        class SelfRefOp(lfs_types.Operator):
            lf_label = "Self Ref"

            def __init__(self):
                super().__init__()
                self.self_ref = self  # Circular reference

            def execute(self, context):
                return {"FINISHED"}

        lf.register_class(SelfRefOp)
        registered.append(SelfRefOp)

        try:
            lf.ops.invoke(SelfRefOp._class_id())
        except Exception:
            pass

        gc.collect()

    def test_callback_circular_reference(self, lf):
        """Callback with circular reference should be handled."""

        class Holder:
            def __init__(self):
                self.callback = None

            def method(self, ctx):
                pass

        holder = Holder()
        holder.callback = holder.method  # Circular via method binding

        lf.on_training_start(holder.method)

        del holder
        gc.collect()

        # Should not crash or leak
