# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for capability registry thread safety.

Targets:
- capabilities.py:119-139 - invoke() releases mutex before calling handler
- Handler deleted while another thread is invoking
"""

import concurrent.futures
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest


@pytest.fixture
def capability_registry():
    """Provide a fresh capability registry instance."""
    scripts_dir = Path(__file__).parent.parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from lfs_plugins.capabilities import CapabilityRegistry

    original_instance = CapabilityRegistry._instance
    CapabilityRegistry._instance = None

    registry = CapabilityRegistry.instance()

    yield registry

    # Cleanup
    with registry._mutex:
        registry._capabilities.clear()

    CapabilityRegistry._instance = original_instance


@pytest.fixture
def mock_context():
    """Provide a mock plugin context."""

    class MockContext:
        def __init__(self):
            pass

    return MockContext()


class TestCapabilityRace:
    """Tests for capability registry race conditions."""

    def test_invoke_during_unregister(self, capability_registry):
        """Invoke capability while it's being unregistered."""
        invoke_count = [0]
        errors = []

        def slow_handler(args, ctx):
            invoke_count[0] += 1
            time.sleep(0.05)
            return {"result": "ok"}

        capability_registry.register("test.slow", slow_handler, plugin_name="test")

        barrier = threading.Barrier(2)

        def invoker():
            try:
                barrier.wait(timeout=5.0)
                for _ in range(10):
                    result = capability_registry.invoke("test.slow", {})
                    if not result.get("success", False) and "Unknown capability" not in result.get("error", ""):
                        errors.append(result)
            except Exception as e:
                errors.append(e)

        def unregisterer():
            try:
                barrier.wait(timeout=5.0)
                time.sleep(0.02)
                capability_registry.unregister("test.slow")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=invoker),
            threading.Thread(target=unregisterer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        # Errors from race conditions (like accessing deleted handler) should not occur
        critical_errors = [e for e in errors if not isinstance(e, dict)]
        assert not critical_errors, f"Critical errors: {critical_errors}"

    def test_concurrent_capability_invocations(self, capability_registry):
        """Multiple threads invoking same capability."""
        call_count = [0]
        lock = threading.Lock()
        errors = []

        def counter_handler(args, ctx):
            with lock:
                call_count[0] += 1
            return {"count": call_count[0]}

        capability_registry.register("test.counter", counter_handler, plugin_name="test")

        N_THREADS = 8
        N_CALLS = 20
        barrier = threading.Barrier(N_THREADS)

        def invoker():
            try:
                barrier.wait(timeout=5.0)
                for _ in range(N_CALLS):
                    result = capability_registry.invoke("test.counter", {})
                    if not result.get("success", False):
                        errors.append(result)
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS) as ex:
            futures = [ex.submit(invoker) for _ in range(N_THREADS)]
            concurrent.futures.wait(futures, timeout=30.0)

        assert not errors, f"Errors occurred: {errors}"
        assert call_count[0] == N_THREADS * N_CALLS

    def test_handler_exception_thread_safety(self, capability_registry):
        """Handler raising exception doesn't corrupt registry state."""
        errors = []
        call_count = [0]

        def failing_handler(args, ctx):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                raise RuntimeError("Intentional failure")
            return {"ok": True}

        capability_registry.register("test.failing", failing_handler, plugin_name="test")

        N_THREADS = 4
        N_CALLS = 10
        barrier = threading.Barrier(N_THREADS)

        def invoker():
            try:
                barrier.wait(timeout=5.0)
                for _ in range(N_CALLS):
                    result = capability_registry.invoke("test.failing", {})
                    # Result should always be a dict with success key
                    assert isinstance(result, dict)
                    assert "success" in result
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS) as ex:
            futures = [ex.submit(invoker) for _ in range(N_THREADS)]
            concurrent.futures.wait(futures, timeout=30.0)

        assert not errors, f"Errors occurred: {errors}"
        # Verify registry still works
        assert capability_registry.has("test.failing")

    def test_unregister_all_while_iterating(self, capability_registry):
        """unregister_all_for_plugin while another thread iterates capabilities."""
        errors = []

        for i in range(10):
            capability_registry.register(
                f"test.cap_{i}",
                lambda args, ctx, i=i: {"id": i},
                plugin_name="test_plugin",
            )

        barrier = threading.Barrier(2)

        def lister():
            try:
                barrier.wait(timeout=5.0)
                for _ in range(20):
                    caps = capability_registry.list_all()
                    # Should not raise during iteration
                    for cap in caps:
                        _ = cap.name
            except Exception as e:
                errors.append(e)

        def unregisterer():
            try:
                barrier.wait(timeout=5.0)
                time.sleep(0.01)
                capability_registry.unregister_all_for_plugin("test_plugin")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=lister),
            threading.Thread(target=unregisterer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert not errors, f"Errors occurred: {errors}"

    def test_concurrent_register_unregister(self, capability_registry):
        """Register and unregister capabilities concurrently."""
        errors = []
        N_THREADS = 4
        N_OPS = 20
        barrier = threading.Barrier(N_THREADS)

        def worker(thread_id):
            try:
                barrier.wait(timeout=5.0)
                for i in range(N_OPS):
                    cap_name = f"test.t{thread_id}_c{i}"
                    capability_registry.register(
                        cap_name, lambda args, ctx: {"ok": True}, plugin_name=f"plugin_{thread_id}"
                    )
                    # Sometimes unregister
                    if i % 3 == 0:
                        capability_registry.unregister(cap_name)
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS) as ex:
            futures = [ex.submit(worker, i) for i in range(N_THREADS)]
            concurrent.futures.wait(futures, timeout=30.0)

        assert not errors, f"Errors occurred: {errors}"


class TestCapabilityHandlerLifecycle:
    """Tests for handler object lifecycle during concurrent access."""

    def test_handler_deleted_during_invoke(self, capability_registry):
        """Handler reference deleted while invocation in progress."""
        invocation_started = threading.Event()
        proceed = threading.Event()
        errors = []

        def blocking_handler(args, ctx):
            invocation_started.set()
            proceed.wait(timeout=5.0)
            return {"completed": True}

        capability_registry.register("test.blocking", blocking_handler, plugin_name="test")

        def invoker():
            try:
                result = capability_registry.invoke("test.blocking", {})
                if not result.get("completed"):
                    errors.append(f"Unexpected result: {result}")
            except Exception as e:
                errors.append(e)

        invoke_thread = threading.Thread(target=invoker)
        invoke_thread.start()

        # Wait for handler to start executing
        invocation_started.wait(timeout=5.0)

        # Unregister while handler is running
        capability_registry.unregister("test.blocking")

        # Let handler complete
        proceed.set()
        invoke_thread.join(timeout=5.0)

        # Handler should have completed successfully despite being unregistered
        assert not errors, f"Errors occurred: {errors}"

    def test_handler_modifies_registry(self, capability_registry):
        """Handler that modifies the registry during execution."""
        errors = []

        def self_modifying_handler(args, ctx):
            # Try to register new capability from within handler
            try:
                capability_registry.register(
                    "test.dynamic",
                    lambda a, c: {"dynamic": True},
                    plugin_name="test",
                )
            except Exception as e:
                errors.append(e)
            return {"modified": True}

        capability_registry.register("test.modifier", self_modifying_handler, plugin_name="test")

        result = capability_registry.invoke("test.modifier", {})
        assert result.get("success", False)
        assert not errors, f"Errors occurred: {errors}"

        # New capability should exist
        assert capability_registry.has("test.dynamic")


class TestCapabilityBrokerThreadLocal:
    """Tests for CapabilityBroker thread-local circular call detection."""

    def test_broker_circular_call_detected(self, capability_registry):
        """Broker detects circular calls within the same thread."""
        from lfs_plugins.context import CapabilityBroker

        broker = CapabilityBroker(capability_registry)

        def recursive_handler(args, ctx):
            return broker.invoke("test.recursive")

        capability_registry.register("test.recursive", recursive_handler, plugin_name="test")
        result = broker.invoke("test.recursive")
        assert result["success"] is False
        assert "Circular" in result["error"]

    def test_broker_no_false_positive_across_threads(self, capability_registry):
        """Two threads invoking the same capability must not trigger false circular detection."""
        from lfs_plugins.context import CapabilityBroker

        barrier = threading.Barrier(2)
        results = [None, None]

        def slow_handler(args, ctx):
            barrier.wait(timeout=5)
            return {"value": "ok"}

        capability_registry.register("test.slow_broker", slow_handler, plugin_name="test")

        def worker(idx):
            broker = CapabilityBroker(capability_registry)
            results[idx] = broker.invoke("test.slow_broker")

        t0 = threading.Thread(target=worker, args=(0,))
        t1 = threading.Thread(target=worker, args=(1,))
        t0.start()
        t1.start()
        t0.join(timeout=10)
        t1.join(timeout=10)

        assert results[0] is not None and results[0]["success"] is True
        assert results[1] is not None and results[1]["success"] is True
