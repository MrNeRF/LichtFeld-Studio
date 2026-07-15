# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the reactive signal system."""

import gc
import sys
import time
import weakref

import pytest

# Import from the build directory
sys.path.insert(0, "build/python")

from lfs_plugins.ui.signals import (
    Signal,
    ComputedSignal,
    ThrottledSignal,
    Batch,
    batch,
)


class TestSignal:
    """Tests for the Signal class."""

    def test_initial_value(self):
        """Signal should return initial value."""
        s = Signal(42)
        assert s.value == 42

    def test_value_change(self):
        """Signal value can be changed."""
        s = Signal(0)
        s.value = 10
        assert s.value == 10

    def test_value_change_notifies(self):
        """Setting value should notify subscribers."""
        s = Signal(0)
        notified = []

        def callback(v):
            notified.append(v)

        s.subscribe(callback)
        s.value = 10
        assert notified == [10]

    def test_no_notification_on_same_value(self):
        """Setting same value should not notify."""
        s = Signal(5)
        notified = []

        def callback(v):
            notified.append(v)

        s.subscribe(callback)
        s.value = 5
        assert notified == []

    def test_multiple_subscribers(self):
        """Multiple subscribers should all be notified."""
        s = Signal(0)
        notified_a = []
        notified_b = []

        def callback_a(v):
            notified_a.append(v)

        def callback_b(v):
            notified_b.append(v)

        s.subscribe(callback_a)
        s.subscribe(callback_b)
        s.value = 10
        assert notified_a == [10]
        assert notified_b == [10]

    def test_unsubscribe(self):
        """Unsubscribe should stop notifications."""
        s = Signal(0)
        notified = []

        def callback(v):
            notified.append(v)

        unsub = s.subscribe(callback)
        unsub()
        s.value = 10
        assert notified == []

    def test_peek_returns_value(self):
        """peek() should return current value."""
        s = Signal(42)
        assert s.peek() == 42

    def test_repr(self):
        """repr should include value."""
        s = Signal(42, name="test")
        assert "42" in repr(s)
        assert "test" in repr(s)

    def test_string_signal(self):
        """Signal should work with string values."""
        s = Signal("hello")
        assert s.value == "hello"
        s.value = "world"
        assert s.value == "world"

    def test_list_signal(self):
        """Signal should work with list values."""
        s = Signal([1, 2, 3])
        assert s.value == [1, 2, 3]

    def test_none_signal(self):
        """Signal should work with None value."""
        s = Signal(None)
        assert s.value is None
        s.value = "not none"
        assert s.value == "not none"


class TestComputedSignal:
    """Tests for the ComputedSignal class."""

    def test_derives_from_dependencies(self):
        """Computed signal should derive from dependencies."""
        a = Signal(2)
        b = Signal(3)
        c = ComputedSignal(lambda: a.value * b.value, [a, b])
        assert c.value == 6

    def test_updates_when_dependency_changes(self):
        """Computed signal should update when dependency changes."""
        a = Signal(2)
        b = Signal(3)
        c = ComputedSignal(lambda: a.value * b.value, [a, b])
        a.value = 4
        assert c.value == 12

    def test_updates_when_any_dependency_changes(self):
        """Should update when any dependency changes."""
        a = Signal(2)
        b = Signal(3)
        c = ComputedSignal(lambda: a.value + b.value, [a, b])
        b.value = 7
        assert c.value == 9

    def test_computed_subscriber_notified(self):
        """Computed signal subscribers should be notified."""
        a = Signal(2)
        c = ComputedSignal(lambda: a.value * 2, [a])
        notified = []

        def callback(v):
            notified.append(v)

        unsub = c.subscribe(callback)
        a.value = 5
        assert 10 in notified
        unsub()

    def test_lazy_evaluation(self):
        """Computed signal should be lazy."""
        call_count = [0]
        a = Signal(2)

        def compute():
            call_count[0] += 1
            return a.value * 2

        c = ComputedSignal(compute, [a])
        assert call_count[0] == 0
        _ = c.value
        assert call_count[0] == 1
        _ = c.value
        assert call_count[0] == 1


class TestThrottledSignal:
    """Tests for the ThrottledSignal class."""

    def test_initial_value(self):
        """Throttled signal should return initial value."""
        s = ThrottledSignal(42, max_rate_hz=10)
        assert s.value == 42

    def test_value_updates(self):
        """Throttled signal value should update."""
        s = ThrottledSignal(0, max_rate_hz=10)
        s.value = 10
        assert s.value == 10

    def test_throttles_rapid_updates(self):
        """Throttled signal should limit notification rate."""
        s = ThrottledSignal(0, max_rate_hz=10)
        notified = []

        def callback(v):
            notified.append(v)

        s.subscribe(callback)
        for i in range(100):
            s.value = i
        assert len(notified) < 100

    def test_flush_sends_pending(self):
        """flush() should send any pending value."""
        s = ThrottledSignal(0, max_rate_hz=1)
        s.value = 1
        time.sleep(0.01)
        for i in range(10):
            s.value = i + 2
        s.flush()
        assert s.value == 11

    def test_subscribe_returns_unsubscribe(self):
        """subscribe should return unsubscribe function."""
        s = ThrottledSignal(0, max_rate_hz=10)
        notified = []

        def callback(v):
            notified.append(v)

        unsub = s.subscribe(callback)
        unsub()
        time.sleep(0.2)
        s.value = 10
        assert notified == []


class TestBatch:
    """Tests for batching signal updates."""

    def test_batch_defers_notifications(self):
        """Batch should defer notifications."""
        s = Signal(0)
        notified = []

        def callback(v):
            notified.append(v)

        s.subscribe(callback)
        with Batch():
            s.value = 1
            s.value = 2
            s.value = 3
            assert notified == []
        assert 3 in notified

    def test_batch_function(self):
        """batch() function should work like Batch()."""
        s = Signal(0)
        notified = []

        def callback(v):
            notified.append(v)

        s.subscribe(callback)
        with batch():
            s.value = 1
            assert notified == []
        assert 1 in notified

    def test_multiple_signals_in_batch(self):
        """Batch should work with multiple signals."""
        a = Signal(0)
        b = Signal(0)
        notified_a = []
        notified_b = []

        def callback_a(v):
            notified_a.append(v)

        def callback_b(v):
            notified_b.append(v)

        a.subscribe(callback_a)
        b.subscribe(callback_b)
        with Batch():
            a.value = 1
            b.value = 2
            assert notified_a == []
            assert notified_b == []
        assert 1 in notified_a
        assert 2 in notified_b


class TestSignalMemory:
    """Memory management tests for signals."""

    def test_callback_exception_handled(self):
        """Signal should handle callback exceptions gracefully."""
        s = Signal(0)
        received = []

        def bad_callback(v):
            raise ValueError("Oops")

        def good_callback(v):
            received.append(v)

        s.subscribe(bad_callback)
        s.subscribe(good_callback)
        s.value = 1

        assert received == [1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
