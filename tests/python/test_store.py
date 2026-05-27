# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the Python reactive app-store facade."""

import pytest

from lfs_plugins.ui import store as store_module
from lfs_plugins.ui.store import StoreSignal, batch_updates


@pytest.fixture(autouse=True)
def fallback_store(monkeypatch):
    monkeypatch.setattr(store_module, "_native_store", lambda: None)
    store_module._batch_context.depth = 0
    store_module._batch_context.pending_notifications.clear()
    yield
    store_module._batch_context.depth = 0
    store_module._batch_context.pending_notifications.clear()


def test_fallback_suppresses_same_value():
    signal = StoreSignal[int]("iteration", 0)
    notified = []

    signal.subscribe(notified.append)

    signal.value = 0
    signal.value = 1
    signal.value = 1

    assert signal.value == 1
    assert notified == [1]


def test_fallback_batch_defers_and_dedups_notifications():
    iteration = StoreSignal[int]("iteration", 0)
    loss = StoreSignal[float]("loss", 0.0)
    notified = []

    iteration.subscribe(lambda value: notified.append(("iteration", value)))
    loss.subscribe(lambda value: notified.append(("loss", value)))

    with batch_updates():
        iteration.value = 1
        iteration.value = 2
        loss.value = 0.5
        assert notified == []

    assert sorted(notified) == [("iteration", 2), ("loss", 0.5)]


def test_nested_fallback_batches_flush_at_outer_exit():
    signal = StoreSignal[int]("iteration", 0)
    notified = []
    signal.subscribe(notified.append)

    with batch_updates():
        signal.value = 1
        with batch_updates():
            signal.value = 2
        assert notified == []

    assert notified == [2]


class _NativeStore:
    def __init__(self):
        self.values = {"fps": 0.0}
        self.unsubscribe_calls = []
        self.batch_events = []

    def get(self, field):
        return self.values[field]

    def set(self, field, value):
        self.values[field] = value

    def subscribe(self, field, callback):
        self.values["subscribed_field"] = field
        self.values["callback"] = callback
        return 42

    def unsubscribe(self, token):
        self.unsubscribe_calls.append(token)

    def begin_batch(self):
        self.batch_events.append("begin")

    def end_batch(self):
        self.batch_events.append("end")


def test_native_store_proxy(monkeypatch):
    native = _NativeStore()
    monkeypatch.setattr(store_module, "_native_store", lambda: native)

    signal = StoreSignal[float]("fps", 0.0)

    signal.value = 59.5
    assert signal.value == 59.5

    unsubscribe = signal.subscribe(lambda value: value)
    assert native.values["subscribed_field"] == "fps"
    unsubscribe()
    assert native.unsubscribe_calls == [42]


def test_native_batch_is_closed_on_exception(monkeypatch):
    native = _NativeStore()
    monkeypatch.setattr(store_module, "_native_store", lambda: native)

    with pytest.raises(RuntimeError):
        with batch_updates():
            raise RuntimeError("boom")

    assert native.batch_events == ["begin", "end"]
