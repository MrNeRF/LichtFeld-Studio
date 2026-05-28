# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the Python reactive app-store facade."""

import pytest

from lfs_plugins.ui import store as store_module
from lfs_plugins.ui.store import AppStore, PanelStoreBinding, StoreSignal, batch_updates, invalidate_panel


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


def test_app_store_exposes_panel_reactive_signals():
    assert isinstance(AppStore.scene_generation, StoreSignal)
    assert isinstance(AppStore.selection_generation, StoreSignal)
    assert isinstance(AppStore.active_tool, StoreSignal)
    assert isinstance(AppStore.active_submode, StoreSignal)
    assert isinstance(AppStore.transform_space, StoreSignal)
    assert isinstance(AppStore.pivot_mode, StoreSignal)
    assert isinstance(AppStore.import_overlay_state, StoreSignal)
    assert isinstance(AppStore.video_export_overlay_state, StoreSignal)
    assert isinstance(AppStore.export_progress_state, StoreSignal)
    assert isinstance(AppStore.mesh2splat_state, StoreSignal)
    assert isinstance(AppStore.splat_simplify_state, StoreSignal)
    assert isinstance(AppStore.scripts_generation, StoreSignal)
    assert isinstance(AppStore.language_generation, StoreSignal)


def test_ui_package_exports_preferred_store_names():
    from lfs_plugins.ui import AppStore as PublicAppStore, NativeAppStore, PanelStoreBinding as PublicBinding

    assert PublicAppStore is AppStore
    assert NativeAppStore is AppStore
    assert PublicBinding is PanelStoreBinding


def test_native_value_helper_does_not_read_fallback_signal_without_native_store(monkeypatch):
    monkeypatch.setattr(AppStore.active_tool, "_fallback", "fallback-only")

    assert store_module.native_value("active_tool", "missing") == "missing"


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


def test_native_value_helper_reads_native_store(monkeypatch):
    native = _NativeStore()
    native.values["active_tool"] = "builtin.translate"
    monkeypatch.setattr(store_module, "_native_store", lambda: native)

    assert store_module.native_value("active_tool", "missing") == "builtin.translate"


def test_native_batch_is_closed_on_exception(monkeypatch):
    native = _NativeStore()
    monkeypatch.setattr(store_module, "_native_store", lambda: native)

    with pytest.raises(RuntimeError):
        with batch_updates():
            raise RuntimeError("boom")

    assert native.batch_events == ["begin", "end"]


class _PanelHandle:
    def __init__(self):
        self.request_count = 0
        self.dirty_fields = []
        self.dirty_all_count = 0

    def request_update(self):
        self.request_count += 1

    def dirty(self, field):
        self.dirty_fields.append(field)

    def dirty_all(self):
        self.dirty_all_count += 1


def test_invalidate_panel_uses_request_update_for_default_dirty_policy():
    handle = _PanelHandle()

    invalidate_panel(handle)

    assert handle.request_count == 1
    assert handle.dirty_fields == []
    assert handle.dirty_all_count == 0


def test_invalidate_panel_supports_field_and_full_model_dirtying():
    handle = _PanelHandle()

    invalidate_panel(handle, "title")
    invalidate_panel(handle, ("rows", "empty_state"))
    invalidate_panel(handle, "*")

    assert handle.dirty_fields == ["title", "rows", "empty_state"]
    assert handle.dirty_all_count == 1


def test_panel_store_binding_owns_subscriptions_and_invalidates_handle():
    signal = StoreSignal[int]("selection_generation", 0)
    handle = _PanelHandle()
    refreshes = []

    binding = PanelStoreBinding(handle).watch(
        signal,
        refresh=lambda: refreshes.append("refresh"),
    )

    signal.value = 1
    binding.close()
    signal.value = 2

    assert refreshes == ["refresh"]
    assert handle.request_count == 1


def test_panel_store_binding_can_dirty_specific_model_fields():
    signal = StoreSignal[int]("language_generation", 0)
    handle = _PanelHandle()

    PanelStoreBinding(handle).watch(signal, dirty=("label", "tooltip"))

    signal.value = 1

    assert handle.request_count == 0
    assert handle.dirty_fields == ["label", "tooltip"]
