# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for GIL lifetime in store subscriptions.
Cover callbacks that unsubscribe themselves. That drops the registry's
strong reference mid-notification, leaving the native lambda as the last
owner of the Python callable, so it performs the final release. Before the
fix that release happened after the GIL had been dropped, which is
undefined behavior.
"""

import gc
import weakref

import pytest

TOUCHED_FIELDS = ("iteration", "splat_simplify_state")


@pytest.fixture
def store(lf):
    """The native store bridge, with every field this module writes restored."""
    bridge = getattr(getattr(lf, "ui", None), "store", None)
    if bridge is None or not hasattr(bridge, "_drain_for_tests"):
        pytest.skip("native lichtfeld.ui.store bridge not available")

    saved = {field: bridge.get(field) for field in TOUCHED_FIELDS}
    yield bridge
    for field, value in saved.items():
        bridge.set(field, value)
    bridge._drain_for_tests()


def _notify_self_unsubscribing_callback(store, field, value):
    """Fire one notification into a callback that unsubscribes itself."""
    state = {"token": None, "calls": 0}

    def on_change(_value):
        state["calls"] += 1
        # Unsubscribe ourselves.
        store.unsubscribe(state["token"])

    state["token"] = store.subscribe(field, on_change)
    probe = weakref.ref(on_change)
    del on_change

    store.set(field, value)
    assert store._drain_for_tests() is True
    return state["calls"], probe


def test_self_unsubscribing_subscription_releases_callback_under_gil(store):
    calls, probe = _notify_self_unsubscribing_callback(
        store, "iteration", store.get("iteration") + 1
    )

    assert calls == 1
    gc.collect()
    assert probe() is None, "callback outlived the notification; release path not exercised"


def test_self_unsubscribing_converting_subscription_releases_callback_under_gil(store):
    calls, probe = _notify_self_unsubscribing_callback(
        store, "splat_simplify_state", {"active": True, "stage": "gil-lifetime-test"}
    )

    assert calls == 1
    gc.collect()
    assert probe() is None, "callback outlived the notification; release path not exercised"
