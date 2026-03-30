# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for the unified Python undo surface."""

import gc

import pytest


@pytest.fixture(autouse=True)
def _clear_history(lf):
    lf.undo.clear()
    yield
    lf.undo.clear()


def test_push_undo_redo_round_trip(lf):
    state = {"value": 0}

    lf.undo.push(
        "custom.step",
        lambda: state.__setitem__("value", 0),
        lambda: state.__setitem__("value", 1),
    )
    state["value"] = 1

    assert lf.undo.undo() is True
    assert state["value"] == 0
    assert lf.undo.redo() is True
    assert state["value"] == 1


def test_transaction_context_groups_python_callbacks(lf):
    state = {"value": 0}

    with lf.undo.transaction("Grouped Changes") as tx:
        tx.add(lambda: state.__setitem__("value", 0), lambda: state.__setitem__("value", 1))
        tx.add(lambda: state.__setitem__("value", 1), lambda: state.__setitem__("value", 2))

    assert state["value"] == 2
    assert lf.undo.undo() is True
    assert state["value"] == 0
    assert lf.undo.redo() is True
    assert state["value"] == 2


def test_subscribe_and_unsubscribe_track_history_changes(lf):
    generations = []
    subscription_id = lf.undo.subscribe(lambda: generations.append(lf.undo.generation()))

    lf.undo.push("custom.step", lambda: None, lambda: None)
    assert lf.undo.undo() is True
    lf.undo.unsubscribe(subscription_id)
    assert lf.undo.redo() is True

    assert len(generations) == 2
    assert generations[0] < generations[1]


def test_invalid_python_callbacks_are_rejected(lf):
    with pytest.raises(TypeError):
        lf.undo.push("custom.step", object(), lambda: None)

    with pytest.raises(TypeError):
        with lf.undo.transaction("Grouped Changes") as tx:
            tx.add(object(), lambda: None)


def test_push_metadata_and_merge_window_are_reflected_in_history(lf):
    state = {"value": 0}

    lf.undo.push(
        "Opacity Drag",
        lambda: state.__setitem__("value", 0),
        lambda: state.__setitem__("value", 1),
        id="python.opacity_drag",
        source="python",
        scope="property",
        estimated_bytes=1024,
        dirty_flags=1,
        merge_window_ms=1000,
    )
    state["value"] = 1

    lf.undo.push(
        "Opacity Drag",
        lambda: state.__setitem__("value", 1),
        lambda: state.__setitem__("value", 2),
        id="python.opacity_drag",
        source="python",
        scope="property",
        estimated_bytes=2048,
        dirty_flags=2,
        merge_window_ms=1000,
    )
    state["value"] = 2

    stack = lf.undo.stack()
    assert len(stack["undo"]) == 1
    assert stack["undo"][0]["id"] == "python.opacity_drag"
    assert stack["undo"][0]["source"] == "python"
    assert stack["undo"][0]["scope"] == "property"
    assert stack["undo"][0]["estimated_bytes"] == 2048
    assert stack["max_bytes"] > 0

    assert lf.undo.undo() is True
    assert state["value"] == 0
    assert lf.undo.redo() is True
    assert state["value"] == 2


def test_transaction_object_rolls_back_on_destruction(lf):
    state = {"value": 0}

    tx = lf.undo.transaction("Leaked Transaction")
    tx.__enter__()
    tx.add(lambda: state.__setitem__("value", 0), lambda: state.__setitem__("value", 1))
    assert state["value"] == 1
    assert lf.undo.has_active_transaction() is True

    del tx
    gc.collect()

    assert state["value"] == 0
    assert lf.undo.has_active_transaction() is False
    assert lf.undo.undo() is False


def test_python_callback_failure_returns_false_and_preserves_history(lf):
    lf.undo.push(
        "custom.step",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        lambda: None,
    )

    assert lf.undo.undo() is False
    assert lf.undo.can_undo() is True
    assert lf.undo.can_redo() is False
