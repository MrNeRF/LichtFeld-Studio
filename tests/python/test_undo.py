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


def test_python_history_details_are_copied_and_merge_snapshots(lf):
    fields = [{"target": "Camera", "property": "history.field.time", "before": 0, "after": 1}]
    lf.undo.push("Custom edit", lambda: None, lambda: None,
                 id="python.detail_edit", merge_window_ms=1000, details=fields)
    fields[0]["after"] = 999
    initial = lf.undo.details("undo", 1, lf.undo.generation())
    assert initial["fields"][0]["after"] == "1"

    lf.undo.push("Custom edit", lambda: None, lambda: None,
                 id="python.detail_edit", merge_window_ms=1000,
                 details=[{"target": "Camera", "property": "history.field.time", "before": 1, "after": 2}])
    merged = lf.undo.details("undo", 1, lf.undo.generation())
    assert len(lf.undo.stack()["undo"]) == 1
    assert merged["fields"] == [{"target": "Camera", "property": "history.field.time", "before": "0", "after": "2"}]


def test_python_history_scalar_details_use_localized_values(lf):
    lf.undo.push("Custom edit", lambda: None, lambda: None,
                 details=[{"property": "history.field.visibility", "before": False, "after": True},
                          {"property": "history.field.parent", "before": None, "after": "Group"}])
    fields = lf.undo.details("undo", 1, lf.undo.generation())["fields"]
    assert fields[0]["before"] == "@tr:history.value.false"
    assert fields[0]["after"] == "@tr:history.value.true"
    assert fields[1]["before"] == "@tr:history.value.none"


@pytest.mark.parametrize("details,exception", [
    ({"property": "time"}, TypeError),
    ([{"property": "time", "before": [], "after": 1}], TypeError),
    ([{"property": "time", "before": 0}], ValueError),
    ([{"property": "time", "before": 0, "after": "x" * 4097}], ValueError),
    ([{"property": "time", "before": 0, "after": 1}] * 129, ValueError),
])
def test_invalid_python_history_details_do_not_push(lf, details, exception):
    with pytest.raises(exception):
        lf.undo.push("Custom edit", lambda: None, lambda: None, details=details)
    assert not lf.undo.can_undo()
