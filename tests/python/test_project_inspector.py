"""Tests for the Projects Inspector cache, models, and action contract."""

from types import SimpleNamespace
import threading
import time

from lfs_plugins.project_inspector import (
    InspectionFactsPipeline,
    details_rows,
    dialog_model,
    inspection_cache_key,
    operation_actions,
    reduce_plan_rows,
)


def _entry(**overrides):
    value = {
        "id": "project",
        "path": "/tmp/project.licht",
        "file_size_bytes": 100,
        "path_mtime_ns": 20,
        "commit_uuid": "commit-a",
        "status": "AVAILABLE",
    }
    value.update(overrides)
    return value


def test_cache_key_uses_stat_identity_and_commit():
    entry = _entry(stat_identity={"size": 101, "mtime_ns": 22, "st_dev": 3, "st_ino": 4})
    assert inspection_cache_key(entry) == (101, 22, 3, 4, "commit-a")
    changed = dict(entry, commit_uuid="commit-b")
    assert inspection_cache_key(changed) != inspection_cache_key(entry)


def test_pipeline_inspects_visible_cards_and_selected_details_once():
    calls = []
    results = []
    card = SimpleNamespace(has_preview=True, physical_file_size=100)
    details = SimpleNamespace(retained_checkpoints=[SimpleNamespace(iteration=30)])
    pipeline = InspectionFactsPipeline(
        lambda path: calls.append(("card", path)) or card,
        lambda path: calls.append(("details", path)) or details,
        lambda asset_id, kind, result, error: results.append((asset_id, kind, result, error)),
        max_background_details=0,
    )
    pipeline.refresh([_entry()], "project")
    assert _wait(lambda: len(results) == 2)
    pipeline.refresh([_entry()], "project")
    time.sleep(0.05)
    assert len(results) == 2
    assert calls == [("card", "/tmp/project.licht"), ("details", "/tmp/project.licht")]
    pipeline.close()


def test_pipeline_cancellation_drops_stale_result():
    started = threading.Event()
    release = threading.Event()
    results = []

    def inspect_card(_path):
        started.set()
        release.wait(1)
        return object()

    pipeline = InspectionFactsPipeline(inspect_card, lambda _path: object(), lambda *args: results.append(args))
    pipeline.refresh([_entry()], "")
    assert started.wait(1)
    pipeline.cancel()
    release.set()
    time.sleep(0.05)
    assert results == []
    pipeline.close()


def test_details_model_hides_metrics_without_samples_and_formats_embedded_dataset():
    details = SimpleNamespace(
        card=SimpleNamespace(saved_at_unix_ns=0, title="Bicycle"),
        storage=SimpleNamespace(physical_bytes=1000, dead_bytes=499, dead_ratio=0.499),
        save_history=[SimpleNamespace(sequence=3, saved_at_unix_ns=0)],
        references=[],
        parameters=SimpleNamespace(
            active_strategy="MRNF", embedded_dataset_present=True,
            embedded_dataset_complete=True, embedded_images=194,
            embedded_normals=194, embedded_sparse=3,
        ),
        scene_graph=SimpleNamespace(dataset_node_name="images"),
        retained_checkpoints=[SimpleNamespace(iteration=30000, gaussians=1000000, sh_degree=3, binds_scene_graph=True)],
        metrics=SimpleNamespace(loss_samples=0, psnr_samples=0),
        license=None,
        autosave_sidecar_present=False,
    )
    model = details_rows(_entry(), details, format_size=lambda n: f"{n} B", format_time=lambda n: "")
    assert model["dataset"] == "embedded, 194 images, 194 normals and 3 sparse, complete"
    assert model["reclaimable_percent"] == "49.9%"
    assert not model["has_metrics"]
    assert model["title"] == "Bicycle"


def test_dialog_models_cover_history_reduce_export_license_and_repair():
    details = SimpleNamespace(save_history=[SimpleNamespace(kind="EXPLICIT", saved_at_unix_ns=0, checkpoint_iteration=30, bytes_added=12, generation=3, holds_checkpoint=True)], license=None)
    plan = SimpleNamespace(
        physical_size=100,
        retained_checkpoints=[SimpleNamespace(iteration=30, bytes=40, scng_bound=True)],
        embedded_dataset=[],
        drop_checkpoints=SimpleNamespace(allowed=True, reclaimable_bytes=20, projected_size=80),
        drop_embedded_dataset=SimpleNamespace(allowed=False, reclaimable_bytes=0, projected_size=100),
        compact=SimpleNamespace(allowed=True, reclaimable_bytes=10, projected_size=90),
    )
    for kind in ("save_history", "reduce_size", "export_as", "update_thumbnail", "set_license", "rename", "repair"):
        model = dialog_model(kind, entry=_entry(), details=details, plan=plan, format_size=lambda n: f"{n} B", format_time=lambda n: "")
        assert model["kind"] == kind
    reduced = reduce_plan_rows(plan, format_size=lambda n: f"{n} B")
    assert reduced["checkpoints"][0]["locked"] is True
    assert reduced["drop_dataset_allowed"] is False


def test_action_table_includes_operations_only_when_details_are_available():
    entry = _entry()
    assert operation_actions(entry, None)[-1]["action"] == "rename"
    details = SimpleNamespace(
        storage=SimpleNamespace(dead_ratio=0.289),
        parameters=SimpleNamespace(embedded_dataset_present=False),
        references=[SimpleNamespace(kind="dataset", reachable=True)],
    )
    actions = {row["action"] for row in operation_actions(entry, details)}
    assert {"save_history", "reduce_size", "embed_dataset", "export_as", "update_thumbnail", "set_license", "rename"} <= actions


def _wait(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False
