import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "benchmark_scene_reconstruction.py"
SPEC = importlib.util.spec_from_file_location("scene_reconstruction_benchmark", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_catalog_cases_follow_backend_specific_presets():
    cases = MODULE.benchmark_cases(
        {
            "backends": [
                {"id": "native", "presets": [{"id": "native", "input_scale": 1.0}]},
                {
                    "id": "future",
                    "presets": [
                        {"id": "quality", "input_scale": 0.75},
                        {"id": "performance", "input_scale": 0.5},
                    ],
                },
            ]
        }
    )
    assert [case.key for case in cases] == [
        "native:native",
        "future:quality",
        "future:performance",
    ]


def test_quality_metrics_identical_and_degraded_images():
    reference = np.linspace(0.0, 1.0, 8 * 6 * 3, dtype=np.float32).reshape(6, 8, 3)
    identical_psnr, identical_ssim = MODULE.quality_metrics(reference, reference)
    assert identical_psnr == pytest.approx(float("inf"))
    assert identical_ssim == pytest.approx(1.0)

    degraded = np.clip(reference * 0.8 + 0.1, 0.0, 1.0)
    degraded_psnr, degraded_ssim = MODULE.quality_metrics(degraded, reference)
    assert degraded_psnr < 30.0
    assert 0.0 < degraded_ssim < 1.0


def test_timing_summary_reports_robust_latency_and_throughput():
    summary = MODULE.summarize_milliseconds(
        [10.0, 11.0, 12.0, 13.0, 100.0], "viewport_fps_from_median"
    )
    assert summary["median_ms"] == pytest.approx(12.0)
    assert summary["p95_ms"] > summary["median_ms"]
    assert summary["sample_count"] == 5
    assert summary["trimmed_mean_ms"] == pytest.approx(29.2)
    assert summary["minimum_ms"] == pytest.approx(10.0)
    assert summary["maximum_ms"] == pytest.approx(100.0)
    assert summary["viewport_fps_from_median"] == pytest.approx(1000.0 / 12.0)


def test_performance_rounds_preserve_total_samples_and_vary_case_order():
    cases = [
        MODULE.BenchmarkCase("native", "native", 1.0),
        MODULE.BenchmarkCase("spatial", "quality", 0.75),
        MODULE.BenchmarkCase("temporal", "quality", 0.75),
    ]
    assert MODULE.split_frame_counts(8, 3) == [3, 3, 2]
    assert [case.key for case in MODULE.performance_case_order(cases, 0)] == [
        "native:native",
        "spatial:quality",
        "temporal:quality",
    ]
    assert [case.key for case in MODULE.performance_case_order(cases, 1)] == [
        "temporal:quality",
        "spatial:quality",
        "native:native",
    ]
    assert [case.key for case in MODULE.performance_case_order(cases, 2)] == [
        "spatial:quality",
        "temporal:quality",
        "native:native",
    ]


def test_command_help_documents_all_benchmark_controls():
    help_text = MODULE.build_parser().format_help()
    for option in (
        "--url",
        "--width",
        "--height",
        "--frames",
        "--quality-frames",
        "--warmup-frames",
        "--performance-rounds",
        "--orbit-degrees",
        "--backends",
        "--timeout",
        "--output",
    ):
        assert option in help_text
    assert "tools/benchmark_scene_reconstruction.py --help" in help_text
    assert "before any PNG capture" in help_text


def test_capture_sends_presented_true_to_render_capture(monkeypatch):
    recorded = {}

    class Client:
        def call_tool(self, name, arguments=None):
            recorded["name"] = name
            recorded["arguments"] = arguments
            return {"success": True, "data": "unused"}

    monkeypatch.setattr(
        MODULE, "decode_png", lambda data: np.zeros((6, 8, 3), dtype=np.float32)
    )
    image = MODULE.capture(Client(), 8, 6)
    assert image.shape == (6, 8, 3)
    assert recorded["name"] == "render.capture"
    assert recorded["arguments"] == {"width": 8, "height": 6, "presented": True}


def test_sample_viewport_frame_requires_positive_in_app_latency():
    class Client:
        def __init__(self, payload):
            self.payload = payload

        def call_tool(self, name, arguments=None):
            assert name == "render.reconstruction.sample_frame"
            assert arguments == {"eye": [1, 2, 3]}
            return self.payload

    view = {"eye": [1, 2, 3]}
    payload = MODULE.sample_viewport_frame(
        Client({"success": True, "frame_latency_ms": 12.5}), view
    )
    assert payload["frame_latency_ms"] == pytest.approx(12.5)

    with pytest.raises(MODULE.BenchmarkError, match="no valid frame latency"):
        MODULE.sample_viewport_frame(
            Client({"success": True, "frame_latency_ms": 0.0}), view
        )

    class ConvergenceClient:
        def __init__(self, remaining):
            self.remaining = iter(remaining)

        def call_tool(self, name, arguments=None):
            assert name == "render.reconstruction.status"
            assert arguments is None
            return {
                "success": True,
                "requested": "temporal",
                "effective": "temporal",
                "fallback": "none",
                "fell_back": False,
                "convergence_remaining": next(self.remaining),
            }

    converged = MODULE.wait_for_convergence(
        ConvergenceClient([2, 1, 0]),
        MODULE.BenchmarkCase("temporal", "quality", 0.75),
        1.0,
    )
    assert converged["convergence_remaining"] == 0

    with pytest.raises(MODULE.BenchmarkError, match="no valid convergence counter"):
        MODULE.wait_for_convergence(
            ConvergenceClient([True]),
            MODULE.BenchmarkCase("temporal", "quality", 0.75),
            1.0,
        )


def test_quality_summary_keeps_exact_matches_valid_json_values():
    exact = MODULE.summarize_quality([float("inf"), float("inf")], [1.0, 1.0])
    assert exact["aggregate_psnr_db"] is None
    assert exact["minimum_psnr_db"] is None
    assert exact["exact_match_frames"] == 2

    mixed = MODULE.summarize_quality([float("inf"), 20.0], [1.0, 0.9])
    assert mixed["aggregate_psnr_db"] == pytest.approx(10.0 * np.log10(200.0))
    assert mixed["minimum_psnr_db"] == pytest.approx(20.0)
    assert mixed["exact_match_frames"] == 1


def test_render_settings_snapshot_unwraps_tool_payload_and_rejects_flat_shape():
    snapshot = MODULE.render_settings_snapshot(
        {
            "success": True,
            "settings": {
                "scene_upscaler": "temporal",
                "scene_upscaler_preset": "balanced",
                "render_scale": 1.0,
            },
        }
    )
    assert snapshot == {
        "scene_upscaler": "temporal",
        "scene_upscaler_preset": "balanced",
    }

    with pytest.raises(MODULE.BenchmarkError, match="no settings object"):
        MODULE.render_settings_snapshot(
            {"scene_upscaler": "temporal", "scene_upscaler_preset": "balanced"}
        )


def test_preflight_requires_a_visible_gaussian_scene_and_reports_temporal_context():
    settings = {
        "success": True,
        "settings": {
            "raster_backend": "3dgs",
            "equirectangular": False,
            "orthographic": True,
            "apply_appearance_correction": False,
            "split_view_mode": 0,
        },
    }
    cases = [MODULE.BenchmarkCase("temporal", "quality", 0.75)]
    preflight = MODULE.benchmark_preflight(
        settings,
        {
            "success": True,
            "nodes": [
                {"type": "splat", "visible": True, "gaussian_count": 1234},
                {"type": "mesh", "visible": True, "gaussian_count": 0},
            ],
        },
        cases,
    )
    assert preflight["visible_gaussian_nodes"] == 1
    assert preflight["visible_gaussians"] == 1234
    assert preflight["temporal_eligible"] is True
    assert preflight["render_context"]["orthographic"] is True

    empty = MODULE.benchmark_preflight(
        settings, {"success": True, "nodes": []}, cases
    )
    assert empty["success"] is False
    assert "No visible Gaussian splat" in empty["errors"][0]


def test_preflight_explains_temporal_ineligibility():
    preflight = MODULE.benchmark_preflight(
        {
            "success": True,
            "settings": {
                "raster_backend": "3dgs",
                "equirectangular": True,
                "orthographic": False,
                "apply_appearance_correction": True,
                "split_view_mode": 0,
            },
        },
        {
            "success": True,
            "nodes": [{"type": "splat", "gaussian_count": 1}],
        },
        [MODULE.BenchmarkCase("temporal", "quality", 0.75)],
    )
    assert preflight["success"] is False
    assert "equirectangular projection is Native-only" in preflight["errors"][0]
    assert "appearance correction is Native-only" in preflight["errors"][0]


def test_restore_state_replays_validated_settings_snapshot_and_camera():
    class RecordingClient:
        def __init__(self):
            self.calls = []

        def call_tool(self, name, arguments=None):
            self.calls.append((name, arguments))
            return {"success": True}

    client = RecordingClient()
    MODULE.restore_state(
        client,
        {"scene_upscaler": "spatial", "scene_upscaler_preset": "performance"},
        {
            "eye": [1.0, 2.0, 3.0],
            "target": [0.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
            "fov_degrees": 55.0,
        },
    )
    assert client.calls == [
        (
            "render.settings.set",
            {
                "scene_upscaler": "spatial",
                "scene_upscaler_preset": "performance",
            },
        ),
        (
            "camera.set_view",
            {
                "eye": [1.0, 2.0, 3.0],
                "target": [0.0, 0.0, 0.0],
                "up": [0.0, 1.0, 0.0],
                "fov_degrees": 55.0,
            },
        ),
    ]


def test_atomic_report_checkpoint_replaces_previous_file(tmp_path):
    output = tmp_path / "benchmark.json"
    MODULE.atomic_write_report(output, {"complete": False, "cases": [1]})
    MODULE.atomic_write_report(output, {"complete": True, "cases": [1, 2]})
    assert output.read_text(encoding="utf-8") == (
        '{\n  "complete": true,\n  "cases": [\n    1,\n    2\n  ]\n}\n'
    )
    assert not (tmp_path / "benchmark.json.tmp").exists()


def test_failed_preflight_replaces_stale_report_without_mutating_state(monkeypatch, tmp_path):
    class EmptySceneClient:
        def __init__(self, _url, _timeout):
            pass

        def initialize(self):
            pass

        def call_tool(self, name, arguments=None):
            if name == "render.reconstruction.catalog":
                return {
                    "success": True,
                    "backends": [
                        {
                            "id": "native",
                            "presets": [{"id": "native", "input_scale": 1.0}],
                        },
                        {
                            "id": "temporal",
                            "presets": [{"id": "quality", "input_scale": 0.75}],
                        },
                    ],
                }
            if name == "render.settings.get":
                return {
                    "success": True,
                    "settings": {
                        "scene_upscaler": "native",
                        "scene_upscaler_preset": "native",
                        "raster_backend": "3dgs",
                        "equirectangular": False,
                        "orthographic": False,
                        "apply_appearance_correction": False,
                        "split_view_mode": 0,
                    },
                }
            if name == "scene.list_nodes":
                return {"success": True, "nodes": []}
            if name == "camera.get":
                return {
                    "success": True,
                    "camera": {
                        "eye": [0.0, 0.0, 4.0],
                        "target": [0.0, 0.0, 0.0],
                        "up": [0.0, 1.0, 0.0],
                        "fov_degrees": 60.0,
                        "width": 8,
                        "height": 6,
                    },
                }
            raise AssertionError(f"preflight must not mutate state: {name}")

    monkeypatch.setattr(MODULE, "McpClient", EmptySceneClient)
    output = tmp_path / "report.json"
    output.write_text('{"complete": true}\n', encoding="utf-8")

    with pytest.raises(MODULE.BenchmarkError, match="No visible Gaussian splat"):
        MODULE.run_benchmark(
            SimpleNamespace(
                url="http://unused.test/mcp",
                timeout=1.0,
                backends="",
                width=0,
                height=0,
                frames=2,
                quality_frames=2,
                warmup_frames=1,
                performance_rounds=2,
                orbit_degrees=2.0,
                output=output,
            )
        )

    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored["complete"] is False
    assert stored["preflight"]["success"] is False
    assert stored["cases"] == []
    assert stored["restoration"] == {
        "attempted": False,
        "success": True,
        "reason": "preflight failed before any benchmark mutation",
    }


def test_progress_checkpoints_are_bounded_and_include_completion():
    checkpoints = [
        current for current in range(1, 25) if MODULE.should_report_progress(current, 24)
    ]
    assert checkpoints == [1, 6, 12, 18, 24]
    assert MODULE.format_duration(0.2) == "0s"
    assert MODULE.format_duration(65.0) == "1m 05s"


def test_completed_report_survives_final_restoration_failure(monkeypatch, tmp_path):
    class FakeClient:
        latest = None

        def __init__(self, _url, _timeout):
            self.backend = "native"
            self.events = []
            FakeClient.latest = self

        def initialize(self):
            pass

        def call_tool(self, name, arguments=None):
            arguments = arguments or {}
            if name == "render.reconstruction.catalog":
                return {
                    "success": True,
                    "backends": [
                        {
                            "id": "native",
                            "presets": [{"id": "native", "input_scale": 1.0}],
                        },
                        {
                            "id": "spatial",
                            "presets": [{"id": "quality", "input_scale": 0.75}],
                        },
                    ],
                }
            if name == "render.settings.get":
                return {
                    "success": True,
                    "settings": {
                        "scene_upscaler": "native",
                        "scene_upscaler_preset": "native",
                        "raster_backend": "3dgs",
                        "equirectangular": False,
                        "orthographic": False,
                        "apply_appearance_correction": False,
                        "split_view_mode": 0,
                    },
                }
            if name == "scene.list_nodes":
                return {
                    "success": True,
                    "nodes": [{"type": "splat", "gaussian_count": 1000}],
                }
            if name == "camera.get":
                return {
                    "success": True,
                    "camera": {
                        "eye": [0.0, 0.0, 4.0],
                        "target": [0.0, 0.0, 0.0],
                        "up": [0.0, 1.0, 0.0],
                        "fov_degrees": 60.0,
                        "width": 8,
                        "height": 6,
                    },
                }
            if name == "render.settings.set":
                self.backend = arguments["scene_upscaler"]
                return {"success": True}
            if name == "camera.set_view":
                raise MODULE.BenchmarkError("synthetic restoration failure")
            if name == "render.reconstruction.sample_frame":
                self.events.append("sample")
                return {
                    "success": True,
                    "requested": self.backend,
                    "effective": self.backend,
                    "fallback": "none",
                    "fell_back": False,
                    "frame_latency_ms": 10.0,
                    "timing_scope": "camera_mutation_to_post_render_without_readback",
                }
            if name == "render.reconstruction.status":
                self.events.append("status")
                return {
                    "success": True,
                    "requested": self.backend,
                    "effective": self.backend,
                    "fallback": "none",
                    "fell_back": False,
                    "convergence_remaining": 0,
                }
            raise AssertionError(f"unexpected tool: {name}")

    monkeypatch.setattr(MODULE, "McpClient", FakeClient)
    captures = []

    def fake_capture(client, width, height):
        client.events.append("capture")
        captures.append((width, height))
        return np.zeros((height, width, 3), dtype=np.float32)

    monkeypatch.setattr(MODULE, "capture", fake_capture)
    output = tmp_path / "report.json"
    report = MODULE.run_benchmark(
        SimpleNamespace(
            url="http://unused.test/mcp",
            timeout=1.0,
            backends="",
            width=0,
            height=0,
            frames=2,
            quality_frames=2,
            warmup_frames=1,
            performance_rounds=2,
            orbit_degrees=2.0,
            output=output,
        )
    )

    assert report["complete"] is True
    assert report["schema_version"] == 4
    assert report["performance"]["timing_scope"] == (
        "camera_mutation_to_post_render_without_readback"
    )
    assert report["preflight"]["visible_gaussians"] == 1000
    assert len(report["cases"]) == 2
    assert len(captures) == 4
    assert report["performance"]["complete"] is True
    assert report["quality"]["complete"] is True
    assert report["performance"]["round_orders"] == [
        ["native:native", "spatial:quality"],
        ["spatial:quality", "native:native"],
    ]
    first_capture = FakeClient.latest.events.index("capture")
    assert FakeClient.latest.events[first_capture - 1] == "status"
    assert all(
        event == "status"
        for index, event in enumerate(FakeClient.latest.events)
        if index + 1 < len(FakeClient.latest.events)
        and FakeClient.latest.events[index + 1] == "capture"
    )
    assert all(
        case["performance"]["viewport_fps_from_median"] == pytest.approx(100.0)
        for case in report["cases"]
    )
    assert all(len(case["performance"]["samples_ms"]) == 2 for case in report["cases"])
    assert all(len(case["performance"]["rounds"]) == 2 for case in report["cases"])
    assert report["restoration"]["success"] is False
    assert "synthetic restoration failure" in report["restoration"]["error"]
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored == report


def test_orbit_path_is_deterministic_and_preserves_radius():
    camera = {
        "eye": [0.0, 0.0, 4.0],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0],
        "fov_degrees": 60.0,
    }
    first = MODULE.orbit_views(camera, 5, 10.0)
    second = MODULE.orbit_views(camera, 5, 10.0)
    assert first == second
    assert len(first) == 5
    for view in first:
        assert np.linalg.norm(np.asarray(view["eye"])) == pytest.approx(4.0)
