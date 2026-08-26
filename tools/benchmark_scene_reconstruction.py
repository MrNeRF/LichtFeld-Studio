#!/usr/bin/env python3
"""Benchmark registered viewport reconstruction backends through the live MCP API.

The benchmark deliberately uses the application's production viewport path. It
keeps performance and image quality as globally separate phases: every performance
case completes before the first PNG capture, then quality uses sparse Native and
reconstructed captures for PSNR/SSIM. Performance cases are distributed over
deterministic rotated/reversed rounds to reduce ordering and thermal bias. Neither
phase is an isolated GPU microbenchmark, but capture encoding and HTTP payload
transfer no longer depress the reported viewport frame rate.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


class BenchmarkError(RuntimeError):
    pass


class McpClient:
    def __init__(self, url: str, timeout_seconds: float = 60.0) -> None:
        self.url = url
        self.timeout_seconds = timeout_seconds
        self._request_id = 0

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params or {},
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                document = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise BenchmarkError(f"MCP request {method!r} failed: {error}") from error
        if "error" in document:
            raise BenchmarkError(f"MCP {method!r}: {document['error']}")
        return document.get("result", {})

    def initialize(self) -> None:
        self.request(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "lichtfeld-scene-reconstruction-benchmark", "version": "1"},
            },
        )

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        result = self.request(
            "tools/call",
            {"name": name, "arguments": arguments or {}},
        )
        structured = result.get("structuredContent", {})
        if result.get("isError") or (isinstance(structured, dict) and structured.get("error")):
            raise BenchmarkError(f"MCP tool {name!r}: {structured.get('error', structured)}")
        if not isinstance(structured, dict):
            raise BenchmarkError(f"MCP tool {name!r} returned no structured content")
        return structured


@dataclass(frozen=True)
class BenchmarkCase:
    backend: str
    preset: str
    input_scale: float

    @property
    def key(self) -> str:
        return f"{self.backend}:{self.preset}"


def benchmark_cases(catalog: dict[str, Any]) -> list[BenchmarkCase]:
    result: list[BenchmarkCase] = []
    for backend in catalog.get("backends", []):
        backend_id = str(backend.get("id", ""))
        for preset in backend.get("presets", []):
            preset_id = str(preset.get("id", ""))
            if backend_id and preset_id:
                result.append(
                    BenchmarkCase(
                        backend=backend_id,
                        preset=preset_id,
                        input_scale=float(preset.get("input_scale", 1.0)),
                    )
                )
    if not result or not any(case.backend == "native" for case in result):
        raise BenchmarkError("The reconstruction catalog has no Native reference case")
    return result


def decode_png(data: str) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as error:
        raise BenchmarkError(
            "Pillow is required to decode MCP captures; run with "
            "`uv run --no-project --with numpy --with pillow python "
            "tools/benchmark_scene_reconstruction.py ...`"
        ) from error
    try:
        with Image.open(io.BytesIO(base64.b64decode(data))) as image:
            return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    except Exception as error:
        raise BenchmarkError(f"Could not decode render.capture PNG: {error}") from error


def quality_metrics(image: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    if image.shape != reference.shape or image.ndim != 3 or image.shape[2] != 3:
        raise BenchmarkError(
            f"Capture shape {image.shape} does not match Native reference {reference.shape}"
        )
    image64 = image.astype(np.float64, copy=False)
    reference64 = reference.astype(np.float64, copy=False)
    difference = image64 - reference64
    mse = float(np.mean(difference * difference))
    psnr = math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse)

    axes = (0, 1)
    image_mean = np.mean(image64, axis=axes)
    reference_mean = np.mean(reference64, axis=axes)
    image_delta = image64 - image_mean
    reference_delta = reference64 - reference_mean
    image_variance = np.mean(image_delta * image_delta, axis=axes)
    reference_variance = np.mean(reference_delta * reference_delta, axis=axes)
    covariance = np.mean(image_delta * reference_delta, axis=axes)
    c1 = 0.01**2
    c2 = 0.03**2
    ssim_channels = (
        (2.0 * image_mean * reference_mean + c1) * (2.0 * covariance + c2)
    ) / (
        (image_mean * image_mean + reference_mean * reference_mean + c1)
        * (image_variance + reference_variance + c2)
    )
    return psnr, float(np.mean(ssim_channels))


def summarize_milliseconds(samples: list[float], rate_name: str) -> dict[str, float]:
    if not samples or any(not math.isfinite(value) or value <= 0.0 for value in samples):
        raise BenchmarkError("Timing samples must be finite positive milliseconds")
    values = np.asarray(samples, dtype=np.float64)
    sorted_values = np.sort(values)
    trim_count = math.floor(len(sorted_values) * 0.1)
    trimmed_values = (
        sorted_values[trim_count:-trim_count]
        if trim_count > 0 and trim_count * 2 < len(sorted_values)
        else sorted_values
    )
    median = float(np.median(values))
    result = {
        "sample_count": len(samples),
        "median_ms": median,
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": float(np.mean(values)),
        "trimmed_mean_ms": float(np.mean(trimmed_values)),
        "stddev_ms": float(np.std(values)),
        "minimum_ms": float(np.min(values)),
        "maximum_ms": float(np.max(values)),
    }
    result[rate_name] = 1000.0 / median
    return result


def split_frame_counts(frame_count: int, round_count: int) -> list[int]:
    if frame_count < 1 or round_count < 1 or round_count > frame_count:
        raise BenchmarkError(
            "Performance rounds must be positive and cannot exceed measured frames"
        )
    base, remainder = divmod(frame_count, round_count)
    return [base + (1 if index < remainder else 0) for index in range(round_count)]


def performance_case_order(
    cases: list[BenchmarkCase], round_index: int
) -> list[BenchmarkCase]:
    if not cases:
        return []
    offset = (round_index // 2) % len(cases)
    ordered = cases[offset:] + cases[:offset]
    if round_index % 2 == 1:
        ordered.reverse()
    return ordered


def summarize_quality(psnr_values: list[float], ssim_values: list[float]) -> dict[str, Any]:
    if not psnr_values or len(psnr_values) != len(ssim_values):
        raise BenchmarkError("Quality samples must be non-empty and paired")
    if any(math.isnan(value) or value < 0.0 for value in psnr_values):
        raise BenchmarkError("PSNR samples must be non-negative numbers")
    if any(not math.isfinite(value) for value in ssim_values):
        raise BenchmarkError("SSIM samples must be finite numbers")

    mse_values = [0.0 if math.isinf(value) else 10.0 ** (-value / 10.0) for value in psnr_values]
    aggregate_mse = float(np.mean(mse_values))
    aggregate_psnr = (
        None if aggregate_mse == 0.0 else 10.0 * math.log10(1.0 / aggregate_mse)
    )
    finite_psnr = [value for value in psnr_values if math.isfinite(value)]
    return {
        "aggregate_psnr_db": aggregate_psnr,
        "minimum_psnr_db": min(finite_psnr) if finite_psnr else None,
        "exact_match_frames": sum(math.isinf(value) for value in psnr_values),
        "mean_ssim": float(np.mean(ssim_values)),
        "minimum_ssim": float(np.min(ssim_values)),
    }


def render_settings_snapshot(payload: dict[str, Any]) -> dict[str, str]:
    settings = payload.get("settings")
    if not isinstance(settings, dict):
        raise BenchmarkError("render.settings.get returned no settings object")
    required = ("scene_upscaler", "scene_upscaler_preset")
    missing = [key for key in required if not isinstance(settings.get(key), str)]
    if missing:
        raise BenchmarkError(
            "render.settings.get omitted required string field(s): " + ", ".join(missing)
        )
    return {key: settings[key] for key in required}


def benchmark_preflight(
    settings_payload: dict[str, Any],
    scene_payload: dict[str, Any],
    cases: list[BenchmarkCase],
) -> dict[str, Any]:
    settings = settings_payload.get("settings")
    if not isinstance(settings, dict):
        raise BenchmarkError("render.settings.get returned no settings object")
    nodes = scene_payload.get("nodes")
    if not scene_payload.get("success") or not isinstance(nodes, list):
        raise BenchmarkError("scene.list_nodes returned no scene node list")

    gaussian_nodes = [
        node
        for node in nodes
        if isinstance(node, dict)
        and node.get("type") in {"splat", "ply_sequence"}
        and isinstance(node.get("gaussian_count"), int)
        and node["gaussian_count"] > 0
    ]
    total_gaussians = sum(int(node["gaussian_count"]) for node in gaussian_nodes)
    temporal_requested = any(case.backend == "temporal" for case in cases)
    temporal_reasons: list[str] = []
    if temporal_requested:
        if settings.get("equirectangular") is True:
            temporal_reasons.append("equirectangular projection is Native-only")
        if settings.get("apply_appearance_correction") is True:
            temporal_reasons.append("appearance correction is Native-only")
        if settings.get("split_view_mode") == 2:
            temporal_reasons.append("ground-truth comparison split is not Temporal-compatible")
        if settings.get("raster_backend") not in {"3dgs", "3dgut"}:
            temporal_reasons.append("the active raster backend is not VkSplat")

    errors: list[str] = []
    if not gaussian_nodes:
        errors.append(
            "No visible Gaussian splat is loaded. Load a non-empty splat scene before "
            "running the scene-reconstruction benchmark."
        )
    if temporal_reasons:
        errors.append(
            "Temporal reconstruction is not eligible in the current viewport: "
            + "; ".join(temporal_reasons)
        )

    return {
        "success": not errors,
        "errors": errors,
        "visible_gaussian_nodes": len(gaussian_nodes),
        "visible_gaussians": total_gaussians,
        "temporal_requested": temporal_requested,
        "temporal_eligible": temporal_requested and bool(gaussian_nodes) and not temporal_reasons,
        "temporal_ineligibility_reasons": temporal_reasons,
        "render_context": {
            "raster_backend": settings.get("raster_backend"),
            "equirectangular": settings.get("equirectangular"),
            "orthographic": settings.get("orthographic"),
            "apply_appearance_correction": settings.get("apply_appearance_correction"),
            "split_view_mode": settings.get("split_view_mode"),
        },
    }


def atomic_write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def format_duration(seconds: float) -> str:
    total_seconds = max(0, round(seconds))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    if minutes:
        return f"{minutes}m {remaining_seconds:02d}s"
    return f"{remaining_seconds}s"


def should_report_progress(current: int, total: int) -> bool:
    interval = max(1, math.ceil(total / 4))
    return current == 1 or current == total or current % interval == 0


def report_phase_progress(
    case_label: str,
    phase: str,
    current: int,
    total: int,
    started_at: float,
) -> None:
    if not should_report_progress(current, total):
        return
    elapsed = time.perf_counter() - started_at
    eta = elapsed * (total - current) / current
    print(
        f"    {case_label} {phase}: {current}/{total} "
        f"(elapsed {format_duration(elapsed)}, ETA {format_duration(eta)})",
        flush=True,
    )


def orbit_views(camera: dict[str, Any], frame_count: int, orbit_degrees: float) -> list[dict[str, Any]]:
    if frame_count < 2:
        raise BenchmarkError("At least two motion frames are required")
    eye = np.asarray(camera["eye"], dtype=np.float64)
    target = np.asarray(camera["target"], dtype=np.float64)
    up = np.asarray(camera["up"], dtype=np.float64)
    up_length = float(np.linalg.norm(up))
    radius = float(np.linalg.norm(eye - target))
    if up_length <= 1e-8 or radius <= 1e-8:
        raise BenchmarkError("Current camera has an invalid up vector or zero orbit radius")
    axis = up / up_length
    offset = eye - target
    angles = np.linspace(-0.5 * orbit_degrees, 0.5 * orbit_degrees, frame_count)
    result: list[dict[str, Any]] = []
    for angle_degrees in angles:
        angle = math.radians(float(angle_degrees))
        rotated = (
            offset * math.cos(angle)
            + np.cross(axis, offset) * math.sin(angle)
            + axis * np.dot(axis, offset) * (1.0 - math.cos(angle))
        )
        result.append(
            {
                "eye": (target + rotated).tolist(),
                "target": target.tolist(),
                "up": axis.tolist(),
                "fov_degrees": float(camera["fov_degrees"]),
            }
        )
    return result


def capture(client: McpClient, width: int, height: int) -> np.ndarray:
    payload = client.call_tool(
        "render.capture", {"width": width, "height": height, "presented": True}
    )
    if not payload.get("success") or not payload.get("data"):
        raise BenchmarkError(f"render.capture failed: {payload}")
    return decode_png(str(payload["data"]))


def select_case(client: McpClient, case: BenchmarkCase) -> None:
    client.call_tool(
        "render.settings.set",
        {"scene_upscaler": case.backend, "scene_upscaler_preset": case.preset},
    )


def set_view(client: McpClient, view: dict[str, Any]) -> None:
    client.call_tool("camera.set_view", view)


def sample_viewport_frame(client: McpClient, view: dict[str, Any]) -> dict[str, Any]:
    payload = client.call_tool("render.reconstruction.sample_frame", view)
    latency_ms = payload.get("frame_latency_ms")
    if (
        not payload.get("success")
        or not isinstance(latency_ms, (int, float))
        or not math.isfinite(float(latency_ms))
        or float(latency_ms) <= 0.0
    ):
        raise BenchmarkError(
            "render.reconstruction.sample_frame returned no valid frame latency: "
            f"{payload}"
        )
    return payload


def wait_for_convergence(
    client: McpClient,
    case: BenchmarkCase,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.perf_counter() + timeout_seconds
    while True:
        payload = client.call_tool("render.reconstruction.status")
        remaining = payload.get("convergence_remaining")
        if (
            not payload.get("success")
            or isinstance(remaining, bool)
            or not isinstance(remaining, int)
            or remaining < 0
        ):
            raise BenchmarkError(
                "render.reconstruction.status returned no valid convergence counter: "
                f"{payload}"
            )
        if payload.get("fell_back") or payload.get("effective") != case.backend:
            raise BenchmarkError(
                f"{case.key} fell back while waiting for convergence: "
                f"effective={payload.get('effective')} fallback={payload.get('fallback')}"
            )
        if remaining == 0:
            return payload
        if time.perf_counter() >= deadline:
            raise BenchmarkError(
                f"{case.key} did not converge within {timeout_seconds:.1f}s "
                f"({remaining} frame(s) remaining)"
            )
        time.sleep(0.01)


def restore_state(
    client: McpClient,
    original_settings: dict[str, Any],
    original_camera: dict[str, Any],
) -> None:
    client.call_tool(
        "render.settings.set",
        {
            "scene_upscaler": original_settings["scene_upscaler"],
            "scene_upscaler_preset": original_settings["scene_upscaler_preset"],
        },
    )
    set_view(
        client,
        {
            "eye": original_camera["eye"],
            "target": original_camera["target"],
            "up": original_camera["up"],
            "fov_degrees": original_camera["fov_degrees"],
        },
    )


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    client = McpClient(args.url, args.timeout)
    client.initialize()
    catalog = client.call_tool("render.reconstruction.catalog")
    cases = benchmark_cases(catalog)
    requested_backends = (
        {backend.strip() for backend in args.backends.split(",") if backend.strip()}
        if args.backends
        else set()
    )
    if requested_backends:
        cases = [case for case in cases if case.backend in requested_backends]
        missing = requested_backends - {case.backend for case in cases}
        if missing:
            raise BenchmarkError(f"Unknown benchmark backend(s): {', '.join(sorted(missing))}")

    settings_payload = client.call_tool("render.settings.get")
    original_settings = render_settings_snapshot(settings_payload)
    scene_payload = client.call_tool(
        "scene.list_nodes", {"include_hidden": False, "include_auxiliary": False}
    )
    preflight = benchmark_preflight(settings_payload, scene_payload, cases)
    original_camera_payload = client.call_tool("camera.get")
    original_camera = original_camera_payload.get("camera")
    if not isinstance(original_camera, dict):
        raise BenchmarkError("camera.get did not return an interactive viewport camera")
    live_width = int(original_camera.get("width", 0))
    live_height = int(original_camera.get("height", 0))
    if live_width <= 0 or live_height <= 0:
        raise BenchmarkError("The live viewport has no measurable render extent")
    if args.width == 0:
        args.width = live_width
    if args.height == 0:
        args.height = live_height
    if args.width != live_width or args.height != live_height:
        raise BenchmarkError(
            f"Requested {args.width}x{args.height}, but the live viewport is "
            f"{live_width}x{live_height}. Resize the viewport first; MCP output resizing "
            "would not benchmark that render resolution."
        )
    performance_views = orbit_views(original_camera, args.frames, args.orbit_degrees)
    quality_views = orbit_views(original_camera, args.quality_frames, args.orbit_degrees)

    native_case = next((case for case in cases if case.backend == "native"), None)
    if native_case is None:
        native_case = next(
            case for case in benchmark_cases(catalog) if case.backend == "native"
        )

    ordered_cases = [native_case] + [case for case in cases if case != native_case]
    results: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "schema_version": 4,
        "complete": False,
        "width": args.width,
        "height": args.height,
        "orbit_degrees": args.orbit_degrees,
        "preflight": preflight,
        "performance": {
            "frames": args.frames,
            "rounds": args.performance_rounds,
            "warmup_frames_per_case_per_round": args.warmup_frames,
            "round_orders": [],
            "complete": False,
            "phase_isolation": "all performance cases complete before any PNG capture",
            "ordering": "deterministic cyclic rotation with alternating reversal",
            "timing_scope": "camera_mutation_to_post_render_without_readback",
            "excludes": ["viewport_readback", "png_encoding", "base64", "http_response"],
        },
        "quality": {
            "frames": args.quality_frames,
            "warmup_frames": args.warmup_frames,
            "complete": False,
            "timing_scope": "end_to_end_mcp_render_readback_png_http",
            "reference": "native_full_resolution_presented_viewport",
            "capture_source": "presented_viewport_crop",
            "convergence": "wait_for_zero_remaining_frames_at_each_capture_pose",
        },
        "restoration": {"attempted": False, "success": False},
        "cases": results,
    }
    if not preflight["success"]:
        message = " ".join(preflight["errors"])
        report["error"] = f"BenchmarkError: {message}"
        report["restoration"] = {
            "attempted": False,
            "success": True,
            "reason": "preflight failed before any benchmark mutation",
        }
        atomic_write_report(args.output, report)
        raise BenchmarkError(message)

    result_by_key: dict[str, dict[str, Any]] = {}
    for case in ordered_cases:
        result = {
            "backend": case.backend,
            "preset": case.preset,
            "input_scale": case.input_scale,
            "performance": None,
            "quality": None,
        }
        result_by_key[case.key] = result
        results.append(result)

    performance_frame_counts = split_frame_counts(args.frames, args.performance_rounds)
    planned_performance_frames = len(ordered_cases) * args.frames
    planned_captures = len(ordered_cases) * args.quality_frames
    print(
        f"Benchmarking {len(ordered_cases)} reconstruction cases at "
        f"{args.width}x{args.height}: {planned_performance_frames} timed viewport frames "
        f"over {args.performance_rounds} performance rounds, followed by "
        f"{planned_captures} quality PNG captures.",
        flush=True,
    )
    atomic_write_report(args.output, report)

    benchmark_error: BaseException | None = None
    try:
        print("\nPhase 1/2: performance (no viewport readback or PNG capture)", flush=True)
        samples_by_key = {case.key: [] for case in ordered_cases}
        rounds_by_key = {case.key: [] for case in ordered_cases}
        performance_status_by_key: dict[str, dict[str, Any]] = {}
        frame_offset = 0
        for round_index, measured_count in enumerate(performance_frame_counts):
            round_cases = performance_case_order(ordered_cases, round_index)
            report["performance"]["round_orders"].append(
                [case.key for case in round_cases]
            )
            print(
                f"  round {round_index + 1}/{args.performance_rounds}: "
                + " -> ".join(case.key for case in round_cases),
                flush=True,
            )
            round_views = performance_views[frame_offset : frame_offset + measured_count]
            for position, case in enumerate(round_cases, start=1):
                result = result_by_key[case.key]
                print(
                    f"    [{position}/{len(round_cases)}] {case.key}: selecting and warming up",
                    flush=True,
                )
                select_case(client, case)
                phase_started = time.perf_counter()
                for index in range(1, args.warmup_frames + 1):
                    warmup_view_index = (
                        frame_offset - args.warmup_frames + index - 1
                    ) % len(performance_views)
                    warmup_status = sample_viewport_frame(
                        client, performance_views[warmup_view_index]
                    )
                    report_phase_progress(
                        case.key,
                        f"round {round_index + 1} warm-up",
                        index,
                        args.warmup_frames,
                        phase_started,
                    )
                if (
                    warmup_status.get("fell_back")
                    or warmup_status.get("effective") != case.backend
                ):
                    result["performance_status"] = warmup_status
                    result["skipped"] = True
                    result["reason"] = "requested backend was not effective after warm-up"
                    atomic_write_report(args.output, report)
                    raise BenchmarkError(
                        f"{case.key} was not effective after warm-up; refusing to publish "
                        "a complete benchmark with a silently skipped backend"
                    )

                round_samples: list[float] = []
                final_status = warmup_status
                phase_started = time.perf_counter()
                for index, view in enumerate(round_views, start=1):
                    final_status = sample_viewport_frame(client, view)
                    if (
                        final_status.get("fell_back")
                        or final_status.get("effective") != case.backend
                    ):
                        raise BenchmarkError(
                            f"{case.key} fell back during performance round "
                            f"{round_index + 1}: effective={final_status.get('effective')} "
                            f"fallback={final_status.get('fallback')}"
                        )
                    round_samples.append(float(final_status["frame_latency_ms"]))
                    report_phase_progress(
                        case.key,
                        f"round {round_index + 1} measured",
                        index,
                        len(round_views),
                        phase_started,
                    )

                samples_by_key[case.key].extend(round_samples)
                round_summary = summarize_milliseconds(
                    round_samples, "viewport_fps_from_median"
                )
                round_summary.update(
                    {
                        "round": round_index + 1,
                        "order_position": position,
                        "samples_ms": round_samples,
                    }
                )
                rounds_by_key[case.key].append(round_summary)
                performance_status_by_key[case.key] = final_status
                result["performance"] = {
                    "complete": False,
                    "samples_ms": samples_by_key[case.key],
                    "rounds": rounds_by_key[case.key],
                }
                result["performance_status"] = final_status
                atomic_write_report(args.output, report)
            frame_offset += measured_count

        for case in ordered_cases:
            summary = summarize_milliseconds(
                samples_by_key[case.key], "viewport_fps_from_median"
            )
            summary.update(
                {
                    "complete": True,
                    "samples_ms": samples_by_key[case.key],
                    "rounds": rounds_by_key[case.key],
                }
            )
            result_by_key[case.key]["performance"] = summary
            result_by_key[case.key]["performance_status"] = performance_status_by_key[
                case.key
            ]
        report["performance"]["complete"] = True
        atomic_write_report(args.output, report)

        print("\nPhase 2/2: quality (sparse full-resolution PNG readback)", flush=True)
        references: list[np.ndarray] = []
        for case_index, case in enumerate(ordered_cases, start=1):
            suffix = " (Native reference)" if case.backend == "native" else ""
            print(f"  [{case_index}/{len(ordered_cases)}] {case.key}{suffix}", flush=True)
            select_case(client, case)
            phase_started = time.perf_counter()
            for index in range(1, args.warmup_frames + 1):
                quality_warmup_status = sample_viewport_frame(
                    client, quality_views[(index - 1) % len(quality_views)]
                )
                report_phase_progress(
                    case.key, "quality warm-up", index, args.warmup_frames, phase_started
                )
            if (
                quality_warmup_status.get("fell_back")
                or quality_warmup_status.get("effective") != case.backend
            ):
                raise BenchmarkError(
                    f"{case.key} fell back before the quality phase: "
                    f"effective={quality_warmup_status.get('effective')} "
                    f"fallback={quality_warmup_status.get('fallback')}"
                )

            capture_ms: list[float] = []
            psnr_values: list[float] = []
            ssim_values: list[float] = []
            final_status = quality_warmup_status
            phase_started = time.perf_counter()
            for index, view in enumerate(quality_views, start=1):
                final_status = sample_viewport_frame(client, view)
                if final_status.get("fell_back") or final_status.get("effective") != case.backend:
                    raise BenchmarkError(
                        f"{case.key} fell back during the quality phase: "
                        f"effective={final_status.get('effective')} "
                        f"fallback={final_status.get('fallback')}"
                    )
                final_status = wait_for_convergence(client, case, args.timeout)
                capture_start = time.perf_counter()
                image = capture(client, args.width, args.height)
                capture_ms.append((time.perf_counter() - capture_start) * 1000.0)
                if case.backend == "native":
                    references.append(image)
                else:
                    psnr, ssim = quality_metrics(image, references[index - 1])
                    psnr_values.append(psnr)
                    ssim_values.append(ssim)
                report_phase_progress(
                    case.key, "quality measured", index, len(quality_views), phase_started
                )

            quality = (
                {
                    "aggregate_psnr_db": None,
                    "minimum_psnr_db": None,
                    "exact_match_frames": args.quality_frames,
                    "mean_ssim": 1.0,
                    "minimum_ssim": 1.0,
                    "reference": True,
                }
                if case.backend == "native"
                else summarize_quality(psnr_values, ssim_values)
            )
            capture_summary = summarize_milliseconds(
                capture_ms, "captures_per_second_from_median"
            )
            capture_summary["samples_ms"] = capture_ms
            quality["capture_timing"] = capture_summary
            result_by_key[case.key]["quality"] = quality
            result_by_key[case.key]["quality_status"] = final_status
            atomic_write_report(args.output, report)
        report["quality"]["complete"] = True
        report["complete"] = True
    except BaseException as error:
        benchmark_error = error
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["restoration"]["attempted"] = True
        try:
            restore_state(client, original_settings, original_camera)
            report["restoration"]["success"] = True
        except Exception as restore_error:
            report["restoration"]["error"] = (
                f"{type(restore_error).__name__}: {restore_error}"
            )
            print(
                f"warning: benchmark state could not be fully restored: {restore_error}",
                file=sys.stderr,
            )
        try:
            atomic_write_report(args.output, report)
        except Exception as report_error:
            if benchmark_error is None:
                raise BenchmarkError(
                    f"Could not write benchmark report: {report_error}"
                ) from report_error
            print(
                f"warning: partial benchmark report could not be written: {report_error}",
                file=sys.stderr,
            )

    return report


def print_results(report: dict[str, Any]) -> None:
    print(
        "backend:preset                 PSNR dB    SSIM    viewport p50   trimmed mean   p95    viewport FPS   PNG p50"
    )
    for case in report["cases"]:
        label = f"{case['backend']}:{case['preset']}"
        if case.get("skipped"):
            print(f"{label:<30} SKIPPED ({case['reason']})")
            continue
        quality = case["quality"]
        performance = case["performance"]
        capture = quality["capture_timing"]
        if quality.get("reference"):
            psnr = "reference"
        elif quality["aggregate_psnr_db"] is None:
            psnr = "exact"
        else:
            psnr = f"{quality['aggregate_psnr_db']:.3f}"
        print(
            f"{label:<30} {psnr:>9}  {quality['mean_ssim']:.5f}  "
            f"{performance['median_ms']:>8.2f} ms  "
            f"{performance['trimmed_mean_ms']:>10.2f} ms  "
            f"{performance['p95_ms']:>7.2f} ms  "
            f"{performance['viewport_fps_from_median']:>10.2f}  "
            f"{capture['median_ms']:>8.2f} ms"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark registered scene-reconstruction backends through the live "
            "LichtFeld viewport and MCP server. The application must already be "
            "running with a visible non-empty Gaussian scene."
        ),
        epilog="""examples:
  uv run --no-project --with numpy --with pillow python tools/benchmark_scene_reconstruction.py --help
  uv run --no-project --with numpy --with pillow python tools/benchmark_scene_reconstruction.py --frames 16 --quality-frames 4 --warmup-frames 4 --performance-rounds 2
  uv run --no-project --with numpy --with pillow python tools/benchmark_scene_reconstruction.py --frames 128 --quality-frames 8 --warmup-frames 8 --performance-rounds 4
  uv run --no-project --with numpy --with pillow python tools/benchmark_scene_reconstruction.py --backends native,spatial,temporal
  uv run --no-project --with numpy --with pillow python tools/benchmark_scene_reconstruction.py --url http://127.0.0.1:45677/mcp --output build/reconstruction.json

The performance phase completes for every case before any PNG capture begins.
--frames is the total measured frame count per case, divided across the requested
rounds; it is not multiplied by --performance-rounds. Quality capture waits for
the reported reconstruction convergence counter to reach zero at every pose.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:45677/mcp",
        help="MCP HTTP endpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Expected live viewport width; 0 discovers it from camera.get (default: 0)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="Expected live viewport height; 0 discovers it from camera.get (default: 0)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=24,
        help="Total measured performance frames per case across all rounds (default: %(default)s)",
    )
    parser.add_argument(
        "--quality-frames",
        type=int,
        default=8,
        help="Sparse full-resolution PNG comparison frames per case (default: %(default)s)",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=8,
        help="Untimed warm-up frames per case and performance round, and per quality case (default: %(default)s)",
    )
    parser.add_argument(
        "--performance-rounds",
        type=int,
        default=3,
        help="Rotated/reversed performance rounds used to reduce order bias (default: %(default)s)",
    )
    parser.add_argument(
        "--orbit-degrees",
        type=float,
        default=8.0,
        help="Total deterministic camera-orbit span in degrees (default: %(default)s)",
    )
    parser.add_argument(
        "--backends",
        default="",
        help="Comma-separated backend IDs to benchmark; Native is always added as the quality reference",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="MCP request and per-pose convergence timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/scene-reconstruction-benchmark.json"),
        help="Atomic JSON report path (default: %(default)s)",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (
        args.width < 0
        or args.height < 0
        or args.frames < 2
        or args.quality_frames < 2
        or args.warmup_frames < 1
        or args.performance_rounds < 1
        or args.performance_rounds > args.frames
        or args.orbit_degrees <= 0.0
        or args.timeout <= 0.0
    ):
        parser.error(
            "width/height cannot be negative, frames/quality-frames >= 2, "
            "warmup-frames >= 1, 1 <= performance-rounds <= frames, and "
            "orbit-degrees/timeout > 0"
        )
    if (args.width == 0) != (args.height == 0):
        parser.error("width and height must either both be 0 or both be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        report = run_benchmark(args)
        print_results(report)
        print(f"\nDetailed report: {args.output}")
        print(
            "Viewport FPS excludes capture/readback/PNG/HTTP response time; "
            "PNG latency is reported separately."
        )
        print(
            "Viewport timing covers camera mutation through post-render command recording, "
            "not isolated GPU execution."
        )
        if not report["restoration"]["success"]:
            print("benchmark completed, but the original viewport state was not restored")
            return 3
        return 0
    except BenchmarkError as error:
        print(f"benchmark failed: {error}")
        if args.output.exists():
            print(f"Partial report: {args.output}")
        return 2
    except Exception as error:
        print(f"benchmark failed unexpectedly: {type(error).__name__}: {error}")
        if args.output.exists():
            print(f"Partial report: {args.output}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
