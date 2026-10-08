"""CPU-only source contracts for render-on-demand."""

from __future__ import annotations

import re
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_render_dirty_requires_a_reason() -> None:
    header = (ROOT / "src/visualizer/rendering/rendering_manager.hpp").read_text()
    assert re.search(r"void\s+markDirty\(DirtyMask\s+flags,\s*FrameReason\s+reason", header)
    assert not re.search(r"void\s+markDirty\(\s*DirtyMask\s+flags\s*\)", header)


def test_frame_ledger_query_does_not_post_gui_work() -> None:
    source = (ROOT / "src/app/mcp_runtime_tools.cpp").read_text()
    start = source.index('.name = "runtime.frame_ledger"')
    end = source.index('.name = "runtime.job.describe"', start)
    assert "post_and_wait" not in source[start:end]
    assert "ledger.snapshot()" in source[start:end]


def test_python_frame_callbacks_have_a_bounded_legacy_lifetime() -> None:
    source = (ROOT / "src/python/runner.cpp").read_text()
    assert "duration_s.value_or(10.0)" in source
    assert "g_frame_callback_warn_on_expiry && !g_frame_callback_deprecation_logged" in source


def test_view_and_gui_presents_are_gated_by_the_ledger_plan() -> None:
    source = (ROOT / "src/visualizer/visualizer_impl.cpp").read_text()
    render = source[source.index("const FrameDemand frame_demand"):source.index("bool VisualizerImpl::allowclose")]
    assert "if (gui_frame_rendered_ && !ledger_plan.present)" in render
    assert "if (ledger_plan.render_views != 0" in render


def test_idle_event_wait_has_no_default_timeout() -> None:
    source = (ROOT / "src/visualizer/visualizer_impl.cpp").read_text()
    assert "is_training ? 0.1 : 0.5" not in source
    assert "window_manager_->waitEvents(0.1)" not in source
    assert not re.search(r"waitEvents\(\s*\d+(?:\.\d+)?\s*\)", source)


def test_python_panel_updates_default_to_dirty() -> None:
    source = (ROOT / "src/python/lfs_plugins/panels.py").read_text()
    assert 'update_policy: str = "dirty"' in source
    assert 'update_interval_ms: int | None = None' in source
    assert "uses interval updates but has no update_interval_ms" in source


def test_infinite_css_animations_have_active_state_selectors() -> None:
    root = ROOT / "src/visualizer/gui/rmlui/resources"
    failures: list[str] = []
    for path in root.rglob("*.rcss"):
        text = path.read_text()
        for match in re.finditer(r"([^{}]+)\{[^{}]*animation:[^;]*infinite", text, re.DOTALL):
            selector = match.group(1).strip()
            if not re.search(r"\.(?:is-[\w-]+|transferring)", selector):
                failures.append(f"{path.relative_to(ROOT)}: {selector}")
    assert not failures, "Infinite animations need a state selector:\n" + "\n".join(failures)


def test_viewport_wait_uses_a_runtime_deadline() -> None:
    source = (ROOT / "src/visualizer/window/window_manager.cpp").read_text()
    assert "SDL_WaitEventTimeout(&event, timeout_ms)" in source
    assert re.search(r"timeout_seconds\s*\?\s*static_cast<int>\(std::ceil\(\*timeout_seconds\s*\*\s*1000\.0\)\)\s*:\s*-1", source)
    assert not re.search(r"SDL_WaitEventTimeout\([^,]+,\s*\d+\s*\)", source)


def test_builtin_plugins_do_not_redraw_from_a_loop() -> None:
    root = ROOT / "src/python/lfs_plugins"
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                continue
            if any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr in {"request_redraw", "request_redraw_after"}
                for call in ast.walk(node)
            ):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, "Built-in plugins must schedule redraws outside loops: " + ", ".join(offenders)
