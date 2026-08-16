# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Structural regressions for training-manager ownership transactions."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_exportable_grow_restores_vulkan_interop_after_detach_failures():
    source = (PROJECT_ROOT / "src/visualizer/training/training_manager.cpp").read_text(
        encoding="utf-8"
    )
    start = source.index("bool TrainerManager::growExportableForDensify")
    end = source.index("void TrainerManager::setupStateMachineCallbacks", start)
    body = source[start:end]

    detach = body.index("if (!rebindExportableCudaOnly())")
    restore_guard = body.index("auto restore_vulkan_interop = ScopeExit", detach)
    guard_reimport = body.index("if (!rebindExportableVulkanInterop())", restore_guard)
    reimport = body.index("if (!rebindExportableVulkanInterop())", guard_reimport + 1)
    release = body.index("restore_vulkan_interop.release()", reimport)

    # The active scope guard spans every early return after a successful detach,
    # and is dismissed only after the Vulkan re-import succeeds.
    assert detach < restore_guard < reimport < release
    assert "void release() noexcept { active_ = false; }" in source[:start]
