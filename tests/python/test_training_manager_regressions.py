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

    assert "const std::size_t old_capacity" in body
    assert "const auto old_bytes" in body
    assert "const std::uint64_t old_generation" in body
    assert "auto grew = splat_storage_->grow(want)" in body
    assert "bindNewExportableChunks(*splat_storage_->block)" in body
    assert "rebindSplatData(*model_ptr, alloc)" in body
    assert body.count("restoreCapacity(old_capacity, old_bytes, old_generation)") == 4
    assert "detachExportable" not in body
    assert "ScopeExit" not in body
