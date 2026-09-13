# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Structural regressions for training-manager ownership transactions."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_pose_preflight_is_shared_by_start_resume_and_mcp_routes():
    manager = (PROJECT_ROOT / "src/visualizer/training/training_manager.cpp").read_text(encoding="utf-8")
    preflight = manager[manager.index("TrainerManager::preflightStartParameters()"):
                        manager.index("TrainerManager::rejectStart(")]
    assert "trainer_->parameterUpdateError(pendingParamsCandidate())" in preflight
    start = manager[manager.index("bool TrainerManager::startTraining()"):
                    manager.index("TrainerManager::preflightStartParameters()")]
    assert start.index("parameterUpdateError(start_params)") < start.index("transitionTo(TrainingState::Starting)")
    resume = manager[manager.index("lfs::Status TrainerManager::resumeTraining()"):
                     manager.index("void TrainerManager::", manager.index("lfs::Status TrainerManager::resumeTraining()"))]
    assert resume.index("preflightStartParameters()") < resume.index("transitionTo(TrainingState::Running)")
    trainer = (PROJECT_ROOT / "src/training/trainer.cpp").read_text(encoding="utf-8")
    assert trainer.count("cameraPoseUpdateErrorLocked(params)") == 2
    mcp = (PROJECT_ROOT / "src/app/mcp_gui_tools.cpp").read_text(encoding="utf-8")
    start_route = mcp[mcp.index(".start_training ="):mcp.index(".start_training =") + 1600]
    assert "viewer->startTraining()" in start_route
    assert "waitForInitialization()" in start_route
    assert "format_for_developer" not in start_route
    runtime = (PROJECT_ROOT / "src/app/mcp_runtime_tools.cpp").read_text(encoding="utf-8")
    assert "if (auto resumed = trainer->resumeTraining(); !resumed)" in runtime
    assert 'return std::unexpected(std::string(resumed.error().user_message()))' in runtime


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
