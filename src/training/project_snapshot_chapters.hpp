/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "core/error.hpp"
#include "core/parameters.hpp"
#include "core/uuid.hpp"
#include "io/project_chapters.hpp"
#include "io/project_container.hpp"
#include "io/scene_chapter_adapter.hpp"
#include "io/selection_chapter.hpp"
#include "io/session_chapters.hpp"
#include "training_snapshot_service.hpp"

#include <filesystem>
#include <optional>
#include <span>
#include <vector>

namespace lfs::core {
    class Scene;
}

namespace lfs::training {

    // GUI-originated safe-point saves carry a detached copy of every
    // non-training chapter that can change outside the optimizer. The writer
    // reopens source_path lazily for clean heavy spans, applies this bundle,
    // then installs SCNG/SELM/PRMS/CKPT from the same optimizer safe point.
    // Training-time periodic autosave does not use this path: it writes a
    // light sidecar and carry-forwards the existing CKPT binding.
    // Save As therefore publishes one destination generation without
    // inheriting an unrelated destination project's identity.
    struct ProjectSnapshotDocumentContext {
        lfs::core::Uuid project_uuid;
        lfs::core::Uuid save_as_project_uuid = {};
        std::optional<std::filesystem::path>
            source_path;
        lfs::io::project::ProjectChapter project;
        lfs::io::project::ReferencesChapter references;
        lfs::io::project::GuiLayoutChapter gui_layout;
        lfs::io::project::ViewSessionChapter view;
        lfs::io::project::EditorSessionChapter editor;
        lfs::io::project::SequencerSessionChapter
            sequencer;
        lfs::io::project::MetricsChapter metrics;
        std::vector<lfs::core::Uuid>
            selected_node_uuids;
        lfs::io::project::ParameterManagerSnapshot
            parameters;
        lfs::io::project::CommitKind
            durable_commit_kind =
                lfs::io::project::CommitKind::Explicit;
        bool allow_existing_destination_replacement = false;
        std::optional<
            lfs::io::project::WriterLockLease>
            writer_lock_lease = std::nullopt;
    };

    struct ProjectSnapshotChapters {
        lfs::core::Uuid snapshot_uuid;
        int iteration = 0;
        lfs::io::project::SceneGraphChapter scene_graph;
        lfs::io::project::SelectionChapter selection;
        lfs::io::project::ParameterManagerSnapshot parameters;
        std::optional<ProjectSnapshotDocumentContext>
            document_context;
    };

    // Detached source of truth captured inside the optimizer safe-point.
    // JSON/DOM assembly is deliberately absent; only owned value state and
    // selection bytes are copied while the optimizer is quiescent.
    struct ProjectSnapshotCpuState {
        lfs::core::Uuid snapshot_uuid;
        int iteration = 0;
        lfs::io::project::CapturedSceneGraphState
            scene_graph;
        lfs::io::project::CapturedSelectionState
            selection;
        lfs::io::project::ParameterManagerSnapshot
            parameters;
    };

    [[nodiscard]] lfs::Result<TrainingSnapshotCpuStateMetrics>
    capture_project_snapshot_cpu_state(
        const lfs::core::Scene& scene,
        const lfs::io::project::ParameterManagerSnapshot&
            parameters,
        const lfs::core::Uuid& snapshot_uuid,
        int iteration,
        ProjectSnapshotCpuState& output,
        std::span<const lfs::core::Uuid>
            selected_node_uuids = {});

    // Fallback for callers without a live ParameterManager (for example,
    // headless periodic checkpoint saves). GUI training saves must use the
    // full role-qualified snapshot overload above.
    [[nodiscard]] lfs::Result<TrainingSnapshotCpuStateMetrics>
    capture_project_snapshot_cpu_state(
        const lfs::core::Scene& scene,
        const lfs::core::param::TrainingParameters&
            checkpoint_params,
        const lfs::core::Uuid& snapshot_uuid,
        int iteration,
        ProjectSnapshotCpuState& output,
        std::span<const lfs::core::Uuid>
            selected_node_uuids = {});

    // Builds JSON/DOM-backed chapters exclusively from a detached safe-point
    // copy. This may run after the optimizer is allowed to mutate again.
    [[nodiscard]] lfs::Result<void>
    materialize_project_snapshot_cpu_chapters(
        ProjectSnapshotCpuState state,
        ProjectSnapshotChapters& output);

    void absolutize_dataset_path_for_snapshot(
        std::filesystem::path& path);

} // namespace lfs::training
