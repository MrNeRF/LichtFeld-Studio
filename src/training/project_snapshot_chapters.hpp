/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "core/error.hpp"
#include "core/parameters.hpp"
#include "core/uuid.hpp"
#include "io/project_chapters.hpp"
#include "io/selection_chapter.hpp"
#include "io/session_chapters.hpp"
#include "training_snapshot_service.hpp"

#include <filesystem>
#include <optional>
#include <span>

namespace lfs::core {
    class Scene;
}

namespace lfs::training {

    // GUI-originated safe-point saves carry a detached copy of every
    // non-training chapter that can change outside the optimizer. The writer
    // reopens source_path lazily for clean heavy spans, applies this bundle,
    // then installs SCNG/SELM/PRMS/CKPT from the same optimizer safe point.
    // Save As therefore publishes one destination generation without
    // inheriting an unrelated destination project's identity.
    struct ProjectSnapshotDocumentContext {
        lfs::core::Uuid project_uuid;
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

    // Captures all CPU-owned training-project chapters transactionally inside
    // the optimizer safe-point window. Nothing in `output` is published until
    // SCNG, SELM, and PRMS have all succeeded for the same UUID/iteration.
    [[nodiscard]] lfs::Result<TrainingSnapshotCpuStateMetrics>
    capture_project_snapshot_cpu_chapters(
        const lfs::core::Scene& scene,
        const lfs::core::param::TrainingParameters&
            checkpoint_params,
        const lfs::core::Uuid& snapshot_uuid,
        int iteration,
        ProjectSnapshotChapters& output,
        std::span<const lfs::core::Uuid>
            selected_node_uuids = {});

} // namespace lfs::training
