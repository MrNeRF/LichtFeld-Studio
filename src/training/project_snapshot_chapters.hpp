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
#include "training_snapshot_service.hpp"

#include <span>

namespace lfs::core {
    class Scene;
}

namespace lfs::training {

    struct ProjectSnapshotChapters {
        lfs::core::Uuid snapshot_uuid;
        int iteration = 0;
        lfs::io::project::SceneGraphChapter scene_graph;
        lfs::io::project::SelectionChapter selection;
        lfs::io::project::ParameterManagerSnapshot parameters;
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
