/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "core/scene.hpp"
#include "io/project_chapters.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace lfs::io::project {

    using ScenePayloadBindings =
        std::unordered_map<lfs::core::Uuid, PayloadBinding>;

    struct ScenePayloadResolver {
        std::function<lfs::Result<std::unique_ptr<lfs::core::SplatData>>(
            const PayloadBinding&)>
            splat;
        std::function<lfs::Result<std::shared_ptr<lfs::core::PointCloud>>(
            const PayloadBinding&)>
            point_cloud;
        std::function<lfs::Result<std::shared_ptr<lfs::core::MeshData>>(
            const PayloadBinding&)>
            mesh;
    };

    // Detached value-only scene state. Capturing this does not allocate or
    // mutate a JSON DOM, so it is safe to use in a bounded optimizer
    // safe-point before chapter materialization resumes off the live scene.
    struct CapturedSceneGraphState {
        std::optional<lfs::core::Uuid> training_model_uuid;
        std::vector<SceneNodeRecord> nodes;
    };

    [[nodiscard]] LFS_IO_API lfs::Result<CapturedSceneGraphState>
    capture_scene_graph_state(
        const lfs::core::Scene& scene,
        const ScenePayloadBindings& payload_bindings,
        std::span<const lfs::core::Uuid> omit_node_uuids = {});

    [[nodiscard]] LFS_IO_API lfs::Result<SceneGraphChapter>
    materialize_scene_graph_chapter(CapturedSceneGraphState state);

    [[nodiscard]] LFS_IO_API lfs::Result<SceneGraphChapter>
    capture_scene_graph(
        const lfs::core::Scene& scene,
        const ScenePayloadBindings& payload_bindings,
        std::span<const lfs::core::Uuid> omit_node_uuids = {});

    // Phase-A API. The returned scene owns all restored nodes and payloads,
    // while its observables are already bound to target. target is not
    // mutated.
    [[nodiscard]] LFS_IO_API
        lfs::Result<std::unique_ptr<lfs::core::Scene>>
        stage_scene_graph(const SceneGraphChapter& chapter,
                          lfs::core::Scene& target,
                          const ScenePayloadResolver& resolver);

    // Phase-A shell restore. Geometry nodes are restored as named, visible
    // unloaded units; no heavy payload resolver is invoked.
    [[nodiscard]] LFS_IO_API
        lfs::Result<std::unique_ptr<lfs::core::Scene>>
        stage_scene_shell(const SceneGraphChapter& chapter,
                          lfs::core::Scene& target);

    // Transactional convenience wrapper for SCNG alone.
    [[nodiscard]] LFS_IO_API lfs::Result<void>
    hydrate_scene_graph(const SceneGraphChapter& chapter, lfs::core::Scene& scene,
                        const ScenePayloadResolver& resolver);

} // namespace lfs::io::project
