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
#include <unordered_map>

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

    [[nodiscard]] LFS_IO_API lfs::Result<SceneGraphChapter>
    capture_scene_graph(const lfs::core::Scene& scene,
                        const ScenePayloadBindings& payload_bindings);

    // Phase-A API. The returned scene owns all restored nodes and payloads,
    // while its observables are already bound to target. target is not
    // mutated.
    [[nodiscard]] LFS_IO_API
        lfs::Result<std::unique_ptr<lfs::core::Scene>>
        stage_scene_graph(const SceneGraphChapter& chapter,
                          lfs::core::Scene& target,
                          const ScenePayloadResolver& resolver);

    // Transactional convenience wrapper for SCNG alone.
    [[nodiscard]] LFS_IO_API lfs::Result<void>
    hydrate_scene_graph(const SceneGraphChapter& chapter, lfs::core::Scene& scene,
                        const ScenePayloadResolver& resolver);

} // namespace lfs::io::project
