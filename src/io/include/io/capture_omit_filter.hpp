/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "core/scene.hpp"
#include "core/uuid.hpp"

#include <span>
#include <unordered_set>

namespace lfs::io::project {

    // Closes omit roots over scene descendants once. Unresolvable roots stay
    // in the set. KEYFRAME exclusion is orthogonal and lives in each capture.
    class CaptureOmitFilter {
    public:
        CaptureOmitFilter(
            const lfs::core::Scene& scene,
            std::span<const lfs::core::Uuid> omit_roots) {
            omitted_.insert(omit_roots.begin(), omit_roots.end());
            for (const auto& uuid : omit_roots) {
                const auto* node = scene.getNodeByUuid(uuid);
                if (node == nullptr) {
                    continue;
                }
                collect_descendants(scene, *node);
            }
        }

        [[nodiscard]] bool omits(
            const lfs::core::Uuid& uuid) const noexcept {
            return omitted_.contains(uuid);
        }

    private:
        void collect_descendants(
            const lfs::core::Scene& scene,
            const lfs::core::SceneNode& node) {
            for (const lfs::core::NodeId child_id : node.children) {
                const lfs::core::SceneNode* child =
                    scene.getNodeById(child_id);
                if (child == nullptr) {
                    continue;
                }
                omitted_.insert(child->uuid);
                collect_descendants(scene, *child);
            }
        }

        std::unordered_set<lfs::core::Uuid> omitted_;
    };

} // namespace lfs::io::project
