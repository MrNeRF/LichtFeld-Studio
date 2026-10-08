/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/scene.hpp"

namespace lfs::vis::gui {

    [[nodiscard]] inline bool showGroupMergeAction(const core::Scene& scene, const core::NodeId id) {
        const auto* node = scene.getNodeById(id);
        if (!node || node->type != core::NodeType::GROUP) {
            return false;
        }
        const auto has_source = [&](const auto& self, const core::SceneNode& current) -> bool {
            if (current.type == core::NodeType::SPLAT && current.model) {
                return true;
            }
            for (const auto child_id : current.children) {
                if (const auto* child = scene.getNodeById(child_id); child && self(self, *child)) {
                    return true;
                }
            }
            return false;
        };
        return has_source(has_source, *node);
    }

} // namespace lfs::vis::gui
