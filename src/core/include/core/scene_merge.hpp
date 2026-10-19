/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/scene.hpp"

namespace lfs::core {

    struct GroupMergeRemovalPlan {
        std::vector<NodeId> removed;
        size_t kept = 0;
    };

    [[nodiscard]] inline GroupMergeRemovalPlan planGroupMergeRemoval(const Scene& scene, const NodeId group) {
        GroupMergeRemovalPlan plan;
        const auto visit = [&](const auto& self, const NodeId id) -> void {
            const auto* node = scene.getNodeById(id);
            if (!node)
                return;
            const auto* parent = scene.getNodeById(node->parent_id);
            const bool splat_helper = (node->type == NodeType::CROPBOX || node->type == NodeType::ELLIPSOID) &&
                                      parent && parent->type == NodeType::SPLAT && parent->model;
            if (node->type == NodeType::GROUP || (node->type == NodeType::SPLAT && node->model) || splat_helper)
                plan.removed.push_back(id);
            else
                ++plan.kept;
            for (const auto child : node->children)
                self(self, child);
        };
        visit(visit, group);
        return plan;
    }

    inline void removeGroupForMerge(Scene& scene, const GroupMergeRemovalPlan& plan) {
        if (plan.removed.empty())
            return;
        if (plan.kept == 0) {
            scene.removeNodeById(plan.removed.front(), false);
            return;
        }
        // Remove consumed nodes from the leaves up. Remaining children retain their
        // attachments and inherit the removed parent's pose without a matrix inverse.
        for (auto it = plan.removed.rbegin(); it != plan.removed.rend(); ++it) {
            const auto* node = scene.getNodeById(*it);
            if (!node)
                continue;
            for (const auto child_id : node->children) {
                scene.setNodeTransform(child_id, node->local_transform.get() * scene.getNodeTransform(child_id));
                if (!node->visible)
                    scene.setNodeVisibility(child_id, false);
            }
            scene.removeNodeById(*it, true);
        }
    }

} // namespace lfs::core
