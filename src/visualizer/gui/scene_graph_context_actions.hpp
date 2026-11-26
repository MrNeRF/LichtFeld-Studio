/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/point_cloud.hpp"
#include "core/scene.hpp"

namespace lfs::vis::gui {

    [[nodiscard]] inline bool isNodeComparableForSplitView(const core::Scene& scene, const core::NodeId id) {
        const auto* node = scene.getNodeById(id);
        if (!node) {
            return false;
        }
        switch (node->type) {
        case core::NodeType::SPLAT:
            if (node->model && node->model->size() > 0) {
                return true;
            }
            for (const auto& slot : scene.getVisibleSplatNodeSlots()) {
                if (slot.node && slot.node->id == node->id) {
                    return true;
                }
            }
            return false;
        case core::NodeType::POINTCLOUD:
            return node->point_cloud && node->point_cloud->size() > 0;
        case core::NodeType::MESH:
            return node->mesh != nullptr;
        default:
            return false;
        }
    }

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
