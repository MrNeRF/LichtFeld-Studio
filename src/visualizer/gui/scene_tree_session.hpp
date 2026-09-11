/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <core/scene.hpp>

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

namespace lfs::vis::gui {

    struct SceneTreeSessionChrome {
        std::vector<std::string> collapsed_uuids;
        bool models_collapsed = false;
        std::string filter_text;
    };

    [[nodiscard]] inline std::unordered_set<core::NodeId>
    collapsedIdsFromUuids(
        const core::Scene& scene,
        const std::vector<std::string>& uuids) {
        std::unordered_set<core::NodeId> ids;
        ids.reserve(uuids.size());
        for (const auto& text : uuids) {
            const auto uuid = core::Uuid::from_string(text);
            if (!uuid || uuid->is_nil())
                continue;
            const auto id = scene.getNodeIdByUuid(*uuid);
            if (id != core::NULL_NODE)
                ids.insert(id);
        }
        return ids;
    }

    [[nodiscard]] inline std::vector<std::string>
    collapsedUuidsFromIds(
        const core::Scene& scene,
        const std::unordered_set<core::NodeId>& ids) {
        std::vector<std::string> uuids;
        uuids.reserve(ids.size());
        for (const auto id : ids) {
            const auto uuid = scene.getNodeUuid(id);
            if (!uuid.is_nil())
                uuids.push_back(uuid.to_string());
        }
        std::sort(uuids.begin(), uuids.end());
        return uuids;
    }

} // namespace lfs::vis::gui
