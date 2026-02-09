/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/scene.hpp"
#include <atomic>
#include <shared_mutex>
#include <span>
#include <unordered_set>
#include <vector>

namespace lfs::vis {

    class LFS_VIS_API SelectionState {
        friend class SceneManager;

    public:
        void selectNode(core::NodeId id);
        void selectNodes(std::span<const core::NodeId> ids);
        void addToSelection(core::NodeId id);
        void removeFromSelection(core::NodeId id);
        void clearNodeSelection();
        [[nodiscard]] bool isNodeSelected(core::NodeId id) const;
        [[nodiscard]] size_t selectedNodeCount() const;

        [[nodiscard]] const std::vector<bool>& getNodeMask(const core::Scene& scene) const;
        void invalidateNodeMask();

        [[nodiscard]] uint32_t generation() const { return generation_.load(std::memory_order_acquire); }

        [[nodiscard]] std::shared_mutex& mutex() const { return mutex_; }

    private:
        // Requires caller to hold shared_lock on mutex()
        [[nodiscard]] const std::unordered_set<core::NodeId>& selectedNodeIds() const;

        void bumpGeneration();

        std::unordered_set<core::NodeId> selected_nodes_;
        mutable std::vector<bool> cached_node_mask_;
        mutable bool node_mask_dirty_ = true;
        std::atomic<uint32_t> generation_{0};
        mutable std::shared_mutex mutex_;
    };

} // namespace lfs::vis
