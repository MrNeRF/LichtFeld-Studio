/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "scene/selection_state.hpp"
#include <cassert>

namespace lfs::vis {

    void SelectionState::selectNode(const core::NodeId id) {
        std::unique_lock lock(mutex_);
        selected_nodes_.clear();
        selected_nodes_.insert(id);
        node_mask_dirty_ = true;
        bumpGeneration();
    }

    void SelectionState::selectNodes(const std::span<const core::NodeId> ids) {
        std::unique_lock lock(mutex_);
        selected_nodes_.clear();
        selected_nodes_.insert(ids.begin(), ids.end());
        node_mask_dirty_ = true;
        bumpGeneration();
    }

    void SelectionState::addToSelection(const core::NodeId id) {
        std::unique_lock lock(mutex_);
        selected_nodes_.insert(id);
        node_mask_dirty_ = true;
        bumpGeneration();
    }

    void SelectionState::removeFromSelection(const core::NodeId id) {
        std::unique_lock lock(mutex_);
        selected_nodes_.erase(id);
        node_mask_dirty_ = true;
        bumpGeneration();
    }

    void SelectionState::clearNodeSelection() {
        std::unique_lock lock(mutex_);
        selected_nodes_.clear();
        node_mask_dirty_ = true;
        bumpGeneration();
    }

    bool SelectionState::isNodeSelected(const core::NodeId id) const {
        std::shared_lock lock(mutex_);
        return selected_nodes_.contains(id);
    }

    const std::unordered_set<core::NodeId>& SelectionState::selectedNodeIds() const {
        return selected_nodes_;
    }

    size_t SelectionState::selectedNodeCount() const {
        std::shared_lock lock(mutex_);
        return selected_nodes_.size();
    }

    const std::vector<bool>& SelectionState::getNodeMask(const core::Scene& scene) const {
        std::shared_lock lock(mutex_);
        if (!node_mask_dirty_)
            return cached_node_mask_;

        // DCLP: release shared lock then acquire exclusive. Another thread may
        // rebuild the cache in the gap — the double-check below handles that.
        lock.unlock();
        std::unique_lock wlock(mutex_);

        if (!node_mask_dirty_)
            return cached_node_mask_;

        // Build name vector from selected IDs for Scene::getSelectedNodeMask
        std::vector<std::string> names;
        names.reserve(selected_nodes_.size());
        for (const auto id : selected_nodes_) {
            const auto* node = scene.getNodeById(id);
            if (node)
                names.push_back(node->name);
        }

        cached_node_mask_ = scene.getSelectedNodeMask(names);
        node_mask_dirty_ = false;
        return cached_node_mask_;
    }

    void SelectionState::invalidateNodeMask() {
        std::unique_lock lock(mutex_);
        node_mask_dirty_ = true;
    }

    void SelectionState::bumpGeneration() {
        generation_.fetch_add(1, std::memory_order_release);
    }

} // namespace lfs::vis
