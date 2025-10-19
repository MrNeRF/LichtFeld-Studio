/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace gs {

    class Scene {
    public:
        struct Node {
            std::string name;
            std::unique_ptr<SplatData> model;
            glm::mat4 transform{1.0f};
            bool visible = true;
            size_t gaussian_count = 0;
        };

        Scene() = default;
        ~Scene() = default;

        // Delete copy operations
        Scene(const Scene&) = delete;
        Scene& operator=(const Scene&) = delete;

        // Allow move operations
        Scene(Scene&&) = default;
        Scene& operator=(Scene&&) = default;

        // Node management
        void addNode(const std::string& name, std::unique_ptr<SplatData> model);
        void removeNode(const std::string& name);
        void setNodeVisibility(const std::string& name, bool visible);
        bool renameNode(const std::string& old_name, const std::string& new_name);
        void clear();
        std::pair<std::string, std::string> cycleVisibilityWithNames();

        // Get combined model for rendering
        const SplatData* getCombinedModel() const;

        // Direct queries
        size_t getNodeCount() const { return nodes_.size(); }
        size_t getTotalGaussianCount() const;
        std::vector<const Node*> getNodes() const;
        const Node* getNode(const std::string& name) const;
        Node* getMutableNode(const std::string& name);
        bool hasNodes() const { return !nodes_.empty(); }

        // Get visible nodes for split view
        std::vector<const Node*> getVisibleNodes() const;

    private:
        std::vector<Node> nodes_;

        // Caching for combined model
        mutable std::unique_ptr<SplatData> cached_combined_;
        mutable bool cache_valid_ = false;

        void invalidateCache() { cache_valid_ = false; }
        void rebuildCacheIfNeeded() const;
    };

} // namespace gs