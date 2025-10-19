/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "scene/scene.hpp"
#include "core/logger.hpp"

#include <algorithm>
#include <print>
#include <ranges>
#include <torch/torch.h>

namespace gs {

    void Scene::addNode(const std::string& name, std::unique_ptr<SplatData> model) {
        // Calculate gaussian count before moving
        size_t gaussian_count = static_cast<size_t>(model->size());

        // Check if name already exists
        auto it = std::find_if(nodes_.begin(), nodes_.end(),
                               [&name](const Node& node) { return node.name == name; });

        if (it != nodes_.end()) {
            // Replace existing
            it->model = std::move(model);
            it->gaussian_count = gaussian_count;
        } else {
            // Add new node
            Node node{
                .name = name,
                .model = std::move(model),
                .transform = glm::mat4(1.0f),
                .visible = true,
                .gaussian_count = gaussian_count};
            nodes_.push_back(std::move(node));
        }

        invalidateCache();
        std::println("Scene: Added node '{}' with {} gaussians", name, gaussian_count);
    }

    void Scene::removeNode(const std::string& name) {
        auto it = std::find_if(nodes_.begin(), nodes_.end(),
                               [&name](const Node& node) { return node.name == name; });

        if (it != nodes_.end()) {
            nodes_.erase(it);
            invalidateCache();
            std::println("Scene: Removed node '{}'", name);
        }
    }

    void Scene::setNodeVisibility(const std::string& name, bool visible) {
        auto it = std::find_if(nodes_.begin(), nodes_.end(),
                               [&name](const Node& node) { return node.name == name; });

        if (it != nodes_.end() && it->visible != visible) {
            it->visible = visible;
            invalidateCache();
        }
    }

    void Scene::clear() {
        nodes_.clear();
        cached_combined_.reset();
        cache_valid_ = false;
    }

    std::pair<std::string, std::string> Scene::cycleVisibilityWithNames() {
        static constexpr std::pair<const char*, const char*> EMPTY_PAIR = {"", ""};

        if (nodes_.size() <= 1) {
            return EMPTY_PAIR;
        }

        std::string hidden_name, shown_name;

        // Find first visible node using modular arithmetic as suggested
        auto visible = std::find_if(nodes_.begin(), nodes_.end(),
                                    [](const Node& n) { return n.visible; });

        if (visible != nodes_.end()) {
            visible->visible = false;
            hidden_name = visible->name;

            auto next_index = (std::distance(nodes_.begin(), visible) + 1) % nodes_.size();
            auto next = nodes_.begin() + next_index;

            next->visible = true;
            shown_name = next->name;
        } else {
            // No visible nodes, show first
            nodes_[0].visible = true;
            shown_name = nodes_[0].name;
        }

        invalidateCache();
        return {hidden_name, shown_name};
    }

    const SplatData* Scene::getCombinedModel() const {
        rebuildCacheIfNeeded();
        return cached_combined_.get();
    }

    size_t Scene::getTotalGaussianCount() const {
        size_t total = 0;
        for (const auto& node : nodes_) {
            if (node.visible) {
                total += node.gaussian_count;
            }
        }
        return total;
    }

    std::vector<const Scene::Node*> Scene::getNodes() const {
        std::vector<const Node*> result;
        result.reserve(nodes_.size());
        for (const auto& node : nodes_) {
            result.push_back(&node);
        }
        return result;
    }

    std::vector<const Scene::Node*> Scene::getVisibleNodes() const {
        std::vector<const Node*> visible;
        for (const auto& node : nodes_) {
            if (node.visible && node.model) {
                visible.push_back(&node);
            }
        }
        return visible;
    }

    const Scene::Node* Scene::getNode(const std::string& name) const {
        auto it = std::find_if(nodes_.begin(), nodes_.end(),
                               [&name](const Node& node) { return node.name == name; });
        return (it != nodes_.end()) ? &(*it) : nullptr;
    }

    Scene::Node* Scene::getMutableNode(const std::string& name) {
        auto it = std::find_if(nodes_.begin(), nodes_.end(),
                               [&name](const Node& node) { return node.name == name; });
        if (it != nodes_.end()) {
            invalidateCache();
            return &(*it);
        }
        return nullptr;
    }

    void Scene::rebuildCacheIfNeeded() const {
        if (cache_valid_)
            return;

        // Collect visible models using ranges
        auto visible_models = nodes_ | std::views::filter([](const auto& node) {
                                  return node.visible && node.model;
                              }) |
                              std::views::transform([](const auto& node) {
                                  return node.model.get();
                              }) |
                              std::ranges::to<std::vector>();

        if (visible_models.empty()) {
            cached_combined_.reset();
            cache_valid_ = true;
            return;
        }

        // Calculate totals and find max SH degree in one pass
        struct ModelStats {
            size_t total_gaussians = 0;
            int max_sh_degree = 0;
            float total_scene_scale = 0.0f;
            bool has_shN = false;
        };

        auto stats = std::accumulate(
            visible_models.begin(), visible_models.end(), ModelStats{},
            [](ModelStats acc, const SplatData* model) {
                acc.total_gaussians += model->size();

                // Calculate SH degree from the actual shN tensor dimensions
                // Degree 0: shN is empty or has 0 coefficients
                // Degree 1: shN has 3 coefficients (for l=1)
                // Degree 2: shN has 8 coefficients (for l=1,2)
                // Degree 3: shN has 15 coefficients (for l=1,2,3)
                int sh_degree = 0;
                if (model->shN().defined() && model->shN().dim() >= 2 && model->shN().size(1) > 0) {
                    int shN_coeffs = model->shN().size(1);
                    // shN contains (degree+1)^2 - 1 coefficients
                    // Solve: shN_coeffs = (degree+1)^2 - 1
                    // Therefore: degree = sqrt(shN_coeffs + 1) - 1
                    sh_degree = static_cast<int>(std::round(std::sqrt(shN_coeffs + 1))) - 1;

                    // Validate the degree is reasonable (0-3)
                    sh_degree = std::clamp(sh_degree, 0, 3);
                }

                acc.max_sh_degree = std::max(acc.max_sh_degree, sh_degree);
                acc.total_scene_scale += model->get_scene_scale();
                acc.has_shN = acc.has_shN || (model->shN().numel() > 0 && model->shN().size(1) > 0);
                return acc;
            });

        std::println("Scene: Combining {} models, {} gaussians, max SH degree {}",
                     visible_models.size(), stats.total_gaussians, stats.max_sh_degree);

        // Setup tensor options from first model
        const auto [device, dtype] = [&] {
            const auto& first = visible_models[0]->means();
            return std::pair{first.device(), first.dtype()};
        }();
        auto opts = torch::TensorOptions().dtype(dtype).device(device);

        // Calculate SH dimensions based on max degree
        // Degree 0: sh0=1, shN=0
        // Degree 1: sh0=1, shN=3
        // Degree 2: sh0=1, shN=8
        // Degree 3: sh0=1, shN=15
        int sh0_coeffs = 1; // Always 1 for l=0
        int shN_coeffs = (stats.max_sh_degree > 0) ? ((stats.max_sh_degree + 1) * (stats.max_sh_degree + 1) - 1) : 0;

        // Pre-allocate all tensors at once
        struct CombinedTensors {
            torch::Tensor means, sh0, shN, opacity, scaling, rotation;
        } combined{
            .means = torch::empty({static_cast<int64_t>(stats.total_gaussians), 3}, opts),
            .sh0 = torch::empty({static_cast<int64_t>(stats.total_gaussians), sh0_coeffs, 3}, opts),
            .shN = (shN_coeffs > 0) ? torch::zeros({static_cast<int64_t>(stats.total_gaussians), shN_coeffs, 3}, opts) : torch::empty({static_cast<int64_t>(stats.total_gaussians), 0, 3}, opts),
            .opacity = torch::empty({static_cast<int64_t>(stats.total_gaussians), 1}, opts),
            .scaling = torch::empty({static_cast<int64_t>(stats.total_gaussians), 3}, opts),
            .rotation = torch::empty({static_cast<int64_t>(stats.total_gaussians), 4}, opts)};

        // Helper to create a slice for current model
        auto make_slice = [](size_t start, size_t size) {
            return torch::indexing::Slice(start, start + size);
        };

        // Copy data from each model
        size_t offset = 0;
        for (const auto* model : visible_models) {
            const auto size = model->size();
            const auto slice = make_slice(offset, size);

            // Direct copy for simple tensors
            combined.means.index({slice}) = model->means();
            combined.opacity.index({slice}) = model->opacity_raw();
            combined.scaling.index({slice}) = model->scaling_raw();
            combined.rotation.index({slice}) = model->rotation_raw();

            // Copy sh0 (always present)
            combined.sh0.index({slice}) = model->sh0();

            // Copy shN if we have coefficients to copy
            if (shN_coeffs > 0) {
                // Check how many coefficients this model has
                int model_shN_coeffs = (model->shN().defined() && model->shN().dim() >= 2) ? model->shN().size(1) : 0;

                if (model_shN_coeffs > 0) {
                    // Copy as many coefficients as the model has, up to our max
                    int coeffs_to_copy = std::min(model_shN_coeffs, shN_coeffs);
                    combined.shN.index({slice, torch::indexing::Slice(0, coeffs_to_copy)}) =
                        model->shN().index({torch::indexing::Slice(), torch::indexing::Slice(0, coeffs_to_copy)});
                }
                // If model has fewer coefficients than max, the rest remain zero (already initialized)
            }

            offset += size;
        }

        // Create the combined model
        cached_combined_ = std::make_unique<SplatData>(
            stats.max_sh_degree,
            std::move(combined.means),
            std::move(combined.sh0),
            std::move(combined.shN),
            std::move(combined.scaling),
            std::move(combined.rotation),
            std::move(combined.opacity),
            stats.total_scene_scale / visible_models.size());

        cache_valid_ = true;
    }

    bool Scene::renameNode(const std::string& old_name, const std::string& new_name) {
        // Check if new name already exists (case-sensitive)
        if (old_name == new_name) {
            return true; // Same name, consider it successful
        }

        // Check if new name already exists
        auto existing_it = std::find_if(nodes_.begin(), nodes_.end(),
                                        [&new_name](const Node& node) {
                                            return node.name == new_name;
                                        });

        if (existing_it != nodes_.end()) {
            LOG_INFO("Scene: Cannot rename '{}' to '{}' - name already exists", old_name, new_name);
            return false; // Name already exists
        }

        // Find the node to rename
        auto it = std::find_if(nodes_.begin(), nodes_.end(),
                               [&old_name](const Node& node) {
                                   return node.name == old_name;
                               });

        if (it != nodes_.end()) {
            std::string prev_name = it->name;
            it->name = new_name;
            invalidateCache();
            LOG_INFO("Scene: Renamed node '{}' to '{}'", prev_name, new_name);
            return true;
        }

        LOG_WARN("Scene: Cannot find node '{}' to rename", old_name);
        return false;
    }
} // namespace gs