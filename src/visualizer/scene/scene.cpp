/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "scene/scene.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/logger.hpp"
#include "core/splat_data_transform.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "io/cache_image_loader.hpp"
#include "training/dataset.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cuda_runtime.h>
#include <functional>
#include <glm/gtc/quaternion.hpp>
#include <limits>
#include <numeric>
#include <ranges>
#include <set>

namespace lfs::vis {

    // SceneNode implementation
    SceneNode::SceneNode(Scene* scene) : scene_(scene) {
        initObservables(scene);
    }

    void SceneNode::initObservables(Scene* scene) {
        scene_ = scene;
        if (!scene_)
            return;

        local_transform.setCallback([this] {
            if (scene_) {
                scene_->invalidateTransformCache();
                scene_->markTransformDirty(id);
            }
        });
        visible.setCallback([this] {
            if (scene_) {
                if (scene_->isConsolidated()) {
                    scene_->invalidateTransformCache();
                } else {
                    scene_->invalidateCache();
                }
            }
        });
    }

    Scene::Scene() {
        // Create default selection group
        addSelectionGroup("Group 1", glm::vec3(0.0f));
    }

    // Helper to compute centroid from model (GPU computation, single copy)
    static glm::vec3 computeCentroid(const lfs::core::SplatData* model) {
        if (!model || model->size() == 0) {
            return glm::vec3(0.0f);
        }
        const auto& means = model->means_raw();
        if (!means.is_valid() || means.size(0) == 0) {
            return glm::vec3(0.0f);
        }
        // Compute mean on GPU, copy only 3 floats back
        auto centroid_tensor = means.mean({0}, false);
        glm::vec3 result(
            centroid_tensor.slice(0, 0, 1).item<float>(),
            centroid_tensor.slice(0, 1, 2).item<float>(),
            centroid_tensor.slice(0, 2, 3).item<float>());
        // Guard against NaN from empty or invalid data
        if (std::isnan(result.x) || std::isnan(result.y) || std::isnan(result.z)) {
            return glm::vec3(0.0f);
        }
        return result;
    }

    void Scene::addNode(const std::string& name, std::unique_ptr<lfs::core::SplatData> model) {
        if (name.empty()) {
            LOG_WARN("Cannot add node with empty name");
            return;
        }

        if (consolidated_) {
            LOG_DEBUG("Adding node invalidates consolidation");
            consolidated_ = false;
            consolidated_node_ids_.clear();
            cached_combined_.reset();
        }

        // Calculate gaussian count and centroid before moving
        const size_t gaussian_count = static_cast<size_t>(model->size());
        const glm::vec3 centroid = computeCentroid(model.get());

        // Check if name already exists
        auto it = std::find_if(nodes_.begin(), nodes_.end(),
                               [&name](const std::unique_ptr<Node>& node) { return node->name == name; });

        if (it != nodes_.end()) {
            // Replace existing model
            (*it)->model = std::move(model);
            (*it)->gaussian_count = gaussian_count;
            (*it)->centroid = centroid;
        } else {
            // Add new splat node
            const NodeId id = next_node_id_++;
            auto node = std::make_unique<Node>();
            node->id = id;
            node->type = NodeType::SPLAT;
            node->name = name;
            node->model = std::move(model);
            node->gaussian_count = gaussian_count;
            node->centroid = centroid;

            id_to_index_[id] = nodes_.size();
            node->initObservables(this); // Initialize before adding (address is stable with unique_ptr)
            nodes_.push_back(std::move(node));
        }

        invalidateCache();
        LOG_DEBUG("Added node '{}': {} gaussians", name, gaussian_count);
    }

    void Scene::removeNode(const std::string& name, const bool keep_children) {
        removeNodeInternal(name, keep_children, false);
    }

    void Scene::removeNodeInternal(const std::string& name, const bool keep_children, const bool force) {
        if (name.empty())
            return;

        const auto it = std::find_if(nodes_.begin(), nodes_.end(),
                                     [&name](const std::unique_ptr<Node>& node) { return node->name == name; });
        if (it == nodes_.end())
            return;

        // Prevent direct deletion of CROPBOX nodes (but allow when deleting parent)
        if (!force && (*it)->type == NodeType::CROPBOX) {
            LOG_WARN("Cannot delete CROPBOX node '{}' - use recalculate instead", name);
            return;
        }

        const NodeId id = (*it)->id;
        const NodeId parent_id = (*it)->parent_id;

        // Remove from parent's children list
        if (parent_id != NULL_NODE) {
            if (auto* parent = getNodeById(parent_id)) {
                auto& children = parent->children;
                children.erase(std::remove(children.begin(), children.end(), id), children.end());
            }
        }

        if (keep_children) {
            // Reparent children to the removed node's parent (or root if no parent)
            for (const NodeId child_id : (*it)->children) {
                if (auto* child = getNodeById(child_id)) {
                    child->parent_id = parent_id;
                    child->transform_dirty = true;
                    // Add to new parent's children list
                    if (parent_id != NULL_NODE) {
                        if (auto* new_parent = getNodeById(parent_id)) {
                            new_parent->children.push_back(child_id);
                        }
                    }
                }
            }
        } else {
            // Recursively remove children (force=true to allow CROPBOX deletion)
            const std::vector<NodeId> children_copy = (*it)->children;
            for (const NodeId child_id : children_copy) {
                if (const auto* child = getNodeById(child_id)) {
                    removeNodeInternal(child->name, false, true);
                }
            }
        }

        // Remove from lookup and vector (re-find iterator since recursive calls may have invalidated it)
        const auto it_final = std::find_if(nodes_.begin(), nodes_.end(),
                                           [&name](const std::unique_ptr<Node>& node) { return node->name == name; });
        if (it_final == nodes_.end())
            return; // Already removed somehow

        // Copy before erase - 'name' may reference the node being deleted
        const std::string name_copy = name;

        id_to_index_.erase(id);
        const size_t removed_index = static_cast<size_t>(std::distance(nodes_.begin(), it_final));
        nodes_.erase(it_final);

        // Update indices for nodes after removed one
        for (auto& [node_id, index] : id_to_index_) {
            if (index > removed_index)
                --index;
        }

        invalidateCache();
        if (!name_copy.empty()) {
            LOG_DEBUG("Removed node '{}'{}", name_copy, keep_children ? " (children kept)" : "");
        }
    }

    void Scene::replaceNodeModel(const std::string& name, std::unique_ptr<lfs::core::SplatData> model) {
        const auto it = std::find_if(nodes_.begin(), nodes_.end(),
                                     [&name](const std::unique_ptr<Node>& node) { return node->name == name; });

        if (it != nodes_.end()) {
            const size_t gaussian_count = static_cast<size_t>(model->size());
            const glm::vec3 centroid = computeCentroid(model.get());
            LOG_DEBUG("replaceNodeModel '{}': {} -> {} gaussians", name, (*it)->gaussian_count, gaussian_count);
            (*it)->model = std::move(model);
            (*it)->gaussian_count = gaussian_count;
            (*it)->centroid = centroid;
            invalidateCache();
        } else {
            LOG_WARN("replaceNodeModel: node '{}' not found", name);
        }
    }

    void Scene::setNodeVisibility(const std::string& name, const bool visible) {
        const auto it = std::find_if(nodes_.begin(), nodes_.end(),
                                     [&name](const std::unique_ptr<Node>& n) { return n->name == name; });
        if (it != nodes_.end()) {
            setNodeVisibilityById((*it)->id, visible);
        }
    }

    void Scene::setNodeVisibilityById(const NodeId id, const bool visible) {
        const auto idx_it = id_to_index_.find(id);
        if (idx_it == id_to_index_.end())
            return;

        Node* node = nodes_[idx_it->second].get();
        node->visible = visible;

        for (const NodeId child_id : node->children) {
            setNodeVisibilityById(child_id, visible);
        }
    }

    void Scene::setNodeTransform(const std::string& name, const glm::mat4& transform) {
        const auto it = std::find_if(nodes_.begin(), nodes_.end(),
                                     [&name](const std::unique_ptr<Node>& node) { return node->name == name; });

        if (it != nodes_.end()) {
            (*it)->local_transform = transform; // Observable auto-invalidates cache and marks transform dirty
        }
    }

    glm::mat4 Scene::getNodeTransform(const std::string& name) const {
        const auto it = std::find_if(nodes_.begin(), nodes_.end(),
                                     [&name](const std::unique_ptr<Node>& node) { return node->name == name; });

        if (it != nodes_.end()) {
            return (*it)->local_transform;
        }
        return glm::mat4(1.0f);
    }

    void Scene::clear() {
        nodes_.clear();
        id_to_index_.clear();
        next_node_id_ = 0;

        cached_combined_.reset();
        cached_transform_indices_.reset();
        cached_transforms_.clear();
        model_cache_valid_ = false;
        transform_cache_valid_ = false;
        consolidated_ = false;
        consolidated_node_ids_.clear();

        selection_mask_.reset();
        has_selection_ = false;
        resetSelectionState();

        train_cameras_.reset();
        val_cameras_.reset();
        initial_point_cloud_.reset();
        training_model_node_.clear();

        if (lfs::io::CacheLoader::hasInstance()) {
            lfs::io::CacheLoader::getInstance().reset_cache();
        }

        // Release GPU memory
        cudaDeviceSynchronize();
        lfs::core::CudaMemoryPool::instance().trim_cached_memory();
        lfs::core::GlobalArenaManager::instance().get_arena().emergency_cleanup();
    }

    std::pair<std::string, std::string> Scene::cycleVisibilityWithNames() {
        static constexpr std::pair<const char*, const char*> EMPTY_PAIR = {"", ""};

        if (nodes_.size() <= 1) {
            return EMPTY_PAIR;
        }

        std::string hidden_name, shown_name;

        // Find first visible node using modular arithmetic as suggested
        auto visible = std::find_if(nodes_.begin(), nodes_.end(),
                                    [](const std::unique_ptr<Node>& n) { return n->visible; });

        if (visible != nodes_.end()) {
            (*visible)->visible = false;
            hidden_name = (*visible)->name;

            auto next_index = (std::distance(nodes_.begin(), visible) + 1) % nodes_.size();
            auto next = nodes_.begin() + next_index;

            (*next)->visible = true;
            shown_name = (*next)->name;
        } else {
            // No visible nodes, show first
            nodes_[0]->visible = true;
            shown_name = nodes_[0]->name;
        }

        invalidateCache();
        return {hidden_name, shown_name};
    }

    const lfs::core::SplatData* Scene::getCombinedModel() const {
        rebuildCacheIfNeeded();
        return single_node_model_ ? single_node_model_ : cached_combined_.get();
    }

    size_t Scene::consolidateNodeModels() {
        rebuildCacheIfNeeded();

        if (single_node_model_ || !cached_combined_) {
            return 0;
        }

        consolidated_node_ids_.clear();
        size_t consolidated = 0;
        for (auto& node : nodes_) {
            if (node->model && isNodeEffectivelyVisible(node->id)) {
                consolidated_node_ids_.push_back(node->id);
                node->model.reset();
                ++consolidated;
            }
        }

        if (consolidated > 0) {
            consolidated_ = true;
            constexpr size_t BYTES_PER_GAUSSIAN = 3 * 4 + 1 * 3 * 4 + 3 * 4 + 4 * 4 + 1 * 4;
            const size_t saved_mb = getTotalGaussianCount() * BYTES_PER_GAUSSIAN / (1024 * 1024);
            LOG_INFO("Consolidated {} nodes, saved ~{} MB VRAM", consolidated, saved_mb);
        }

        return consolidated;
    }

    std::vector<bool> Scene::getNodeVisibilityMask() const {
        if (!consolidated_ || consolidated_node_ids_.empty()) {
            return {};
        }

        std::vector<bool> mask;
        mask.reserve(consolidated_node_ids_.size());
        for (const NodeId id : consolidated_node_ids_) {
            if (const auto* node = getNodeById(id)) {
                mask.push_back(node->visible.get());
            } else {
                mask.push_back(true);
            }
        }
        return mask;
    }

    const lfs::core::PointCloud* Scene::getVisiblePointCloud() const {
        for (const auto& node : nodes_) {
            if (node->type == NodeType::POINTCLOUD && isNodeEffectivelyVisible(node->id) && node->point_cloud) {
                return node->point_cloud.get();
            }
        }
        return nullptr;
    }

    size_t Scene::getTotalGaussianCount() const {
        size_t total = 0;
        for (const auto& node : nodes_) {
            if (node->visible) {
                total += node->gaussian_count;
            }
        }
        return total;
    }

    std::vector<const Scene::Node*> Scene::getNodes() const {
        std::vector<const Node*> result;
        result.reserve(nodes_.size());
        for (const auto& node : nodes_) {
            result.push_back(node.get());
        }
        return result;
    }

    std::vector<const Scene::Node*> Scene::getVisibleNodes() const {
        std::vector<const Node*> visible;
        for (const auto& node : nodes_) {
            if (node->visible && node->model) {
                visible.push_back(node.get());
            }
        }
        return visible;
    }

    std::unordered_set<int> Scene::getVisibleCameraIndices() const {
        std::unordered_set<int> visible_indices;
        for (const auto& node : nodes_) {
            if (node->type == NodeType::CAMERA && node->camera_index >= 0 &&
                isNodeEffectivelyVisible(node->id)) {
                visible_indices.insert(node->camera_index);
            }
        }
        return visible_indices;
    }

    const Scene::Node* Scene::getNode(const std::string& name) const {
        auto it = std::find_if(nodes_.begin(), nodes_.end(),
                               [&name](const std::unique_ptr<Node>& node) { return node->name == name; });
        return (it != nodes_.end()) ? it->get() : nullptr;
    }

    Scene::Node* Scene::getMutableNode(const std::string& name) {
        auto it = std::find_if(nodes_.begin(), nodes_.end(),
                               [&name](const std::unique_ptr<Node>& node) { return node->name == name; });
        if (it != nodes_.end()) {
            invalidateCache();
            return it->get();
        }
        return nullptr;
    }

    void Scene::rebuildModelCacheIfNeeded() const {
        if (model_cache_valid_)
            return;

        if (consolidated_ && cached_combined_) {
            model_cache_valid_ = true;
            return;
        }

        LOG_DEBUG("Rebuilding combined model cache");

        single_node_model_ = nullptr;

        std::vector<const Node*> visible_nodes;
        for (const auto& node : nodes_) {
            if (node->model && isNodeEffectivelyVisible(node->id)) {
                visible_nodes.push_back(node.get());
            }
        }

        if (visible_nodes.empty()) {
            cached_combined_.reset();
            cached_transform_indices_.reset();
            model_cache_valid_ = true;
            transform_cache_valid_ = false;
            return;
        }

        if (visible_nodes.size() == 1) {
            const auto* node = visible_nodes[0];
            single_node_model_ = node->model.get();
            cached_combined_.reset();

            const size_t n = node->model->size();
            cached_transform_indices_ = std::make_shared<lfs::core::Tensor>(
                lfs::core::Tensor::zeros({n}, lfs::core::Device::CUDA, lfs::core::DataType::Int32));

            LOG_DEBUG("Single node: {} ({} gaussians)", node->name, n);
            model_cache_valid_ = true;
            transform_cache_valid_ = false;
            return;
        }

        // Cache model sizes to avoid race with training thread
        struct ModelStats {
            size_t total_gaussians = 0;
            int max_sh_degree = 0;
            float total_scene_scale = 0.0f;
            bool has_shN = false;
        };

        std::vector<size_t> cached_sizes;
        cached_sizes.reserve(visible_nodes.size());
        ModelStats stats{};

        for (const auto* node : visible_nodes) {
            const auto* model = node->model.get();
            const size_t node_size = model->size();
            cached_sizes.push_back(node_size);
            stats.total_gaussians += node_size;

            const auto& shN_tensor = model->shN_raw();
            if (shN_tensor.is_valid() && shN_tensor.ndim() >= 2 && shN_tensor.size(1) > 0) {
                const int shN_coeffs = static_cast<int>(shN_tensor.size(1));
                const int sh_degree = std::clamp(
                    static_cast<int>(std::round(std::sqrt(shN_coeffs + 1))) - 1, 0, 3);
                stats.max_sh_degree = std::max(stats.max_sh_degree, sh_degree);
            }

            stats.total_scene_scale += model->get_scene_scale();
            stats.has_shN = stats.has_shN || (shN_tensor.numel() > 0 && shN_tensor.size(1) > 0);
        }

        const lfs::core::Device device = visible_nodes[0]->model->means_raw().device();
        constexpr int SH0_COEFFS = 1;
        const int shN_coeffs = (stats.max_sh_degree > 0)
                                   ? ((stats.max_sh_degree + 1) * (stats.max_sh_degree + 1) - 1)
                                   : 0;

        using lfs::core::Tensor;
        Tensor means = Tensor::empty({static_cast<size_t>(stats.total_gaussians), 3}, device);
        Tensor sh0 = Tensor::empty({static_cast<size_t>(stats.total_gaussians), static_cast<size_t>(SH0_COEFFS), 3}, device);
        Tensor shN = (shN_coeffs > 0) ? Tensor::zeros({static_cast<size_t>(stats.total_gaussians), static_cast<size_t>(shN_coeffs), 3}, device) : Tensor::empty({static_cast<size_t>(stats.total_gaussians), 0, 3}, device);
        Tensor opacity = Tensor::empty({static_cast<size_t>(stats.total_gaussians), 1}, device);
        Tensor scaling = Tensor::empty({static_cast<size_t>(stats.total_gaussians), 3}, device);
        Tensor rotation = Tensor::empty({static_cast<size_t>(stats.total_gaussians), 4}, device);

        const bool has_any_deleted = std::any_of(visible_nodes.begin(), visible_nodes.end(),
                                                 [](const Node* node) { return node->model->has_deleted_mask(); });

        Tensor deleted = has_any_deleted
                             ? Tensor::zeros({static_cast<size_t>(stats.total_gaussians)}, device, lfs::core::DataType::Bool)
                             : Tensor();

        std::vector<int> transform_indices_data(stats.total_gaussians);

        size_t offset = 0;
        for (size_t i = 0; i < visible_nodes.size(); ++i) {
            const auto* model = visible_nodes[i]->model.get();
            const size_t size = cached_sizes[i];

            std::fill(transform_indices_data.begin() + offset,
                      transform_indices_data.begin() + offset + size,
                      static_cast<int>(i));

            means.slice(0, offset, offset + size) = model->means_raw();
            scaling.slice(0, offset, offset + size) = model->scaling_raw();
            rotation.slice(0, offset, offset + size) = model->rotation_raw();
            sh0.slice(0, offset, offset + size) = model->sh0_raw();
            opacity.slice(0, offset, offset + size) = model->opacity_raw();

            if (shN_coeffs > 0) {
                const auto& model_shN = model->shN_raw();
                const int model_shN_coeffs = (model_shN.is_valid() && model_shN.ndim() >= 2)
                                                 ? static_cast<int>(model_shN.size(1))
                                                 : 0;
                if (model_shN_coeffs > 0) {
                    const int coeffs_to_copy = std::min(model_shN_coeffs, shN_coeffs);
                    shN.slice(0, offset, offset + size).slice(1, 0, coeffs_to_copy) =
                        model_shN.slice(1, 0, coeffs_to_copy);
                }
            }

            if (has_any_deleted && model->has_deleted_mask()) {
                deleted.slice(0, offset, offset + size) = model->deleted();
            }

            offset += size;
        }

        cached_transform_indices_ = std::make_shared<Tensor>(
            Tensor::from_vector(transform_indices_data, {stats.total_gaussians}, lfs::core::Device::CPU).cuda());

        cached_combined_ = std::make_unique<lfs::core::SplatData>(
            stats.max_sh_degree,
            std::move(means),
            std::move(sh0),
            std::move(shN),
            std::move(scaling),
            std::move(rotation),
            std::move(opacity),
            stats.total_scene_scale / visible_nodes.size());

        if (has_any_deleted) {
            cached_combined_->deleted() = std::move(deleted);
        }

        model_cache_valid_ = true;
        transform_cache_valid_ = false; // Force transform rebuild after model rebuild
    }

    void Scene::rebuildTransformCacheIfNeeded() const {
        if (transform_cache_valid_)
            return;

        cached_transforms_.clear();
        for (const auto& node : nodes_) {
            // Include both SPLAT nodes (with model) and POINTCLOUD nodes (with point_cloud)
            const bool has_renderable = node->model || node->point_cloud;
            if (has_renderable && isNodeEffectivelyVisible(node->id)) {
                cached_transforms_.push_back(getWorldTransform(node->id));
            }
        }
        transform_cache_valid_ = true;
    }

    void Scene::rebuildCacheIfNeeded() const {
        rebuildModelCacheIfNeeded();
        rebuildTransformCacheIfNeeded();
    }

    std::vector<glm::mat4> Scene::getVisibleNodeTransforms() const {
        rebuildCacheIfNeeded();
        return cached_transforms_;
    }

    std::shared_ptr<lfs::core::Tensor> Scene::getTransformIndices() const {
        rebuildCacheIfNeeded();
        return cached_transform_indices_;
    }

    int Scene::getVisibleNodeIndex(const std::string& name) const {
        int index = 0;
        for (const auto& node : nodes_) {
            if (!node->visible || !node->model) {
                continue;
            }
            if (node->name == name) {
                return index;
            }
            ++index;
        }
        return -1;
    }

    std::vector<bool> Scene::getSelectedNodeMask(const std::string& selected_node_name) const {
        const size_t visible_count = std::count_if(nodes_.begin(), nodes_.end(),
                                                   [](const auto& n) { return n->visible && n->model; });

        if (selected_node_name.empty()) {
            return std::vector<bool>(visible_count, false);
        }

        const Node* selected = getNode(selected_node_name);
        if (!selected) {
            return std::vector<bool>(visible_count, false);
        }

        if (selected->type == NodeType::CROPBOX && selected->parent_id != NULL_NODE) {
            selected = getNodeById(selected->parent_id);
            if (!selected)
                return {};
        }

        const NodeId selected_id = selected->id;
        const auto isSelectedOrDescendant = [this, selected_id](const Node* node) {
            for (const Node* n = node; n; n = (n->parent_id != NULL_NODE) ? getNodeById(n->parent_id) : nullptr) {
                if (n->id == selected_id)
                    return true;
            }
            return false;
        };

        std::vector<bool> mask;
        mask.reserve(visible_count);
        for (const auto& node : nodes_) {
            if (node->visible && node->model) {
                mask.push_back(isSelectedOrDescendant(node.get()));
            }
        }
        return mask;
    }

    std::vector<bool> Scene::getSelectedNodeMask(const std::vector<std::string>& selected_node_names) const {
        const size_t visible_count = std::count_if(nodes_.begin(), nodes_.end(),
                                                   [](const auto& n) { return n->visible && n->model; });

        if (selected_node_names.empty()) {
            return std::vector<bool>(visible_count, false);
        }

        std::set<NodeId> selected_ids;
        for (const auto& name : selected_node_names) {
            const Node* selected = getNode(name);
            if (!selected)
                continue;

            if (selected->type == NodeType::CROPBOX && selected->parent_id != NULL_NODE) {
                selected = getNodeById(selected->parent_id);
                if (!selected)
                    continue;
            }
            selected_ids.insert(selected->id);
        }

        if (selected_ids.empty()) {
            return std::vector<bool>(visible_count, false);
        }

        const auto isSelectedOrDescendant = [this, &selected_ids](const Node* node) {
            for (const Node* n = node; n; n = (n->parent_id != NULL_NODE) ? getNodeById(n->parent_id) : nullptr) {
                if (selected_ids.count(n->id) > 0)
                    return true;
            }
            return false;
        };

        std::vector<bool> mask;
        mask.reserve(visible_count);
        for (const auto& node : nodes_) {
            if (node->visible && node->model) {
                mask.push_back(isSelectedOrDescendant(node.get()));
            }
        }
        return mask;
    }

    std::shared_ptr<lfs::core::Tensor> Scene::getSelectionMask() const {
        if (!has_selection_) {
            return nullptr;
        }
        return selection_mask_;
    }

    void Scene::setSelection(const std::vector<size_t>& selected_indices) {
        // Get total gaussian count
        size_t total = getTotalGaussianCount();
        if (total == 0) {
            clearSelection();
            return;
        }

        // Create or resize selection mask
        if (!selection_mask_ || selection_mask_->size(0) != total) {
            // Create new mask (all zeros on CPU first, then move to GPU)
            selection_mask_ = std::make_shared<lfs::core::Tensor>(
                lfs::core::Tensor::zeros({total}, lfs::core::Device::CPU, lfs::core::DataType::UInt8));
        } else {
            // Clear existing mask (set all to 0)
            auto mask_cpu = selection_mask_->cpu();
            std::memset(mask_cpu.ptr<uint8_t>(), 0, total);
            *selection_mask_ = mask_cpu;
        }

        if (!selected_indices.empty()) {
            auto mask_cpu = selection_mask_->cpu();
            uint8_t* mask_data = mask_cpu.ptr<uint8_t>();
            for (size_t idx : selected_indices) {
                if (idx < total) {
                    mask_data[idx] = 1;
                }
            }
            *selection_mask_ = mask_cpu.cuda();
            has_selection_ = true;
        } else {
            has_selection_ = false;
        }
    }

    void Scene::setSelectionMask(std::shared_ptr<lfs::core::Tensor> mask) {
        selection_mask_ = std::move(mask);
        has_selection_ = selection_mask_ && selection_mask_->is_valid() && selection_mask_->numel() > 0;
    }

    void Scene::clearSelection() {
        selection_mask_.reset();
        has_selection_ = false;
    }

    bool Scene::hasSelection() const {
        return has_selection_;
    }

    bool Scene::renameNode(const std::string& old_name, const std::string& new_name) {
        // Check if new name already exists (case-sensitive)
        if (old_name == new_name) {
            return true; // Same name, consider it successful
        }

        // Check if new name already exists
        auto existing_it = std::find_if(nodes_.begin(), nodes_.end(),
                                        [&new_name](const std::unique_ptr<Node>& node) {
                                            return node->name == new_name;
                                        });

        if (existing_it != nodes_.end()) {
            LOG_WARN("Cannot rename '{}' to '{}' - name exists", old_name, new_name);
            return false; // Name already exists
        }

        // Find the node to rename
        auto it = std::find_if(nodes_.begin(), nodes_.end(),
                               [&old_name](const std::unique_ptr<Node>& node) {
                                   return node->name == old_name;
                               });

        if (it != nodes_.end()) {
            std::string prev_name = (*it)->name;
            (*it)->name = new_name;
            invalidateCache();
            LOG_DEBUG("Renamed node '{}' to '{}'", prev_name, new_name);
            return true;
        }

        LOG_WARN("Scene: Cannot find node '{}' to rename", old_name);
        return false;
    }

    size_t Scene::applyDeleted() {
        size_t total_removed = 0;

        for (auto& node : nodes_) {
            if (node->model && node->model->has_deleted_mask()) {
                const size_t removed = node->model->apply_deleted();
                if (removed > 0) {
                    node->gaussian_count = node->model->size();
                    node->centroid = computeCentroid(node->model.get());
                    total_removed += removed;
                }
            }
        }

        if (total_removed > 0) {
            invalidateCache();
            clearSelection();
        }

        return total_removed;
    }

    // Selection group color palette
    static constexpr std::array<glm::vec3, 8> GROUP_COLOR_PALETTE = {{
        {1.0f, 0.3f, 0.3f}, // Red
        {0.3f, 1.0f, 0.3f}, // Green
        {0.3f, 0.5f, 1.0f}, // Blue
        {1.0f, 1.0f, 0.3f}, // Yellow
        {1.0f, 0.5f, 0.0f}, // Orange
        {0.8f, 0.3f, 1.0f}, // Purple
        {0.3f, 1.0f, 1.0f}, // Cyan
        {1.0f, 0.5f, 0.8f}, // Pink
    }};

    SelectionGroup* Scene::findGroup(const uint8_t id) {
        const auto it = std::find_if(selection_groups_.begin(), selection_groups_.end(),
                                     [id](const SelectionGroup& g) { return g.id == id; });
        return (it != selection_groups_.end()) ? &(*it) : nullptr;
    }

    const SelectionGroup* Scene::findGroup(const uint8_t id) const {
        const auto it = std::find_if(selection_groups_.begin(), selection_groups_.end(),
                                     [id](const SelectionGroup& g) { return g.id == id; });
        return (it != selection_groups_.end()) ? &(*it) : nullptr;
    }

    uint8_t Scene::addSelectionGroup(const std::string& name, const glm::vec3& color) {
        if (next_group_id_ == 0) {
            LOG_WARN("Maximum selection groups reached");
            return 0;
        }

        SelectionGroup group;
        group.id = next_group_id_++;
        group.name = name.empty() ? "Group " + std::to_string(group.id) : name;
        group.color = (color == glm::vec3(0.0f))
                          ? GROUP_COLOR_PALETTE[(group.id - 1) % GROUP_COLOR_PALETTE.size()]
                          : color;
        group.count = 0;

        selection_groups_.push_back(group);
        active_selection_group_ = group.id;

        LOG_DEBUG("Added selection group '{}' (ID {})", group.name, group.id);
        return group.id;
    }

    void Scene::removeSelectionGroup(const uint8_t id) {
        const auto it = std::find_if(selection_groups_.begin(), selection_groups_.end(),
                                     [id](const SelectionGroup& g) { return g.id == id; });
        if (it == selection_groups_.end())
            return;

        clearSelectionGroup(id);
        const std::string name = it->name;
        selection_groups_.erase(it);

        if (active_selection_group_ == id) {
            active_selection_group_ = selection_groups_.empty() ? 0 : selection_groups_.back().id;
        }

        LOG_DEBUG("Removed selection group '{}' (ID {})", name, id);
    }

    void Scene::renameSelectionGroup(const uint8_t id, const std::string& name) {
        if (auto* group = findGroup(id)) {
            group->name = name;
        }
    }

    void Scene::setSelectionGroupColor(const uint8_t id, const glm::vec3& color) {
        if (auto* group = findGroup(id)) {
            group->color = color;
        }
    }

    void Scene::setSelectionGroupLocked(const uint8_t id, const bool locked) {
        if (auto* group = findGroup(id)) {
            group->locked = locked;
        }
    }

    bool Scene::isSelectionGroupLocked(const uint8_t id) const {
        const auto* group = findGroup(id);
        return group ? group->locked : false;
    }

    const SelectionGroup* Scene::getSelectionGroup(const uint8_t id) const {
        return findGroup(id);
    }

    void Scene::updateSelectionGroupCounts() {
        for (auto& group : selection_groups_) {
            group.count = 0;
        }

        if (!selection_mask_ || !selection_mask_->is_valid())
            return;

        const auto mask_cpu = selection_mask_->cpu();
        const uint8_t* data = mask_cpu.ptr<uint8_t>();
        const size_t n = mask_cpu.numel();

        for (size_t i = 0; i < n; ++i) {
            const uint8_t group_id = data[i];
            if (auto* group = findGroup(group_id)) {
                group->count++;
            }
        }
    }

    void Scene::clearSelectionGroup(const uint8_t id) {
        if (!selection_mask_ || !selection_mask_->is_valid())
            return;

        auto mask_cpu = selection_mask_->cpu();
        uint8_t* data = mask_cpu.ptr<uint8_t>();
        const size_t n = mask_cpu.numel();

        bool any_remaining = false;
        for (size_t i = 0; i < n; ++i) {
            if (data[i] == id) {
                data[i] = 0;
            } else if (data[i] > 0) {
                any_remaining = true;
            }
        }

        *selection_mask_ = mask_cpu.cuda();
        has_selection_ = any_remaining;

        if (auto* group = findGroup(id)) {
            group->count = 0;
        }
    }

    void Scene::resetSelectionState() {
        selection_mask_.reset();
        has_selection_ = false;
        selection_groups_.clear();
        next_group_id_ = 1;
        addSelectionGroup("Group 1", glm::vec3(0.0f));
    }

    // ========== Scene Graph Operations ==========

    NodeId Scene::addGroup(const std::string& name, const NodeId parent) {
        const NodeId id = next_node_id_++;
        auto node = std::make_unique<Node>();
        node->id = id;
        node->parent_id = parent;
        node->type = NodeType::GROUP;
        node->name = name;

        // Add to parent's children
        if (parent != NULL_NODE) {
            if (auto* p = getNodeById(parent)) {
                p->children.push_back(id);
            }
        }

        id_to_index_[id] = nodes_.size();
        node->initObservables(this);
        nodes_.push_back(std::move(node));
        invalidateCache();

        LOG_DEBUG("Added group node '{}' (id={})", name, id);
        return id;
    }

    NodeId Scene::addSplat(const std::string& name, std::unique_ptr<lfs::core::SplatData> model, const NodeId parent) {
        if (consolidated_) {
            LOG_DEBUG("Adding splat invalidates consolidation");
            consolidated_ = false;
            consolidated_node_ids_.clear();
            cached_combined_.reset();
        }

        const size_t gaussian_count = static_cast<size_t>(model->size());
        const glm::vec3 centroid = computeCentroid(model.get());

        const NodeId id = next_node_id_++;
        auto node = std::make_unique<Node>();
        node->id = id;
        node->parent_id = parent;
        node->type = NodeType::SPLAT;
        node->name = name;
        node->model = std::move(model);
        node->gaussian_count = gaussian_count;
        node->centroid = centroid;

        // Add to parent's children
        if (parent != NULL_NODE) {
            if (auto* p = getNodeById(parent)) {
                p->children.push_back(id);
            }
        }

        id_to_index_[id] = nodes_.size();
        node->initObservables(this);
        nodes_.push_back(std::move(node));
        invalidateCache();

        LOG_DEBUG("Added splat node '{}' (id={}, {} gaussians)", name, id, gaussian_count);
        return id;
    }

    NodeId Scene::addPointCloud(const std::string& name, std::shared_ptr<lfs::core::PointCloud> point_cloud, const NodeId parent) {
        if (!point_cloud) {
            LOG_WARN("Cannot add point cloud node '{}': point cloud is null", name);
            return NULL_NODE;
        }

        const size_t point_count = point_cloud->size();
        const glm::vec3 centroid = [&]() {
            if (point_count == 0)
                return glm::vec3(0.0f);
            auto means_cpu = point_cloud->means.cpu();
            auto acc = means_cpu.accessor<float, 2>();
            glm::vec3 sum(0.0f);
            for (size_t i = 0; i < point_count; ++i) {
                sum.x += acc(i, 0);
                sum.y += acc(i, 1);
                sum.z += acc(i, 2);
            }
            return sum / static_cast<float>(point_count);
        }();

        const NodeId id = next_node_id_++;
        auto node = std::make_unique<Node>();
        node->id = id;
        node->parent_id = parent;
        node->type = NodeType::POINTCLOUD;
        node->name = name;
        node->point_cloud = std::move(point_cloud);
        node->gaussian_count = point_count; // Reuse field for point count
        node->centroid = centroid;

        // Add to parent's children
        if (parent != NULL_NODE) {
            if (auto* p = getNodeById(parent)) {
                p->children.push_back(id);
            }
        }

        id_to_index_[id] = nodes_.size();
        node->initObservables(this);
        nodes_.push_back(std::move(node));
        invalidateCache();

        LOG_DEBUG("Added point cloud node '{}' (id={}, {} points)", name, id, point_count);
        return id;
    }

    NodeId Scene::addCropBox(const std::string& name, const NodeId parent_node) {
        // Verify parent is a SPLAT or POINTCLOUD node
        const auto* parent = getNodeById(parent_node);
        if (!parent || (parent->type != NodeType::SPLAT && parent->type != NodeType::POINTCLOUD)) {
            LOG_WARN("Cannot add cropbox '{}': parent must be a SPLAT or POINTCLOUD node", name);
            return NULL_NODE;
        }

        // Check if this node already has a cropbox
        for (const NodeId child_id : parent->children) {
            if (const auto* child = getNodeById(child_id)) {
                if (child->type == NodeType::CROPBOX) {
                    LOG_DEBUG("Node '{}' already has cropbox '{}'", parent->name, child->name);
                    return child_id;
                }
            }
        }

        const NodeId id = next_node_id_++;
        auto node = std::make_unique<Node>();
        node->id = id;
        node->parent_id = parent_node;
        node->type = NodeType::CROPBOX;
        node->name = name;
        node->cropbox = std::make_unique<CropBoxData>();

        // Initialize cropbox bounds from parent's bounding box
        glm::vec3 bounds_min, bounds_max;
        if (getNodeBounds(parent_node, bounds_min, bounds_max)) {
            node->cropbox->min = bounds_min;
            node->cropbox->max = bounds_max;
        }

        // Add to parent's children (must re-get parent as vector may have changed)
        if (auto* p = getNodeById(parent_node)) {
            p->children.push_back(id);
        }

        id_to_index_[id] = nodes_.size();
        node->initObservables(this);
        nodes_.push_back(std::move(node));

        LOG_DEBUG("Added cropbox node '{}' (id={}) as child of '{}'", name, id, parent->name);
        return id;
    }

    NodeId Scene::addDataset(const std::string& name) {
        const NodeId id = next_node_id_++;
        auto node = std::make_unique<Node>();
        node->id = id;
        node->parent_id = NULL_NODE;
        node->type = NodeType::DATASET;
        node->name = name;

        id_to_index_[id] = nodes_.size();
        node->initObservables(this);
        nodes_.push_back(std::move(node));

        LOG_DEBUG("Added dataset node '{}' (id={})", name, id);
        return id;
    }

    NodeId Scene::addCameraGroup(const std::string& name, const NodeId parent, const size_t camera_count) {
        const NodeId id = next_node_id_++;
        auto node = std::make_unique<Node>();
        node->id = id;
        node->parent_id = parent;
        node->type = NodeType::CAMERA_GROUP;
        node->name = name;
        node->gaussian_count = camera_count; // Reuse gaussian_count to store camera count

        // Add to parent's children
        if (parent != NULL_NODE) {
            if (auto* p = getNodeById(parent)) {
                p->children.push_back(id);
            }
        }

        id_to_index_[id] = nodes_.size();
        node->initObservables(this);
        nodes_.push_back(std::move(node));

        LOG_DEBUG("Added camera group '{}' (id={}, {} cameras)", name, id, camera_count);
        return id;
    }

    NodeId Scene::addCamera(const std::string& name, const NodeId parent, const int camera_index, const int camera_uid,
                            const std::string& image_path, const std::string& mask_path) {
        const NodeId id = next_node_id_++;
        auto node = std::make_unique<Node>();
        node->id = id;
        node->parent_id = parent;
        node->type = NodeType::CAMERA;
        node->name = name;
        node->camera_index = camera_index;
        node->camera_uid = camera_uid;
        node->image_path = image_path;
        node->mask_path = mask_path;

        // Add to parent's children
        if (parent != NULL_NODE) {
            if (auto* p = getNodeById(parent)) {
                p->children.push_back(id);
            }
        }

        id_to_index_[id] = nodes_.size();
        node->initObservables(this);
        nodes_.push_back(std::move(node));

        return id;
    }

    std::string Scene::duplicateNode(const std::string& name) {
        const auto* src_node = getNode(name);
        if (!src_node)
            return "";

        // Helper to generate unique name
        auto generate_unique_name = [this](const std::string& base_name) -> std::string {
            std::string new_name = base_name + "_copy";
            int counter = 2;
            while (std::any_of(nodes_.begin(), nodes_.end(),
                               [&new_name](const std::unique_ptr<Node>& n) { return n->name == new_name; })) {
                new_name = base_name + "_copy_" + std::to_string(counter++);
            }
            return new_name;
        };

        // Recursive helper to duplicate node and children
        // IMPORTANT: Pass NodeId, not reference, because nodes_ vector may reallocate!
        std::function<NodeId(NodeId, NodeId)> duplicate_recursive =
            [&](const NodeId src_id, const NodeId parent_id) -> NodeId {
            // Must re-lookup node each time as vector may have reallocated
            const auto* src = getNodeById(src_id);
            if (!src)
                return NULL_NODE;

            // Copy data we need BEFORE calling addGroup/addSplat (which may reallocate)
            const std::string src_name_copy = src->name;
            const NodeType src_type = src->type;
            const glm::mat4 src_transform = src->local_transform;
            const bool src_visible = src->visible;
            const bool src_locked = src->locked;
            const std::vector<NodeId> src_children = src->children; // Copy children list

            const std::string new_name = generate_unique_name(src_name_copy);

            NodeId new_id = NULL_NODE;
            if (src_type == NodeType::GROUP) {
                new_id = addGroup(new_name, parent_id);
            } else if (src_type == NodeType::CROPBOX) {
                // Clone cropbox - only valid if parent is a SPLAT
                const auto* src_for_cropbox = getNodeById(src_id);
                if (src_for_cropbox && src_for_cropbox->cropbox && parent_id != NULL_NODE) {
                    new_id = addCropBox(new_name, parent_id);
                    // Copy cropbox data
                    if (auto* new_node = getNodeById(new_id)) {
                        if (new_node->cropbox) {
                            *new_node->cropbox = *src_for_cropbox->cropbox;
                        }
                    }
                }
            } else {
                // Re-lookup src after potential reallocation check
                const auto* src_for_model = getNodeById(src_id);
                if (src_for_model && src_for_model->model) {
                    // Clone SplatData
                    const auto& model = *src_for_model->model;
                    auto cloned = std::make_unique<lfs::core::SplatData>(
                        model.get_max_sh_degree(),
                        model.means_raw().clone(), model.sh0_raw().clone(), model.shN_raw().clone(),
                        model.scaling_raw().clone(), model.rotation_raw().clone(), model.opacity_raw().clone(),
                        model.get_scene_scale());
                    cloned->set_active_sh_degree(model.get_active_sh_degree());
                    new_id = addSplat(new_name, std::move(cloned), parent_id);
                }
            }

            // Copy transform and visibility (re-lookup new node as vector may have changed)
            if (auto* new_node = getNodeById(new_id)) {
                new_node->local_transform = src_transform;
                new_node->visible = src_visible;
                new_node->locked = src_locked;
                new_node->transform_dirty = true;
            }

            // Recursively duplicate children (using copied children list)
            for (const NodeId child_id : src_children) {
                duplicate_recursive(child_id, new_id);
            }

            return new_id;
        };

        // Store source info before any modifications
        const NodeId src_id = src_node->id;
        const NodeId src_parent_id = src_node->parent_id;
        const std::string result_name = generate_unique_name(src_node->name);

        duplicate_recursive(src_id, src_parent_id);

        invalidateCache();
        LOG_DEBUG("Duplicated node '{}' as '{}'", name, result_name);
        return result_name;
    }

    std::string Scene::mergeGroup(const std::string& group_name) {
        const auto* const group_node = getNode(group_name);
        if (!group_node || group_node->type != NodeType::GROUP) {
            return "";
        }

        std::vector<std::pair<const lfs::core::SplatData*, glm::mat4>> splats;
        const std::function<void(NodeId)> collect = [&](const NodeId id) {
            const auto* const node = getNodeById(id);
            if (!node)
                return;
            if (node->type == NodeType::SPLAT && node->model && node->visible) {
                splats.emplace_back(node->model.get(), getWorldTransform(id));
            }
            for (const NodeId cid : node->children)
                collect(cid);
        };

        const NodeId parent_id = group_node->parent_id;
        collect(group_node->id);

        auto merged = mergeSplatsWithTransforms(splats);
        if (!merged) {
            return "";
        }

        removeNode(group_name, false);
        addSplat(group_name, std::move(merged), parent_id);
        invalidateCache();

        return group_name;
    }

    std::unique_ptr<lfs::core::SplatData> Scene::createMergedModelWithTransforms() const {
        std::vector<std::pair<const lfs::core::SplatData*, glm::mat4>> splats;
        for (const auto& node : nodes_) {
            if (node->type == NodeType::SPLAT && node->model && isNodeEffectivelyVisible(node->id)) {
                splats.emplace_back(node->model.get(), getWorldTransform(node->id));
            }
        }
        return mergeSplatsWithTransforms(splats);
    }

    std::unique_ptr<lfs::core::SplatData> Scene::mergeSplatsWithTransforms(
        const std::vector<std::pair<const lfs::core::SplatData*, glm::mat4>>& splats) {
        if (splats.empty()) {
            return nullptr;
        }

        int max_sh = 0;
        for (const auto& [model, _] : splats) {
            max_sh = std::max(max_sh, model->get_max_sh_degree());
        }

        // Fast path: single splat with identity transform
        static const glm::mat4 IDENTITY{1.0f};
        if (splats.size() == 1 && splats[0].second == IDENTITY) {
            const auto* const src = splats[0].first;

            // Filter out deleted splats if deletion mask exists
            if (src->has_deleted_mask()) {
                const auto keep_mask = src->deleted().logical_not();
                auto result = std::make_unique<lfs::core::SplatData>(
                    src->get_max_sh_degree(),
                    src->means_raw().index_select(0, keep_mask),
                    src->sh0_raw().index_select(0, keep_mask),
                    src->shN_raw().is_valid() ? src->shN_raw().index_select(0, keep_mask) : lfs::core::Tensor(),
                    src->scaling_raw().index_select(0, keep_mask),
                    src->rotation_raw().index_select(0, keep_mask),
                    src->opacity_raw().index_select(0, keep_mask),
                    src->get_scene_scale());
                result->set_active_sh_degree(src->get_active_sh_degree());
                return result;
            } else {
                auto result = std::make_unique<lfs::core::SplatData>(
                    src->get_max_sh_degree(),
                    src->means_raw().clone(),
                    src->sh0_raw().clone(),
                    src->shN_raw().is_valid() ? src->shN_raw().clone() : lfs::core::Tensor(),
                    src->scaling_raw().clone(),
                    src->rotation_raw().clone(),
                    src->opacity_raw().clone(),
                    src->get_scene_scale());
                result->set_active_sh_degree(src->get_active_sh_degree());
                return result;
            }
        }

        const int shN_coeffs = (max_sh > 0) ? ((max_sh + 1) * (max_sh + 1) - 1) : 0;
        std::vector<lfs::core::Tensor> means_list, sh0_list, shN_list, scaling_list, rotation_list, opacity_list;
        means_list.reserve(splats.size());
        sh0_list.reserve(splats.size());
        scaling_list.reserve(splats.size());
        rotation_list.reserve(splats.size());
        opacity_list.reserve(splats.size());
        if (shN_coeffs > 0)
            shN_list.reserve(splats.size());

        float total_scale = 0.0f;

        for (const auto& [model, world_transform] : splats) {
            // Filter out deleted splats first
            lfs::core::Tensor means, sh0, shN, scaling, rotation, opacity;
            if (model->has_deleted_mask()) {
                const auto keep_mask = model->deleted().logical_not();
                means = model->means_raw().index_select(0, keep_mask);
                sh0 = model->sh0_raw().index_select(0, keep_mask);
                shN = model->shN_raw().is_valid() ? model->shN_raw().index_select(0, keep_mask) : lfs::core::Tensor();
                scaling = model->scaling_raw().index_select(0, keep_mask);
                rotation = model->rotation_raw().index_select(0, keep_mask);
                opacity = model->opacity_raw().index_select(0, keep_mask);
            } else {
                means = model->means_raw().clone();
                sh0 = model->sh0_raw().clone();
                shN = model->shN_raw().is_valid() ? model->shN_raw().clone() : lfs::core::Tensor();
                scaling = model->scaling_raw().clone();
                rotation = model->rotation_raw().clone();
                opacity = model->opacity_raw().clone();
            }

            lfs::core::SplatData transformed(
                model->get_max_sh_degree(),
                std::move(means),
                std::move(sh0),
                std::move(shN),
                std::move(scaling),
                std::move(rotation),
                std::move(opacity),
                model->get_scene_scale());

            lfs::core::transform(transformed, world_transform);

            means_list.push_back(transformed.means_raw().clone());
            sh0_list.push_back(transformed.sh0_raw().clone());
            scaling_list.push_back(transformed.scaling_raw().clone());
            rotation_list.push_back(transformed.rotation_raw().clone());
            opacity_list.push_back(transformed.opacity_raw().clone());

            if (shN_coeffs > 0) {
                const auto& src_shN = transformed.shN_raw();
                const int src_coeffs = (src_shN.is_valid() && src_shN.ndim() >= 2)
                                           ? static_cast<int>(src_shN.size(1))
                                           : 0;

                if (src_coeffs == shN_coeffs) {
                    shN_list.push_back(src_shN.clone());
                } else if (src_coeffs > 0) {
                    const int copy_coeffs = std::min(src_coeffs, shN_coeffs);
                    auto padded = lfs::core::Tensor::zeros(
                        {transformed.size(), static_cast<size_t>(shN_coeffs), 3},
                        lfs::core::Device::CUDA);
                    padded.slice(1, 0, copy_coeffs).copy_from(src_shN.slice(1, 0, copy_coeffs));
                    shN_list.push_back(std::move(padded));
                } else {
                    shN_list.push_back(lfs::core::Tensor::zeros(
                        {transformed.size(), static_cast<size_t>(shN_coeffs), 3},
                        lfs::core::Device::CUDA));
                }
            }

            total_scale += model->get_scene_scale();
        }

        auto result = std::make_unique<lfs::core::SplatData>(
            max_sh,
            lfs::core::Tensor::cat(means_list, 0),
            lfs::core::Tensor::cat(sh0_list, 0),
            (shN_coeffs > 0) ? lfs::core::Tensor::cat(shN_list, 0) : lfs::core::Tensor(),
            lfs::core::Tensor::cat(scaling_list, 0),
            lfs::core::Tensor::cat(rotation_list, 0),
            lfs::core::Tensor::cat(opacity_list, 0),
            total_scale / static_cast<float>(splats.size()));
        result->set_active_sh_degree(max_sh);

        return result;
    }

    void Scene::reparent(const NodeId node_id, const NodeId new_parent) {
        auto* node = getNodeById(node_id);
        if (!node)
            return;

        // Prevent circular references
        if (new_parent != NULL_NODE) {
            NodeId check = new_parent;
            while (check != NULL_NODE) {
                if (check == node_id) {
                    LOG_WARN("Cannot reparent: would create cycle");
                    return;
                }
                const auto* check_node = getNodeById(check);
                check = check_node ? check_node->parent_id : NULL_NODE;
            }
        }

        // Remove from old parent
        if (node->parent_id != NULL_NODE) {
            if (auto* old_parent = getNodeById(node->parent_id)) {
                auto& children = old_parent->children;
                children.erase(std::remove(children.begin(), children.end(), node_id), children.end());
            }
        }

        // Add to new parent
        node->parent_id = new_parent;
        if (new_parent != NULL_NODE) {
            if (auto* p = getNodeById(new_parent)) {
                p->children.push_back(node_id);
            }
        }

        markTransformDirty(node_id);
        invalidateCache();
    }

    const glm::mat4& Scene::getWorldTransform(const NodeId node_id) const {
        const auto* node = getNodeById(node_id);
        if (!node) {
            static const glm::mat4 IDENTITY{1.0f};
            return IDENTITY;
        }
        updateWorldTransform(*node);
        return node->world_transform;
    }

    std::vector<NodeId> Scene::getRootNodes() const {
        std::vector<NodeId> roots;
        for (const auto& node : nodes_) {
            if (node->parent_id == NULL_NODE) {
                roots.push_back(node->id);
            }
        }
        return roots;
    }

    Scene::Node* Scene::getNodeById(const NodeId id) {
        const auto it = id_to_index_.find(id);
        if (it == id_to_index_.end())
            return nullptr;
        return nodes_[it->second].get();
    }

    const Scene::Node* Scene::getNodeById(const NodeId id) const {
        const auto it = id_to_index_.find(id);
        if (it == id_to_index_.end())
            return nullptr;
        return nodes_[it->second].get();
    }

    bool Scene::isNodeEffectivelyVisible(const NodeId id) const {
        const auto* node = getNodeById(id);
        if (!node)
            return false;

        // Check this node's visibility
        if (!node->visible)
            return false;

        // Recursively check parent visibility
        if (node->parent_id != NULL_NODE) {
            return isNodeEffectivelyVisible(node->parent_id);
        }

        return true;
    }

    void Scene::markTransformDirty(const NodeId node_id) {
        auto* node = getNodeById(node_id);
        if (!node || node->transform_dirty)
            return;

        node->transform_dirty = true;
        for (const NodeId child_id : node->children) {
            markTransformDirty(child_id);
        }
    }

    void Scene::updateWorldTransform(const Node& node) const {
        if (!node.transform_dirty)
            return;

        if (node.parent_id == NULL_NODE) {
            node.world_transform = node.local_transform;
        } else {
            const auto* parent = getNodeById(node.parent_id);
            if (parent) {
                updateWorldTransform(*parent);
                node.world_transform = parent->world_transform * node.local_transform;
            } else {
                node.world_transform = node.local_transform;
            }
        }
        node.transform_dirty = false;
    }

    bool Scene::getNodeBounds(const NodeId id, glm::vec3& out_min, glm::vec3& out_max) const {
        const auto* node = getNodeById(id);
        if (!node)
            return false;

        bool has_bounds = false;
        glm::vec3 total_min(std::numeric_limits<float>::max());
        glm::vec3 total_max(std::numeric_limits<float>::lowest());

        // Helper to expand bounds
        const auto expand_bounds = [&](const glm::vec3& min_b, const glm::vec3& max_b) {
            total_min = glm::min(total_min, min_b);
            total_max = glm::max(total_max, max_b);
            has_bounds = true;
        };

        // If this node has a model (SPLAT), include its bounds
        if (node->model && node->model->size() > 0) {
            glm::vec3 model_min, model_max;
            if (lfs::core::compute_bounds(*node->model, model_min, model_max)) {
                expand_bounds(model_min, model_max);
            }
        }

        // If this node has a point cloud (POINTCLOUD), include its bounds
        if (node->point_cloud && node->point_cloud->size() > 0) {
            auto means_cpu = node->point_cloud->means.cpu();
            auto acc = means_cpu.accessor<float, 2>();
            glm::vec3 pc_min(std::numeric_limits<float>::max());
            glm::vec3 pc_max(std::numeric_limits<float>::lowest());
            for (int64_t i = 0; i < node->point_cloud->size(); ++i) {
                pc_min.x = std::min(pc_min.x, acc(i, 0));
                pc_min.y = std::min(pc_min.y, acc(i, 1));
                pc_min.z = std::min(pc_min.z, acc(i, 2));
                pc_max.x = std::max(pc_max.x, acc(i, 0));
                pc_max.y = std::max(pc_max.y, acc(i, 1));
                pc_max.z = std::max(pc_max.z, acc(i, 2));
            }
            expand_bounds(pc_min, pc_max);
        }

        // If this node is a CROPBOX, include its bounds
        if (node->type == NodeType::CROPBOX && node->cropbox) {
            expand_bounds(node->cropbox->min, node->cropbox->max);
        }

        // Recursively include children bounds
        for (const NodeId child_id : node->children) {
            glm::vec3 child_min, child_max;
            if (getNodeBounds(child_id, child_min, child_max)) {
                // Transform child bounds by child's local transform relative to this node
                const auto* child = getNodeById(child_id);
                if (child) {
                    // Transform the 8 corners of child bounding box
                    const glm::mat4& child_transform = child->local_transform;
                    glm::vec3 corners[8] = {
                        {child_min.x, child_min.y, child_min.z},
                        {child_max.x, child_min.y, child_min.z},
                        {child_min.x, child_max.y, child_min.z},
                        {child_max.x, child_max.y, child_min.z},
                        {child_min.x, child_min.y, child_max.z},
                        {child_max.x, child_min.y, child_max.z},
                        {child_min.x, child_max.y, child_max.z},
                        {child_max.x, child_max.y, child_max.z}};
                    for (const auto& corner : corners) {
                        const glm::vec3 transformed = glm::vec3(child_transform * glm::vec4(corner, 1.0f));
                        expand_bounds(transformed, transformed);
                    }
                }
            }
        }

        if (has_bounds) {
            out_min = total_min;
            out_max = total_max;
        }
        return has_bounds;
    }

    glm::vec3 Scene::getNodeBoundsCenter(const NodeId id) const {
        glm::vec3 min_bounds, max_bounds;
        if (getNodeBounds(id, min_bounds, max_bounds)) {
            return (min_bounds + max_bounds) * 0.5f;
        }
        return glm::vec3(0.0f);
    }

    // ========== CropBox Operations ==========

    NodeId Scene::getCropBoxForSplat(const NodeId splat_id) const {
        const auto* node = getNodeById(splat_id);
        // Support both SPLAT and POINTCLOUD nodes
        if (!node || (node->type != NodeType::SPLAT && node->type != NodeType::POINTCLOUD)) {
            return NULL_NODE;
        }

        for (const NodeId child_id : node->children) {
            if (const auto* child = getNodeById(child_id)) {
                if (child->type == NodeType::CROPBOX) {
                    return child_id;
                }
            }
        }
        return NULL_NODE;
    }

    NodeId Scene::getOrCreateCropBoxForSplat(const NodeId splat_id) {
        NodeId existing = getCropBoxForSplat(splat_id);
        if (existing != NULL_NODE) {
            return existing;
        }

        const auto* node = getNodeById(splat_id);
        // Support both SPLAT and POINTCLOUD nodes
        if (!node || (node->type != NodeType::SPLAT && node->type != NodeType::POINTCLOUD)) {
            return NULL_NODE;
        }

        // Create a cropbox with a name based on the node name
        const std::string cropbox_name = node->name + "_cropbox";
        return addCropBox(cropbox_name, splat_id);
    }

    CropBoxData* Scene::getCropBoxData(const NodeId cropbox_id) {
        auto* node = getNodeById(cropbox_id);
        if (!node || node->type != NodeType::CROPBOX) {
            return nullptr;
        }
        return node->cropbox.get();
    }

    const CropBoxData* Scene::getCropBoxData(const NodeId cropbox_id) const {
        const auto* node = getNodeById(cropbox_id);
        if (!node || node->type != NodeType::CROPBOX) {
            return nullptr;
        }
        return node->cropbox.get();
    }

    void Scene::setCropBoxData(const NodeId cropbox_id, const CropBoxData& data) {
        auto* node = getNodeById(cropbox_id);
        if (!node || node->type != NodeType::CROPBOX || !node->cropbox) {
            return;
        }
        *node->cropbox = data;
        // Note: cropbox changes don't invalidate the combined model cache
        // since cropboxes are not part of the splat data
    }

    std::vector<Scene::RenderableCropBox> Scene::getVisibleCropBoxes() const {
        std::vector<RenderableCropBox> result;

        for (const auto& node : nodes_) {
            if (node->type != NodeType::CROPBOX)
                continue;
            if (!node->visible)
                continue;
            if (!node->cropbox)
                continue;

            // Check if parent (splat or pointcloud) is effectively visible
            if (!isNodeEffectivelyVisible(node->parent_id))
                continue;

            RenderableCropBox rcb;
            rcb.node_id = node->id;
            rcb.parent_splat_id = node->parent_id;
            rcb.data = node->cropbox.get();
            rcb.world_transform = getWorldTransform(node->id);
            rcb.local_transform = node->local_transform.get();
            result.push_back(rcb);
        }

        return result;
    }

    // ========== Training Data Storage ==========

    void Scene::setTrainCameras(std::shared_ptr<lfs::training::CameraDataset> dataset) {
        train_cameras_ = std::move(dataset);
        LOG_DEBUG("Scene: set train cameras ({})", train_cameras_ ? "valid" : "null");
    }

    void Scene::setValCameras(std::shared_ptr<lfs::training::CameraDataset> dataset) {
        val_cameras_ = std::move(dataset);
        LOG_DEBUG("Scene: set val cameras ({})", val_cameras_ ? "valid" : "null");
    }

    void Scene::setInitialPointCloud(std::shared_ptr<lfs::core::PointCloud> point_cloud) {
        initial_point_cloud_ = std::move(point_cloud);
        LOG_DEBUG("Scene: set initial point cloud ({})", initial_point_cloud_ ? "valid" : "null");
    }

    void Scene::setSceneCenter(lfs::core::Tensor scene_center) {
        scene_center_ = std::move(scene_center);
        if (scene_center_.is_valid()) {
            auto sc_cpu = scene_center_.cpu();
            const float* ptr = sc_cpu.ptr<float>();
            LOG_DEBUG("Scene: set scene center to [{:.3f}, {:.3f}, {:.3f}]", ptr[0], ptr[1], ptr[2]);
        } else {
            LOG_DEBUG("Scene: set scene center (invalid/empty)");
        }
    }

    void Scene::setTrainingModelNode(const std::string& name) {
        training_model_node_ = name;
        LOG_DEBUG("Scene: set training model node to '{}'", name);
    }

    void Scene::setTrainingModel(std::unique_ptr<lfs::core::SplatData> splat_data, const std::string& name) {
        // Add as a new SPLAT node
        addNode(name, std::move(splat_data));
        // Set it as the training model
        setTrainingModelNode(name);
        LOG_INFO("Scene: created training model node '{}' from checkpoint", name);
    }

    lfs::core::SplatData* Scene::getTrainingModel() {
        if (training_model_node_.empty())
            return nullptr;
        auto* node = getMutableNode(training_model_node_);
        if (!node || !isNodeEffectivelyVisible(node->id))
            return nullptr;
        return node->model.get();
    }

    const lfs::core::SplatData* Scene::getTrainingModel() const {
        if (training_model_node_.empty())
            return nullptr;
        const auto* node = getNode(training_model_node_);
        if (!node || !isNodeEffectivelyVisible(node->id))
            return nullptr;
        return node->model.get();
    }

    std::shared_ptr<const lfs::core::Camera> Scene::getCameraByUid(int uid) const {
        if (!train_cameras_) {
            return nullptr;
        }
        for (const auto& cam : train_cameras_->get_cameras()) {
            if (cam->uid() == uid) {
                return cam;
            }
        }
        return nullptr;
    }

    std::vector<std::shared_ptr<const lfs::core::Camera>> Scene::getAllCameras() const {
        std::vector<std::shared_ptr<const lfs::core::Camera>> result;
        if (train_cameras_) {
            const auto& cams = train_cameras_->get_cameras();
            result.reserve(cams.size());
            for (const auto& cam : cams) {
                result.push_back(cam);
            }
        }
        return result;
    }

} // namespace lfs::vis