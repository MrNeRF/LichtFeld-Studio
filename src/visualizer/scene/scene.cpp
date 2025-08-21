#include "scene/scene.hpp"
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
                acc.max_sh_degree = std::max(acc.max_sh_degree, model->get_active_sh_degree());
                acc.total_scene_scale += model->get_scene_scale();
                acc.has_shN = acc.has_shN || (model->shN().numel() > 0);
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

        // Calculate SH dimensions
        constexpr auto sh_coefficients = [](int degree) -> std::pair<int, int> {
            int total = (degree + 1) * (degree + 1);
            return {1, total - 1}; // sh0 always 1, shN is the rest
        };
        auto [sh0_coeffs, shN_coeffs] = sh_coefficients(stats.max_sh_degree);

        // Pre-allocate all tensors at once
        struct CombinedTensors {
            torch::Tensor means, sh0, shN, opacity, scaling, rotation;
        } combined{
            .means = torch::empty({static_cast<int64_t>(stats.total_gaussians), 3}, opts),
            .sh0 = torch::empty({static_cast<int64_t>(stats.total_gaussians), sh0_coeffs, 3}, opts),
            .shN = torch::zeros({static_cast<int64_t>(stats.total_gaussians),
                                 stats.has_shN ? shN_coeffs : 0, 3},
                                opts),
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

            // Copy sh0 (should always match for valid PLYs)
            combined.sh0.index({slice}) = model->sh0();

            // Copy shN if present (leave zeros for degree 0 models)
            if (shN_coeffs > 0 && model->shN().numel() > 0) {
                const auto copy_coeffs = std::min<int64_t>(model->shN().size(1), shN_coeffs);
                combined.shN.index({slice, torch::indexing::Slice(0, copy_coeffs)}) =
                    model->shN().index({torch::indexing::Slice(), torch::indexing::Slice(0, copy_coeffs)});
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
} // namespace gs
