/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "scene/scene_manager.hpp"
#include "command/command_history.hpp"
#include "command/commands/crop_command.hpp"
#include "command/commands/composite_command.hpp"
#include "command/commands/mirror_command.hpp"
#include "core/logger.hpp"
#include "core/parameter_manager.hpp"
#include "core/services.hpp"
#include "core/splat_data_export.hpp"
#include "core/splat_data_transform.hpp"
#include "geometry/bounding_box.hpp"
#include "geometry/euclidean_transform.hpp"
#include "gui/panels/gizmo_toolbar.hpp"
#include "io/loader.hpp"
#include "rendering/rendering_manager.hpp"
#include "training/checkpoint.hpp"
#include "training/trainer.hpp"
#include "training/training_manager.hpp"
#include "training/training_setup.hpp"
#include <algorithm>
#include <format>
#include <glm/gtc/quaternion.hpp>
#include <set>
#include <stdexcept>

namespace lfs::vis {

    using namespace lfs::core::events;

    SceneManager::SceneManager() {
        setupEventHandlers();
        LOG_DEBUG("SceneManager initialized");
    }

    SceneManager::~SceneManager() = default;

    void SceneManager::setupEventHandlers() {

        // Handle PLY commands
        cmd::AddPLY::when([this](const auto& cmd) {
            addSplatFile(cmd.path, cmd.name);
        });

        cmd::RemovePLY::when([this](const auto& cmd) {
            removePLY(cmd.name, cmd.keep_children);
        });

        cmd::SetPLYVisibility::when([this](const auto& cmd) {
            setPLYVisibility(cmd.name, cmd.visible);
        });

        cmd::ClearScene::when([this](const auto&) {
            clear();
        });

        cmd::SwitchToEditMode::when([this](const auto&) {
            switchToEditMode();
        });

        // Handle PLY cycling with proper event emission for UI updates
        cmd::CyclePLY::when([this](const auto&) {
            // Check if rendering manager has split view enabled (in PLY comparison mode)
            if (services().renderingOrNull()) {
                auto settings = services().renderingOrNull()->getSettings();
                if (settings.split_view_mode == lfs::vis::SplitViewMode::PLYComparison) {
                    // In split mode: advance the offset
                    services().renderingOrNull()->advanceSplitOffset();
                    LOG_DEBUG("Advanced split view offset");
                    return; // Don't cycle visibility when in split view
                }
            }

            // Normal mode: existing cycle code
            if (content_type_ == ContentType::SplatFiles) {
                auto [hidden, shown] = scene_.cycleVisibilityWithNames();

                if (!hidden.empty()) {
                    cmd::SetPLYVisibility{.name = hidden, .visible = false}.emit();
                }
                if (!shown.empty()) {
                    cmd::SetPLYVisibility{.name = shown, .visible = true}.emit();
                    LOG_DEBUG("Cycled to: {}", shown);
                }

                emitSceneChanged();
            }
        });

        cmd::CropPLY::when([this](const auto& cmd) {
            handleCropActivePly(cmd.crop_box, cmd.inverse);
        });

        cmd::FitCropBoxToScene::when([this](const auto& cmd) {
            updateCropBoxToFitScene(cmd.use_percentile);
        });

        cmd::RenamePLY::when([this](const auto& cmd) {
            handleRenamePly(cmd);
        });

        cmd::ReparentNode::when([this](const auto& cmd) {
            handleReparentNode(cmd.node_name, cmd.new_parent_name);
        });

        cmd::AddGroup::when([this](const auto& cmd) {
            handleAddGroup(cmd.name, cmd.parent_name);
        });

        cmd::DuplicateNode::when([this](const auto& cmd) {
            handleDuplicateNode(cmd.name);
        });

        cmd::MergeGroup::when([this](const auto& cmd) {
            handleMergeGroup(cmd.name);
        });

        // Handle node selection from scene panel (both PLYs and Groups)
        ui::NodeSelected::when([this](const auto& event) {
            // Don't allow selection changes during training
            if (services().trainerOrNull() && services().trainerOrNull()->isRunning()) {
                LOG_TRACE("Ignoring selection change during training");
                return;
            }

            if (event.type == "PLY" || event.type == "Group" || event.type == "Dataset" ||
                event.type == "PointCloud" || event.type == "CameraGroup") {
                // Skip if this is a multi-select event (already handled by selectNodes)
                auto it = event.metadata.find("multi_select");
                if (it != event.metadata.end() && it->second != "1") {
                    // Multi-select already set selected_nodes_, just sync cropbox
                    syncCropBoxToRenderSettings();
                    return;
                }

                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    selected_nodes_.clear();
                    selected_nodes_.insert(event.path);
                }
                // Sync selected node's cropbox to render settings
                syncCropBoxToRenderSettings();
            }
        });

        // Handle node deselection (but not during training)
        ui::NodeDeselected::when([this](const auto&) {
            // Don't clear selection during training - dataset and model must stay selected
            if (services().trainerOrNull() && services().trainerOrNull()->isRunning()) {
                LOG_TRACE("Ignoring deselection during training");
                return;
            }
            std::lock_guard<std::mutex> lock(state_mutex_);
            selected_nodes_.clear();
        });
    }

    void SceneManager::changeContentType(const ContentType& type) {
        std::lock_guard<std::mutex> lock(state_mutex_);

        const char* type_str = (type == ContentType::Empty) ? "Empty" : (type == ContentType::SplatFiles) ? "SplatFiles"
                                                                                                          : "Dataset";
        LOG_DEBUG("Changing content type to: {}", type_str);

        content_type_ = type;
    }

    void SceneManager::loadSplatFile(const std::filesystem::path& path) {
        LOG_TIMER("SceneManager::loadSplatFile");

        try {
            LOG_INFO("Loading splat file: {}", path.string());

            // Clear existing scene
            clear();

            // Load the file
            LOG_DEBUG("Creating loader for splat file");
            auto loader = lfs::io::Loader::create();
            lfs::io::LoadOptions options{
                .resize_factor = -1,
                .max_width = 3840,
                .images_folder = "images",
                .validate_only = false};

            LOG_TRACE("Loading splat file with loader");
            auto load_result = loader->load(path, options);
            if (!load_result) {
                LOG_ERROR("Failed to load splat file: {}", load_result.error().format());
                throw std::runtime_error(load_result.error().format());
            }

            auto* splat_data = std::get_if<std::shared_ptr<lfs::core::SplatData>>(&load_result->data);
            if (!splat_data || !*splat_data) {
                LOG_ERROR("Expected splat file but got different data type from: {}", path.string());
                throw std::runtime_error("Expected splat file but got different data type");
            }

            // Add to scene
            std::string name = path.stem().string();
            size_t gaussian_count = (*splat_data)->size();
            LOG_DEBUG("Adding '{}' to scene with {} gaussians", name, gaussian_count);

            scene_.addNode(name, std::make_unique<lfs::core::SplatData>(std::move(**splat_data)));

            // Create cropbox as child of this splat
            const auto* splat_node = scene_.getNode(name);
            if (splat_node) {
                const NodeId cropbox_id = scene_.getOrCreateCropBoxForSplat(splat_node->id);
                if (cropbox_id != NULL_NODE) {
                    LOG_DEBUG("Created cropbox for '{}'", name);
                }
            }

            // Update content state
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                content_type_ = ContentType::SplatFiles;
                splat_paths_.clear();
                splat_paths_[name] = path;
            }

            // Determine file type for event
            auto ext = path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            state::SceneLoaded::Type file_type = state::SceneLoaded::Type::PLY;
            if (ext == ".sog") {
                file_type = state::SceneLoaded::Type::SOG;
            } else if (ext == ".spz") {
                file_type = state::SceneLoaded::Type::SPZ;
            }

            // Emit events
            state::SceneLoaded{
                .scene = nullptr,
                .path = path,
                .type = file_type,
                .num_gaussians = scene_.getTotalGaussianCount()}
                .emit();

            state::PLYAdded{
                .name = name,
                .node_gaussians = gaussian_count,
                .total_gaussians = scene_.getTotalGaussianCount(),
                .is_visible = true,
                .parent_name = "",
                .is_group = false,
                .node_type = 0} // SPLAT
                .emit();

            // Emit PLYAdded for the cropbox (re-lookup splat as vector may have reallocated)
            const auto* splat_for_cropbox = scene_.getNode(name);
            if (splat_for_cropbox) {
                const NodeId cropbox_id = scene_.getCropBoxForSplat(splat_for_cropbox->id);
                if (cropbox_id != NULL_NODE) {
                    const auto* cropbox_node = scene_.getNodeById(cropbox_id);
                    if (cropbox_node) {
                        LOG_DEBUG("Emitting PLYAdded for cropbox '{}'", cropbox_node->name);
                        state::PLYAdded{
                            .name = cropbox_node->name,
                            .node_gaussians = 0,
                            .total_gaussians = scene_.getTotalGaussianCount(),
                            .is_visible = true,
                            .parent_name = name,
                            .is_group = false,
                            .node_type = 2} // CROPBOX
                            .emit();
                    }
                }
            }

            emitSceneChanged();
            updateCropBoxToFitScene(true);
            selectNode(name);
            tools::SetToolbarTool{.tool_mode = static_cast<int>(gui::panels::ToolType::CropBox)}.emit();

            LOG_INFO("Loaded '{}' with {} gaussians", name, gaussian_count);

        } catch (const std::exception& e) {
            LOG_ERROR("Failed to load splat file: {} (path: {})", e.what(), path.string());
            throw;
        }
    }

    std::string SceneManager::addSplatFile(const std::filesystem::path& path, const std::string& name_hint,
                                           const bool is_visible) {
        LOG_TIMER_TRACE("SceneManager::addSplatFile");

        try {
            if (content_type_ != ContentType::SplatFiles) {
                loadSplatFile(path);
                return path.stem().string();
            }

            auto loader = lfs::io::Loader::create();
            const lfs::io::LoadOptions options{
                .resize_factor = -1,
                .max_width = 3840,
                .images_folder = "images",
                .validate_only = false};

            auto load_result = loader->load(path, options);
            if (!load_result) {
                throw std::runtime_error(load_result.error().format());
            }

            auto* splat_data = std::get_if<std::shared_ptr<lfs::core::SplatData>>(&load_result->data);
            if (!splat_data || !*splat_data) {
                throw std::runtime_error("Expected splat file");
            }

            // Generate unique name
            const std::string base_name = name_hint.empty() ? path.stem().string() : name_hint;
            std::string name = base_name;
            int counter = 1;
            while (scene_.getNode(name) != nullptr) {
                name = std::format("{}_{}", base_name, counter++);
            }

            const size_t gaussian_count = (*splat_data)->size();
            scene_.addNode(name, std::make_unique<lfs::core::SplatData>(std::move(**splat_data)));

            // Create cropbox as child of this splat
            if (const auto* splat_node = scene_.getNode(name)) {
                [[maybe_unused]] const auto cropbox_id = scene_.getOrCreateCropBoxForSplat(splat_node->id);
            }

            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                splat_paths_[name] = path;
            }

            state::PLYAdded{
                .name = name,
                .node_gaussians = gaussian_count,
                .total_gaussians = scene_.getTotalGaussianCount(),
                .is_visible = is_visible,
                .parent_name = "",
                .is_group = false,
                .node_type = 0}
                .emit();

            // Emit PLYAdded for the cropbox
            const auto* splat_for_cropbox = scene_.getNode(name);
            if (splat_for_cropbox) {
                const NodeId cropbox_id = scene_.getCropBoxForSplat(splat_for_cropbox->id);
                if (cropbox_id != NULL_NODE) {
                    const auto* cropbox_node = scene_.getNodeById(cropbox_id);
                    if (cropbox_node) {
                        state::PLYAdded{
                            .name = cropbox_node->name,
                            .node_gaussians = 0,
                            .total_gaussians = scene_.getTotalGaussianCount(),
                            .is_visible = true,
                            .parent_name = name,
                            .is_group = false,
                            .node_type = 2}
                            .emit();
                    }
                }
            }

            emitSceneChanged();
            updateCropBoxToFitScene(true);
            selectNode(name);

            LOG_INFO("Added '{}' ({} gaussians)", name, gaussian_count);
            return name;

        } catch (const std::exception& e) {
            LOG_ERROR("Failed to add splat file: {} (path: {})", e.what(), path.string());
            throw;
        }
    }

    void SceneManager::removePLY(const std::string& name, const bool keep_children) {
        const auto& training_name = scene_.getTrainingModelNodeName();

        // Check if node is or contains training model
        const auto isTrainingNode = [&]() -> bool {
            if (training_name.empty())
                return false;
            if (name == training_name)
                return true;
            for (const auto* n = scene_.getNode(training_name); n && n->parent_id != NULL_NODE;) {
                n = scene_.getNodeById(n->parent_id);
                if (n && n->name == name)
                    return true;
            }
            return false;
        };

        const bool affects_training = isTrainingNode();

        // Use state machine to check if deletion is allowed
        if (affects_training && services().trainerOrNull()) {
            if (!services().trainerOrNull()->canPerform(TrainingAction::DeleteTrainingNode)) {
                LOG_WARN("Cannot delete '{}': {}", name,
                         services().trainerOrNull()->getActionBlockedReason(TrainingAction::DeleteTrainingNode));
                return;
            }

            // Clean up training state if deleting training model (e.g., while paused)
            LOG_INFO("Stopping training due to node deletion: {}", name);
            services().trainerOrNull()->stopTraining();
            services().trainerOrNull()->waitForCompletion();
            services().trainerOrNull()->clearTrainer();
            scene_.setTrainingModelNode("");
        }

        std::string parent_name;
        if (const auto* node = scene_.getNode(name)) {
            if (node->parent_id != NULL_NODE) {
                if (const auto* p = scene_.getNodeById(node->parent_id)) {
                    parent_name = p->name;
                }
            }
        }

        scene_.removeNode(name, keep_children);
        {
            std::lock_guard lock(state_mutex_);
            splat_paths_.erase(name);
            selected_nodes_.erase(name);
        }

        if (scene_.getNodeCount() == 0) {
            std::lock_guard lock(state_mutex_);
            content_type_ = ContentType::Empty;
            dataset_path_.clear();
        }

        state::PLYRemoved{.name = name, .children_kept = keep_children, .parent_of_removed = parent_name}.emit();
        emitSceneChanged();
    }

    void SceneManager::setPLYVisibility(const std::string& name, const bool visible) {
        scene_.setNodeVisibility(name, visible);
        emitSceneChanged();
    }

    // ========== Node Selection ==========

    void SceneManager::selectNode(const std::string& name) {
        const auto* node = scene_.getNode(name);
        if (node != nullptr) {
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                selected_nodes_.clear();
                selected_nodes_.insert(name);
            }
            ui::NodeSelected{
                .path = name,
                .type = "PLY",
                .metadata = {
                    {"name", name},
                    {"gaussians", std::to_string(node->model ? node->model->size() : 0)},
                    {"visible", node->visible ? "true" : "false"}}}
                .emit();
        }
    }

    void SceneManager::selectNodes(const std::vector<std::string>& names) {
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            selected_nodes_.clear();
            for (const auto& name : names) {
                if (scene_.getNode(name) != nullptr) {
                    selected_nodes_.insert(name);
                }
            }
        }
        if (services().renderingOrNull())
            services().renderingOrNull()->triggerSelectionFlash();
    }

    void SceneManager::addToSelection(const std::string& name) {
        if (scene_.getNode(name) == nullptr)
            return;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            selected_nodes_.insert(name);
        }
        if (services().renderingOrNull())
            services().renderingOrNull()->triggerSelectionFlash();
    }

    void SceneManager::clearSelection() {
        std::lock_guard<std::mutex> lock(state_mutex_);
        selected_nodes_.clear();
        LOG_TRACE("Cleared node selection");
    }

    std::string SceneManager::getSelectedNodeName() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (selected_nodes_.empty())
            return "";
        return *selected_nodes_.begin();
    }

    std::vector<std::string> SceneManager::getSelectedNodeNames() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return std::vector<std::string>(selected_nodes_.begin(), selected_nodes_.end());
    }

    bool SceneManager::hasSelectedNode() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (selected_nodes_.empty())
            return false;
        // Check if at least one selected node still exists
        for (const auto& name : selected_nodes_) {
            if (scene_.getNode(name) != nullptr)
                return true;
        }
        return false;
    }

    NodeType SceneManager::getSelectedNodeType() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (selected_nodes_.empty()) {
            return NodeType::SPLAT;
        }
        const auto* node = scene_.getNode(*selected_nodes_.begin());
        return node ? node->type : NodeType::SPLAT;
    }

    int SceneManager::getSelectedNodeIndex() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (selected_nodes_.empty()) {
            return -1;
        }
        return scene_.getVisibleNodeIndex(*selected_nodes_.begin());
    }

    std::vector<bool> SceneManager::getSelectedNodeMask() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return scene_.getSelectedNodeMask(std::vector<std::string>(selected_nodes_.begin(), selected_nodes_.end()));
    }

    std::string SceneManager::pickNodeAtWorldPosition(const glm::vec3& world_pos) const {
        constexpr float INVALID_COORD = -1e9f;
        if (world_pos.x <= INVALID_COORD) {
            return "";
        }

        for (const auto* node : scene_.getVisibleNodes()) {
            if (node->type != NodeType::SPLAT)
                continue;

            glm::vec3 local_min, local_max;
            if (!scene_.getNodeBounds(node->id, local_min, local_max))
                continue;

            const glm::mat4 world_to_local = glm::inverse(scene_.getWorldTransform(node->id));
            const glm::vec3 local_pos = glm::vec3(world_to_local * glm::vec4(world_pos, 1.0f));

            if (local_pos.x >= local_min.x && local_pos.x <= local_max.x &&
                local_pos.y >= local_min.y && local_pos.y <= local_max.y &&
                local_pos.z >= local_min.z && local_pos.z <= local_max.z) {
                return node->name;
            }
        }
        return "";
    }

    std::vector<std::string> SceneManager::pickNodesInScreenRect(
        const glm::vec2& rect_min, const glm::vec2& rect_max,
        const glm::mat4& view, const glm::mat4& proj,
        const glm::ivec2& viewport_size) const {

        constexpr float BEHIND_CAMERA = -1e10f;
        constexpr int BBOX_CORNERS = 8;

        std::vector<std::string> result;

        const auto projectToScreen = [&](const glm::vec3& world_pos) -> glm::vec2 {
            const glm::vec4 clip = proj * view * glm::vec4(world_pos, 1.0f);
            if (clip.w <= 0.0f)
                return glm::vec2(BEHIND_CAMERA);
            const glm::vec3 ndc = glm::vec3(clip) / clip.w;
            return glm::vec2(
                (ndc.x * 0.5f + 0.5f) * static_cast<float>(viewport_size.x),
                (1.0f - (ndc.y * 0.5f + 0.5f)) * static_cast<float>(viewport_size.y));
        };

        const auto rectsOverlap = [](const glm::vec2& a_min, const glm::vec2& a_max,
                                     const glm::vec2& b_min, const glm::vec2& b_max) {
            return !(a_max.x < b_min.x || b_max.x < a_min.x ||
                     a_max.y < b_min.y || b_max.y < a_min.y);
        };

        for (const auto* node : scene_.getVisibleNodes()) {
            if (node->type != NodeType::SPLAT)
                continue;

            glm::vec3 local_min, local_max;
            if (!scene_.getNodeBounds(node->id, local_min, local_max))
                continue;

            const glm::mat4 world_transform = scene_.getWorldTransform(node->id);

            glm::vec2 screen_min(1e10f);
            glm::vec2 screen_max(-1e10f);
            bool any_visible = false;

            for (int i = 0; i < BBOX_CORNERS; ++i) {
                const glm::vec3 corner(
                    (i & 1) ? local_max.x : local_min.x,
                    (i & 2) ? local_max.y : local_min.y,
                    (i & 4) ? local_max.z : local_min.z);
                const glm::vec3 world_corner = glm::vec3(world_transform * glm::vec4(corner, 1.0f));
                const glm::vec2 screen_pos = projectToScreen(world_corner);

                if (screen_pos.x > BEHIND_CAMERA + 1e5f) {
                    screen_min = glm::min(screen_min, screen_pos);
                    screen_max = glm::max(screen_max, screen_pos);
                    any_visible = true;
                }
            }

            if (!any_visible)
                continue;

            // Check if the screen-space bbox overlaps with the selection rectangle
            if (rectsOverlap(rect_min, rect_max, screen_min, screen_max)) {
                result.push_back(node->name);
            }
        }

        return result;
    }

    void SceneManager::ensureCropBoxForSelectedNode() {
        std::string node_name;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            // If nothing selected, try to find a POINTCLOUD node (dataset mode)
            if (selected_nodes_.empty()) {
                for (const auto* node : scene_.getNodes()) {
                    if (node->type == NodeType::POINTCLOUD) {
                        node_name = node->name;
                        break;
                    }
                }
                if (node_name.empty())
                    return;
            } else {
                node_name = *selected_nodes_.begin();
            }
        }
        if (node_name.empty())
            return;

        const auto* node = scene_.getNode(node_name);
        if (!node)
            return;

        // For CROPBOX nodes, use parent SPLAT/POINTCLOUD
        NodeId target_id = node->id;
        if (node->type == NodeType::CROPBOX && node->parent_id != NULL_NODE) {
            target_id = node->parent_id;
        } else if (node->type == NodeType::GROUP) {
            // For groups, find first child SPLAT or POINTCLOUD
            for (const NodeId child_id : node->children) {
                if (const auto* child = scene_.getNodeById(child_id)) {
                    if (child->type == NodeType::SPLAT || child->type == NodeType::POINTCLOUD) {
                        target_id = child_id;
                        break;
                    }
                }
            }
        }

        const auto* target = scene_.getNodeById(target_id);
        if (!target || (target->type != NodeType::SPLAT && target->type != NodeType::POINTCLOUD))
            return;

        // Check if cropbox already exists
        const NodeId existing = scene_.getCropBoxForSplat(target_id);
        if (existing != NULL_NODE)
            return;

        // Create cropbox and fit to bounds
        const NodeId cropbox_id = scene_.getOrCreateCropBoxForSplat(target_id);
        if (cropbox_id == NULL_NODE)
            return;

        // Fit cropbox to node bounds
        glm::vec3 min_bounds, max_bounds;
        if (scene_.getNodeBounds(target_id, min_bounds, max_bounds)) {
            CropBoxData data;
            data.min = min_bounds;
            data.max = max_bounds;
            data.enabled = true;
            scene_.setCropBoxData(cropbox_id, data);
        }

        // Emit PLYAdded for the new cropbox
        if (const auto* cropbox = scene_.getNodeById(cropbox_id)) {
            state::PLYAdded{
                .name = cropbox->name,
                .node_gaussians = 0,
                .total_gaussians = scene_.getTotalGaussianCount(),
                .is_visible = cropbox->visible,
                .parent_name = target->name,
                .is_group = false,
                .node_type = static_cast<int>(NodeType::CROPBOX)}
                .emit();
        }

        LOG_DEBUG("Created cropbox for node '{}'", target->name);
    }

    void SceneManager::selectCropBoxForCurrentNode() {
        NodeId target_id = NULL_NODE;

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (!selected_nodes_.empty()) {
                const auto* node = scene_.getNode(*selected_nodes_.begin());
                if (node) {
                    if (node->type == NodeType::SPLAT || node->type == NodeType::POINTCLOUD) {
                        target_id = node->id;
                    } else if (node->type == NodeType::CROPBOX) {
                        return; // Already a cropbox selected
                    }
                }
            }
        }

        // Fall back to first POINTCLOUD
        if (target_id == NULL_NODE) {
            for (const auto* node : scene_.getNodes()) {
                if (node->type == NodeType::POINTCLOUD) {
                    target_id = node->id;
                    break;
                }
            }
        }

        if (target_id == NULL_NODE)
            return;

        const NodeId cropbox_id = scene_.getCropBoxForSplat(target_id);
        if (cropbox_id == NULL_NODE)
            return;

        const auto* cropbox = scene_.getNodeById(cropbox_id);
        if (!cropbox)
            return;

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            selected_nodes_.clear();
            selected_nodes_.insert(cropbox->name);
        }

        LOG_DEBUG("Auto-selected cropbox '{}'", cropbox->name);
        emitSceneChanged();
    }

    // ========== Node Transforms ==========

    void SceneManager::setNodeTransform(const std::string& name, const glm::mat4& transform) {
        scene_.setNodeTransform(name, transform);
        emitSceneChanged();
    }

    glm::mat4 SceneManager::getNodeTransform(const std::string& name) const {
        return scene_.getNodeTransform(name);
    }

    void SceneManager::setSelectedNodeTranslation(const glm::vec3& translation) {
        std::string node_name;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (selected_nodes_.empty()) {
                LOG_TRACE("No node selected for translation");
                return;
            }
            node_name = *selected_nodes_.begin();
        }

        if (node_name.empty()) {
            LOG_TRACE("No node selected for translation");
            return;
        }

        // Create translation matrix
        glm::mat4 transform = glm::mat4(1.0f);
        transform[3][0] = translation.x;
        transform[3][1] = translation.y;
        transform[3][2] = translation.z;

        scene_.setNodeTransform(node_name, transform);
        emitSceneChanged();
    }

    glm::vec3 SceneManager::getSelectedNodeTranslation() const {
        std::string node_name;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (selected_nodes_.empty())
                return glm::vec3(0.0f);
            node_name = *selected_nodes_.begin();
        }

        if (node_name.empty()) {
            return glm::vec3(0.0f);
        }

        glm::mat4 transform = scene_.getNodeTransform(node_name);
        return glm::vec3(transform[3][0], transform[3][1], transform[3][2]);
    }

    glm::vec3 SceneManager::getSelectedNodeCentroid() const {
        std::string node_name;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (selected_nodes_.empty())
                return glm::vec3(0.0f);
            node_name = *selected_nodes_.begin();
        }

        const auto* node = scene_.getNode(node_name);
        if (!node || !node->model)
            return glm::vec3(0.0f);
        return node->centroid;
    }

    glm::vec3 SceneManager::getSelectedNodeCenter() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (selected_nodes_.empty())
            return glm::vec3(0.0f);

        const std::string& node_name = *selected_nodes_.begin();
        const auto* const node = scene_.getNode(node_name);
        if (!node)
            return glm::vec3(0.0f);

        return scene_.getNodeBoundsCenter(node->id);
    }

    void SceneManager::setSelectedNodeTransform(const glm::mat4& transform) {
        std::string node_name;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (selected_nodes_.empty()) {
                LOG_TRACE("No node selected for transform");
                return;
            }
            node_name = *selected_nodes_.begin();
        }

        LOG_DEBUG("setSelectedNodeTransform '{}': pos=[{:.2f}, {:.2f}, {:.2f}]",
                  node_name, transform[3][0], transform[3][1], transform[3][2]);
        scene_.setNodeTransform(node_name, transform);
        emitSceneChanged();
    }

    glm::mat4 SceneManager::getSelectedNodeTransform() const {
        std::string node_name;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (selected_nodes_.empty())
                return glm::mat4(1.0f);
            node_name = *selected_nodes_.begin();
        }

        return scene_.getNodeTransform(node_name);
    }

    glm::mat4 SceneManager::getSelectedNodeWorldTransform() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (selected_nodes_.empty())
            return glm::mat4(1.0f);

        const auto* node = scene_.getNode(*selected_nodes_.begin());
        if (!node)
            return glm::mat4(1.0f);

        return scene_.getWorldTransform(node->id);
    }

    glm::vec3 SceneManager::getSelectionCenter() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (selected_nodes_.empty())
            return glm::vec3(0.0f);

        // Single selection - use node's local bounds center
        if (selected_nodes_.size() == 1) {
            const auto* node = scene_.getNode(*selected_nodes_.begin());
            if (!node)
                return glm::vec3(0.0f);
            return scene_.getNodeBoundsCenter(node->id);
        }

        // Multi-selection - compute combined bounds center in local space
        glm::vec3 total_min(std::numeric_limits<float>::max());
        glm::vec3 total_max(std::numeric_limits<float>::lowest());
        bool has_bounds = false;

        for (const auto& name : selected_nodes_) {
            const auto* node = scene_.getNode(name);
            if (!node)
                continue;

            glm::vec3 node_min, node_max;
            if (scene_.getNodeBounds(node->id, node_min, node_max)) {
                total_min = glm::min(total_min, node_min);
                total_max = glm::max(total_max, node_max);
                has_bounds = true;
            }
        }

        return has_bounds ? (total_min + total_max) * 0.5f : glm::vec3(0.0f);
    }

    glm::vec3 SceneManager::getSelectionWorldCenter() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (selected_nodes_.empty())
            return glm::vec3(0.0f);

        glm::vec3 total_min(std::numeric_limits<float>::max());
        glm::vec3 total_max(std::numeric_limits<float>::lowest());
        bool has_bounds = false;

        for (const auto& name : selected_nodes_) {
            const auto* node = scene_.getNode(name);
            if (!node)
                continue;

            glm::vec3 local_min, local_max;
            if (!scene_.getNodeBounds(node->id, local_min, local_max))
                continue;

            const glm::mat4 world_transform = scene_.getWorldTransform(node->id);
            const glm::vec3 corners[8] = {
                {local_min.x, local_min.y, local_min.z},
                {local_max.x, local_min.y, local_min.z},
                {local_min.x, local_max.y, local_min.z},
                {local_max.x, local_max.y, local_min.z},
                {local_min.x, local_min.y, local_max.z},
                {local_max.x, local_min.y, local_max.z},
                {local_min.x, local_max.y, local_max.z},
                {local_max.x, local_max.y, local_max.z}};

            for (const auto& corner : corners) {
                const glm::vec3 world_corner = glm::vec3(world_transform * glm::vec4(corner, 1.0f));
                total_min = glm::min(total_min, world_corner);
                total_max = glm::max(total_max, world_corner);
            }
            has_bounds = true;
        }

        return has_bounds ? (total_min + total_max) * 0.5f : glm::vec3(0.0f);
    }

    // ========== Cropbox Operations ==========

    NodeId SceneManager::getSelectedNodeCropBoxId() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (selected_nodes_.empty())
            return NULL_NODE;

        const auto* node = scene_.getNode(*selected_nodes_.begin());
        if (!node)
            return NULL_NODE;

        // If selected node is a cropbox, return its ID
        if (node->type == NodeType::CROPBOX) {
            return node->id;
        }

        // If selected node is a splat or pointcloud, return its cropbox child
        if (node->type == NodeType::SPLAT || node->type == NodeType::POINTCLOUD) {
            return scene_.getCropBoxForSplat(node->id);
        }

        // For groups, no cropbox
        return NULL_NODE;
    }

    CropBoxData* SceneManager::getSelectedNodeCropBox() {
        const NodeId cropbox_id = getSelectedNodeCropBoxId();
        if (cropbox_id == NULL_NODE)
            return nullptr;
        return scene_.getCropBoxData(cropbox_id);
    }

    const CropBoxData* SceneManager::getSelectedNodeCropBox() const {
        const NodeId cropbox_id = getSelectedNodeCropBoxId();
        if (cropbox_id == NULL_NODE)
            return nullptr;
        return scene_.getCropBoxData(cropbox_id);
    }

    void SceneManager::syncCropBoxToRenderSettings() {
        // Scene graph is single source of truth - just trigger re-render
        if (services().renderingOrNull()) {
            services().renderingOrNull()->markDirty();
        }
    }

    void SceneManager::loadDataset(const std::filesystem::path& path,
                                   const lfs::core::param::TrainingParameters& params) {
        LOG_TIMER("SceneManager::loadDataset");

        try {
            LOG_INFO("Loading dataset: {}", path.string());

            // Setup training parameters
            auto dataset_params = params;
            dataset_params.dataset.data_path = path;

            // Validate dataset BEFORE clearing scene
            auto validation_result = lfs::training::validateDatasetPath(dataset_params);
            if (!validation_result) {
                LOG_ERROR("Dataset validation failed: {}", validation_result.error());
                state::DatasetLoadCompleted{
                    .path = path,
                    .success = false,
                    .error = validation_result.error(),
                    .num_images = 0,
                    .num_points = 0}
                    .emit();
                return;
            }

            // Validation passed - now clear and load
            if (services().trainerOrNull()) {
                services().trainerOrNull()->clearTrainer();
            }
            clear();

            cached_params_ = dataset_params;

            auto load_result = lfs::training::loadTrainingDataIntoScene(dataset_params, scene_);
            if (!load_result) {
                LOG_ERROR("Failed to load training data: {}", load_result.error());
                state::DatasetLoadCompleted{
                    .path = path,
                    .success = false,
                    .error = load_result.error(),
                    .num_images = 0,
                    .num_points = 0}
                    .emit();
                return;
            }

            // Create cropbox for PointCloud node if present
            const auto* pointcloud_node = scene_.getNode("PointCloud");
            if (pointcloud_node && pointcloud_node->type == NodeType::POINTCLOUD) {
                [[maybe_unused]] auto cropbox_id = scene_.getOrCreateCropBoxForSplat(pointcloud_node->id);
            }

            // Create Trainer from Scene
            auto trainer = std::make_unique<lfs::training::Trainer>(scene_);
            trainer->setParams(dataset_params);

            // Pass trainer to manager
            if (services().trainerOrNull()) {
                LOG_DEBUG("Setting trainer in manager");
                services().trainerOrNull()->setScene(&scene_);
                services().trainerOrNull()->setTrainer(std::move(trainer));
            } else {
                LOG_ERROR("No trainer manager available");
                throw std::runtime_error("No trainer manager available");
            }

            // Update content state
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                content_type_ = ContentType::Dataset;
                dataset_path_ = path;
            }

            // Get info from scene
            const auto* training_model = scene_.getTrainingModel();
            const size_t num_gaussians = training_model ? training_model->size() : 0;
            const size_t num_cameras = scene_.getTrainCameras() ? scene_.getTrainCameras()->size() : 0;

            LOG_INFO("Dataset loaded successfully - {} images, {} initial gaussians",
                     num_cameras, num_gaussians);

            state::SceneLoaded{
                .scene = nullptr,
                .path = path,
                .type = state::SceneLoaded::Type::Dataset,
                .num_gaussians = num_gaussians}
                .emit();

            state::DatasetLoadCompleted{
                .path = path,
                .success = true,
                .error = std::nullopt,
                .num_images = num_cameras,
                .num_points = num_gaussians}
                .emit();

            emitSceneChanged();

            // Switch to point cloud rendering mode by default for datasets
            // Re-enabled with debug logging to investigate dimension mismatch
            if (num_gaussians > 0 && services().trainerOrNull() && services().trainerOrNull()->getTrainer()) {
                ui::PointCloudModeChanged{
                    .enabled = true,
                    .voxel_size = 0.03f}
                    .emit();
                LOG_INFO("Switched to point cloud rendering mode for dataset ({} gaussians)", num_gaussians);
            }

        } catch (const std::exception& e) {
            LOG_ERROR("Failed to load dataset: {} (path: {})", e.what(), path.string());

            // Emit failure event instead of throwing
            state::DatasetLoadCompleted{
                .path = path,
                .success = false,
                .error = e.what(),
                .num_images = 0,
                .num_points = 0}
                .emit();
        }
    }

    void SceneManager::loadCheckpointForTraining(const std::filesystem::path& path,
                                                 const lfs::core::param::TrainingParameters& params) {
        LOG_TIMER("SceneManager::loadCheckpointForTraining");

        try {
            // === Phase 1: Validate checkpoint BEFORE clearing scene ===
            const auto header_result = lfs::training::load_checkpoint_header(path);
            if (!header_result) {
                throw std::runtime_error("Failed to load checkpoint header: " + header_result.error());
            }
            const int checkpoint_iteration = header_result->iteration;

            auto params_result = lfs::training::load_checkpoint_params(path);
            if (!params_result) {
                throw std::runtime_error("Failed to load checkpoint params: " + params_result.error());
            }
            auto checkpoint_params = *params_result;

            // CLI path overrides
            if (!params.dataset.data_path.empty()) {
                checkpoint_params.dataset.data_path = params.dataset.data_path;
            }
            if (!params.dataset.output_path.empty()) {
                checkpoint_params.dataset.output_path = params.dataset.output_path;
            }

            if (checkpoint_params.dataset.data_path.empty()) {
                throw std::runtime_error("Checkpoint has no dataset path and none provided");
            }
            if (!std::filesystem::exists(checkpoint_params.dataset.data_path)) {
                throw std::runtime_error("Dataset path does not exist: " +
                                         checkpoint_params.dataset.data_path.string());
            }

            // Validate dataset structure before clearing
            const auto validation_result = lfs::training::validateDatasetPath(checkpoint_params);
            if (!validation_result) {
                throw std::runtime_error("Failed to load training data: " + validation_result.error());
            }

            // === Phase 2: Clear scene (validation passed) ===
            if (services().trainerOrNull()) {
                services().trainerOrNull()->clearTrainer();
            }
            clear();

            cached_params_ = checkpoint_params;

            // === Phase 3: Load data ===
            const auto load_result = lfs::training::loadTrainingDataIntoScene(checkpoint_params, scene_);
            if (!load_result) {
                throw std::runtime_error("Failed to load training data: " + load_result.error());
            }

            // Remove POINTCLOUD node (checkpoint model replaces it)
            for (const auto* node : scene_.getNodes()) {
                if (node->type == lfs::vis::NodeType::POINTCLOUD) {
                    scene_.removeNode(node->name, false);
                    break;
                }
            }

            auto splat_result = lfs::training::load_checkpoint_splat_data(path);
            if (!splat_result) {
                throw std::runtime_error("Failed to load checkpoint SplatData: " + splat_result.error());
            }

            auto splat_data = std::make_unique<lfs::core::SplatData>(std::move(*splat_result));
            const size_t num_gaussians = splat_data->size();
            constexpr const char* MODEL_NAME = "Model";

            scene_.addSplat(MODEL_NAME, std::move(splat_data), lfs::vis::NULL_NODE);
            scene_.setTrainingModelNode(MODEL_NAME);

            // Mark as checkpoint restore for sparsity handling
            checkpoint_params.resume_checkpoint = path;

            auto trainer = std::make_unique<lfs::training::Trainer>(scene_);
            const auto init_result = trainer->initialize(checkpoint_params);
            if (!init_result) {
                throw std::runtime_error("Failed to initialize trainer: " + init_result.error());
            }

            const auto ckpt_load_result = trainer->load_checkpoint(path);
            if (!ckpt_load_result) {
                LOG_WARN("Failed to restore checkpoint state: {}", ckpt_load_result.error());
            }

            if (!services().trainerOrNull()) {
                throw std::runtime_error("No trainer manager available");
            }
            services().trainerOrNull()->setScene(&scene_);
            services().trainerOrNull()->setTrainerFromCheckpoint(std::move(trainer), checkpoint_iteration);

            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                content_type_ = ContentType::Dataset;
                dataset_path_ = checkpoint_params.dataset.data_path;
            }

            // Update current params from checkpoint (session params remain unchanged)
            if (auto* param_mgr = services().paramsOrNull()) {
                param_mgr->setCurrentParams(checkpoint_params.optimization);
            }

            LOG_INFO("Checkpoint loaded: {} gaussians, iteration {}", num_gaussians, checkpoint_iteration);

            state::SceneLoaded{
                .scene = nullptr,
                .path = path,
                .type = state::SceneLoaded::Type::Checkpoint,
                .num_gaussians = num_gaussians,
                .checkpoint_iteration = checkpoint_iteration}
                .emit();

            emitSceneChanged();

            ui::PointCloudModeChanged{.enabled = false, .voxel_size = 0.03f}.emit();
            selectNode(MODEL_NAME);
            ui::FocusTrainingPanel{}.emit();

        } catch (const std::exception& e) {
            LOG_ERROR("Failed to load checkpoint: {}", e.what());
            throw;
        }
    }

    void SceneManager::clear() {
        LOG_DEBUG("Clearing scene");

        // Check if clearing is allowed via state machine
        if (services().trainerOrNull() && content_type_ == ContentType::Dataset) {
            if (!services().trainerOrNull()->canPerform(TrainingAction::ClearScene)) {
                LOG_WARN("Cannot clear scene: {}",
                         services().trainerOrNull()->getActionBlockedReason(TrainingAction::ClearScene));
                return;
            }
            LOG_DEBUG("Clearing trainer before scene");
            // clearTrainer() handles stop, wait, and cleanup internally
            services().trainerOrNull()->clearTrainer();
        }

        scene_.clear();

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            content_type_ = ContentType::Empty;
            splat_paths_.clear();
            dataset_path_.clear();
        }

        state::SceneCleared{}.emit();
        emitSceneChanged();

        LOG_INFO("Scene cleared");
    }

    void SceneManager::switchToEditMode() {
        if (content_type_ != ContentType::Dataset) {
            LOG_WARN("switchToEditMode: not in dataset mode");
            return;
        }

        const std::string model_name = scene_.getTrainingModelNodeName();
        auto* model_node = model_name.empty() ? nullptr : scene_.getMutableNode(model_name);
        if (!model_node || !model_node->model) {
            LOG_WARN("switchToEditMode: no training model");
            return;
        }

        auto splat_data = std::move(model_node->model);
        const size_t num_gaussians = splat_data->size();

        if (services().trainerOrNull()) {
            services().trainerOrNull()->clearTrainer();
        }
        scene_.clear();

        constexpr const char* MODEL_NAME = "Trained Model";
        scene_.addNode(MODEL_NAME, std::move(splat_data));
        selectNode(MODEL_NAME);

        {
            std::lock_guard lock(state_mutex_);
            content_type_ = ContentType::SplatFiles;
            dataset_path_.clear();
            splat_paths_.clear();
        }

        state::SceneLoaded{
            .scene = nullptr,
            .path = {},
            .type = state::SceneLoaded::Type::PLY,
            .num_gaussians = num_gaussians}
            .emit();
        emitSceneChanged();

        LOG_INFO("Switched to Edit Mode: {} gaussians", num_gaussians);
    }

    const lfs::core::SplatData* SceneManager::getModelForRendering() const {
        std::lock_guard<std::mutex> lock(state_mutex_);

        if (content_type_ == ContentType::SplatFiles) {
            return scene_.getCombinedModel();
        } else if (content_type_ == ContentType::Dataset) {
            // For dataset mode, get model from scene directly (Scene owns the model)
            return scene_.getTrainingModel();
        }

        return nullptr;
    }

    SceneRenderState SceneManager::buildRenderState() const {
        std::lock_guard<std::mutex> lock(state_mutex_);

        SceneRenderState state;

        // Get combined model or point cloud (before training starts)
        if (content_type_ == ContentType::SplatFiles) {
            state.combined_model = scene_.getCombinedModel();
        } else if (content_type_ == ContentType::Dataset) {
            // For dataset mode, first try training model (SplatData)
            state.combined_model = scene_.getTrainingModel();
            // If no SplatData yet, try PointCloud (pre-training mode)
            if (!state.combined_model) {
                state.point_cloud = scene_.getVisiblePointCloud();
            }
        }

        // Get transforms and indices
        state.model_transforms = scene_.getVisibleNodeTransforms();
        state.transform_indices = scene_.getTransformIndices();
        state.visible_splat_count = state.model_transforms.size();

        // Get selection mask
        state.selection_mask = scene_.getSelectionMask();
        state.has_selection = scene_.hasSelection();

        // Get selected node info for desaturation
        // Use mask-based approach: when a group is selected, all descendant SPLAT nodes are marked
        state.selected_node_name = selected_nodes_.empty() ? "" : *selected_nodes_.begin();
        state.selected_node_mask = scene_.getSelectedNodeMask(std::vector<std::string>(selected_nodes_.begin(), selected_nodes_.end()));

        // Get cropboxes
        state.cropboxes = scene_.getVisibleCropBoxes();

        // Find selected cropbox index
        if (!selected_nodes_.empty()) {
            const auto* selected = scene_.getNode(*selected_nodes_.begin());
            if (selected) {
                NodeId cropbox_id = NULL_NODE;
                if (selected->type == NodeType::CROPBOX) {
                    cropbox_id = selected->id;
                } else if (selected->type == NodeType::SPLAT) {
                    cropbox_id = scene_.getCropBoxForSplat(selected->id);
                }
                if (cropbox_id != NULL_NODE) {
                    for (size_t i = 0; i < state.cropboxes.size(); ++i) {
                        if (state.cropboxes[i].node_id == cropbox_id) {
                            state.selected_cropbox_index = static_cast<int>(i);
                            break;
                        }
                    }
                }
            }
        }

        return state;
    }

    SceneManager::SceneInfo SceneManager::getSceneInfo() const {
        std::lock_guard<std::mutex> lock(state_mutex_);

        SceneInfo info;

        switch (content_type_) {
        case ContentType::Empty:
            info.source_type = "Empty";
            break;

        case ContentType::SplatFiles:
            info.has_model = scene_.hasNodes();
            info.num_gaussians = scene_.getTotalGaussianCount();
            info.num_nodes = scene_.getNodeCount();
            info.source_type = "Splat";
            if (!splat_paths_.empty()) {
                info.source_path = splat_paths_.rbegin()->second; // get the "last" element of the splat_paths_
                // Determine specific type from extension
                auto ext = info.source_path.extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".sog") {
                    info.source_type = "SOG";
                } else if (ext == ".ply") {
                    info.source_type = "PLY";
                } else if (ext == ".spz") {
                    info.source_type = "SPZ";
                }
            }
            break;

        case ContentType::Dataset:
            // For dataset mode, get info from scene directly (Scene owns the model)
            info.has_model = scene_.hasNodes();
            if (info.has_model) {
                const auto* training_model = scene_.getTrainingModel();
                info.num_gaussians = training_model ? training_model->size() : 0;
            }
            info.num_nodes = scene_.getNodeCount();
            info.source_type = "Dataset";
            info.source_path = dataset_path_;
            break;
        }

        LOG_TRACE("Scene info - type: {}, gaussians: {}, nodes: {}",
                  info.source_type, info.num_gaussians, info.num_nodes);

        return info;
    }

    void SceneManager::emitSceneChanged() {
        state::SceneChanged{}.emit();
    }

    void SceneManager::handleCropActivePly(const lfs::geometry::BoundingBox& crop_box, const bool inverse) {
        std::vector<std::string> splat_node_names;
        std::vector<std::string> pointcloud_node_names;
        bool had_selection = false;

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (!selected_nodes_.empty()) {
                had_selection = true;
                for (const auto& node_name : selected_nodes_) {
                    const auto* selected = scene_.getNode(node_name);
                    if (!selected)
                        continue;

                    if (selected->type == NodeType::SPLAT) {
                        splat_node_names.push_back(node_name);
                    } else if (selected->type == NodeType::POINTCLOUD) {
                        pointcloud_node_names.push_back(node_name);
                    } else if (selected->type == NodeType::CROPBOX) {
                        // Use parent SPLAT/POINTCLOUD
                        const auto* parent = scene_.getNodeById(selected->parent_id);
                        if (parent && parent->type == NodeType::SPLAT) {
                            splat_node_names.push_back(parent->name);
                        } else if (parent && parent->type == NodeType::POINTCLOUD) {
                            pointcloud_node_names.push_back(parent->name);
                        }
                    }
                }
            }
        }

        // Fall back to visible nodes if no selection
        if (splat_node_names.empty() && pointcloud_node_names.empty() && !had_selection) {
            for (const auto* node : scene_.getVisibleNodes()) {
                if (node->type == NodeType::SPLAT) {
                    splat_node_names.push_back(node->name);
                } else if (node->type == NodeType::POINTCLOUD) {
                    pointcloud_node_names.push_back(node->name);
                }
            }
        }

        // Crop point cloud data (GPU-accelerated)
        for (const auto& node_name : pointcloud_node_names) {
            auto* node = scene_.getMutableNode(node_name);
            if (!node || !node->point_cloud)
                continue;

            const NodeId cropbox_id = scene_.getCropBoxForSplat(node->id);
            if (cropbox_id == NULL_NODE)
                continue;

            const auto* cropbox_node = scene_.getNodeById(cropbox_id);
            if (!cropbox_node || !cropbox_node->cropbox)
                continue;

            const auto& cb = *cropbox_node->cropbox;
            const glm::mat4 m = glm::inverse(cropbox_node->local_transform.get());
            const auto& means = node->point_cloud->means;
            const auto& colors = node->point_cloud->colors;
            const size_t num_points = node->point_cloud->size();
            const auto device = means.device();

            // GLM column-major -> row-major for tensor matmul
            const auto transform = lfs::core::Tensor::from_vector({m[0][0], m[1][0], m[2][0], m[3][0],
                                                                   m[0][1], m[1][1], m[2][1], m[3][1],
                                                                   m[0][2], m[1][2], m[2][2], m[3][2],
                                                                   m[0][3], m[1][3], m[2][3], m[3][3]},
                                                                  {4, 4}, device);

            // Transform and filter on GPU
            const auto ones = lfs::core::Tensor::ones({num_points, 1}, device);
            const auto local_pos = transform.mm(means.cat(ones, 1).t()).t();

            const auto x = local_pos.slice(1, 0, 1).squeeze(1);
            const auto y = local_pos.slice(1, 1, 2).squeeze(1);
            const auto z = local_pos.slice(1, 2, 3).squeeze(1);

            auto mask = (x >= cb.min.x) && (x <= cb.max.x) &&
                        (y >= cb.min.y) && (y <= cb.max.y) &&
                        (z >= cb.min.z) && (z <= cb.max.z);
            if (inverse)
                mask = mask.logical_not();

            const auto indices = mask.nonzero().squeeze(1);
            const size_t filtered_count = indices.size(0);

            if (filtered_count > 0 && filtered_count < num_points) {
                node->point_cloud = std::make_shared<lfs::core::PointCloud>(
                    means.index_select(0, indices), colors.index_select(0, indices));
                node->gaussian_count = filtered_count;

                LOG_INFO("Cropped PointCloud '{}': {} -> {} points", node_name, num_points, filtered_count);

                if (auto* cb_mutable = scene_.getMutableNode(cropbox_node->name)) {
                    if (cb_mutable->cropbox)
                        cb_mutable->cropbox->enabled = false;
                }
            }
        }

        if (splat_node_names.empty()) {
            if (!pointcloud_node_names.empty()) {
                emitSceneChanged();
                if (services().renderingOrNull()) {
                    services().renderingOrNull()->markDirty();
                }
            }
            return;
        }

        // Only change content type when cropping SPLAT nodes
        changeContentType(ContentType::SplatFiles);

        for (const auto& node_name : splat_node_names) {
            auto* node = scene_.getMutableNode(node_name);
            if (!node || !node->model) {
                continue;
            }

            try {
                const size_t original_count = node->model->size();
                const size_t original_visible = node->model->visible_count();

                // Capture old deletion mask for undo
                lfs::core::Tensor old_deleted_mask = node->model->has_deleted_mask()
                                                         ? node->model->deleted().clone()
                                                         : lfs::core::Tensor::zeros({original_count}, lfs::core::Device::CUDA, lfs::core::DataType::Bool);

                // Transform crop box to node's local space if node has a transform
                lfs::geometry::BoundingBox local_crop_box = crop_box;
                const glm::mat4 node_world_transform = scene_.getWorldTransform(node->id);
                static const glm::mat4 IDENTITY_MATRIX(1.0f);

                if (node_world_transform != IDENTITY_MATRIX) {
                    // Combine: node_local -> world -> cropbox_local
                    // world2bbox transforms world -> cropbox_local
                    // node_world transforms node_local -> world
                    // So we need: world2bbox * node_world to go node_local -> cropbox_local
                    if (crop_box.hasFullTransform()) {
                        local_crop_box.setworld2BBox(crop_box.getworld2BBoxMat4() * node_world_transform);
                    } else {
                        const lfs::geometry::EuclideanTransform node_to_world(node_world_transform);
                        local_crop_box.setworld2BBox(crop_box.getworld2BBox() * node_to_world);
                    }
                }

                const auto applied_mask = lfs::core::soft_crop_by_cropbox(*node->model, local_crop_box, inverse);
                if (!applied_mask.is_valid()) {
                    continue;
                }

                const size_t new_visible = node->model->visible_count();
                if (new_visible == original_visible) {
                    continue;
                }

                LOG_INFO("Cropped '{}': {} -> {} visible", node_name, original_visible, new_visible);

                if (services().commandsOrNull()) {
                    lfs::core::Tensor new_deleted_mask = node->model->deleted().clone();
                    auto cmd = std::make_unique<command::CropCommand>(
                        node_name, std::move(old_deleted_mask), std::move(new_deleted_mask));
                    services().commandsOrNull()->execute(std::move(cmd));
                }

                state::PLYAdded{
                    .name = node_name,
                    .node_gaussians = new_visible,
                    .total_gaussians = scene_.getTotalGaussianCount(),
                    .is_visible = true,
                    .parent_name = "",
                    .is_group = false,
                    .node_type = 0 // SPLAT
                }
                    .emit();

            } catch (const std::exception& e) {
                LOG_ERROR("Failed to crop '{}': {}", node_name, e.what());
            }
        }

        emitSceneChanged();
    }

    void SceneManager::updatePlyPath(const std::string& ply_name, const std::filesystem::path& ply_path) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        auto it = splat_paths_.find(ply_name);
        if (it != splat_paths_.end()) {
            it->second = ply_path;
        } else {
            LOG_WARN("ply name was not found {}", ply_name);
        }
    }

    size_t SceneManager::applyDeleted() {
        const size_t removed = scene_.applyDeleted();
        if (removed > 0 && services().renderingOrNull()) {
            services().renderingOrNull()->markDirty();
        }
        return removed;
    }

    bool SceneManager::renamePLY(const std::string& old_name, const std::string& new_name) {
        LOG_DEBUG("Renaming '{}' to '{}'", old_name, new_name);

        // Attempt to rename in the scene
        bool success = scene_.renameNode(old_name, new_name);

        if (success && old_name != new_name) {
            // Update the splat_paths_ map to use the new name
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                auto it = splat_paths_.find(old_name);
                if (it != splat_paths_.end()) {
                    auto path = it->second;

                    splat_paths_.erase(it);
                    splat_paths_[new_name] = path;
                }
            }

            emitSceneChanged();

            LOG_INFO("Successfully renamed '{}' to '{}'", old_name, new_name);
        } else if (!success) {
            LOG_WARN("Failed to rename '{}' to '{}' - name may already exist", old_name, new_name);
        }

        return success;
    }
    void SceneManager::handleRenamePly(const cmd::RenamePLY& event) {
        renamePLY(event.old_name, event.new_name);
    }

    void SceneManager::handleReparentNode(const std::string& node_name, const std::string& new_parent_name) {
        auto* node = scene_.getMutableNode(node_name);
        if (!node)
            return;

        std::string old_parent_name;
        if (node->parent_id != NULL_NODE) {
            if (const auto* p = scene_.getNodeById(node->parent_id)) {
                old_parent_name = p->name;
            }
        }

        NodeId parent_id = NULL_NODE;
        if (!new_parent_name.empty()) {
            const auto* parent = scene_.getNode(new_parent_name);
            if (!parent)
                return;
            parent_id = parent->id;
        }

        scene_.reparent(node->id, parent_id);
        state::NodeReparented{.name = node_name, .old_parent = old_parent_name, .new_parent = new_parent_name}.emit();
        emitSceneChanged();
    }

    void SceneManager::handleAddGroup(const std::string& name, const std::string& parent_name) {
        NodeId parent_id = NULL_NODE;
        if (!parent_name.empty()) {
            const auto* parent = scene_.getNode(parent_name);
            if (!parent)
                return;
            parent_id = parent->id;
        }

        std::string unique_name = name;
        for (int i = 1; scene_.getNode(unique_name); ++i) {
            unique_name = std::format("{} {}", name, i);
        }

        scene_.addGroup(unique_name, parent_id);
        state::PLYAdded{
            .name = unique_name,
            .node_gaussians = 0,
            .total_gaussians = scene_.getTotalGaussianCount(),
            .is_visible = true,
            .parent_name = parent_name,
            .is_group = true,
            .node_type = 1 // GROUP
        }
            .emit();
    }

    void SceneManager::handleDuplicateNode(const std::string& name) {
        const auto* src = scene_.getNode(name);
        if (!src)
            return;

        std::string parent_name;
        if (src->parent_id != NULL_NODE) {
            if (const auto* p = scene_.getNodeById(src->parent_id)) {
                parent_name = p->name;
            }
        }

        const std::string new_name = scene_.duplicateNode(name);
        if (new_name.empty())
            return;

        // Emit PLYAdded for duplicated node tree
        std::function<void(const std::string&, const std::string&)> emit_added =
            [&](const std::string& n, const std::string& pn) {
                const auto* node = scene_.getNode(n);
                if (!node)
                    return;

                state::PLYAdded{
                    .name = node->name,
                    .node_gaussians = node->gaussian_count,
                    .total_gaussians = scene_.getTotalGaussianCount(),
                    .is_visible = node->visible,
                    .parent_name = pn,
                    .is_group = node->type == NodeType::GROUP,
                    .node_type = static_cast<int>(node->type)}
                    .emit();

                for (const NodeId cid : node->children) {
                    if (const auto* c = scene_.getNodeById(cid)) {
                        emit_added(c->name, node->name);
                    }
                }
            };

        emit_added(new_name, parent_name);
        emitSceneChanged();
    }

    void SceneManager::handleMergeGroup(const std::string& name) {
        const auto* group = scene_.getNode(name);
        if (!group || group->type != NodeType::GROUP) {
            return;
        }

        std::string parent_name;
        if (group->parent_id != NULL_NODE) {
            if (const auto* p = scene_.getNodeById(group->parent_id)) {
                parent_name = p->name;
            }
        }

        // Check if the group being merged is currently selected
        bool was_selected = false;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            was_selected = (selected_nodes_.count(name) > 0);
            if (was_selected) {
                selected_nodes_.erase(name);
            }
        }

        // Collect children to emit PLYRemoved events
        std::vector<std::string> children_to_remove;
        std::function<void(const SceneNode*)> collect_children = [&](const SceneNode* n) {
            for (const NodeId cid : n->children) {
                if (const auto* c = scene_.getNodeById(cid)) {
                    children_to_remove.push_back(c->name);
                    collect_children(c);
                }
            }
        };
        collect_children(group);

        const std::string merged_name = scene_.mergeGroup(name);
        if (merged_name.empty()) {
            LOG_WARN("Failed to merge group '{}'", name);
            return;
        }

        // Emit PLYRemoved for all original children and the group
        for (const auto& child_name : children_to_remove) {
            state::PLYRemoved{.name = child_name}.emit();
        }
        state::PLYRemoved{.name = name}.emit();

        // Emit PLYAdded for merged node
        const auto* merged = scene_.getNode(merged_name);
        if (merged) {
            state::PLYAdded{
                .name = merged->name,
                .node_gaussians = merged->gaussian_count,
                .total_gaussians = scene_.getTotalGaussianCount(),
                .is_visible = merged->visible,
                .parent_name = parent_name,
                .is_group = false,
                .node_type = static_cast<int>(merged->type)}
                .emit();

            // Re-select the merged node if the group was selected
            if (was_selected) {
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    selected_nodes_.insert(merged_name);
                }
                ui::NodeSelected{
                    .path = merged_name,
                    .type = "PLY",
                    .metadata = {{"name", merged_name}}}
                    .emit();
            }
        }

        emitSceneChanged();
        LOG_INFO("Merged group '{}' -> '{}'", name, merged_name);
    }

    void SceneManager::updateCropBoxToFitScene(const bool use_percentile) {
        if (!services().renderingOrNull())
            return;

        // Find selected cropbox or parent with cropbox child
        const SceneNode* cropbox = nullptr;
        const SceneNode* parent = nullptr;

        for (const auto& name : selected_nodes_) {
            const auto* node = scene_.getNode(name);
            if (!node)
                continue;

            if (node->type == NodeType::CROPBOX) {
                cropbox = node;
                parent = scene_.getNodeById(node->parent_id);
                break;
            }

            if (node->type == NodeType::SPLAT || node->type == NodeType::POINTCLOUD) {
                for (const NodeId child_id : node->children) {
                    if (const auto* child = scene_.getNodeById(child_id);
                        child && child->type == NodeType::CROPBOX) {
                        cropbox = child;
                        parent = node;
                        break;
                    }
                }
                if (cropbox)
                    break;
            }
        }

        if (!cropbox || !parent) {
            LOG_WARN("No cropbox found in selection");
            return;
        }

        glm::vec3 min_bounds, max_bounds;
        bool bounds_valid = false;

        if (parent->type == NodeType::SPLAT && parent->model && parent->model->size() > 0) {
            bounds_valid = lfs::core::compute_bounds(*parent->model, min_bounds, max_bounds, 0.0f, use_percentile);
        } else if (parent->type == NodeType::POINTCLOUD && parent->point_cloud && parent->point_cloud->size() > 0) {
            bounds_valid = lfs::core::compute_bounds(*parent->point_cloud, min_bounds, max_bounds, 0.0f, use_percentile);
        }

        if (!bounds_valid) {
            LOG_WARN("Cannot compute bounds for '{}'", parent->name);
            return;
        }

        const glm::vec3 center = (min_bounds + max_bounds) * 0.5f;
        const glm::vec3 half_size = (max_bounds - min_bounds) * 0.5f;

        if (auto* node = scene_.getMutableNode(cropbox->name); node && node->cropbox) {
            node->cropbox->min = -half_size;
            node->cropbox->max = half_size;
            node->local_transform = glm::translate(glm::mat4(1.0f), center);
            node->transform_dirty = true;
        }

        services().renderingOrNull()->markDirty();

        LOG_INFO("Fit '{}' to '{}': center({:.2f},{:.2f},{:.2f}) size({:.2f},{:.2f},{:.2f})",
                 cropbox->name, parent->name, center.x, center.y, center.z,
                 half_size.x * 2, half_size.y * 2, half_size.z * 2);
    }

    SceneManager::ClipboardEntry::HierarchyNode SceneManager::copyNodeHierarchy(const SceneNode* node) {
        ClipboardEntry::HierarchyNode result;
        result.type = node->type;
        result.local_transform = node->local_transform.get();

        if (node->cropbox) {
            result.cropbox = std::make_unique<CropBoxData>(*node->cropbox);
        }

        for (const NodeId child_id : node->children) {
            if (const auto* child = scene_.getNodeById(child_id)) {
                result.children.push_back(copyNodeHierarchy(child));
            }
        }

        return result;
    }

    void SceneManager::pasteNodeHierarchy(const ClipboardEntry::HierarchyNode& src, const NodeId parent_id) {
        for (const auto& child : src.children) {
            if (child.type == NodeType::CROPBOX && child.cropbox) {
                const NodeId cropbox_id = scene_.getOrCreateCropBoxForSplat(parent_id);
                if (cropbox_id == NULL_NODE)
                    continue;

                const auto* cropbox_info = scene_.getNodeById(cropbox_id);
                if (!cropbox_info)
                    continue;

                auto* cropbox_node = scene_.getMutableNode(cropbox_info->name);
                if (cropbox_node && cropbox_node->cropbox) {
                    *cropbox_node->cropbox = *child.cropbox;
                    cropbox_node->local_transform = child.local_transform;
                    cropbox_node->transform_dirty = true;
                }
            }
        }
    }

    bool SceneManager::copySelectedNodes() {
        static constexpr glm::mat4 IDENTITY{1.0f};

        std::lock_guard<std::mutex> lock(state_mutex_);
        if (selected_nodes_.empty()) {
            clipboard_.clear();
            return false;
        }

        clipboard_.clear();
        clipboard_.reserve(selected_nodes_.size());

        for (const auto& node_name : selected_nodes_) {
            const auto* node = scene_.getNode(node_name);
            if (!node || !node->model || node->model->size() == 0)
                continue;

            const auto& src = *node->model;
            auto cloned = std::make_unique<lfs::core::SplatData>(
                src.get_max_sh_degree(),
                src.means_raw().clone(), src.sh0_raw().clone(), src.shN_raw().clone(),
                src.scaling_raw().clone(), src.rotation_raw().clone(), src.opacity_raw().clone(),
                src.get_scene_scale());
            cloned->set_active_sh_degree(src.get_active_sh_degree());

            ClipboardEntry entry;
            entry.data = std::move(cloned);
            entry.transform = node->local_transform.get();
            entry.hierarchy = copyNodeHierarchy(node);

            clipboard_.push_back(std::move(entry));
        }

        LOG_INFO("Copied {} nodes to clipboard", clipboard_.size());
        return !clipboard_.empty();
    }

    bool SceneManager::copySelectedGaussians() {
        gaussian_clipboard_.reset();

        if (!scene_.hasSelection())
            return false;

        const auto* combined = scene_.getCombinedModel();
        if (!combined || combined->size() == 0)
            return false;

        const auto mask = scene_.getSelectionMask();
        if (!mask || !mask->is_valid())
            return false;

        // Extract selected indices from mask
        const auto mask_cpu = mask->cpu();
        const auto* mask_ptr = mask_cpu.ptr<uint8_t>();
        const size_t n = mask_cpu.size(0);

        std::vector<int> indices_vec;
        indices_vec.reserve(n / 10);
        for (size_t i = 0; i < n; ++i) {
            if (mask_ptr[i] > 0) {
                indices_vec.push_back(static_cast<int>(i));
            }
        }

        if (indices_vec.empty())
            return false;

        const auto indices = lfs::core::Tensor::from_vector(
            indices_vec, {indices_vec.size()}, lfs::core::Device::CUDA);

        const auto& src = *combined;
        lfs::core::Tensor shN_selected = src.shN_raw().is_valid()
                                             ? src.shN_raw().index_select(0, indices).contiguous()
                                             : lfs::core::Tensor{};

        gaussian_clipboard_ = std::make_unique<lfs::core::SplatData>(
            src.get_max_sh_degree(),
            src.means_raw().index_select(0, indices).contiguous(),
            src.sh0_raw().index_select(0, indices).contiguous(),
            std::move(shN_selected),
            src.scaling_raw().index_select(0, indices).contiguous(),
            src.rotation_raw().index_select(0, indices).contiguous(),
            src.opacity_raw().index_select(0, indices).contiguous(),
            src.get_scene_scale());
        gaussian_clipboard_->set_active_sh_degree(src.get_active_sh_degree());

        LOG_INFO("Copied {} Gaussians", indices_vec.size());
        return true;
    }

    std::vector<std::string> SceneManager::pasteGaussians() {
        if (!gaussian_clipboard_ || gaussian_clipboard_->size() == 0)
            return {};

        const auto& src = *gaussian_clipboard_;
        auto data = std::make_unique<lfs::core::SplatData>(
            src.get_max_sh_degree(),
            src.means_raw().clone(), src.sh0_raw().clone(), src.shN_raw().clone(),
            src.scaling_raw().clone(), src.rotation_raw().clone(), src.opacity_raw().clone(),
            src.get_scene_scale());
        data->set_active_sh_degree(src.get_active_sh_degree());

        const std::string name = std::format("Selection_{}", ++clipboard_counter_);
        const size_t count = data->size();
        scene_.addNode(name, std::move(data));

        state::PLYAdded{
            .name = name,
            .node_gaussians = count,
            .total_gaussians = scene_.getTotalGaussianCount(),
            .is_visible = true,
            .parent_name = "",
            .is_group = false,
            .node_type = 0}
            .emit();

        {
            std::lock_guard lock(state_mutex_);
            if (content_type_ == ContentType::Empty) {
                content_type_ = ContentType::SplatFiles;
            }
        }

        scene_.invalidateCache();
        emitSceneChanged();
        if (services().renderingOrNull())
            services().renderingOrNull()->markDirty();

        LOG_INFO("Pasted {} Gaussians as '{}'", count, name);
        return {name};
    }

    bool SceneManager::executeMirror(const lfs::core::MirrorAxis axis) {
        std::vector<Scene::Node*> nodes;
        nodes.reserve(selected_nodes_.size());
        for (const auto& name : selected_nodes_) {
            if (auto* n = scene_.getMutableNode(name); n && n->type == NodeType::SPLAT && n->model) {
                nodes.push_back(n);
            }
        }

        if (nodes.empty()) {
            LOG_WARN("Mirror: no SPLAT nodes selected");
            return false;
        }

        // Cache selection mask count to avoid redundant GPU->CPU syncs
        const auto scene_mask = scene_.getSelectionMask();
        const size_t selection_count =
            (scene_mask && scene_mask->is_valid()) ? static_cast<size_t>(scene_mask->ne(0).sum_scalar()) : 0;
        const bool use_selection = selection_count > 0 && nodes.size() == 1 &&
                                   static_cast<size_t>(scene_mask->size(0)) == nodes[0]->model->size();

        auto composite_cmd = std::make_unique<command::CompositeCommand>();
        size_t total_count = 0;

        for (auto* node : nodes) {
            auto& model = *node->model;
            const size_t count = use_selection ? selection_count : model.size();
            total_count += count;

            auto mask = use_selection
                            ? scene_mask
                            : std::make_shared<lfs::core::Tensor>(lfs::core::Tensor::ones(
                                  {model.size()}, model.means().device(), lfs::core::DataType::UInt8));

            const auto center = lfs::core::compute_selection_center(model, *mask);

            // Snapshot for undo (sh0 excluded - DC component is isotropic)
            auto old_means = std::make_shared<lfs::core::Tensor>(model.means_raw().clone());
            auto old_rotation = std::make_shared<lfs::core::Tensor>(model.rotation_raw().clone());
            auto old_shN =
                model.shN_raw().is_valid() ? std::make_shared<lfs::core::Tensor>(model.shN_raw().clone()) : nullptr;

            lfs::core::mirror_gaussians(model, *mask, axis, center);

            composite_cmd->add(std::make_unique<command::MirrorCommand>(
                this, node->name, axis, center, std::make_shared<lfs::core::Tensor>(mask->clone()),
                std::move(old_means), std::move(old_rotation), std::move(old_shN)));
        }

        if (auto* history = getCommandHistory(); !composite_cmd->empty()) {
            history->execute(std::move(composite_cmd));
        }

        scene_.invalidateCache();
        if (auto* rendering = services().renderingOrNull()) {
            rendering->markDirty();
        }

        static constexpr const char* AXIS_NAMES[] = {"X", "Y", "Z"};
        LOG_INFO("Mirrored {} gaussians ({} nodes) along {} axis", total_count, nodes.size(),
                 AXIS_NAMES[static_cast<int>(axis)]);
        return true;
    }

    std::vector<std::string> SceneManager::pasteNodes() {
        std::vector<std::string> pasted_names;
        if (clipboard_.empty()) {
            return pasted_names;
        }

        pasted_names.reserve(clipboard_.size());

        for (const auto& entry : clipboard_) {
            if (!entry.data || entry.data->size() == 0)
                continue;

            auto paste_data = std::make_unique<lfs::core::SplatData>(
                entry.data->get_max_sh_degree(),
                entry.data->means_raw().clone(), entry.data->sh0_raw().clone(), entry.data->shN_raw().clone(),
                entry.data->scaling_raw().clone(), entry.data->rotation_raw().clone(), entry.data->opacity_raw().clone(),
                entry.data->get_scene_scale());
            paste_data->set_active_sh_degree(entry.data->get_active_sh_degree());

            ++clipboard_counter_;
            const std::string name = std::format("Pasted_{}", clipboard_counter_);
            const size_t count = entry.data->size();
            scene_.addNode(name, std::move(paste_data));

            // Apply original transform
            static constexpr glm::mat4 IDENTITY{1.0f};
            if (entry.transform != IDENTITY) {
                scene_.setNodeTransform(name, entry.transform);
            }

            // Paste node hierarchy (cropbox)
            const auto* splat_node = scene_.getNode(name);
            if (splat_node && entry.hierarchy) {
                pasteNodeHierarchy(*entry.hierarchy, splat_node->id);
            }

            state::PLYAdded{
                .name = name,
                .node_gaussians = count,
                .total_gaussians = scene_.getTotalGaussianCount(),
                .is_visible = true,
                .parent_name = "",
                .is_group = false,
                .node_type = 0}
                .emit();

            // Emit PLYAdded for cropbox
            const auto* pasted_splat = scene_.getNode(name);
            if (pasted_splat) {
                const NodeId cropbox_id = scene_.getCropBoxForSplat(pasted_splat->id);
                if (cropbox_id != NULL_NODE) {
                    if (const auto* cropbox_node = scene_.getNodeById(cropbox_id)) {
                        state::PLYAdded{
                            .name = cropbox_node->name,
                            .node_gaussians = 0,
                            .total_gaussians = scene_.getTotalGaussianCount(),
                            .is_visible = true,
                            .parent_name = name,
                            .is_group = false,
                            .node_type = 2}
                            .emit();
                    }
                }
            }

            pasted_names.push_back(name);
        }

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (content_type_ == ContentType::Empty && !pasted_names.empty()) {
                content_type_ = ContentType::SplatFiles;
            }
        }

        scene_.invalidateCache();
        emitSceneChanged();

        if (services().renderingOrNull()) {
            services().renderingOrNull()->markDirty();
        }

        LOG_DEBUG("Pasted {} nodes", pasted_names.size());
        return pasted_names;
    }

} // namespace lfs::vis