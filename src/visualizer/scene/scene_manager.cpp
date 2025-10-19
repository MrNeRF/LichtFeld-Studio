/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "scene/scene_manager.hpp"
#include "core/logger.hpp"
#include "loader/loader.hpp"
#include "rendering/rendering_manager.hpp"
#include "training/training_manager.hpp"
#include "training_setup.hpp"
#include <stdexcept>

namespace gs {

    SceneManager::SceneManager() {
        setupEventHandlers();
        LOG_DEBUG("SceneManager initialized");
    }

    SceneManager::~SceneManager() = default;

    void SceneManager::setupEventHandlers() {
        using namespace events;

        // Handle PLY commands
        cmd::AddPLY::when([this](const auto& cmd) {
            addSplatFile(cmd.path, cmd.name);
        });

        cmd::RemovePLY::when([this](const auto& cmd) {
            removePLY(cmd.name);
        });

        cmd::SetPLYVisibility::when([this](const auto& cmd) {
            setPLYVisibility(cmd.name, cmd.visible);
        });

        cmd::ClearScene::when([this](const auto&) {
            clear();
        });

        // Handle PLY cycling with proper event emission for UI updates
        cmd::CyclePLY::when([this](const auto&) {
            // Check if rendering manager has split view enabled (in PLY comparison mode)
            if (rendering_manager_) {
                auto settings = rendering_manager_->getSettings();
                if (settings.split_view_mode == visualizer::SplitViewMode::PLYComparison) {
                    // In split mode: advance the offset
                    rendering_manager_->advanceSplitOffset();
                    LOG_DEBUG("Advanced split view offset");
                    return; // Don't cycle visibility when in split view
                }
            }

            // Normal mode: existing cycle code
            if (content_type_ == ContentType::SplatFiles) {
                auto [hidden, shown] = scene_.cycleVisibilityWithNames();

                if (!hidden.empty()) {
                    events::cmd::SetPLYVisibility{.name = hidden, .visible = false}.emit();
                }
                if (!shown.empty()) {
                    events::cmd::SetPLYVisibility{.name = shown, .visible = true}.emit();
                    LOG_DEBUG("Cycled to: {}", shown);
                }

                emitSceneChanged();
            }
        });

        cmd::CropPLY::when([this](const auto& cmd) {
            handleCropActivePly(cmd.crop_box);
        });

        cmd::RenamePLY::when([this](const auto& cmd) {
            handleRenamePly(cmd);
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
            auto loader = gs::loader::Loader::create();
            gs::loader::LoadOptions options{
                .resize_factor = -1,
                .max_width = 3840,
                .images_folder = "images",
                .validate_only = false};

            LOG_TRACE("Loading splat file with loader");
            auto load_result = loader->load(path, options);
            if (!load_result) {
                LOG_ERROR("Failed to load splat file: {}", load_result.error());
                throw std::runtime_error(load_result.error());
            }

            auto* splat_data = std::get_if<std::shared_ptr<gs::SplatData>>(&load_result->data);
            if (!splat_data || !*splat_data) {
                LOG_ERROR("Expected splat file but got different data type from: {}", path.string());
                throw std::runtime_error("Expected splat file but got different data type");
            }

            // Add to scene
            std::string name = path.stem().string();
            size_t gaussian_count = (*splat_data)->size();
            LOG_DEBUG("Adding '{}' to scene with {} gaussians", name, gaussian_count);

            scene_.addNode(name, std::make_unique<SplatData>(std::move(**splat_data)));

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
            auto file_type = (ext == ".sog") ? events::state::SceneLoaded::Type::SOG : events::state::SceneLoaded::Type::PLY;

            // Emit events
            events::state::SceneLoaded{
                .scene = nullptr,
                .path = path,
                .type = file_type,
                .num_gaussians = scene_.getTotalGaussianCount()}
                .emit();

            events::state::PLYAdded{
                .name = name,
                .node_gaussians = gaussian_count,
                .total_gaussians = scene_.getTotalGaussianCount(),
                .is_visible = true}
                .emit();

            emitSceneChanged();

            LOG_INFO("Successfully loaded '{}' with {} gaussians", name, gaussian_count);

        } catch (const std::exception& e) {
            LOG_ERROR("Failed to load splat file: {} (path: {})", e.what(), path.string());
            throw;
        }
    }

    void SceneManager::addSplatFile(const std::filesystem::path& path, const std::string& name_hint,
                                    bool is_visible) {
        LOG_TIMER_TRACE("SceneManager::addSplatFile");

        try {
            // If not in splat mode, switch to it
            if (content_type_ != ContentType::SplatFiles) {
                LOG_DEBUG("Not in splat mode, switching to splat mode and loading");
                loadSplatFile(path);
                return;
            }

            LOG_INFO("Adding splat file to scene: {}", path.string());

            // Load the file
            auto loader = gs::loader::Loader::create();
            gs::loader::LoadOptions options{
                .resize_factor = -1,
                .max_width = 3840,
                .images_folder = "images",
                .validate_only = false};

            LOG_TRACE("Loading splat data");
            auto load_result = loader->load(path, options);
            if (!load_result) {
                LOG_ERROR("Failed to load splat file: {}", load_result.error());
                throw std::runtime_error(load_result.error());
            }

            auto* splat_data = std::get_if<std::shared_ptr<gs::SplatData>>(&load_result->data);
            if (!splat_data || !*splat_data) {
                LOG_ERROR("Expected splat file from: {}", path.string());
                throw std::runtime_error("Expected splat file");
            }

            // Generate unique name
            std::string base_name = name_hint.empty() ? path.stem().string() : name_hint;
            std::string name = base_name;
            int counter = 1;

            while (scene_.getNode(name) != nullptr) {
                name = std::format("{}_{}", base_name, counter++);
                LOG_TRACE("Name '{}' already exists, trying '{}'", base_name, name);
            }

            size_t gaussian_count = (*splat_data)->size();
            LOG_DEBUG("Adding node '{}' with {} gaussians", name, gaussian_count);

            scene_.addNode(name, std::make_unique<SplatData>(std::move(**splat_data)));

            // Update paths
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                splat_paths_[name] = path;
            }

            events::state::PLYAdded{
                .name = name,
                .node_gaussians = gaussian_count,
                .total_gaussians = scene_.getTotalGaussianCount(),
                .is_visible = is_visible}
                .emit();

            emitSceneChanged();

            LOG_INFO("Added '{}' to scene ({} gaussians, visible: {})",
                     name, gaussian_count, is_visible);

        } catch (const std::exception& e) {
            LOG_ERROR("Failed to add splat file: {} (path: {})", e.what(), path.string());
            throw;
        }
    }

    void SceneManager::removePLY(const std::string& name) {
        LOG_DEBUG("Removing '{}' from scene", name);

        scene_.removeNode(name);
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            splat_paths_.erase(name);
        }

        if (lfs_project_) {
            lfs_project_->removePly(name);
        }

        // If no nodes left, transition to empty
        if (scene_.getNodeCount() == 0) {
            std::lock_guard<std::mutex> lock(state_mutex_);
            content_type_ = ContentType::Empty;
            LOG_DEBUG("No nodes remaining, transitioning to empty state");
        }

        events::state::PLYRemoved{.name = name}.emit();
        emitSceneChanged();

        LOG_INFO("Removed '{}' from scene", name);
    }

    void SceneManager::setPLYVisibility(const std::string& name, bool visible) {
        LOG_TRACE("Setting '{}' visibility to: {}", name, visible);
        scene_.setNodeVisibility(name, visible);
        emitSceneChanged();
    }

    void SceneManager::loadDataset(const std::filesystem::path& path,
                                   const param::TrainingParameters& params) {
        LOG_TIMER("SceneManager::loadDataset");

        try {
            LOG_INFO("Loading dataset: {}", path.string());

            // Stop any existing training
            if (trainer_manager_) {
                LOG_DEBUG("Clearing existing trainer");
                trainer_manager_->clearTrainer();
            }

            // Clear scene
            clear();

            // Setup training
            auto dataset_params = params;
            dataset_params.dataset.data_path = path;
            cached_params_ = dataset_params;

            LOG_DEBUG("Setting up training with parameters");
            LOG_TRACE("Dataset path: {}", path.string());
            LOG_TRACE("Iterations: {}", dataset_params.optimization.iterations);

            auto setup_result = gs::training::setupTraining(dataset_params);
            if (!setup_result) {
                LOG_ERROR("Failed to setup training: {}", setup_result.error());
                throw std::runtime_error(setup_result.error());
            }

            // Pass trainer to manager
            if (trainer_manager_) {
                LOG_DEBUG("Setting trainer in manager");
                trainer_manager_->setTrainer(std::move(setup_result->trainer));
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

            // Emit events
            const size_t num_gaussians = trainer_manager_->getTrainer()
                                             ->get_strategy()
                                             .get_model()
                                             .size();

            LOG_INFO("Dataset loaded successfully - {} images, {} initial gaussians",
                     setup_result->dataset->size().value(), num_gaussians);

            events::state::SceneLoaded{
                .scene = nullptr,
                .path = path,
                .type = events::state::SceneLoaded::Type::Dataset,
                .num_gaussians = num_gaussians}
                .emit();

            events::state::DatasetLoadCompleted{
                .path = path,
                .success = true,
                .error = std::nullopt,
                .num_images = setup_result->dataset->size().value(),
                .num_points = num_gaussians}
                .emit();

            emitSceneChanged();

        } catch (const std::exception& e) {
            LOG_ERROR("Failed to load dataset: {} (path: {})", e.what(), path.string());
            throw;
        }
    }

    void SceneManager::clear() {
        LOG_DEBUG("Clearing scene");

        // Stop training if active
        if (trainer_manager_ && content_type_ == ContentType::Dataset) {
            LOG_DEBUG("Stopping training before clearing");
            events::cmd::StopTraining{}.emit();
            trainer_manager_->clearTrainer();
        }

        scene_.clear();

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            content_type_ = ContentType::Empty;
            splat_paths_.clear();
            dataset_path_.clear();
        }

        events::state::SceneCleared{}.emit();
        emitSceneChanged();

        LOG_INFO("Scene cleared");
    }

    const SplatData* SceneManager::getModelForRendering() const {
        std::lock_guard<std::mutex> lock(state_mutex_);

        if (content_type_ == ContentType::SplatFiles) {
            return scene_.getCombinedModel();
        } else if (content_type_ == ContentType::Dataset) {
            if (trainer_manager_ && trainer_manager_->getTrainer()) {
                return &trainer_manager_->getTrainer()->get_strategy().get_model();
            }
        }

        return nullptr;
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
                }
            }
            break;

        case ContentType::Dataset:
            info.has_model = trainer_manager_ && trainer_manager_->getTrainer();
            if (info.has_model) {
                info.num_gaussians = trainer_manager_->getTrainer()
                                         ->get_strategy()
                                         .get_model()
                                         .size();
            }
            info.num_nodes = 1;
            info.source_type = "Dataset";
            info.source_path = dataset_path_;
            break;
        }

        LOG_TRACE("Scene info - type: {}, gaussians: {}, nodes: {}",
                  info.source_type, info.num_gaussians, info.num_nodes);

        return info;
    }

    void SceneManager::emitSceneChanged() {
        events::state::SceneChanged{}.emit();
    }

    void SceneManager::setRenderingManager(visualizer::RenderingManager* rm) {
        rendering_manager_ = rm;
    }

    void SceneManager::handleCropActivePly(const gs::geometry::BoundingBox& crop_box) {
        LOG_DEBUG("Starting crop operation for active PLY files");

        changeContentType(ContentType::SplatFiles);

        // Get all visible nodes from the scene
        auto visible_nodes = scene_.getVisibleNodes();
        if (visible_nodes.empty()) {
            LOG_DEBUG("No visible nodes to crop");
            return;
        }

        LOG_INFO("Cropping {} visible splat files", visible_nodes.size());

        std::map<std::string, std::filesystem::path> crop_name_to_ply_path;

        for (const auto* node : visible_nodes) {
            const std::string& node_name = node->name;
            const SplatData* node_data = node->model.get();

            try {

                LOG_DEBUG("Processing node: {}", node_name);

                // Generate new name with "_crop" suffix
                std::string crop_name = node_name + "_cropped";

                // Check if cropped version already exists
                if (scene_.getNode(crop_name) != nullptr) {
                    LOG_DEBUG("Cropped version '{}' already exists, skipping", crop_name);
                    continue;
                }

                // Perform the crop operation
                auto cropped_splat = node_data->crop_by_cropbox(crop_box);

                // Check if the number of splats changed after cropping
                if (cropped_splat.size() == node_data->size()) {
                    LOG_DEBUG("No splats were removed during cropping for '{}', skipping", node_name);
                    continue;
                }

                LOG_DEBUG("Cropped '{}': {} -> {} splats", node_name, node_data->size(), cropped_splat.size());

                // Find the original file path for this node
                std::filesystem::path original_path;
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    // accessing splat_paths_ is critical code
                    if (!splat_paths_.empty()) {
                        original_path = splat_paths_[node_name];
                    }
                }

                if (original_path.empty()) {
                    LOG_ERROR("Could not determine original path for node '{}'", node_name);
                    continue;
                }

                // Create the new file path next to the original
                std::filesystem::path crop_path = original_path.parent_path() / (crop_name + original_path.extension().string());

                LOG_DEBUG("Saving cropped splat to: {}", crop_path.string());

                // Save the cropped splat data
                // We need to save as PLY format - using iteration 0 as a placeholder
                std::filesystem::path temp_dir = crop_path.parent_path();
                cropped_splat.save_ply(temp_dir, 0, true, crop_name); // join_threads = true for synchronous save

                std::filesystem::path actual_saved_path = temp_dir / (crop_name + ".ply");

                if (std::filesystem::exists(actual_saved_path)) {
                    crop_name_to_ply_path[crop_name] = actual_saved_path;
                    LOG_INFO("Successfully saved cropped splat: {}", actual_saved_path.string());
                } else {
                    LOG_ERROR("Failed to save cropped splat file: {}", actual_saved_path.string());
                }

            } catch (const std::exception& e) {
                LOG_ERROR("Failed to crop node '{}': {}", node_name, e.what());
                continue;
            }
        }

        // Add all successfully cropped files to the scene
        for (const auto& [crop_name, crop_path] : crop_name_to_ply_path) {
            try {
                addSplatFile(crop_path, crop_name, true); // Use default name, make visible
                if (lfs_project_) {
                    if (!lfs_project_->addPly(false, crop_path, -1, crop_name)) {
                        LOG_ERROR("Failed to add cropped ply '{}' to project", crop_name);
                    }
                }
            } catch (const std::exception& e) {
                LOG_ERROR("Failed to add cropped file '{}' to scene: {}", crop_path.string(), e.what());
            }
        }

        if (!crop_name_to_ply_path.empty()) {
            LOG_INFO("Crop operation completed. Added {} new cropped splat files", crop_name_to_ply_path.size());
        } else {
            LOG_INFO("Crop operation completed. No new files were created");
        }
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
                    std::filesystem::path new_ply_path;
                    // chaning ply name and path
                    std::filesystem::path new_path = path;
                    // resposibility of file systems changes should be on the project
                    if (lfs_project_) {
                        auto parent = path.parent_path();
                        auto extension = path.extension();
                        new_path = parent / (new_name + extension.string());
                        if (!lfs_project_->updatePlyPath(old_name, new_path)) {
                            // os rename failed - reducing to old path
                            new_path = path;
                        }
                        lfs_project_->renamePly(old_name, new_name);
                    }

                    splat_paths_[new_name] = new_path;
                }
            }

            emitSceneChanged();

            LOG_INFO("Successfully renamed '{}' to '{}'", old_name, new_name);
        } else if (!success) {
            LOG_WARN("Failed to rename '{}' to '{}' - name may already exist", old_name, new_name);
        }

        return success;
    }
    void SceneManager::handleRenamePly(const events::cmd::RenamePLY& event) {
        renamePLY(event.old_name, event.new_name);
    }
} // namespace gs