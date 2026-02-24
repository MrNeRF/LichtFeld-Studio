/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "scene/scene_manager.hpp"
#include "core/checkpoint_format.hpp"
#include "core/editor_context.hpp"
#include "core/logger.hpp"
#include "core/mesh_data.hpp"
#include "core/parameter_manager.hpp"
#include "core/path_utils.hpp"
#include "core/services.hpp"
#include "core/splat_data_transform.hpp"
#include "geometry/bounding_box.hpp"
#include "geometry/euclidean_transform.hpp"
#include "io/cache_image_loader.hpp"
#include "io/formats/colmap.hpp"
#include "io/loader.hpp"
#include "operation/undo_entry.hpp"
#include "operation/undo_history.hpp"
#include "python/python_runtime.hpp"
#include "rendering/rendering_manager.hpp"
#include "training/checkpoint.hpp"
#include "training/components/ppisp.hpp"
#include "training/components/ppisp_controller.hpp"
#include "training/components/ppisp_file.hpp"
#include "training/trainer.hpp"
#include "training/training_manager.hpp"
#include "training/training_setup.hpp"
#include <algorithm>
#include <format>
#include <glm/gtc/quaternion.hpp>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>

namespace lfs::vis {

    namespace {
        // Voxel size for point cloud rendering (in scene units)
        constexpr float DEFAULT_VOXEL_SIZE = 0.01f;
    } // namespace

    using namespace lfs::core::events;

    SceneManager::SceneManager() {
        setupEventHandlers();
        python::set_application_scene(&scene_);
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

        cmd::ImportColmapCameras::when([this](const auto& cmd) {
            loadColmapCamerasOnly(cmd.sparse_path);
        });

        cmd::PrepareTrainingFromScene::when([this](const auto&) {
            prepareTrainingFromScene();
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
            }
        });

        cmd::CropPLY::when([this](const auto& cmd) {
            handleCropActivePly(cmd.crop_box, cmd.inverse);
        });

        cmd::CropPLYEllipsoid::when([this](const auto& cmd) {
            handleCropByEllipsoid(cmd.world_transform, cmd.radii, cmd.inverse);
        });

        cmd::FitCropBoxToScene::when([this](const auto& cmd) {
            updateCropBoxToFitScene(cmd.use_percentile);
        });

        cmd::FitEllipsoidToScene::when([this](const auto& cmd) {
            updateEllipsoidToFitScene(cmd.use_percentile);
        });

        cmd::AddCropBox::when([this](const auto& cmd) {
            handleAddCropBox(cmd.node_name);
        });

        cmd::AddCropEllipsoid::when([this](const auto& cmd) {
            handleAddCropEllipsoid(cmd.node_name);
        });

        cmd::ResetCropBox::when([this](const auto&) {
            handleResetCropBox();
        });

        cmd::ResetEllipsoid::when([this](const auto&) {
            handleResetEllipsoid();
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
            if (services().trainerOrNull() && services().trainerOrNull()->isRunning()) {
                LOG_INFO("Selection blocked while training is active");
                return;
            }

            if (event.type == "PLY" || event.type == "Group" || event.type == "Dataset" ||
                event.type == "PointCloud" || event.type == "CameraGroup" || event.type == "Camera") {
                const core::NodeId id = scene_.getNodeIdByName(event.path);
                if (id != core::NULL_NODE)
                    selection_.selectNode(id);
                syncCropBoxToRenderSettings();
            }
        });

        // Handle node deselection (but not during training)
        ui::NodeDeselected::when([this](const auto&) {
            if (services().trainerOrNull() && services().trainerOrNull()->isRunning()) {
                LOG_INFO("Selection blocked while training is active");
                return;
            }
            selection_.clearNodeSelection();
        });

        // Gaussian-level selection operations
        cmd::DeleteSelected::when([this](const auto&) { deleteSelectedGaussians(); });
        cmd::InvertSelection::when([this](const auto&) { invertSelection(); });
        cmd::DeselectAll::when([this](const auto&) { deselectAllGaussians(); });
        cmd::SelectAll::when([this](const auto&) { selectAllGaussians(); });
        cmd::CopySelection::when([this](const auto&) { copySelectionToClipboard(); });
        cmd::PasteSelection::when([this](const auto&) { pasteSelectionFromClipboard(); });
        cmd::SelectRect::when([this](const auto& e) { selectRect(e.x0, e.y0, e.x1, e.y1, e.mode); });
        cmd::ApplySelectionMask::when([this](const auto& e) { applySelectionMask(e.mask); });

        state::SelectionChanged::when([](const auto& event) {
            python::update_selection(event.has_selection, event.count);
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
            LOG_INFO("Loading splat file: {}", lfs::core::path_to_utf8(path));

            core::Scene::Transaction txn(scene_);

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

            std::string name = lfs::core::path_to_utf8(path.stem());

            auto ext = path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            state::SceneLoaded::Type file_type = state::SceneLoaded::Type::PLY;
            if (ext == ".sog") {
                file_type = state::SceneLoaded::Type::SOG;
            } else if (ext == ".spz") {
                file_type = state::SceneLoaded::Type::SPZ;
            }

            auto* mesh_data = std::get_if<std::shared_ptr<lfs::core::MeshData>>(&load_result->data);
            if (mesh_data && *mesh_data) {
                LOG_INFO("Adding mesh '{}' ({} vertices, {} faces)", name,
                         (*mesh_data)->vertex_count(), (*mesh_data)->face_count());
                scene_.addMesh(name, *mesh_data);

                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    content_type_ = ContentType::SplatFiles;
                    splat_paths_.clear();
                }

                state::SceneLoaded{
                    .scene = nullptr,
                    .path = path,
                    .type = file_type,
                    .num_gaussians = 0}
                    .emit();

                python::set_application_scene(&scene_);

                state::PLYAdded{
                    .name = name,
                    .node_gaussians = 0,
                    .total_gaussians = scene_.getTotalGaussianCount(),
                    .is_visible = true,
                    .parent_name = "",
                    .is_group = false,
                    .node_type = static_cast<int>(core::NodeType::MESH)}
                    .emit();

                selectNode(name);

                LOG_INFO("Loaded mesh '{}'", name);
            } else {
                auto* splat_data = std::get_if<std::shared_ptr<lfs::core::SplatData>>(&load_result->data);
                if (!splat_data || !*splat_data) {
                    LOG_ERROR("Expected splat/mesh file but got different data type from: {}", lfs::core::path_to_utf8(path));
                    throw std::runtime_error("Expected splat/mesh file but got different data type");
                }

                const size_t gaussian_count = (*splat_data)->size();
                LOG_DEBUG("Adding '{}' to scene with {} gaussians", name, gaussian_count);

                scene_.addNode(name, std::make_unique<lfs::core::SplatData>(std::move(**splat_data)));

                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    content_type_ = ContentType::SplatFiles;
                    splat_paths_.clear();
                    splat_paths_[name] = path;
                }

                state::SceneLoaded{
                    .scene = nullptr,
                    .path = path,
                    .type = file_type,
                    .num_gaussians = scene_.getTotalGaussianCount()}
                    .emit();

                python::set_application_scene(&scene_);

                state::PLYAdded{
                    .name = name,
                    .node_gaussians = gaussian_count,
                    .total_gaussians = scene_.getTotalGaussianCount(),
                    .is_visible = true,
                    .parent_name = "",
                    .is_group = false,
                    .node_type = 0}
                    .emit();

                const auto* splat_for_cropbox = scene_.getNode(name);
                if (splat_for_cropbox) {
                    const core::NodeId cropbox_id = scene_.getCropBoxForSplat(splat_for_cropbox->id);
                    if (cropbox_id != core::NULL_NODE) {
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
                                .node_type = 2}
                                .emit();
                        }
                    }
                }

                updateCropBoxToFitScene(true);
                selectNode(name);

                // Check for companion PPISP file
                auto ppisp_path = lfs::training::find_ppisp_companion(path);
                if (!ppisp_path.empty()) {
                    LOG_INFO("Found PPISP companion file: {}", lfs::core::path_to_utf8(ppisp_path));
                    loadPPISPCompanion(ppisp_path);
                }

                LOG_INFO("Loaded '{}' with {} gaussians", name, gaussian_count);
            }

        } catch (const std::exception& e) {
            LOG_ERROR("Failed to load splat file: {} (path: {})", e.what(), lfs::core::path_to_utf8(path));
            throw;
        }
    }

    void SceneManager::loadPPISPCompanion(const std::filesystem::path& ppisp_path) {
        try {
            // Read header to get dimensions
            std::ifstream file;
            if (!lfs::core::open_file_for_read(ppisp_path, std::ios::binary, file)) {
                LOG_ERROR("Failed to open PPISP file: {}", lfs::core::path_to_utf8(ppisp_path));
                return;
            }

            lfs::training::PPISPFileHeader header{};
            file.read(reinterpret_cast<char*>(&header), sizeof(header));
            file.close();

            if (header.magic != lfs::training::PPISP_FILE_MAGIC) {
                LOG_ERROR("Invalid PPISP file: wrong magic");
                return;
            }

            // Create PPISP for inference (total_iterations=1 since we won't be training)
            // deserialize_inference will set up internal maps from the file
            auto ppisp = std::make_unique<lfs::training::PPISP>(1);

            // Create controller pool if present in file
            std::unique_ptr<lfs::training::PPISPControllerPool> controller_pool;
            if (lfs::training::has_flag(header.flags, lfs::training::PPISPFileFlags::HAS_CONTROLLER)) {
                controller_pool = std::make_unique<lfs::training::PPISPControllerPool>(
                    static_cast<int>(header.num_cameras), 1);
            }

            // Load the actual data
            auto result = lfs::training::load_ppisp_file(ppisp_path, *ppisp, controller_pool.get());
            if (!result) {
                LOG_ERROR("Failed to load PPISP file: {}", result.error());
                return;
            }

            // Allocate CNN buffers for controller if present
            if (controller_pool) {
                // Use a reasonable default size for viewport rendering
                // Buffers will be reallocated if larger images are needed
                constexpr size_t DEFAULT_MAX_H = 1080;
                constexpr size_t DEFAULT_MAX_W = 1920;
                controller_pool->allocate_buffers(DEFAULT_MAX_H, DEFAULT_MAX_W);
            }

            const bool has_controller = (controller_pool != nullptr);
            setAppearanceModel(std::move(ppisp), std::move(controller_pool));
            ui::AppearanceModelLoaded{.has_controller = has_controller}.emit();

        } catch (const std::exception& e) {
            LOG_ERROR("Failed to load PPISP companion: {}", e.what());
        }
    }

    std::string SceneManager::addSplatFile(const std::filesystem::path& path, const std::string& name_hint,
                                           const bool is_visible) {
        LOG_TIMER_TRACE("SceneManager::addSplatFile");

        try {
            if (content_type_ != ContentType::SplatFiles) {
                loadSplatFile(path);
                return lfs::core::path_to_utf8(path.stem());
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

            const std::string base_name = name_hint.empty() ? lfs::core::path_to_utf8(path.stem()) : name_hint;
            std::string name = base_name;
            int counter = 1;
            while (scene_.getNode(name) != nullptr) {
                name = std::format("{}_{}", base_name, counter++);
            }

            auto* mesh_data = std::get_if<std::shared_ptr<lfs::core::MeshData>>(&load_result->data);
            if (mesh_data && *mesh_data) {
                scene_.addMesh(name, *mesh_data);

                state::PLYAdded{
                    .name = name,
                    .node_gaussians = 0,
                    .total_gaussians = scene_.getTotalGaussianCount(),
                    .is_visible = is_visible,
                    .parent_name = "",
                    .is_group = false,
                    .node_type = static_cast<int>(core::NodeType::MESH)}
                    .emit();

                selectNode(name);

                LOG_INFO("Added mesh '{}' ({} vertices, {} faces)", name,
                         (*mesh_data)->vertex_count(), (*mesh_data)->face_count());
                return name;
            }

            auto* splat_data = std::get_if<std::shared_ptr<lfs::core::SplatData>>(&load_result->data);
            if (!splat_data || !*splat_data) {
                throw std::runtime_error("Expected splat or mesh file");
            }

            const size_t gaussian_count = (*splat_data)->size();
            scene_.addNode(name, std::make_unique<lfs::core::SplatData>(std::move(**splat_data)));

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

            selectNode(name);

            auto ppisp_path = lfs::training::find_ppisp_companion(path);
            if (!ppisp_path.empty()) {
                LOG_INFO("Found PPISP companion file: {}", lfs::core::path_to_utf8(ppisp_path));
                loadPPISPCompanion(ppisp_path);
            }

            LOG_INFO("Added '{}' ({} gaussians)", name, gaussian_count);
            return name;

        } catch (const std::exception& e) {
            LOG_ERROR("Failed to add splat file: {} (path: {})", e.what(), lfs::core::path_to_utf8(path));
            throw;
        }
    }

    size_t SceneManager::consolidateNodeModels() {
        return scene_.consolidateNodeModels();
    }

    void SceneManager::removePLY(const std::string& name, const bool keep_children) {
        const auto& training_name = scene_.getTrainingModelNodeName();

        // Check if node is or contains training model
        const auto isTrainingNode = [&]() -> bool {
            if (training_name.empty())
                return false;
            if (name == training_name)
                return true;
            for (const auto* n = scene_.getNode(training_name); n && n->parent_id != core::NULL_NODE;) {
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
            if (node->parent_id != core::NULL_NODE) {
                if (const auto* p = scene_.getNodeById(node->parent_id)) {
                    parent_name = p->name;
                }
            }
        }

        // Collect all descendant IDs before removal (they'll be gone after removeNode)
        const core::NodeId removed_id = scene_.getNodeIdByName(name);
        std::vector<core::NodeId> ids_to_deselect;
        if (removed_id != core::NULL_NODE && !keep_children) {
            std::function<void(core::NodeId)> collect = [&](core::NodeId id) {
                ids_to_deselect.push_back(id);
                if (const auto* node = scene_.getNodeById(id)) {
                    for (core::NodeId child_id : node->children)
                        collect(child_id);
                }
            };
            collect(removed_id);
        } else if (removed_id != core::NULL_NODE) {
            ids_to_deselect.push_back(removed_id);
        }

        scene_.removeNode(name, keep_children);
        {
            std::lock_guard lock(state_mutex_);
            splat_paths_.erase(name);
        }
        for (core::NodeId id : ids_to_deselect)
            selection_.removeFromSelection(id);
        if (!ids_to_deselect.empty())
            selection_.invalidateNodeMask();

        if (scene_.getNodeCount() == 0) {
            std::lock_guard lock(state_mutex_);
            content_type_ = ContentType::Empty;
            dataset_path_.clear();
        }

        state::PLYRemoved{.name = name, .children_kept = keep_children, .parent_of_removed = parent_name}.emit();
    }

    void SceneManager::setPLYVisibility(const std::string& name, const bool visible) {
        const auto* node = scene_.getNode(name);
        scene_.setNodeVisibility(name, visible);

        if (visible)
            syncCropToolRenderSettings(node);
    }

    // ========== Node Selection ==========

    void SceneManager::selectNode(const std::string& name) {
        const core::NodeId id = scene_.getNodeIdByName(name);
        if (id == core::NULL_NODE)
            return;

        selection_.selectNode(id);

        const auto* node = scene_.getNodeById(id);
        assert(node);
        syncCropToolRenderSettings(node);
        python::invalidate_poll_caches(1);

        ui::NodeSelected{
            .path = name,
            .type = "PLY",
            .metadata = {
                {"name", name},
                {"gaussians", std::to_string(node->model ? node->model->size() : 0)},
                {"visible", node->visible ? "true" : "false"}}}
            .emit();
    }

    void SceneManager::selectNodes(const std::vector<std::string>& names) {
        std::vector<core::NodeId> ids;
        ids.reserve(names.size());
        for (const auto& name : names) {
            const core::NodeId id = scene_.getNodeIdByName(name);
            if (id != core::NULL_NODE)
                ids.push_back(id);
        }
        selection_.selectNodes(ids);
        python::invalidate_poll_caches(1);
        if (services().renderingOrNull())
            services().renderingOrNull()->triggerSelectionFlash();
    }

    void SceneManager::addToSelection(const std::string& name) {
        const core::NodeId id = scene_.getNodeIdByName(name);
        if (id == core::NULL_NODE)
            return;
        selection_.addToSelection(id);
        python::invalidate_poll_caches(1);
        if (services().renderingOrNull())
            services().renderingOrNull()->triggerSelectionFlash();
    }

    void SceneManager::clearSelection() {
        selection_.clearNodeSelection();
        python::invalidate_poll_caches(1);
        if (auto* rm = services().renderingOrNull())
            rm->markDirty();
        LOG_TRACE("Cleared node selection");
    }

    std::string SceneManager::getSelectedNodeName() const {
        std::shared_lock lock(selection_.mutex());
        const auto& ids = selection_.selectedNodeIds();
        if (ids.empty())
            return "";
        const auto* node = scene_.getNodeById(*ids.begin());
        return node ? node->name : "";
    }

    std::vector<std::string> SceneManager::getSelectedNodeNames() const {
        std::shared_lock lock(selection_.mutex());
        const auto& ids = selection_.selectedNodeIds();
        std::vector<std::string> names;
        names.reserve(ids.size());
        for (const auto id : ids) {
            const auto* node = scene_.getNodeById(id);
            if (node)
                names.push_back(node->name);
        }
        return names;
    }

    bool SceneManager::hasSelectedNode() const {
        std::shared_lock lock(selection_.mutex());
        const auto& ids = selection_.selectedNodeIds();
        for (const auto id : ids) {
            if (scene_.getNodeById(id) != nullptr)
                return true;
        }
        return false;
    }

    core::NodeType SceneManager::getSelectedNodeType() const {
        std::shared_lock lock(selection_.mutex());
        const auto& ids = selection_.selectedNodeIds();
        if (ids.empty())
            return core::NodeType::SPLAT;
        const auto* node = scene_.getNodeById(*ids.begin());
        return node ? node->type : core::NodeType::SPLAT;
    }

    int SceneManager::getSelectedNodeIndex() const {
        std::shared_lock lock(selection_.mutex());
        const auto& ids = selection_.selectedNodeIds();
        if (ids.empty())
            return -1;
        return scene_.getVisibleNodeIndex(*ids.begin());
    }

    std::vector<bool> SceneManager::getSelectedNodeMask() const {
        return selection_.getNodeMask(scene_);
    }

    int SceneManager::getSelectedCameraUid() const {
        std::shared_lock lock(selection_.mutex());
        const auto& ids = selection_.selectedNodeIds();
        if (ids.size() != 1)
            return -1;
        const auto* node = scene_.getNodeById(*ids.begin());
        if (node && node->type == core::NodeType::CAMERA)
            return node->camera_uid;
        return -1;
    }

    namespace {
        constexpr size_t MAX_MESH_CPU_CACHE_ENTRIES = 64;

        struct CachedMeshCpu {
            uint32_t generation = 0;
            core::Tensor verts_cpu;
            core::Tensor idx_cpu;
            glm::vec3 aabb_min{0.0f};
            glm::vec3 aabb_max{0.0f};
        };

        std::unordered_map<const core::MeshData*, CachedMeshCpu> g_mesh_cpu_cache;

        // Möller-Trumbore ray-triangle intersection, returns distance or -1
        float rayTriangleIntersect(const glm::vec3& origin, const glm::vec3& dir,
                                   const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2) {
            constexpr float EPS = 1e-7f;
            const glm::vec3 e1 = v1 - v0;
            const glm::vec3 e2 = v2 - v0;
            const glm::vec3 h = glm::cross(dir, e2);
            const float a = glm::dot(e1, h);
            if (a > -EPS && a < EPS)
                return -1.0f;

            const float f = 1.0f / a;
            const glm::vec3 s = origin - v0;
            const float u = f * glm::dot(s, h);
            if (u < 0.0f || u > 1.0f)
                return -1.0f;

            const glm::vec3 q = glm::cross(s, e1);
            const float v = f * glm::dot(dir, q);
            if (v < 0.0f || u + v > 1.0f)
                return -1.0f;

            const float t = f * glm::dot(e2, q);
            return t > EPS ? t : -1.0f;
        }

        bool rayAABBIntersect(const glm::vec3& origin, const glm::vec3& dir,
                              const glm::vec3& aabb_min, const glm::vec3& aabb_max,
                              float& t_hit_out) {
            const glm::vec3 inv_dir = 1.0f / dir;
            const glm::vec3 t1 = (aabb_min - origin) * inv_dir;
            const glm::vec3 t2 = (aabb_max - origin) * inv_dir;
            const glm::vec3 t_min_v = glm::min(t1, t2);
            const glm::vec3 t_max_v = glm::max(t1, t2);
            const float t_enter = std::max({t_min_v.x, t_min_v.y, t_min_v.z});
            const float t_exit = std::min({t_max_v.x, t_max_v.y, t_max_v.z});
            if (t_enter > t_exit || t_exit < 0.0f)
                return false;
            t_hit_out = t_enter >= 0.0f ? t_enter : t_exit;
            return true;
        }

        struct CpuMeshAccessor {
            core::Tensor verts_cpu;
            core::Tensor idx_cpu;
            glm::vec3 aabb_min{0.0f};
            glm::vec3 aabb_max{0.0f};

            static std::optional<CpuMeshAccessor> from(const core::MeshData& mesh) {
                if (!mesh.vertices.is_valid() || mesh.vertex_count() == 0)
                    return std::nullopt;

                auto it = g_mesh_cpu_cache.find(&mesh);
                if (it != g_mesh_cpu_cache.end() && it->second.generation == mesh.generation()) {
                    CpuMeshAccessor a;
                    a.verts_cpu = it->second.verts_cpu;
                    a.idx_cpu = it->second.idx_cpu;
                    a.aabb_min = it->second.aabb_min;
                    a.aabb_max = it->second.aabb_max;
                    return a;
                }

                if (g_mesh_cpu_cache.size() >= MAX_MESH_CPU_CACHE_ENTRIES)
                    g_mesh_cpu_cache.clear();

                CpuMeshAccessor a;
                a.verts_cpu = mesh.vertices.to(core::Device::CPU).contiguous();
                if (mesh.indices.is_valid() && mesh.face_count() > 0)
                    a.idx_cpu = mesh.indices.to(core::Device::CPU).contiguous();

                const int64_t nv = a.verts_cpu.size(0);
                a.aabb_min = a.aabb_max = a.vertex(0);
                for (int64_t i = 1; i < nv; ++i) {
                    const glm::vec3 v = a.vertex(i);
                    a.aabb_min = glm::min(a.aabb_min, v);
                    a.aabb_max = glm::max(a.aabb_max, v);
                }

                auto& entry = g_mesh_cpu_cache[&mesh];
                entry.generation = mesh.generation();
                entry.verts_cpu = a.verts_cpu;
                entry.idx_cpu = a.idx_cpu;
                entry.aabb_min = a.aabb_min;
                entry.aabb_max = a.aabb_max;
                return a;
            }

            glm::vec3 vertex(int64_t i) const {
                assert(i >= 0 && i < verts_cpu.size(0));
                const float* p = verts_cpu.ptr<float>() + i * 3;
                return {p[0], p[1], p[2]};
            }

            void getBounds(glm::vec3& out_min, glm::vec3& out_max) const {
                out_min = aabb_min;
                out_max = aabb_max;
            }

            float rayIntersect(const glm::vec3& origin, const glm::vec3& dir) {
                if (!idx_cpu.is_valid())
                    return -1.0f;
                auto va = verts_cpu.accessor<float, 2>();
                auto ia = idx_cpu.accessor<int32_t, 2>();
                const int64_t nf = idx_cpu.size(0);
                float closest = std::numeric_limits<float>::max();
                for (int64_t f = 0; f < nf; ++f) {
                    const glm::vec3 v0(va(ia(f, 0), 0), va(ia(f, 0), 1), va(ia(f, 0), 2));
                    const glm::vec3 v1(va(ia(f, 1), 0), va(ia(f, 1), 1), va(ia(f, 1), 2));
                    const glm::vec3 v2(va(ia(f, 2), 0), va(ia(f, 2), 1), va(ia(f, 2), 2));
                    const float t = rayTriangleIntersect(origin, dir, v0, v1, v2);
                    if (t > 0.0f && t < closest)
                        closest = t;
                }
                return closest < std::numeric_limits<float>::max() ? closest : -1.0f;
            }
        };
    } // namespace

    std::string SceneManager::pickNodeByRay(const glm::vec3& ray_origin, const glm::vec3& ray_dir) const {
        float closest_world_dist = std::numeric_limits<float>::max();
        std::string closest_name;

        for (const auto* node : scene_.getNodes()) {
            if (node->type != core::NodeType::SPLAT && node->type != core::NodeType::MESH)
                continue;
            if (!scene_.isNodeEffectivelyVisible(node->id))
                continue;

            const glm::mat4 local_to_world = scene_.getWorldTransform(node->id);
            const glm::mat4 world_to_local = glm::inverse(local_to_world);
            const glm::vec3 local_origin = glm::vec3(world_to_local * glm::vec4(ray_origin, 1.0f));
            const glm::vec3 local_dir = glm::vec3(world_to_local * glm::vec4(ray_dir, 0.0f));

            auto toWorldDist = [&](float local_t) {
                const glm::vec3 local_hit = local_origin + local_t * local_dir;
                const glm::vec3 world_hit = glm::vec3(local_to_world * glm::vec4(local_hit, 1.0f));
                return glm::length(world_hit - ray_origin);
            };

            if (node->type == core::NodeType::MESH && node->mesh) {
                auto accessor = CpuMeshAccessor::from(*node->mesh);
                if (!accessor)
                    continue;

                glm::vec3 aabb_min, aabb_max;
                accessor->getBounds(aabb_min, aabb_max);

                float aabb_t;
                if (!rayAABBIntersect(local_origin, local_dir, aabb_min, aabb_max, aabb_t))
                    continue;

                const float t_hit = accessor->rayIntersect(local_origin, local_dir);
                if (t_hit > 0.0f) {
                    const float world_dist = toWorldDist(t_hit);
                    if (world_dist < closest_world_dist) {
                        closest_world_dist = world_dist;
                        closest_name = node->name;
                    }
                }
            } else {
                glm::vec3 local_min, local_max;
                if (!scene_.getNodeBounds(node->id, local_min, local_max))
                    continue;

                float t_hit;
                if (!rayAABBIntersect(local_origin, local_dir, local_min, local_max, t_hit))
                    continue;

                const float world_dist = toWorldDist(t_hit);
                if (world_dist < closest_world_dist) {
                    closest_world_dist = world_dist;
                    closest_name = node->name;
                }
            }
        }
        return closest_name;
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

        for (const auto* node : scene_.getNodes()) {
            if (node->type != core::NodeType::SPLAT && node->type != core::NodeType::MESH)
                continue;
            if (!scene_.isNodeEffectivelyVisible(node->id))
                continue;

            const glm::mat4 world_transform = scene_.getWorldTransform(node->id);

            if (node->type == core::NodeType::MESH && node->mesh) {
                auto accessor = CpuMeshAccessor::from(*node->mesh);
                if (!accessor)
                    continue;

                glm::vec3 aabb_min, aabb_max;
                accessor->getBounds(aabb_min, aabb_max);

                glm::vec2 screen_aabb_min(1e10f);
                glm::vec2 screen_aabb_max(-1e10f);
                bool aabb_visible = false;
                for (int i = 0; i < BBOX_CORNERS; ++i) {
                    const glm::vec3 corner(
                        (i & 1) ? aabb_max.x : aabb_min.x,
                        (i & 2) ? aabb_max.y : aabb_min.y,
                        (i & 4) ? aabb_max.z : aabb_min.z);
                    const glm::vec2 sp = projectToScreen(
                        glm::vec3(world_transform * glm::vec4(corner, 1.0f)));
                    if (sp.x > BEHIND_CAMERA + 1e5f) {
                        screen_aabb_min = glm::min(screen_aabb_min, sp);
                        screen_aabb_max = glm::max(screen_aabb_max, sp);
                        aabb_visible = true;
                    }
                }
                if (!aabb_visible || !rectsOverlap(rect_min, rect_max, screen_aabb_min, screen_aabb_max))
                    continue;

                const int64_t nv = accessor->verts_cpu.size(0);
                bool hit = false;
                for (int64_t vi = 0; vi < nv; ++vi) {
                    const glm::vec2 sp = projectToScreen(
                        glm::vec3(world_transform * glm::vec4(accessor->vertex(vi), 1.0f)));
                    if (sp.x > BEHIND_CAMERA + 1e5f &&
                        sp.x >= rect_min.x && sp.x <= rect_max.x &&
                        sp.y >= rect_min.y && sp.y <= rect_max.y) {
                        hit = true;
                        break;
                    }
                }
                if (hit)
                    result.push_back(node->name);
            } else {
                glm::vec3 local_min, local_max;
                if (!scene_.getNodeBounds(node->id, local_min, local_max))
                    continue;

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

                if (any_visible && rectsOverlap(rect_min, rect_max, screen_min, screen_max))
                    result.push_back(node->name);
            }
        }

        glm::mat4 cam_scene_transform(1.0f);
        auto visible_transforms = scene_.getVisibleNodeTransforms();
        if (!visible_transforms.empty())
            cam_scene_transform = visible_transforms[0];

        for (const auto* node : scene_.getNodes()) {
            if (node->type != core::NodeType::CAMERA || !node->camera)
                continue;
            if (!scene_.isNodeEffectivelyVisible(node->id))
                continue;

            auto R_tensor = node->camera->R();
            auto T_tensor = node->camera->T();
            if (!R_tensor.is_valid() || !T_tensor.is_valid())
                continue;

            if (R_tensor.device() != lfs::core::Device::CPU)
                R_tensor = R_tensor.cpu();
            if (T_tensor.device() != lfs::core::Device::CPU)
                T_tensor = T_tensor.cpu();

            glm::mat4 w2c(1.0f);
            auto R_acc = R_tensor.accessor<float, 2>();
            auto T_acc = T_tensor.accessor<float, 1>();
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j)
                    w2c[j][i] = R_acc(i, j);
                w2c[3][i] = T_acc(i);
            }
            const glm::vec3 cam_pos = glm::vec3((cam_scene_transform * glm::inverse(w2c))[3]);

            const glm::vec2 screen_pos = projectToScreen(cam_pos);
            if (screen_pos.x <= BEHIND_CAMERA + 1e5f)
                continue;

            if (screen_pos.x >= rect_min.x && screen_pos.x <= rect_max.x &&
                screen_pos.y >= rect_min.y && screen_pos.y <= rect_max.y) {
                result.push_back(node->name);
            }
        }

        return result;
    }

    // ========== Node Transforms ==========

    void SceneManager::setNodeTransform(const std::string& name, const glm::mat4& transform) {
        scene_.setNodeTransform(name, transform);
    }

    glm::mat4 SceneManager::getNodeTransform(const std::string& name) const {
        return scene_.getNodeTransform(name);
    }

    void SceneManager::setSelectedNodeTranslation(const glm::vec3& translation) {
        std::string node_name;
        {
            std::shared_lock slock(selection_.mutex());
            const auto& ids = selection_.selectedNodeIds();
            if (ids.empty()) {
                LOG_TRACE("No node selected for translation");
                return;
            }
            const auto* node = scene_.getNodeById(*ids.begin());
            if (!node)
                return;
            node_name = node->name;
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
    }

    glm::vec3 SceneManager::getSelectedNodeTranslation() const {
        std::string node_name;
        {
            std::shared_lock slock(selection_.mutex());
            const auto& ids = selection_.selectedNodeIds();
            if (ids.empty())
                return glm::vec3(0.0f);
            const auto* node = scene_.getNodeById(*ids.begin());
            if (!node)
                return glm::vec3(0.0f);
            node_name = node->name;
        }

        if (node_name.empty()) {
            return glm::vec3(0.0f);
        }

        glm::mat4 transform = scene_.getNodeTransform(node_name);
        return glm::vec3(transform[3][0], transform[3][1], transform[3][2]);
    }

    glm::vec3 SceneManager::getSelectedNodeCentroid() const {
        std::shared_lock slock(selection_.mutex());
        const auto& ids = selection_.selectedNodeIds();
        if (ids.empty())
            return glm::vec3(0.0f);

        const auto* node = scene_.getNodeById(*ids.begin());
        if (!node || !node->model)
            return glm::vec3(0.0f);
        return node->centroid;
    }

    glm::vec3 SceneManager::getSelectedNodeCenter() const {
        std::shared_lock slock(selection_.mutex());
        const auto& ids = selection_.selectedNodeIds();
        if (ids.empty())
            return glm::vec3(0.0f);

        const auto* node = scene_.getNodeById(*ids.begin());
        if (!node)
            return glm::vec3(0.0f);

        return scene_.getNodeBoundsCenter(node->id);
    }

    void SceneManager::setSelectedNodeTransform(const glm::mat4& transform) {
        std::string node_name;
        {
            std::shared_lock slock(selection_.mutex());
            const auto& ids = selection_.selectedNodeIds();
            if (ids.empty()) {
                LOG_TRACE("No node selected for transform");
                return;
            }
            const auto* node = scene_.getNodeById(*ids.begin());
            if (!node)
                return;
            node_name = node->name;
        }

        LOG_DEBUG("setSelectedNodeTransform '{}': pos=[{:.2f}, {:.2f}, {:.2f}]",
                  node_name, transform[3][0], transform[3][1], transform[3][2]);
        scene_.setNodeTransform(node_name, transform);
    }

    glm::mat4 SceneManager::getSelectedNodeTransform() const {
        std::string node_name;
        {
            std::shared_lock slock(selection_.mutex());
            const auto& ids = selection_.selectedNodeIds();
            if (ids.empty())
                return glm::mat4(1.0f);
            const auto* node = scene_.getNodeById(*ids.begin());
            if (!node)
                return glm::mat4(1.0f);
            node_name = node->name;
        }

        return scene_.getNodeTransform(node_name);
    }

    glm::mat4 SceneManager::getSelectedNodeWorldTransform() const {
        std::shared_lock slock(selection_.mutex());
        const auto& ids = selection_.selectedNodeIds();
        if (ids.empty())
            return glm::mat4(1.0f);

        const auto* node = scene_.getNodeById(*ids.begin());
        if (!node)
            return glm::mat4(1.0f);

        return scene_.getWorldTransform(node->id);
    }

    glm::vec3 SceneManager::getSelectionCenter() const {
        std::shared_lock slock(selection_.mutex());
        const auto& ids = selection_.selectedNodeIds();
        if (ids.empty())
            return glm::vec3(0.0f);

        if (ids.size() == 1) {
            const auto* node = scene_.getNodeById(*ids.begin());
            if (!node)
                return glm::vec3(0.0f);
            return scene_.getNodeBoundsCenter(node->id);
        }

        glm::vec3 total_min(std::numeric_limits<float>::max());
        glm::vec3 total_max(std::numeric_limits<float>::lowest());
        bool has_bounds = false;

        for (const core::NodeId id : ids) {
            const auto* node = scene_.getNodeById(id);
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
        std::shared_lock slock(selection_.mutex());
        const auto& ids = selection_.selectedNodeIds();
        if (ids.empty())
            return glm::vec3(0.0f);

        glm::vec3 total_min(std::numeric_limits<float>::max());
        glm::vec3 total_max(std::numeric_limits<float>::lowest());
        bool has_bounds = false;

        for (const core::NodeId id : ids) {
            const auto* node = scene_.getNodeById(id);
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

    core::NodeId SceneManager::getSelectedNodeCropBoxId() const {
        std::shared_lock slock(selection_.mutex());
        const auto& ids = selection_.selectedNodeIds();
        if (ids.empty())
            return core::NULL_NODE;

        const auto* node = scene_.getNodeById(*ids.begin());
        if (!node)
            return core::NULL_NODE;

        // If selected node is a cropbox, return its ID
        if (node->type == core::NodeType::CROPBOX) {
            return node->id;
        }

        // If selected node is a splat or pointcloud, return its cropbox child
        if (node->type == core::NodeType::SPLAT || node->type == core::NodeType::POINTCLOUD) {
            return scene_.getCropBoxForSplat(node->id);
        }

        // For groups, no cropbox
        return core::NULL_NODE;
    }

    core::CropBoxData* SceneManager::getSelectedNodeCropBox() {
        const core::NodeId cropbox_id = getSelectedNodeCropBoxId();
        if (cropbox_id == core::NULL_NODE)
            return nullptr;
        return scene_.getCropBoxData(cropbox_id);
    }

    const core::CropBoxData* SceneManager::getSelectedNodeCropBox() const {
        const core::NodeId cropbox_id = getSelectedNodeCropBoxId();
        if (cropbox_id == core::NULL_NODE)
            return nullptr;
        return scene_.getCropBoxData(cropbox_id);
    }

    void SceneManager::syncCropBoxToRenderSettings() {
        // Scene graph is single source of truth - just trigger re-render
        if (services().renderingOrNull()) {
            services().renderingOrNull()->markDirty();
        }
    }

    // ========== Ellipsoid Operations ==========

    core::NodeId SceneManager::getSelectedNodeEllipsoidId() const {
        std::shared_lock slock(selection_.mutex());
        const auto& ids = selection_.selectedNodeIds();
        if (ids.empty())
            return core::NULL_NODE;

        const auto* node = scene_.getNodeById(*ids.begin());
        if (!node)
            return core::NULL_NODE;

        if (node->type == core::NodeType::ELLIPSOID) {
            return node->id;
        }

        if (node->type == core::NodeType::SPLAT || node->type == core::NodeType::POINTCLOUD) {
            return scene_.getEllipsoidForSplat(node->id);
        }

        return core::NULL_NODE;
    }

    core::EllipsoidData* SceneManager::getSelectedNodeEllipsoid() {
        const core::NodeId ellipsoid_id = getSelectedNodeEllipsoidId();
        if (ellipsoid_id == core::NULL_NODE)
            return nullptr;
        return scene_.getEllipsoidData(ellipsoid_id);
    }

    const core::EllipsoidData* SceneManager::getSelectedNodeEllipsoid() const {
        const core::NodeId ellipsoid_id = getSelectedNodeEllipsoidId();
        if (ellipsoid_id == core::NULL_NODE)
            return nullptr;
        return scene_.getEllipsoidData(ellipsoid_id);
    }

    void SceneManager::syncEllipsoidToRenderSettings() {
        if (services().renderingOrNull()) {
            services().renderingOrNull()->markDirty();
        }
    }

    std::expected<void, std::string> SceneManager::applyLoadedDataset(
        const std::filesystem::path& path,
        const lfs::core::param::TrainingParameters& params,
        lfs::io::LoadResult&& load_result) {
        LOG_TIMER("SceneManager::applyLoadedDataset");

        try {
            core::Scene::Transaction txn(scene_);

            if (services().trainerOrNull()) {
                services().trainerOrNull()->clearTrainer();
            }
            clear();

            auto dataset_params = params;
            dataset_params.dataset.data_path = path;
            cached_params_ = dataset_params;

            auto apply_result = lfs::training::applyLoadResultToScene(dataset_params, scene_, std::move(load_result));
            if (!apply_result) {
                return std::unexpected(apply_result.error());
            }

            if (scene_.hasTrainingData()) {
                auto trainer = std::make_unique<lfs::training::Trainer>(scene_);
                trainer->setParams(dataset_params);

                if (!services().trainerOrNull()) {
                    return std::unexpected("No trainer manager");
                }
                services().trainerOrNull()->setScene(&scene_);
                services().trainerOrNull()->setTrainer(std::move(trainer));
            }

            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                content_type_ = ContentType::Dataset;
                dataset_path_ = path;
            }

            const auto* training_model = scene_.getTrainingModel();
            const size_t num_gaussians = training_model ? training_model->size() : 0;
            const auto* point_cloud = scene_.getVisiblePointCloud();
            const size_t num_points = point_cloud ? point_cloud->size() : 0;

            state::SceneLoaded{
                .scene = nullptr,
                .path = path,
                .type = state::SceneLoaded::Type::Dataset,
                .num_gaussians = num_gaussians}
                .emit();

            python::set_application_scene(&scene_);

            if ((num_gaussians > 0 || num_points > 0) && services().trainerOrNull() && services().trainerOrNull()->getTrainer()) {
                ui::PointCloudModeChanged{.enabled = true, .voxel_size = DEFAULT_VOXEL_SIZE}.emit();
            }

            return {};

        } catch (const std::exception& e) {
            LOG_ERROR("applyLoadedDataset failed: {}", e.what());
            return std::unexpected(e.what());
        }
    }

    void SceneManager::loadDataset(const std::filesystem::path& path,
                                   const lfs::core::param::TrainingParameters& params) {
        LOG_TIMER("SceneManager::loadDataset");

        // Emit start event for progress tracking
        state::DatasetLoadStarted{.path = path}.emit();

        try {
            LOG_INFO("Loading dataset: {}", lfs::core::path_to_utf8(path));

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
            core::Scene::Transaction txn(scene_);

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
            const auto* point_cloud = scene_.getVisiblePointCloud();
            const size_t num_points = point_cloud ? point_cloud->size() : 0;
            const size_t num_cameras = scene_.getAllCameras().size();

            LOG_INFO("Dataset loaded successfully - {} images, {} initial points/gaussians",
                     num_cameras, num_gaussians > 0 ? num_gaussians : num_points);

            state::SceneLoaded{
                .scene = nullptr,
                .path = path,
                .type = state::SceneLoaded::Type::Dataset,
                .num_gaussians = num_gaussians}
                .emit();

            python::set_application_scene(&scene_);

            state::DatasetLoadCompleted{
                .path = path,
                .success = true,
                .error = std::nullopt,
                .num_images = num_cameras,
                .num_points = num_gaussians > 0 ? num_gaussians : num_points}
                .emit();

            // Switch to point cloud rendering mode by default for datasets
            if ((num_gaussians > 0 || num_points > 0) && services().trainerOrNull() && services().trainerOrNull()->getTrainer()) {
                ui::PointCloudModeChanged{.enabled = true, .voxel_size = DEFAULT_VOXEL_SIZE}.emit();
                LOG_INFO("Switched to point cloud mode ({} points)", num_gaussians > 0 ? num_gaussians : num_points);
            }

        } catch (const std::exception& e) {
            LOG_ERROR("Failed to load dataset: {} (path: {})", e.what(), lfs::core::path_to_utf8(path));

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

    void SceneManager::loadColmapCamerasOnly(const std::filesystem::path& sparse_path) {
        LOG_TIMER("SceneManager::loadColmapCamerasOnly");

        try {
            auto result = lfs::io::read_colmap_cameras_only(sparse_path);
            if (!result) {
                LOG_ERROR("Failed to load COLMAP cameras: {}", result.error().format());
                state::FileDropFailed{
                    .files = {lfs::core::path_to_utf8(sparse_path)},
                    .error = result.error().format()}
                    .emit();
                return;
            }

            auto [cameras, scene_center] = std::move(*result);

            if (cameras.empty()) {
                LOG_WARN("No cameras found in COLMAP sparse folder");
                return;
            }

            {
                core::Scene::Transaction txn(scene_);
                const core::NodeId group_id = scene_.addCameraGroup("Imported Cameras", core::NULL_NODE, cameras.size());
                for (const auto& cam : cameras) {
                    scene_.addCamera(cam->image_name(), group_id, cam);
                }
            }
            selection_.invalidateNodeMask();

            scene_.setSceneCenter(std::move(scene_center));

            state::SceneLoaded{
                .scene = nullptr,
                .path = sparse_path,
                .type = state::SceneLoaded::Type::Dataset,
                .num_gaussians = 0}
                .emit();

            python::set_application_scene(&scene_);

            LOG_INFO("Imported {} cameras from COLMAP (no images required)", cameras.size());

        } catch (const std::exception& e) {
            LOG_ERROR("Failed to import COLMAP cameras: {}", e.what());
            state::FileDropFailed{
                .files = {lfs::core::path_to_utf8(sparse_path)},
                .error = e.what()}
                .emit();
        }
    }

    void SceneManager::prepareTrainingFromScene() {
        if (!scene_.hasTrainingData()) {
            LOG_ERROR("Cannot prepare training: scene has no cameras");
            return;
        }

        auto* trainer_mgr = services().trainerOrNull();
        if (!trainer_mgr) {
            LOG_ERROR("Cannot prepare training: no trainer manager");
            return;
        }

        try {
            auto trainer = std::make_unique<lfs::training::Trainer>(scene_);
            trainer_mgr->setScene(&scene_);
            trainer_mgr->setTrainer(std::move(trainer));

            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                content_type_ = ContentType::Dataset;
            }

            const auto* point_cloud = scene_.getVisiblePointCloud();
            const size_t num_points = point_cloud ? point_cloud->size() : 0;
            const size_t num_cameras = scene_.getAllCameras().size();

            state::SceneLoaded{
                .scene = nullptr,
                .path = {},
                .type = state::SceneLoaded::Type::Dataset,
                .num_gaussians = 0}
                .emit();

            python::set_application_scene(&scene_);

            LOG_INFO("Trainer prepared from scene: {} cameras, {} points", num_cameras, num_points);
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to prepare training from scene: {}", e.what());
        }
    }

    void SceneManager::loadCheckpointForTraining(const std::filesystem::path& path,
                                                 const lfs::core::param::TrainingParameters& params) {
        LOG_TIMER("SceneManager::loadCheckpointForTraining");

        try {
            // === Phase 1: Validate checkpoint BEFORE clearing scene ===
            const auto header_result = lfs::core::load_checkpoint_header(path);
            if (!header_result) {
                throw std::runtime_error("Failed to load checkpoint header: " + header_result.error());
            }
            const int checkpoint_iteration = header_result->iteration;

            auto params_result = lfs::core::load_checkpoint_params(path);
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
                                         lfs::core::path_to_utf8(checkpoint_params.dataset.data_path));
            }

            // Validate dataset structure before clearing
            const auto validation_result = lfs::training::validateDatasetPath(checkpoint_params);
            if (!validation_result) {
                throw std::runtime_error("Failed to load training data: " + validation_result.error());
            }

            // === Phase 2: Clear scene (validation passed) ===
            core::Scene::Transaction txn(scene_);

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
                if (node->type == lfs::core::NodeType::POINTCLOUD) {
                    scene_.removeNode(node->name, false);
                    break;
                }
            }

            auto splat_result = lfs::core::load_checkpoint_splat_data(path);
            if (!splat_result) {
                throw std::runtime_error("Failed to load checkpoint SplatData: " + splat_result.error());
            }

            auto splat_data = std::make_unique<lfs::core::SplatData>(std::move(*splat_result));
            const size_t num_gaussians = splat_data->size();
            constexpr const char* MODEL_NAME = "Model";

            scene_.addSplat(MODEL_NAME, std::move(splat_data), lfs::core::NULL_NODE);
            selection_.invalidateNodeMask();
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

            // when loading checkpoint with sparsity enabled, adjust total iterations
            checkpoint_params.optimization.iterations = checkpoint_params.optimization.enable_sparsity ? checkpoint_params.optimization.iterations - checkpoint_params.optimization.sparsify_steps : checkpoint_params.optimization.iterations;

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

            python::set_application_scene(&scene_);

            ui::PointCloudModeChanged{.enabled = false, .voxel_size = DEFAULT_VOXEL_SIZE}.emit();
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

        selection_.clearNodeSelection();
        python::set_application_scene(nullptr);
        clearAppearanceModel();
        scene_.clear();

        if (lfs::io::CacheLoader::hasInstance()) {
            lfs::io::CacheLoader::getInstance().reset_cache();
        }

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            content_type_ = ContentType::Empty;
            splat_paths_.clear();
            dataset_path_.clear();
        }

        state::SceneCleared{}.emit();

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

        core::Scene::Transaction txn(scene_);

        auto splat_data = std::move(model_node->model);
        const size_t num_gaussians = splat_data->size();

        // Extract PPISP models from trainer before clearing
        std::unique_ptr<lfs::training::PPISP> ppisp;
        std::unique_ptr<lfs::training::PPISPControllerPool> controller_pool;
        if (auto* trainer_mgr = services().trainerOrNull()) {
            if (auto* trainer = trainer_mgr->getTrainer(); trainer && trainer->hasPPISP()) {
                ppisp = trainer->takePPISP();
                controller_pool = trainer->takePPISPControllerPool();
            }
            trainer_mgr->clearTrainer();
        }
        scene_.clear();

        constexpr const char* MODEL_NAME = "Trained Model";
        scene_.addNode(MODEL_NAME, std::move(splat_data));
        selectNode(MODEL_NAME);

        if (ppisp) {
            setAppearanceModel(std::move(ppisp), std::move(controller_pool));
        }

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

        LOG_INFO("Switched to Edit Mode: {} gaussians", num_gaussians);
    }

    const lfs::core::SplatData* SceneManager::getModelForRendering() const {
        std::lock_guard<std::mutex> lock(state_mutex_);

        switch (content_type_) {
        case ContentType::SplatFiles:
            return scene_.getCombinedModel();
        case ContentType::Dataset:
            return scene_.getTrainingModel();
        case ContentType::Empty:
            return scene_.hasNodes() ? scene_.getCombinedModel() : nullptr;
        }
        return nullptr;
    }

    SceneRenderState SceneManager::buildRenderState() const {
        std::lock_guard<std::mutex> lock(state_mutex_);

        SceneRenderState state;

        // Get combined model or point cloud
        if (content_type_ == ContentType::SplatFiles) {
            state.combined_model = scene_.getCombinedModel();
        } else if (content_type_ == ContentType::Dataset) {
            state.combined_model = scene_.getTrainingModel();
        }

        // Always try to get point cloud if no model (supports plugin-added point clouds)
        if (!state.combined_model) {
            state.point_cloud = scene_.getVisiblePointCloud();
        }

        state.meshes = scene_.getVisibleMeshes();
        for (auto& vm : state.meshes) {
            vm.is_selected = selection_.isNodeSelected(vm.node_id);
        }

        // Get transforms and indices
        state.model_transforms = scene_.getVisibleNodeTransforms();
        state.transform_indices = scene_.getTransformIndices();
        state.visible_splat_count = state.model_transforms.size();

        // Get node visibility mask (for consolidated models)
        state.node_visibility_mask = scene_.getNodeVisibilityMask();

        // Get selection mask
        state.selection_mask = scene_.getSelectionMask();
        state.has_selection = scene_.hasSelection();

        // Get cropboxes (before lock — no selection dependency)
        state.cropboxes = scene_.getVisibleCropBoxes();

        // Read selection-dependent state
        {
            std::shared_lock slock(selection_.mutex());
            const auto& sel_ids = selection_.selectedNodeIds();
            if (!sel_ids.empty()) {
                const auto* first = scene_.getNodeById(*sel_ids.begin());
                state.selected_node_name = first ? first->name : "";

                if (first) {
                    core::NodeId cropbox_id = core::NULL_NODE;
                    if (first->type == core::NodeType::CROPBOX) {
                        cropbox_id = first->id;
                    } else if (first->type == core::NodeType::SPLAT) {
                        cropbox_id = scene_.getCropBoxForSplat(first->id);
                    }
                    if (cropbox_id != core::NULL_NODE) {
                        for (size_t i = 0; i < state.cropboxes.size(); ++i) {
                            if (state.cropboxes[i].node_id == cropbox_id) {
                                state.selected_cropbox_index = static_cast<int>(i);
                                break;
                            }
                        }
                    }
                }
            }
        }
        // getNodeMask() may promote shared→exclusive internally, call outside shared_lock
        state.selected_node_mask = selection_.getNodeMask(scene_);

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

    void SceneManager::syncCropToolRenderSettings(const core::SceneNode* node) {
        if (!node)
            return;
        auto* rm = services().renderingOrNull();
        if (!rm)
            return;

        auto settings = rm->getSettings();
        if (node->type == core::NodeType::CROPBOX && !settings.show_crop_box) {
            settings.show_crop_box = true;
            rm->updateSettings(settings);
        } else if (node->type == core::NodeType::ELLIPSOID && !settings.show_ellipsoid) {
            settings.show_ellipsoid = true;
            rm->updateSettings(settings);
        }
    }

    void SceneManager::handleCropActivePly(const lfs::geometry::BoundingBox& crop_box, const bool inverse) {
        std::vector<std::string> splat_node_names;
        std::vector<std::string> pointcloud_node_names;
        bool had_selection = false;

        {
            std::shared_lock slock(selection_.mutex());
            const auto& sel_ids = selection_.selectedNodeIds();
            if (!sel_ids.empty()) {
                had_selection = true;
                for (const core::NodeId nid : sel_ids) {
                    const auto* selected = scene_.getNodeById(nid);
                    if (!selected)
                        continue;

                    if (selected->type == core::NodeType::SPLAT) {
                        splat_node_names.push_back(selected->name);
                    } else if (selected->type == core::NodeType::POINTCLOUD) {
                        pointcloud_node_names.push_back(selected->name);
                    } else if (selected->type == core::NodeType::CROPBOX) {
                        const auto* parent = scene_.getNodeById(selected->parent_id);
                        if (parent && parent->type == core::NodeType::SPLAT) {
                            splat_node_names.push_back(parent->name);
                        } else if (parent && parent->type == core::NodeType::POINTCLOUD) {
                            pointcloud_node_names.push_back(parent->name);
                        }
                    }
                }
            }
        }

        // Fall back to visible nodes if no selection
        if (splat_node_names.empty() && pointcloud_node_names.empty() && !had_selection) {
            for (const auto* node : scene_.getVisibleNodes()) {
                if (node->type == core::NodeType::SPLAT) {
                    splat_node_names.push_back(node->name);
                } else if (node->type == core::NodeType::POINTCLOUD) {
                    pointcloud_node_names.push_back(node->name);
                }
            }
        }

        // Crop point cloud data (GPU-accelerated)
        for (const auto& node_name : pointcloud_node_names) {
            auto* node = scene_.getMutableNode(node_name);
            if (!node || !node->point_cloud)
                continue;

            const core::NodeId cropbox_id = scene_.getCropBoxForSplat(node->id);
            if (cropbox_id == core::NULL_NODE)
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
                scene_.notifyMutation(core::Scene::MutationType::MODEL_CHANGED);
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
                const size_t original_visible = node->model->visible_count();

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

        scene_.notifyMutation(core::Scene::MutationType::MODEL_CHANGED);
    }

    void SceneManager::handleCropByEllipsoid(const glm::mat4& world_transform, const glm::vec3& radii, const bool inverse) {
        std::vector<std::string> splat_node_names;
        std::vector<std::string> pointcloud_node_names;
        bool had_selection = false;

        {
            std::shared_lock slock(selection_.mutex());
            const auto& sel_ids = selection_.selectedNodeIds();
            if (!sel_ids.empty()) {
                had_selection = true;
                for (const core::NodeId nid : sel_ids) {
                    const auto* selected = scene_.getNodeById(nid);
                    if (!selected)
                        continue;

                    if (selected->type == core::NodeType::SPLAT) {
                        splat_node_names.push_back(selected->name);
                    } else if (selected->type == core::NodeType::POINTCLOUD) {
                        pointcloud_node_names.push_back(selected->name);
                    } else if (selected->type == core::NodeType::ELLIPSOID) {
                        const auto* parent = scene_.getNodeById(selected->parent_id);
                        if (parent && parent->type == core::NodeType::SPLAT) {
                            splat_node_names.push_back(parent->name);
                        } else if (parent && parent->type == core::NodeType::POINTCLOUD) {
                            pointcloud_node_names.push_back(parent->name);
                        }
                    }
                }
            }
        }

        if (splat_node_names.empty() && pointcloud_node_names.empty() && !had_selection) {
            for (const auto* node : scene_.getVisibleNodes()) {
                if (node->type == core::NodeType::SPLAT) {
                    splat_node_names.push_back(node->name);
                } else if (node->type == core::NodeType::POINTCLOUD) {
                    pointcloud_node_names.push_back(node->name);
                }
            }
        }

        const glm::mat4 inv_world = glm::inverse(world_transform);

        // Crop point clouds
        for (const auto& node_name : pointcloud_node_names) {
            auto* node = scene_.getMutableNode(node_name);
            if (!node || !node->point_cloud)
                continue;

            const auto& means = node->point_cloud->means;
            const auto& colors = node->point_cloud->colors;
            const size_t num_points = node->point_cloud->size();
            const auto device = means.device();

            // Transform to ellipsoid local space
            const auto transform = lfs::core::Tensor::from_vector(
                {inv_world[0][0], inv_world[1][0], inv_world[2][0], inv_world[3][0],
                 inv_world[0][1], inv_world[1][1], inv_world[2][1], inv_world[3][1],
                 inv_world[0][2], inv_world[1][2], inv_world[2][2], inv_world[3][2],
                 inv_world[0][3], inv_world[1][3], inv_world[2][3], inv_world[3][3]},
                {4, 4}, device);

            const auto ones = lfs::core::Tensor::ones({num_points, 1}, device);
            const auto local_pos = transform.mm(means.cat(ones, 1).t()).t();

            const auto x = local_pos.slice(1, 0, 1).squeeze(1) / radii.x;
            const auto y = local_pos.slice(1, 1, 2).squeeze(1) / radii.y;
            const auto z = local_pos.slice(1, 2, 3).squeeze(1) / radii.z;

            auto mask = (x * x + y * y + z * z) <= 1.0f;
            if (inverse)
                mask = mask.logical_not();

            const auto indices = mask.nonzero().squeeze(1);
            const size_t filtered_count = indices.size(0);

            if (filtered_count > 0 && filtered_count < num_points) {
                node->point_cloud = std::make_shared<lfs::core::PointCloud>(
                    means.index_select(0, indices), colors.index_select(0, indices));
                node->gaussian_count = filtered_count;
                LOG_INFO("Ellipsoid cropped PointCloud '{}': {} -> {} points", node_name, num_points, filtered_count);
            }
        }

        if (splat_node_names.empty()) {
            if (!pointcloud_node_names.empty()) {
                scene_.notifyMutation(core::Scene::MutationType::MODEL_CHANGED);
                if (services().renderingOrNull()) {
                    services().renderingOrNull()->markDirty();
                }
            }
            return;
        }

        changeContentType(ContentType::SplatFiles);

        for (const auto& node_name : splat_node_names) {
            auto* node = scene_.getMutableNode(node_name);
            if (!node || !node->model)
                continue;

            try {
                const size_t original_visible = node->model->visible_count();

                // Transform means to ellipsoid local space and apply mask
                const glm::mat4 node_world_transform = scene_.getWorldTransform(node->id);
                const glm::mat4 combined_transform = inv_world * node_world_transform;

                const auto applied_mask = lfs::core::soft_crop_by_ellipsoid(*node->model, combined_transform, radii, inverse);
                if (!applied_mask.is_valid())
                    continue;

                const size_t new_visible = node->model->visible_count();
                if (new_visible == original_visible)
                    continue;

                LOG_INFO("Ellipsoid cropped '{}': {} -> {} visible", node_name, original_visible, new_visible);

            } catch (const std::exception& e) {
                LOG_ERROR("Failed to ellipsoid crop '{}': {}", node_name, e.what());
            }
        }

        scene_.notifyMutation(core::Scene::MutationType::MODEL_CHANGED);
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
        if (node->parent_id != core::NULL_NODE) {
            if (const auto* p = scene_.getNodeById(node->parent_id)) {
                old_parent_name = p->name;
            }
        }

        core::NodeId parent_id = core::NULL_NODE;
        if (!new_parent_name.empty()) {
            const auto* parent = scene_.getNode(new_parent_name);
            if (!parent)
                return;
            parent_id = parent->id;
        }

        scene_.reparent(node->id, parent_id);
        selection_.invalidateNodeMask();
        state::NodeReparented{.name = node_name, .old_parent = old_parent_name, .new_parent = new_parent_name}.emit();
    }

    void SceneManager::handleAddGroup(const std::string& name, const std::string& parent_name) {
        core::NodeId parent_id = core::NULL_NODE;
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
        selection_.invalidateNodeMask();
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
        if (src->parent_id != core::NULL_NODE) {
            if (const auto* p = scene_.getNodeById(src->parent_id)) {
                parent_name = p->name;
            }
        }

        const std::string new_name = scene_.duplicateNode(name);
        if (new_name.empty())
            return;
        selection_.invalidateNodeMask();

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
                    .is_group = node->type == core::NodeType::GROUP,
                    .node_type = static_cast<int>(node->type)}
                    .emit();

                for (const core::NodeId cid : node->children) {
                    if (const auto* c = scene_.getNodeById(cid)) {
                        emit_added(c->name, node->name);
                    }
                }
            };

        emit_added(new_name, parent_name);
    }

    void SceneManager::handleMergeGroup(const std::string& name) {
        const auto* group = scene_.getNode(name);
        if (!group || group->type != core::NodeType::GROUP) {
            return;
        }

        std::string parent_name;
        if (group->parent_id != core::NULL_NODE) {
            if (const auto* p = scene_.getNodeById(group->parent_id)) {
                parent_name = p->name;
            }
        }

        // Check if the group being merged is currently selected
        bool was_selected = false;
        {
            const core::NodeId group_nid = scene_.getNodeIdByName(name);
            if (group_nid != core::NULL_NODE && selection_.isNodeSelected(group_nid)) {
                was_selected = true;
                selection_.removeFromSelection(group_nid);
            }
        }

        // Collect children to emit PLYRemoved events
        std::vector<std::string> children_to_remove;
        std::function<void(const core::SceneNode*)> collect_children = [&](const core::SceneNode* n) {
            for (const core::NodeId cid : n->children) {
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
        selection_.invalidateNodeMask();

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
                    const core::NodeId merged_nid = scene_.getNodeIdByName(merged_name);
                    assert(merged_nid != core::NULL_NODE);
                    selection_.addToSelection(merged_nid);
                }
                ui::NodeSelected{
                    .path = merged_name,
                    .type = "PLY",
                    .metadata = {{"name", merged_name}}}
                    .emit();
            }
        }

        LOG_INFO("Merged group '{}' -> '{}'", name, merged_name);
    }

    void SceneManager::handleAddCropBox(const std::string& node_name) {
        const auto* node = scene_.getNode(node_name);
        if (!node)
            return;

        if (node->type != core::NodeType::SPLAT && node->type != core::NodeType::POINTCLOUD) {
            LOG_WARN("Cannot add cropbox to node type: {}", static_cast<int>(node->type));
            return;
        }

        // Check if cropbox already exists for this node
        const core::NodeId existing = scene_.getCropBoxForSplat(node->id);
        if (existing != core::NULL_NODE) {
            LOG_DEBUG("Cropbox already exists for '{}'", node_name);
            selectNode(scene_.getNodeById(existing)->name);
            return;
        }

        const std::string cropbox_name = node_name + "_cropbox";
        const core::NodeId cropbox_id = scene_.addCropBox(cropbox_name, node->id);
        if (cropbox_id == core::NULL_NODE)
            return;

        // Fit cropbox to parent bounds and enable it
        core::CropBoxData data;
        glm::vec3 min_bounds, max_bounds;
        if (scene_.getNodeBounds(node->id, min_bounds, max_bounds)) {
            data.min = min_bounds;
            data.max = max_bounds;
        }
        data.enabled = true;
        scene_.setCropBoxData(cropbox_id, data);

        // Emit PLYAdded event
        if (const auto* cropbox = scene_.getNodeById(cropbox_id)) {
            state::PLYAdded{
                .name = cropbox->name,
                .node_gaussians = 0,
                .total_gaussians = scene_.getTotalGaussianCount(),
                .is_visible = cropbox->visible,
                .parent_name = node_name,
                .is_group = false,
                .node_type = static_cast<int>(core::NodeType::CROPBOX)}
                .emit();
        }

        // Enable cropbox visibility in render settings
        if (auto* rm = services().renderingOrNull()) {
            auto settings = rm->getSettings();
            settings.show_crop_box = true;
            rm->updateSettings(settings);
        }

        selectNode(cropbox_name);
        LOG_INFO("Added cropbox '{}' as child of '{}'", cropbox_name, node_name);
    }

    void SceneManager::handleAddCropEllipsoid(const std::string& node_name) {
        const auto* node = scene_.getNode(node_name);
        if (!node)
            return;

        if (node->type != core::NodeType::SPLAT && node->type != core::NodeType::POINTCLOUD) {
            LOG_WARN("Cannot add ellipsoid to node type: {}", static_cast<int>(node->type));
            return;
        }

        // Check if ellipsoid already exists for this node
        const core::NodeId existing = scene_.getEllipsoidForSplat(node->id);
        if (existing != core::NULL_NODE) {
            LOG_DEBUG("Ellipsoid already exists for '{}'", node_name);
            selectNode(scene_.getNodeById(existing)->name);
            return;
        }

        const std::string ellipsoid_name = node_name + "_ellipsoid";
        const core::NodeId ellipsoid_id = scene_.addEllipsoid(ellipsoid_name, node->id);
        if (ellipsoid_id == core::NULL_NODE)
            return;

        // Fit ellipsoid to parent bounds and enable it
        core::EllipsoidData data;
        glm::vec3 min_bounds, max_bounds;
        if (scene_.getNodeBounds(node->id, min_bounds, max_bounds)) {
            constexpr float CIRCUMSCRIBE_FACTOR = 1.732050808f; // sqrt(3)
            const glm::vec3 half_size = (max_bounds - min_bounds) * 0.5f;
            data.radii = half_size * CIRCUMSCRIBE_FACTOR;

            // Position ellipsoid at center of bounds
            if (auto* ellipsoid_node = scene_.getMutableNode(ellipsoid_name)) {
                const glm::vec3 center = (min_bounds + max_bounds) * 0.5f;
                ellipsoid_node->local_transform = glm::translate(glm::mat4(1.0f), center);
                ellipsoid_node->transform_dirty = true;
            }
        }
        data.enabled = true;
        scene_.setEllipsoidData(ellipsoid_id, data);

        // Emit PLYAdded event
        if (const auto* ellipsoid = scene_.getNodeById(ellipsoid_id)) {
            state::PLYAdded{
                .name = ellipsoid->name,
                .node_gaussians = 0,
                .total_gaussians = scene_.getTotalGaussianCount(),
                .is_visible = ellipsoid->visible,
                .parent_name = node_name,
                .is_group = false,
                .node_type = static_cast<int>(core::NodeType::ELLIPSOID)}
                .emit();
        }

        // Enable ellipsoid visibility in render settings
        if (auto* rm = services().renderingOrNull()) {
            auto settings = rm->getSettings();
            settings.show_ellipsoid = true;
            rm->updateSettings(settings);
        }

        selectNode(ellipsoid_name);
        LOG_INFO("Added ellipsoid '{}' as child of '{}'", ellipsoid_name, node_name);
    }

    void SceneManager::handleResetCropBox() {
        const core::SceneNode* cropbox_node = nullptr;
        {
            std::shared_lock slock(selection_.mutex());
            for (const auto id : selection_.selectedNodeIds()) {
                const auto* node = scene_.getNodeById(id);
                if (node && node->type == core::NodeType::CROPBOX && node->cropbox) {
                    cropbox_node = node;
                    break;
                }
            }
        }

        if (!cropbox_node) {
            LOG_WARN("No cropbox selected for reset");
            return;
        }

        auto* node = scene_.getMutableNode(cropbox_node->name);
        if (!node || !node->cropbox)
            return;

        node->cropbox->min = glm::vec3(-1.0f);
        node->cropbox->max = glm::vec3(1.0f);
        node->cropbox->inverse = false;
        node->local_transform = glm::mat4(1.0f);
        node->transform_dirty = true;

        if (auto* rm = services().renderingOrNull()) {
            auto settings = rm->getSettings();
            settings.use_crop_box = false;
            rm->updateSettings(settings);
            rm->markDirty();
        }

        scene_.notifyMutation(core::Scene::MutationType::MODEL_CHANGED);
        LOG_INFO("Reset cropbox '{}'", cropbox_node->name);
    }

    void SceneManager::handleResetEllipsoid() {
        const core::SceneNode* ellipsoid_node = nullptr;
        {
            std::shared_lock slock(selection_.mutex());
            for (const auto id : selection_.selectedNodeIds()) {
                const auto* node = scene_.getNodeById(id);
                if (node && node->type == core::NodeType::ELLIPSOID && node->ellipsoid) {
                    ellipsoid_node = node;
                    break;
                }
            }
        }

        if (!ellipsoid_node) {
            LOG_WARN("No ellipsoid selected for reset");
            return;
        }

        auto* node = scene_.getMutableNode(ellipsoid_node->name);
        if (!node || !node->ellipsoid)
            return;

        node->ellipsoid->radii = glm::vec3(1.0f);
        node->ellipsoid->inverse = false;
        node->local_transform = glm::mat4(1.0f);
        node->transform_dirty = true;

        if (auto* rm = services().renderingOrNull()) {
            auto settings = rm->getSettings();
            settings.use_ellipsoid = false;
            rm->updateSettings(settings);
            rm->markDirty();
        }

        scene_.notifyMutation(core::Scene::MutationType::MODEL_CHANGED);
        LOG_INFO("Reset ellipsoid '{}'", ellipsoid_node->name);
    }

    void SceneManager::updateCropBoxToFitScene(const bool use_percentile) {
        if (!services().renderingOrNull())
            return;

        // Find selected cropbox
        const core::SceneNode* cropbox_node = nullptr;
        const core::SceneNode* target_node = nullptr;

        {
            std::shared_lock slock(selection_.mutex());
            for (const auto id : selection_.selectedNodeIds()) {
                const auto* node = scene_.getNodeById(id);
                if (!node)
                    continue;
                if (node->type == core::NodeType::CROPBOX && node->cropbox) {
                    cropbox_node = node;
                    if (node->parent_id != core::NULL_NODE)
                        target_node = scene_.getNodeById(node->parent_id);
                    break;
                }
            }
        }

        if (!cropbox_node) {
            LOG_WARN("No cropbox found in selection");
            return;
        }

        if (!target_node) {
            for (const auto* node : scene_.getNodes()) {
                if (node->type == core::NodeType::SPLAT || node->type == core::NodeType::POINTCLOUD) {
                    target_node = node;
                    break;
                }
            }
        }

        if (!target_node) {
            LOG_WARN("No target splat found for cropbox '{}'", cropbox_node->name);
            return;
        }

        glm::vec3 min_bounds, max_bounds;
        bool bounds_valid = false;

        if (target_node->type == core::NodeType::SPLAT && target_node->model && target_node->model->size() > 0) {
            bounds_valid = lfs::core::compute_bounds(*target_node->model, min_bounds, max_bounds, 0.0f, use_percentile);
        } else if (target_node->type == core::NodeType::POINTCLOUD && target_node->point_cloud && target_node->point_cloud->size() > 0) {
            bounds_valid = lfs::core::compute_bounds(*target_node->point_cloud, min_bounds, max_bounds, 0.0f, use_percentile);
        }

        if (!bounds_valid) {
            LOG_WARN("Cannot compute bounds for '{}'", target_node->name);
            return;
        }

        const glm::vec3 center = (min_bounds + max_bounds) * 0.5f;
        const glm::vec3 half_size = (max_bounds - min_bounds) * 0.5f;

        if (auto* node = scene_.getMutableNode(cropbox_node->name); node && node->cropbox) {
            node->cropbox->min = -half_size;
            node->cropbox->max = half_size;
            node->local_transform = glm::translate(glm::mat4(1.0f), center);
            node->transform_dirty = true;
        }

        services().renderingOrNull()->markDirty();

        LOG_INFO("Fit '{}' to '{}': center({:.2f},{:.2f},{:.2f}) size({:.2f},{:.2f},{:.2f})",
                 cropbox_node->name, target_node->name, center.x, center.y, center.z,
                 half_size.x * 2, half_size.y * 2, half_size.z * 2);
    }

    void SceneManager::updateEllipsoidToFitScene(const bool use_percentile) {
        if (!services().renderingOrNull())
            return;

        // Find selected ellipsoid
        const core::SceneNode* ellipsoid_node = nullptr;
        const core::SceneNode* target_node = nullptr;

        {
            std::shared_lock slock(selection_.mutex());
            for (const auto id : selection_.selectedNodeIds()) {
                const auto* node = scene_.getNodeById(id);
                if (!node)
                    continue;
                if (node->type == core::NodeType::ELLIPSOID && node->ellipsoid) {
                    ellipsoid_node = node;
                    if (node->parent_id != core::NULL_NODE)
                        target_node = scene_.getNodeById(node->parent_id);
                    break;
                }
            }
        }

        if (!ellipsoid_node) {
            LOG_WARN("No ellipsoid found in selection");
            return;
        }

        // If no target splat set, try to find first SPLAT or POINTCLOUD
        if (!target_node) {
            for (const auto* node : scene_.getNodes()) {
                if (node->type == core::NodeType::SPLAT || node->type == core::NodeType::POINTCLOUD) {
                    target_node = node;
                    break;
                }
            }
        }

        if (!target_node) {
            LOG_WARN("No target splat found for ellipsoid '{}'", ellipsoid_node->name);
            return;
        }

        glm::vec3 min_bounds, max_bounds;
        bool bounds_valid = false;

        if (target_node->type == core::NodeType::SPLAT && target_node->model && target_node->model->size() > 0) {
            bounds_valid = lfs::core::compute_bounds(*target_node->model, min_bounds, max_bounds, 0.0f, use_percentile);
        } else if (target_node->type == core::NodeType::POINTCLOUD && target_node->point_cloud && target_node->point_cloud->size() > 0) {
            bounds_valid = lfs::core::compute_bounds(*target_node->point_cloud, min_bounds, max_bounds, 0.0f, use_percentile);
        }

        if (!bounds_valid) {
            LOG_WARN("Cannot compute bounds for '{}'", target_node->name);
            return;
        }

        const glm::vec3 center = (min_bounds + max_bounds) * 0.5f;
        const glm::vec3 half_size = (max_bounds - min_bounds) * 0.5f;

        // Scale radii by sqrt(3) so ellipsoid circumscribes the bounding box
        // (contains all corners, not just face centers)
        constexpr float CIRCUMSCRIBE_FACTOR = 1.732050808f; // sqrt(3)
        const glm::vec3 radii = half_size * CIRCUMSCRIBE_FACTOR;

        if (auto* node = scene_.getMutableNode(ellipsoid_node->name); node && node->ellipsoid) {
            node->ellipsoid->radii = radii;
            node->local_transform = glm::translate(glm::mat4(1.0f), center);
            node->transform_dirty = true;
        }

        if (auto* rm = services().renderingOrNull()) {
            rm->markDirty();
        }

        LOG_INFO("Fit ellipsoid '{}' to '{}': center({:.2f},{:.2f},{:.2f}) radii({:.2f},{:.2f},{:.2f})",
                 ellipsoid_node->name, target_node->name, center.x, center.y, center.z,
                 radii.x, radii.y, radii.z);
    }

    SceneManager::ClipboardEntry::HierarchyNode SceneManager::copyNodeHierarchy(const core::SceneNode* node) {
        ClipboardEntry::HierarchyNode result;
        result.type = node->type;
        result.local_transform = node->local_transform.get();

        if (node->cropbox) {
            result.cropbox = std::make_unique<core::CropBoxData>(*node->cropbox);
        }

        for (const core::NodeId child_id : node->children) {
            if (const auto* child = scene_.getNodeById(child_id)) {
                result.children.push_back(copyNodeHierarchy(child));
            }
        }

        return result;
    }

    void SceneManager::pasteNodeHierarchy(const ClipboardEntry::HierarchyNode& src, const core::NodeId parent_id) {
        for (const auto& child : src.children) {
            if (child.type == core::NodeType::CROPBOX && child.cropbox) {
                const core::NodeId cropbox_id = scene_.getOrCreateCropBoxForSplat(parent_id);
                if (cropbox_id == core::NULL_NODE)
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
        std::shared_lock slock(selection_.mutex());
        const auto& sel_ids = selection_.selectedNodeIds();
        if (sel_ids.empty()) {
            clipboard_.clear();
            return false;
        }

        clipboard_.clear();
        clipboard_.reserve(sel_ids.size());

        for (const auto id : sel_ids) {
            const auto* node = scene_.getNodeById(id);
            if (!node)
                continue;

            ClipboardEntry entry;
            entry.transform = node->local_transform.get();
            entry.hierarchy = copyNodeHierarchy(node);

            if (node->type == core::NodeType::MESH && node->mesh) {
                const auto& sm = *node->mesh;
                auto cloned = std::make_shared<core::MeshData>();
                cloned->vertices = sm.vertices.clone();
                cloned->indices = sm.indices.clone();
                if (sm.has_normals())
                    cloned->normals = sm.normals.clone();
                if (sm.has_tangents())
                    cloned->tangents = sm.tangents.clone();
                if (sm.has_texcoords())
                    cloned->texcoords = sm.texcoords.clone();
                if (sm.has_colors())
                    cloned->colors = sm.colors.clone();
                cloned->materials = sm.materials;
                cloned->submeshes = sm.submeshes;
                cloned->texture_images = sm.texture_images;
                entry.mesh = std::move(cloned);
            } else if (node->model && node->model->size() > 0) {
                const auto& src = *node->model;
                auto cloned = std::make_unique<lfs::core::SplatData>(
                    src.get_max_sh_degree(),
                    src.means_raw().clone(), src.sh0_raw().clone(), src.shN_raw().clone(),
                    src.scaling_raw().clone(), src.rotation_raw().clone(), src.opacity_raw().clone(),
                    src.get_scene_scale());
                cloned->set_active_sh_degree(src.get_active_sh_degree());
                entry.data = std::move(cloned);
            } else {
                continue;
            }

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
        selection_.invalidateNodeMask();

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

        LOG_INFO("Pasted {} Gaussians as '{}'", count, name);
        return {name};
    }

    bool SceneManager::executeMirror(const lfs::core::MirrorAxis axis) {
        std::vector<core::SceneNode*> nodes;
        {
            std::shared_lock slock(selection_.mutex());
            const auto& sel_ids = selection_.selectedNodeIds();
            nodes.reserve(sel_ids.size());
            for (const auto id : sel_ids) {
                auto* n = scene_.getNodeById(id);
                if (n && n->type == core::NodeType::SPLAT && n->model)
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
            lfs::core::mirror_gaussians(model, *mask, axis, center);
        }

        scene_.notifyMutation(core::Scene::MutationType::MODEL_CHANGED);

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
        core::Scene::Transaction txn(scene_);

        for (const auto& entry : clipboard_) {
            ++clipboard_counter_;
            const std::string name = std::format("Pasted_{}", clipboard_counter_);

            if (entry.mesh) {
                auto cloned = std::make_shared<core::MeshData>();
                cloned->vertices = entry.mesh->vertices.clone();
                cloned->indices = entry.mesh->indices.clone();
                if (entry.mesh->has_normals())
                    cloned->normals = entry.mesh->normals.clone();
                if (entry.mesh->has_tangents())
                    cloned->tangents = entry.mesh->tangents.clone();
                if (entry.mesh->has_texcoords())
                    cloned->texcoords = entry.mesh->texcoords.clone();
                if (entry.mesh->has_colors())
                    cloned->colors = entry.mesh->colors.clone();
                cloned->materials = entry.mesh->materials;
                cloned->submeshes = entry.mesh->submeshes;
                cloned->texture_images = entry.mesh->texture_images;
                scene_.addMesh(name, std::move(cloned));
            } else if (entry.data && entry.data->size() > 0) {
                auto paste_data = std::make_unique<lfs::core::SplatData>(
                    entry.data->get_max_sh_degree(),
                    entry.data->means_raw().clone(), entry.data->sh0_raw().clone(), entry.data->shN_raw().clone(),
                    entry.data->scaling_raw().clone(), entry.data->rotation_raw().clone(), entry.data->opacity_raw().clone(),
                    entry.data->get_scene_scale());
                paste_data->set_active_sh_degree(entry.data->get_active_sh_degree());

                scene_.addNode(name, std::move(paste_data));
            } else {
                --clipboard_counter_;
                continue;
            }

            selection_.invalidateNodeMask();

            static constexpr glm::mat4 IDENTITY{1.0f};
            if (entry.transform != IDENTITY) {
                scene_.setNodeTransform(name, entry.transform);
            }

            const auto* pasted_node = scene_.getNode(name);
            if (pasted_node && entry.hierarchy) {
                pasteNodeHierarchy(*entry.hierarchy, pasted_node->id);
            }

            state::PLYAdded{
                .name = name,
                .node_gaussians = pasted_node ? pasted_node->gaussian_count : 0,
                .total_gaussians = scene_.getTotalGaussianCount(),
                .is_visible = true,
                .parent_name = "",
                .is_group = false,
                .node_type = 0}
                .emit();

            if (pasted_node && pasted_node->type == core::NodeType::SPLAT) {
                const core::NodeId cropbox_id = scene_.getCropBoxForSplat(pasted_node->id);
                if (cropbox_id != core::NULL_NODE) {
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

        LOG_DEBUG("Pasted {} nodes", pasted_names.size());
        return pasted_names;
    }

    void SceneManager::setAppearanceModel(std::unique_ptr<lfs::training::PPISP> ppisp,
                                          std::unique_ptr<lfs::training::PPISPControllerPool> controller_pool) {
        appearance_ppisp_ = std::move(ppisp);
        appearance_controller_pool_ = std::move(controller_pool);
    }

    void SceneManager::clearAppearanceModel() {
        appearance_ppisp_.reset();
        appearance_controller_pool_.reset();
    }

    // --- Selection service and gaussian-level selection operations ---

    void SceneManager::initSelectionService() {
        if (selection_service_)
            return;
        auto* rm = services().renderingOrNull();
        if (!rm)
            return;
        selection_service_ = std::make_unique<SelectionService>(this, rm);
        python::set_selection_service(selection_service_.get());
    }

    void SceneManager::deleteSelectedGaussians() {
        auto selection = scene_.getSelectionMask();
        if (!selection || !selection->is_valid()) {
            LOG_INFO("No Gaussians selected to delete");
            return;
        }

        auto nodes = scene_.getVisibleNodes();
        if (nodes.empty())
            return;

        auto entry = std::make_unique<op::SceneSnapshot>(*this, "edit.delete");
        entry->captureTopology();
        entry->captureSelection();

        size_t offset = 0;
        bool any_deleted = false;

        for (const auto* node : nodes) {
            if (!node || !node->model)
                continue;

            const size_t node_size = node->model->size();
            if (node_size == 0)
                continue;

            auto node_selection = selection->slice(0, offset, offset + node_size);
            auto bool_mask = node_selection.to(lfs::core::DataType::Bool);
            node->model->soft_delete(bool_mask);

            any_deleted = true;
            offset += node_size;
        }

        if (any_deleted) {
            LOG_INFO("Deleted selected Gaussians");
            scene_.markDirty();
            scene_.clearSelection();

            entry->captureAfter();
            op::undoHistory().push(std::move(entry));

            if (auto* rm = services().renderingOrNull())
                rm->markDirty();
        }
    }

    void SceneManager::invertSelection() {
        const size_t total = scene_.getTotalGaussianCount();
        if (total == 0)
            return;

        auto entry = std::make_unique<op::SceneSnapshot>(*this, "select.invert");
        entry->captureSelection();

        const auto old_mask = scene_.getSelectionMask();
        const auto ones = lfs::core::Tensor::ones({total}, lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
        auto new_mask = std::make_shared<lfs::core::Tensor>(
            (old_mask && old_mask->is_valid()) ? ones - *old_mask : ones);

        scene_.setSelectionMask(new_mask);

        entry->captureAfter();
        op::undoHistory().push(std::move(entry));

        if (auto* rm = services().renderingOrNull())
            rm->markDirty();
    }

    void SceneManager::deselectAllGaussians() {
        if (!scene_.hasSelection())
            return;

        auto entry = std::make_unique<op::SceneSnapshot>(*this, "select.none");
        entry->captureSelection();

        scene_.clearSelection();

        entry->captureAfter();
        op::undoHistory().push(std::move(entry));

        if (auto* rm = services().renderingOrNull())
            rm->markDirty();
    }

    void SceneManager::selectAllGaussians() {
        auto* editor = services().editorOrNull();
        const auto tool = editor ? editor->getActiveTool() : ToolType::None;
        const bool is_selection_tool = (tool == ToolType::Selection || tool == ToolType::Brush);

        if (is_selection_tool) {
            const size_t total = scene_.getTotalGaussianCount();
            if (total == 0)
                return;

            const auto& selected_name = getSelectedNodeName();
            if (selected_name.empty())
                return;

            const int node_index = scene_.getVisibleNodeIndex(selected_name);
            if (node_index < 0)
                return;

            const auto transform_indices = scene_.getTransformIndices();
            if (!transform_indices || transform_indices->numel() != total)
                return;

            auto entry = std::make_unique<op::SceneSnapshot>(*this, "select.all");
            entry->captureSelection();

            auto new_mask = std::make_shared<lfs::core::Tensor>(transform_indices->eq(node_index));
            scene_.setSelectionMask(new_mask);

            entry->captureAfter();
            op::undoHistory().push(std::move(entry));
        } else {
            const auto nodes = scene_.getNodes();
            std::vector<std::string> splat_names;
            splat_names.reserve(nodes.size());
            for (const auto* node : nodes) {
                if (node->type == core::NodeType::SPLAT)
                    splat_names.push_back(node->name);
            }
            if (!splat_names.empty())
                selectNodes(splat_names);
        }

        if (auto* rm = services().renderingOrNull())
            rm->markDirty();
    }

    void SceneManager::copySelectionToClipboard() {
        auto* editor = services().editorOrNull();
        const auto tool = editor ? editor->getActiveTool() : ToolType::None;
        const bool is_selection_tool = (tool == ToolType::Selection || tool == ToolType::Brush);

        if (is_selection_tool && scene_.hasSelection()) {
            copySelectedGaussians();
        } else {
            copySelectedNodes();
        }
    }

    void SceneManager::pasteSelectionFromClipboard() {
        const auto pasted = hasGaussianClipboard() ? pasteGaussians() : pasteNodes();
        if (pasted.empty())
            return;

        scene_.resetSelectionState();

        clearSelection();
        for (const auto& name : pasted)
            addToSelection(name);

        if (auto* rm = services().renderingOrNull())
            rm->markDirty();
    }

    void SceneManager::selectRect(float x0, float y0, float x1, float y1, const std::string& mode) {
        if (!selection_service_)
            return;

        SelectionMode sel_mode = SelectionMode::Replace;
        if (mode == "add")
            sel_mode = SelectionMode::Add;
        else if (mode == "remove")
            sel_mode = SelectionMode::Remove;

        (void)selection_service_->selectRect(x0, y0, x1, y1, sel_mode);
    }

    void SceneManager::applySelectionMask(const std::vector<uint8_t>& mask) {
        if (!selection_service_)
            return;

        (void)selection_service_->applyMask(mask, SelectionMode::Replace);
    }

} // namespace lfs::vis
