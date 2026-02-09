/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/events.hpp"
#include "core/export.hpp"
#include "core/parameters.hpp"
#include "core/scene.hpp"
#include "core/services.hpp"
#include "core/splat_data_mirror.hpp"
#include "geometry/bounding_box.hpp"
#include "io/loader.hpp"
#include "scene/scene_render_state.hpp"
#include "training/components/ppisp.hpp"
#include "training/components/ppisp_controller_pool.hpp"
#include <filesystem>
#include <mutex>
#include <set>

namespace lfs::vis {

    // Forward declarations
    class Trainer;

    class LFS_VIS_API SceneManager {
    public:
        // Content type - what's loaded, not execution state
        enum class ContentType {
            Empty,
            SplatFiles, // Changed from PLYFiles to be more generic
            Dataset
        };

        SceneManager();
        ~SceneManager();

        // Delete copy operations
        SceneManager(const SceneManager&) = delete;
        SceneManager& operator=(const SceneManager&) = delete;

        // Content queries - direct, no events
        ContentType getContentType() const {
            std::lock_guard<std::mutex> lock(state_mutex_);
            return content_type_;
        }
        bool isEmpty() const {
            std::lock_guard<std::mutex> lock(state_mutex_);
            return content_type_ == ContentType::Empty && !scene_.hasNodes();
        }

        bool hasSplatFiles() const {
            std::lock_guard<std::mutex> lock(state_mutex_);
            return content_type_ == ContentType::SplatFiles;
        }

        // Legacy compatibility
        bool hasPLYFiles() const { return hasSplatFiles(); }

        bool hasDataset() const {
            std::lock_guard<std::mutex> lock(state_mutex_);
            return content_type_ == ContentType::Dataset;
        }

        // Path accessors
        std::vector<std::filesystem::path> getSplatPaths() const {
            std::lock_guard<std::mutex> lock(state_mutex_);

            std::vector<std::filesystem::path> values;
            values.reserve(splat_paths_.size());

            for (const auto& [key, value] : splat_paths_) {
                values.push_back(value);
            }

            return values;
        }

        // Legacy compatibility
        std::vector<std::filesystem::path> getPLYPaths() const { return getSplatPaths(); }

        std::filesystem::path getDatasetPath() const {
            std::lock_guard<std::mutex> lock(state_mutex_);
            return dataset_path_;
        }

        // Scene access
        core::Scene& getScene() { return scene_; }
        const core::Scene& getScene() const { return scene_; }

        // Service accessors (via service locator)
        TrainerManager* getTrainerManager() { return services().trainerOrNull(); }
        const TrainerManager* getTrainerManager() const { return services().trainerOrNull(); }
        RenderingManager* getRenderingManager() { return services().renderingOrNull(); }

        void changeContentType(const ContentType& type);

        // Operations - Generic splat file loading
        void loadSplatFile(const std::filesystem::path& path);
        std::string addSplatFile(const std::filesystem::path& path, const std::string& name = "", bool is_visible = true);
        size_t consolidateNodeModels();

        void removePLY(const std::string& name, bool keep_children = false);
        void setPLYVisibility(const std::string& name, bool visible);

        // Node selection
        void selectNode(const std::string& name);
        void selectNodes(const std::vector<std::string>& names);
        void addToSelection(const std::string& name);
        void clearSelection();
        [[nodiscard]] std::string getSelectedNodeName() const;
        [[nodiscard]] std::vector<std::string> getSelectedNodeNames() const;
        [[nodiscard]] bool hasSelectedNode() const;
        [[nodiscard]] core::NodeType getSelectedNodeType() const;
        [[nodiscard]] int getSelectedNodeIndex() const;
        [[nodiscard]] std::vector<bool> getSelectedNodeMask() const;
        [[nodiscard]] int getSelectedCameraUid() const;
        void ensureCropBoxForSelectedNode();
        void selectCropBoxForCurrentNode();

        // Node picking
        [[nodiscard]] std::string pickNodeAtWorldPosition(const glm::vec3& world_pos) const;
        [[nodiscard]] std::vector<std::string> pickNodesInScreenRect(
            const glm::vec2& rect_min, const glm::vec2& rect_max,
            const glm::mat4& view, const glm::mat4& proj,
            const glm::ivec2& viewport_size) const;

        // Node transforms
        void setNodeTransform(const std::string& name, const glm::mat4& transform);
        glm::mat4 getNodeTransform(const std::string& name) const;
        void setSelectedNodeTranslation(const glm::vec3& translation);
        glm::vec3 getSelectedNodeTranslation() const;
        glm::vec3 getSelectedNodeCentroid() const;
        glm::vec3 getSelectedNodeCenter() const;

        // Full transform for selected node (includes rotation and scale)
        void setSelectedNodeTransform(const glm::mat4& transform);
        glm::mat4 getSelectedNodeTransform() const;      // Returns local transform
        glm::mat4 getSelectedNodeWorldTransform() const; // Returns world transform

        // Multi-selection support
        [[nodiscard]] glm::vec3 getSelectionCenter() const;
        [[nodiscard]] glm::vec3 getSelectionWorldCenter() const;

        // Cropbox operations for selected node
        core::NodeId getSelectedNodeCropBoxId() const;
        core::CropBoxData* getSelectedNodeCropBox();
        const core::CropBoxData* getSelectedNodeCropBox() const;
        void syncCropBoxToRenderSettings();

        // Ellipsoid operations for selected node
        void ensureEllipsoidForSelectedNode();
        void selectEllipsoidForCurrentNode();
        core::NodeId getSelectedNodeEllipsoidId() const;
        core::EllipsoidData* getSelectedNodeEllipsoid();
        const core::EllipsoidData* getSelectedNodeEllipsoid() const;
        void syncEllipsoidToRenderSettings();

        void loadDataset(const std::filesystem::path& path,
                         const lfs::core::param::TrainingParameters& params);

        // Import COLMAP cameras only (no images required)
        // Loads cameras from sparse folder and displays frustums without needing image files
        void loadColmapCamerasOnly(const std::filesystem::path& sparse_path);

        // Apply pre-loaded dataset to scene (for async loading)
        // The LoadResult comes from background thread, scene modification happens on main thread
        std::expected<void, std::string> applyLoadedDataset(
            const std::filesystem::path& path,
            const lfs::core::param::TrainingParameters& params,
            lfs::io::LoadResult&& load_result);

        void loadCheckpointForTraining(const std::filesystem::path& path,
                                       const lfs::core::param::TrainingParameters& params);
        void clear();
        void switchToEditMode(); // Keep trained model, discard dataset

        // For rendering - gets appropriate model
        const lfs::core::SplatData* getModelForRendering() const;

        // Build complete render state from scene graph
        // This is the single source of truth for all rendering data
        SceneRenderState buildRenderState() const;

        // Direct info queries
        struct SceneInfo {
            bool has_model = false;
            size_t num_gaussians = 0;
            size_t num_nodes = 0;
            std::string source_type;
            std::filesystem::path source_path;
        };

        SceneInfo getSceneInfo() const;

        bool renamePLY(const std::string& old_name, const std::string& new_name);
        void updatePlyPath(const std::string& ply_name, const std::filesystem::path& ply_path);

        // Permanently remove soft-deleted gaussians from all nodes
        size_t applyDeleted();

        // Clipboard - node-level copy/paste
        bool copySelectedNodes();
        std::vector<std::string> pasteNodes();
        [[nodiscard]] bool hasClipboard() const { return !clipboard_.empty(); }

        // Gaussian-level copy/paste (for selection tools)
        bool copySelectedGaussians();
        std::vector<std::string> pasteGaussians();
        [[nodiscard]] bool hasGaussianClipboard() const { return gaussian_clipboard_ != nullptr; }

        /// Mirror selected gaussians along specified axis
        bool executeMirror(lfs::core::MirrorAxis axis);

        // Appearance model (PPISP) - owned by scene_manager, not scene
        void setAppearanceModel(std::unique_ptr<lfs::training::PPISP> ppisp,
                                std::unique_ptr<lfs::training::PPISPControllerPool> controller_pool = nullptr);
        void clearAppearanceModel();
        [[nodiscard]] lfs::training::PPISP* getAppearancePPISP() { return appearance_ppisp_.get(); }
        [[nodiscard]] const lfs::training::PPISP* getAppearancePPISP() const { return appearance_ppisp_.get(); }
        [[nodiscard]] lfs::training::PPISPControllerPool* getAppearanceControllerPool() { return appearance_controller_pool_.get(); }
        [[nodiscard]] const lfs::training::PPISPControllerPool* getAppearanceControllerPool() const { return appearance_controller_pool_.get(); }
        [[nodiscard]] bool hasAppearanceController() const { return appearance_controller_pool_ != nullptr; }
        [[nodiscard]] bool hasAppearanceModel() const { return appearance_ppisp_ != nullptr; }

    private:
        void setupEventHandlers();
        void syncCropToolRenderSettings(const core::SceneNode* node);
        void loadPPISPCompanion(const std::filesystem::path& ppisp_path);
        void handleCropActivePly(const lfs::geometry::BoundingBox& crop_box, bool inverse);
        void handleCropByEllipsoid(const glm::mat4& world_transform, const glm::vec3& radii, bool inverse);
        void handleRenamePly(const lfs::core::events::cmd::RenamePLY& event);
        void handleReparentNode(const std::string& node_name, const std::string& new_parent_name);
        void handleAddGroup(const std::string& name, const std::string& parent_name);
        void handleDuplicateNode(const std::string& name);
        void handleMergeGroup(const std::string& name);
        void handleAddCropBox(const std::string& node_name);
        void handleAddCropEllipsoid(const std::string& node_name);
        void handleResetCropBox();
        void handleResetEllipsoid();
        void updateCropBoxToFitScene(bool use_percentile);
        void updateEllipsoidToFitScene(bool use_percentile);

        core::Scene scene_;
        mutable std::mutex state_mutex_;

        ContentType content_type_ = ContentType::Empty;
        // splat name to splat path
        std::map<std::string, std::filesystem::path> splat_paths_;
        std::filesystem::path dataset_path_;

        // Cache for parameters
        std::optional<lfs::core::param::TrainingParameters> cached_params_;

        std::set<std::string> selected_nodes_;

        // Clipboard for copy/paste (supports multi-selection)
        struct ClipboardEntry {
            std::unique_ptr<lfs::core::SplatData> data;
            glm::mat4 transform{1.0f};
            struct HierarchyNode {
                core::NodeType type = core::NodeType::SPLAT;
                glm::mat4 local_transform{1.0f};
                std::unique_ptr<core::CropBoxData> cropbox;
                std::vector<HierarchyNode> children;
            };
            std::optional<HierarchyNode> hierarchy;
        };
        std::vector<ClipboardEntry> clipboard_;
        int clipboard_counter_ = 0;

        // Gaussian-level clipboard (selected Gaussians only)
        std::unique_ptr<lfs::core::SplatData> gaussian_clipboard_;

        ClipboardEntry::HierarchyNode copyNodeHierarchy(const core::SceneNode* node);
        void pasteNodeHierarchy(const ClipboardEntry::HierarchyNode& src, core::NodeId parent_id);

        // Standalone appearance model (for viewing without training)
        std::unique_ptr<lfs::training::PPISP> appearance_ppisp_;
        std::unique_ptr<lfs::training::PPISPControllerPool> appearance_controller_pool_;
    };

} // namespace lfs::vis