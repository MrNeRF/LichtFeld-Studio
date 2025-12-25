/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/events.hpp"
#include "core/parameters.hpp"
#include "core/services.hpp"
#include "core/splat_data_mirror.hpp"
#include "geometry/bounding_box.hpp"
#include "scene/scene.hpp"
#include "scene/scene_render_state.hpp"
#include <filesystem>
#include <mutex>
#include <set>

namespace lfs::vis {

    // Forward declarations
    class Trainer;

    class SceneManager {
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
            return content_type_ == ContentType::Empty;
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
        Scene& getScene() { return scene_; }
        const Scene& getScene() const { return scene_; }

        // Service accessors (via service locator)
        TrainerManager* getTrainerManager() { return services().trainerOrNull(); }
        const TrainerManager* getTrainerManager() const { return services().trainerOrNull(); }
        RenderingManager* getRenderingManager() { return services().renderingOrNull(); }
        command::CommandHistory* getCommandHistory() { return services().commandsOrNull(); }

        void changeContentType(const ContentType& type);

        // Operations - Generic splat file loading
        void loadSplatFile(const std::filesystem::path& path);
        std::string addSplatFile(const std::filesystem::path& path, const std::string& name = "", bool is_visible = true);

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
        [[nodiscard]] NodeType getSelectedNodeType() const;
        [[nodiscard]] int getSelectedNodeIndex() const;
        [[nodiscard]] std::vector<bool> getSelectedNodeMask() const;
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
        NodeId getSelectedNodeCropBoxId() const;
        CropBoxData* getSelectedNodeCropBox();
        const CropBoxData* getSelectedNodeCropBox() const;
        void syncCropBoxToRenderSettings(); // Sync selected node's cropbox to render settings

        void loadDataset(const std::filesystem::path& path,
                         const lfs::core::param::TrainingParameters& params);
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

    private:
        void setupEventHandlers();
        void emitSceneChanged();
        void handleCropActivePly(const lfs::geometry::BoundingBox& crop_box, bool inverse);
        void handleRenamePly(const lfs::core::events::cmd::RenamePLY& event);
        void handleReparentNode(const std::string& node_name, const std::string& new_parent_name);
        void handleAddGroup(const std::string& name, const std::string& parent_name);
        void handleDuplicateNode(const std::string& name);
        void handleMergeGroup(const std::string& name);
        void updateCropBoxToFitScene(bool use_percentile);

        Scene scene_;
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
                NodeType type = NodeType::SPLAT;
                glm::mat4 local_transform{1.0f};
                std::unique_ptr<CropBoxData> cropbox;
                std::vector<HierarchyNode> children;
            };
            std::optional<HierarchyNode> hierarchy;
        };
        std::vector<ClipboardEntry> clipboard_;
        int clipboard_counter_ = 0;

        // Gaussian-level clipboard (selected Gaussians only)
        std::unique_ptr<lfs::core::SplatData> gaussian_clipboard_;

        ClipboardEntry::HierarchyNode copyNodeHierarchy(const SceneNode* node);
        void pasteNodeHierarchy(const ClipboardEntry::HierarchyNode& src, NodeId parent_id);
    };

} // namespace lfs::vis