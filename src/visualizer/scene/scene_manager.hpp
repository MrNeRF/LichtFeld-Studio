/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/events.hpp"
#include "core/parameters.hpp"
#include "scene/scene.hpp"
#include <filesystem>
#include <mutex>
#include <project/project.hpp>

namespace gs {

    // Forward declarations
    class Trainer;
    class TrainerManager;
    class SplatData;

    namespace visualizer {
        class RenderingManager;
    }

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

        // Trainer manager link
        void setTrainerManager(TrainerManager* tm) { trainer_manager_ = tm; }
        TrainerManager* getTrainerManager() { return trainer_manager_; }
        const TrainerManager* getTrainerManager() const { return trainer_manager_; }

        // Rendering manager link
        void setRenderingManager(visualizer::RenderingManager* rm);
        visualizer::RenderingManager* getRenderingManager() { return rendering_manager_; }

        void changeContentType(const ContentType& type);

        // Operations - Generic splat file loading
        void loadSplatFile(const std::filesystem::path& path);
        void addSplatFile(const std::filesystem::path& path, const std::string& name = "", bool is_visible = true);

        void removePLY(const std::string& name);
        void setPLYVisibility(const std::string& name, bool visible);

        void loadDataset(const std::filesystem::path& path,
                         const param::TrainingParameters& params);
        void clear();

        // For rendering - gets appropriate model
        const SplatData* getModelForRendering() const;

        // Direct info queries
        struct SceneInfo {
            bool has_model = false;
            size_t num_gaussians = 0;
            size_t num_nodes = 0;
            std::string source_type;
            std::filesystem::path source_path;
        };

        SceneInfo getSceneInfo() const;

        void setProject(std::shared_ptr<gs::management::Project> project) { lfs_project_ = project; }

        [[nodiscard]] std::shared_ptr<gs::management::Project> getProject() const { return lfs_project_; }

        bool renamePLY(const std::string& old_name, const std::string& new_name);
        void updatePlyPath(const std::string& ply_name, const std::filesystem::path& ply_path);

    private:
        void setupEventHandlers();
        void emitSceneChanged();
        void handleCropActivePly(const gs::geometry::BoundingBox& crop_box);
        void handleRenamePly(const events::cmd::RenamePLY& event);

        Scene scene_;
        mutable std::mutex state_mutex_;

        ContentType content_type_ = ContentType::Empty;
        // splat name to splat path
        std::map<std::string, std::filesystem::path> splat_paths_;
        std::filesystem::path dataset_path_;

        // Training support
        TrainerManager* trainer_manager_ = nullptr;

        // Rendering support
        visualizer::RenderingManager* rendering_manager_ = nullptr;

        // Cache for parameters
        std::optional<param::TrainingParameters> cached_params_;
        // project
        std::shared_ptr<gs::management::Project> lfs_project_ = nullptr;
    };

} // namespace gs