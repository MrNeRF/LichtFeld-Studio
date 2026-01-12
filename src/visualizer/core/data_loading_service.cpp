/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/data_loading_service.hpp"
#include "core/logger.hpp"
#include "core/parameter_manager.hpp"
#include "core/path_utils.hpp"
#include "core/services.hpp"
#include "scene/scene_manager.hpp"
#include <algorithm>
#include <stdexcept>

namespace lfs::vis {

    DataLoadingService::DataLoadingService(SceneManager* scene_manager)
        : scene_manager_(scene_manager) {
        setupEventHandlers();
    }

    DataLoadingService::~DataLoadingService() = default;

    void DataLoadingService::setupEventHandlers() {
        using namespace lfs::core::events;

        // Listen for file load commands
        cmd::LoadFile::when([this](const auto& cmd) {
            handleLoadFileCommand(cmd.is_dataset, cmd.path);
        });

        // Listen for checkpoint load for training commands
        cmd::LoadCheckpointForTraining::when([this](const auto& cmd) {
            handleLoadCheckpointForTrainingCommand(cmd.checkpoint_path, cmd.dataset_path, cmd.output_path);
        });
    }

    void DataLoadingService::handleLoadFileCommand(const bool is_dataset, const std::filesystem::path& path) {
        if (is_dataset) {
            return; // Handled async by GuiManager
        }

        // Checkpoint files get special handling - redirect to training resume flow
        if (isCheckpointFile(path)) {
            handleLoadCheckpointForTrainingCommand(path, {}, {});
            return;
        }

        if (scene_manager_->getContentType() == SceneManager::ContentType::Dataset) {
            scene_manager_->clear();
        }
        scene_manager_->changeContentType(SceneManager::ContentType::SplatFiles);

        const std::string name = lfs::core::path_to_utf8(path.stem());
        scene_manager_->addSplatFile(path, name);
    }

    void DataLoadingService::handleLoadCheckpointForTrainingCommand(
        const std::filesystem::path& checkpoint_path,
        const std::filesystem::path& dataset_path,
        const std::filesystem::path& output_path) {
        LOG_INFO("Loading checkpoint for training: {}", lfs::core::path_to_utf8(checkpoint_path));
        if (auto result = loadCheckpointForTraining(checkpoint_path, dataset_path, output_path); !result) {
            LOG_ERROR("Failed to load checkpoint for training: {}", result.error());
        }
    }

    bool DataLoadingService::isSOGFile(const std::filesystem::path& path) const {
        // Check for .sog extension
        auto ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".sog") {
            return true;
        }

        // Check for SOG directory (with meta.json and WebP files)
        if (std::filesystem::is_directory(path)) {
            if (std::filesystem::exists(path / "meta.json")) {
                // Check for SOG-specific files
                if (std::filesystem::exists(path / "means_l.webp") ||
                    std::filesystem::exists(path / "means_u.webp") ||
                    std::filesystem::exists(path / "quats.webp") ||
                    std::filesystem::exists(path / "scales.webp") ||
                    std::filesystem::exists(path / "sh0.webp")) {
                    return true;
                }
            }
        }

        // Check if it's a meta.json file that's part of a SOG dataset
        if (path.filename() == "meta.json") {
            auto parent = path.parent_path();
            if (std::filesystem::exists(parent / "means_l.webp") ||
                std::filesystem::exists(parent / "means_u.webp") ||
                std::filesystem::exists(parent / "quats.webp") ||
                std::filesystem::exists(parent / "scales.webp") ||
                std::filesystem::exists(parent / "sh0.webp")) {
                return true;
            }
        }

        return false;
    }

    bool DataLoadingService::isPLYFile(const std::filesystem::path& path) const {
        auto ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        return ext == ".ply";
    }

    bool DataLoadingService::isCheckpointFile(const std::filesystem::path& path) const {
        auto ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        return ext == ".resume";
    }

    std::expected<void, std::string> DataLoadingService::loadPLY(const std::filesystem::path& path) {
        LOG_TIMER("LoadPLY");

        try {
            LOG_INFO("Loading PLY file: {}", lfs::core::path_to_utf8(path));

            // Load through scene manager
            scene_manager_->loadSplatFile(path);

            LOG_INFO("Successfully loaded PLY: {} (from: {})",
                     lfs::core::path_to_utf8(path.filename()),
                     lfs::core::path_to_utf8(path.parent_path()));

            return {};
        } catch (const std::exception& e) {
            std::string error_msg = std::format("Failed to load PLY: {}", e.what());
            LOG_ERROR("{} (Path: {})", error_msg, lfs::core::path_to_utf8(path));
            throw std::runtime_error(error_msg);
        }
    }

    std::expected<void, std::string> DataLoadingService::loadSOG(const std::filesystem::path& path) {
        LOG_TIMER("LoadSOG");

        try {
            LOG_INFO("Loading SOG file: {}", lfs::core::path_to_utf8(path));

            // Load through scene manager
            scene_manager_->loadSplatFile(path);

            LOG_INFO("Successfully loaded SOG: {} (from: {})",
                     lfs::core::path_to_utf8(path.filename()),
                     lfs::core::path_to_utf8(path.parent_path()));

            return {};
        } catch (const std::exception& e) {
            std::string error_msg = std::format("Failed to load SOG: {}", e.what());
            LOG_ERROR("{} (Path: {})", error_msg, lfs::core::path_to_utf8(path));
            throw std::runtime_error(error_msg);
        }
    }

    std::expected<void, std::string> DataLoadingService::loadSplatFile(const std::filesystem::path& path) {
        LOG_TIMER("LoadSplatFile");

        try {
            // Determine file type
            if (isSOGFile(path)) {
                return loadSOG(path);
            } else if (isPLYFile(path)) {
                return loadPLY(path);
            } else {
                // Let the scene manager figure it out with the generic loader
                LOG_INFO("Loading splat file: {}", lfs::core::path_to_utf8(path));
                scene_manager_->loadSplatFile(path);

                LOG_INFO("Successfully loaded splat file: {}", lfs::core::path_to_utf8(path.filename()));
                return {};
            }
        } catch (const std::exception& e) {
            std::string error_msg = std::format("Failed to load splat file: {}", e.what());
            LOG_ERROR("{} (Path: {})", error_msg, lfs::core::path_to_utf8(path));
            throw std::runtime_error(error_msg);
        }
    }

    void DataLoadingService::addPLYToScene(const std::filesystem::path& path) {
        LOG_TIMER_TRACE("AddPLYToScene");

        try {
            LOG_DEBUG("Adding PLY to scene: {}", lfs::core::path_to_utf8(path));

            // Extract name from path
            std::string name = lfs::core::path_to_utf8(path.stem());
            LOG_TRACE("Extracted PLY name: {}", name);

            // Add through scene manager
            scene_manager_->addSplatFile(path, name);

            LOG_INFO("Added PLY '{}' to scene", name);

        } catch (const std::exception& e) {
            std::string error_msg = std::format("Failed to add PLY: {}", e.what());
            LOG_ERROR("{} (Path: {})", error_msg, lfs::core::path_to_utf8(path));
            throw std::runtime_error(error_msg);
        }
    }

    void DataLoadingService::addSOGToScene(const std::filesystem::path& path) {
        LOG_TIMER_TRACE("AddSOGToScene");

        try {
            LOG_DEBUG("Adding SOG to scene: {}", lfs::core::path_to_utf8(path));

            // Extract name from path
            std::string name = lfs::core::path_to_utf8(path.stem());
            LOG_TRACE("Extracted SOG name: {}", name);

            // Add through scene manager
            scene_manager_->addSplatFile(path, name);

            LOG_INFO("Added SOG '{}' to scene", name);

        } catch (const std::exception& e) {
            std::string error_msg = std::format("Failed to add SOG: {}", e.what());
            LOG_ERROR("{} (Path: {})", error_msg, lfs::core::path_to_utf8(path));
            throw std::runtime_error(error_msg);
        }
    }

    void DataLoadingService::addSplatFileToScene(const std::filesystem::path& path) {
        if (isSOGFile(path)) {
            addSOGToScene(path);
        } else if (isPLYFile(path)) {
            addPLYToScene(path);
        } else {
            // Generic add
            std::string name = lfs::core::path_to_utf8(path.stem());
            scene_manager_->addSplatFile(path, name);
        }
    }

    std::expected<void, std::string> DataLoadingService::loadDataset(const std::filesystem::path& path) {
        LOG_TIMER("LoadDataset");

        LOG_INFO("Loading dataset from: {}", lfs::core::path_to_utf8(path));

        // Validate parameters
        if (params_.dataset.data_path.empty() && path.empty()) {
            LOG_ERROR("No dataset path specified");
            return std::unexpected("No dataset path specified");
        }

        // Load through scene manager (it emits DatasetLoadCompleted event on success/failure)
        LOG_DEBUG("Passing dataset to scene manager with parameters");
        scene_manager_->loadDataset(path, params_);

        return {};
    }

    void DataLoadingService::clearScene() {
        try {
            LOG_DEBUG("Clearing scene");
            scene_manager_->clear();
            LOG_INFO("Scene cleared");
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to clear scene: {}", e.what());
            throw std::runtime_error(std::format("Failed to clear scene: {}", e.what()));
        }
    }

    std::expected<void, std::string> DataLoadingService::loadCheckpointForTraining(
        const std::filesystem::path& checkpoint_path,
        const std::filesystem::path& dataset_path,
        const std::filesystem::path& output_path) {
        LOG_TIMER("LoadCheckpointForTraining");
        try {
            // Override dataset/output paths if provided by user
            lfs::core::param::TrainingParameters params;
            if (!dataset_path.empty()) {
                params.dataset.data_path = dataset_path;
            }
            if (!output_path.empty()) {
                params.dataset.output_path = output_path;
            }
            scene_manager_->loadCheckpointForTraining(checkpoint_path, params);
            return {};
        } catch (const std::exception& e) {
            const std::string error = std::format("Checkpoint load failed: {}", e.what());
            LOG_ERROR("{}", error);
            return std::unexpected(error);
        }
    }

} // namespace lfs::vis