/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/data_loading_service.hpp"
#include "core/logger.hpp"
#include "scene/scene_manager.hpp"
#include <algorithm>
#include <stdexcept>

namespace gs::visualizer {

    DataLoadingService::DataLoadingService(SceneManager* scene_manager)
        : scene_manager_(scene_manager) {
        setupEventHandlers();
    }

    DataLoadingService::~DataLoadingService() = default;

    void DataLoadingService::setupEventHandlers() {
        using namespace events;

        // Listen for file load commands
        cmd::LoadFile::when([this](const auto& cmd) {
            handleLoadFileCommand(cmd.is_dataset, cmd.path);
        });
    }

    void DataLoadingService::handleLoadFileCommand(bool is_dataset, const std::filesystem::path& path) {
        if (is_dataset) {
            loadDataset(path);
        } else {
            scene_manager_->changeContentType(SceneManager::ContentType::SplatFiles);
            // Determine file type and load appropriately
            if (isSOGFile(path) || isPLYFile(path)) {
                std::string ply_name = path.stem().string();
                scene_manager_->addSplatFile(path, ply_name);
                scene_manager_->getProject()->addPly(true, path, -1, ply_name);
            } else {
                // Let scene manager determine the type
                scene_manager_->addSplatFile(path);
            }
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

    std::expected<void, std::string> DataLoadingService::loadPLY(const std::filesystem::path& path) {
        LOG_TIMER("LoadPLY");

        try {
            LOG_INFO("Loading PLY file: {}", path.string());

            // Load through scene manager
            scene_manager_->loadSplatFile(path);

            LOG_INFO("Successfully loaded PLY: {} (from: {})",
                     path.filename().string(),
                     path.parent_path().string());

            return {};
        } catch (const std::exception& e) {
            std::string error_msg = std::format("Failed to load PLY: {}", e.what());
            LOG_ERROR("{} (Path: {})", error_msg, path.string());
            throw std::runtime_error(error_msg);
        }
    }

    std::expected<void, std::string> DataLoadingService::loadSOG(const std::filesystem::path& path) {
        LOG_TIMER("LoadSOG");

        try {
            LOG_INFO("Loading SOG file: {}", path.string());

            // Load through scene manager
            scene_manager_->loadSplatFile(path);

            LOG_INFO("Successfully loaded SOG: {} (from: {})",
                     path.filename().string(),
                     path.parent_path().string());

            return {};
        } catch (const std::exception& e) {
            std::string error_msg = std::format("Failed to load SOG: {}", e.what());
            LOG_ERROR("{} (Path: {})", error_msg, path.string());
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
                LOG_INFO("Loading splat file: {}", path.string());
                scene_manager_->loadSplatFile(path);

                LOG_INFO("Successfully loaded splat file: {}", path.filename().string());
                return {};
            }
        } catch (const std::exception& e) {
            std::string error_msg = std::format("Failed to load splat file: {}", e.what());
            LOG_ERROR("{} (Path: {})", error_msg, path.string());
            throw std::runtime_error(error_msg);
        }
    }

    void DataLoadingService::addPLYToScene(const std::filesystem::path& path) {
        LOG_TIMER_TRACE("AddPLYToScene");

        try {
            LOG_DEBUG("Adding PLY to scene: {}", path.string());

            // Extract name from path
            std::string name = path.stem().string();
            LOG_TRACE("Extracted PLY name: {}", name);

            // Add through scene manager
            scene_manager_->addSplatFile(path, name);

            LOG_INFO("Added PLY '{}' to scene", name);

        } catch (const std::exception& e) {
            std::string error_msg = std::format("Failed to add PLY: {}", e.what());
            LOG_ERROR("{} (Path: {})", error_msg, path.string());
            throw std::runtime_error(error_msg);
        }
    }

    void DataLoadingService::addSOGToScene(const std::filesystem::path& path) {
        LOG_TIMER_TRACE("AddSOGToScene");

        try {
            LOG_DEBUG("Adding SOG to scene: {}", path.string());

            // Extract name from path
            std::string name = path.stem().string();
            LOG_TRACE("Extracted SOG name: {}", name);

            // Add through scene manager
            scene_manager_->addSplatFile(path, name);

            LOG_INFO("Added SOG '{}' to scene", name);

        } catch (const std::exception& e) {
            std::string error_msg = std::format("Failed to add SOG: {}", e.what());
            LOG_ERROR("{} (Path: {})", error_msg, path.string());
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
            std::string name = path.stem().string();
            scene_manager_->addSplatFile(path, name);
        }
    }

    std::expected<void, std::string> DataLoadingService::loadDataset(const std::filesystem::path& path) {
        LOG_TIMER("LoadDataset");

        try {
            LOG_INFO("Loading dataset from: {}", path.string());

            // Validate parameters
            if (params_.dataset.data_path.empty() && path.empty()) {
                LOG_ERROR("No dataset path specified");
                throw std::runtime_error("No dataset path specified");
            }

            // Load through scene manager
            LOG_DEBUG("Passing dataset to scene manager with parameters");
            scene_manager_->loadDataset(path, params_);

            return {};
        } catch (const std::exception& e) {
            std::string error_msg = std::format("Failed to load dataset: {}", e.what());
            LOG_ERROR("{} (Path: {})", error_msg, path.string());
            throw std::runtime_error(error_msg);
        }
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

} // namespace gs::visualizer