/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/parameters.hpp"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>

namespace lfs::mcp {

    struct SharedSceneToolBackend {
        using LoadDatasetHandler =
            std::function<std::expected<void, std::string>(const std::filesystem::path&,
                                                           const core::param::TrainingParameters&)>;
        using PathHandler =
            std::function<std::expected<void, std::string>(const std::filesystem::path&)>;
        using SaveCheckpointHandler =
            std::function<std::expected<std::filesystem::path, std::string>(const std::optional<std::filesystem::path>&)>;
        using StartTrainingHandler =
            std::function<std::expected<void, std::string>()>;
        using RenderCaptureHandler =
            std::function<std::expected<std::string, std::string>(int camera_index, int width, int height)>;
        using GaussianCountHandler =
            std::function<std::expected<int64_t, std::string>()>;

        std::string runtime = "shared";
        std::string thread_affinity = "any";

        LoadDatasetHandler load_dataset;
        PathHandler load_checkpoint;
        SaveCheckpointHandler save_checkpoint;
        PathHandler save_ply;
        StartTrainingHandler start_training;
        RenderCaptureHandler render_capture;
        GaussianCountHandler gaussian_count;
    };

    LFS_MCP_API void register_shared_scene_tools(const SharedSceneToolBackend& backend);

} // namespace lfs::mcp
