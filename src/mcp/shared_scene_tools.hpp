/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "core/parameters.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <utility>

namespace lfs::mcp {

    struct SharedSceneToolBackend {
        using LoadDatasetHandler =
            std::function<std::expected<void, std::string>(const std::filesystem::path&,
                                                           const core::param::TrainingParameters&)>;
        using PathHandler =
            std::function<std::expected<void, std::string>(const std::filesystem::path&)>;
        using SavePlyHandler =
            std::function<std::expected<void, std::string>(const std::filesystem::path&, bool include_provenance)>;
        using StartTrainingHandler =
            std::function<std::expected<void, std::string>(bool overwrite)>;
        // A size the capture source cannot satisfy fails as InvalidArgument.
        using RenderCaptureHandler =
            std::function<lfs::Result<std::string>(int width, int height, bool presented)>;
        using GaussianCountHandler =
            std::function<std::expected<int64_t, std::string>()>;
        using LastTrainingErrorHandler =
            std::function<std::optional<lfs::Error>()>;

        std::string runtime = "shared";
        std::string thread_affinity = "any";

        LoadDatasetHandler load_dataset;
        PathHandler load_checkpoint;
        SavePlyHandler save_ply;
        StartTrainingHandler start_training;
        RenderCaptureHandler render_capture;
        GaussianCountHandler gaussian_count;
        LastTrainingErrorHandler last_training_error;
    };

    // Largest accepted capture edge in pixels. Captures are resampled on the CPU and
    // PNG-encoded in memory, so this bounds the buffers (16384^2 RGBA is 1 GiB).
    inline constexpr int MAX_CAPTURE_DIMENSION = 16384;

    // Schema for an optional capture width or height of 1..MAX_CAPTURE_DIMENSION pixels.
    [[nodiscard]] inline nlohmann::json capture_size_schema(std::string description) {
        return nlohmann::json{{"type", "integer"},
                              {"minimum", 1},
                              {"maximum", MAX_CAPTURE_DIMENSION},
                              {"description", std::move(description)}};
    }

    LFS_MCP_API void register_shared_scene_tools(const SharedSceneToolBackend& backend);

} // namespace lfs::mcp
