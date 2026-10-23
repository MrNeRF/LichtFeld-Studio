/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "las_prior_sidecar.hpp"

#include "core/logger.hpp"
#include "core/path_utils.hpp"

#include <fstream>
#include <nlohmann/json.hpp>
#include <string_view>

namespace lfs::training {

    namespace {
        constexpr std::string_view kSidecarName = "supervision.json";

        [[nodiscard]] std::string string_field(const nlohmann::json& json, const char* key) {
            const auto it = json.find(key);
            return it != json.end() && it->is_string() ? it->get<std::string>() : std::string{};
        }
    } // namespace

    std::optional<LasPriorSidecar> read_las_prior_sidecar(const std::filesystem::path& dataset_path) {
        const auto path = dataset_path / kSidecarName;
        std::error_code ec;
        if (!lfs::core::native_fs::exists(path, ec)) {
            return std::nullopt;
        }
        std::ifstream file;
        if (!lfs::core::open_file_for_read(path, file)) {
            LOG_WARN("Could not open {}", lfs::core::path_to_utf8(path));
            return std::nullopt;
        }
        const auto json = nlohmann::json::parse(file, nullptr, false);
        if (json.is_discarded() || !json.is_object()) {
            LOG_WARN("Ignoring {}: not a JSON object", lfs::core::path_to_utf8(path));
            return std::nullopt;
        }
        LasPriorSidecar sidecar;
        sidecar.depth_millimetres = string_field(json, "depth_format") == "uint16_png_mm";
        sidecar.ray_distance = string_field(json, "depth_type") == "ray_distance";
        sidecar.normals_camera_opencv = string_field(json, "normal_frame").starts_with("opencv_camera");
        if (!sidecar.depth_millimetres && !sidecar.normals_camera_opencv) {
            return std::nullopt;
        }
        return sidecar;
    }

    std::vector<std::string> apply_las_prior_defaults(
        lfs::core::param::OptimizationParameters& opt,
        const LasPriorSidecar& sidecar) {
        using lfs::core::param::NormalLossSpace;
        std::vector<std::string> changes;
        if (sidecar.depth_millimetres) {
            if (opt.depth_loss_mode == "ssi") {
                opt.depth_loss_mode = "metric";
                changes.emplace_back("depth_loss_mode ssi -> metric");
            }
            if (opt.depth_auto_generate) {
                opt.depth_auto_generate = false;
                changes.emplace_back("depth_auto_generate off");
            }
        }
        if (sidecar.normals_camera_opencv) {
            if (opt.normal_loss_space == NormalLossSpace::Auto) {
                opt.normal_loss_space = NormalLossSpace::CameraOpenCV;
                changes.emplace_back("normal_loss_space auto -> camera-opencv");
            }
            if (opt.normal_auto_generate) {
                opt.normal_auto_generate = false;
                changes.emplace_back("normal_auto_generate off");
            }
        }
        return changes;
    }

} // namespace lfs::training
