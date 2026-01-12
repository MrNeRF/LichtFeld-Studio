/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_data_export.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/sogs.hpp"
#include "core/splat_data.hpp"
#include "io/exporter.hpp"

#include <filesystem>

namespace lfs::core {

    void save_ply(const SplatData& splat_data,
                  const std::filesystem::path& root,
                  const int iteration,
                  const bool join_threads,
                  std::string stem) {
        namespace fs = std::filesystem;
        fs::create_directories(root);

        const fs::path output_path = stem.empty()
                                         ? root / ("splat_" + std::to_string(iteration) + ".ply")
                                         : root / utf8_to_path(stem + ".ply"); // UTF-8 stem -> path

        const lfs::io::PlySaveOptions options{
            .output_path = output_path,
            .binary = true,
            .async = !join_threads};

        if (auto result = lfs::io::save_ply(splat_data, options); !result) {
            LOG_ERROR("Failed to save PLY to {}: {}",
                      path_to_utf8(output_path), result.error().message);
        }
    }

    std::filesystem::path save_sog(const SplatData& splat_data,
                                   const std::filesystem::path& root,
                                   const int iteration,
                                   const int kmeans_iterations,
                                   bool /* join_threads */) {
        namespace fs = std::filesystem;

        const fs::path sog_dir = root / "sog";
        fs::create_directories(sog_dir);

        const fs::path sog_out_path = sog_dir / ("splat_" + std::to_string(iteration) + "_sog.sog");

        const SogWriteOptions options{
            .iterations = kmeans_iterations,
            .output_path = sog_out_path};

        if (auto result = write_sog(splat_data, options); !result) {
            LOG_ERROR("Failed to save SOG to {}: {}",
                      path_to_utf8(sog_out_path), result.error());
        }
        return sog_out_path;
    }

    PointCloud to_point_cloud(const SplatData& splat_data) {
        return lfs::io::to_point_cloud(splat_data);
    }

    std::vector<std::string> get_attribute_names(const SplatData& splat_data) {
        return lfs::io::get_ply_attribute_names(splat_data);
    }

} // namespace lfs::core
