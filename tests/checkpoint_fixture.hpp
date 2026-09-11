/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/path_utils.hpp"
#include "training/checkpoint.hpp"

#include <expected>
#include <filesystem>
#include <fstream>
#include <string>

namespace lfs::test {

    [[nodiscard]] inline std::filesystem::path
    checkpoint_fixture_path(const std::filesystem::path& output_directory) {
        return output_directory / "checkpoint.resume";
    }

    // Test-only legacy fixture generator. Product code embeds the same LFKP
    // stream in CKPT and has no standalone checkpoint writer.
    [[nodiscard]] inline std::expected<void, std::string>
    write_checkpoint_fixture(
        const std::filesystem::path& output_directory,
        const int iteration,
        const lfs::training::IStrategy& strategy,
        const lfs::core::param::TrainingParameters& params,
        const lfs::training::BilateralGrid* bilateral_grid,
        const lfs::training::PPISP* ppisp,
        const lfs::training::PPISPControllerPool* controller_pool,
        const lfs::training::ADMMSparsityOptimizer* sparsity_optimizer) {
        const auto path = checkpoint_fixture_path(output_directory);
        std::error_code error;
        std::filesystem::create_directories(
            path.parent_path(), error);
        if (error) {
            return std::unexpected(
                "Cannot create checkpoint fixture directory: " +
                error.message());
        }

        std::ofstream output(
            path,
            std::ios::binary |
                std::ios::trunc);
        if (!output) {
            return std::unexpected(
                "Cannot open checkpoint fixture: " +
                lfs::core::path_to_utf8(path));
        }
        auto serialized =
            lfs::training::serialize_checkpoint(
                output, iteration, strategy, params,
                bilateral_grid, ppisp,
                controller_pool,
                sparsity_optimizer);
        if (!serialized) {
            return std::unexpected(
                lfs::format_for_developer(
                    serialized.error()));
        }
        output.close();
        if (!output) {
            return std::unexpected(
                "Cannot finish checkpoint fixture: " +
                lfs::core::path_to_utf8(path));
        }
        return {};
    }

} // namespace lfs::test
