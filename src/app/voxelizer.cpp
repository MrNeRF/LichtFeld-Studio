/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "app/voxelizer.hpp"
#include "app/converter_overwrite.hpp"
#include "app/headless_run_coordinator.hpp"
#include "app/terminal_progress_bar.hpp"
#include "core/logger.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include "io/atomic_output.hpp"
#include "io/collision_export.hpp"
#include <iostream>
#include <print>
namespace lfs::app {
    int run_voxelizer(const io::args::VoxelizeMode& options) {
        HeadlessRunCoordinator run;
        try {
            options.settings.validate();
            auto binary = options.output_path;
            binary.replace_extension(".bin");
            std::filesystem::path existing;
            for (const auto& target : {options.output_path, binary}) {
                std::error_code ec;
                if (target == options.input_path || core::native_fs::equivalent(target, options.input_path, ec))
                    throw std::invalid_argument("Voxel output must not replace the input splat file");
                if (core::native_fs::exists(target)) {
                    if (!core::native_fs::is_regular_file(target))
                        throw std::invalid_argument("Not a regular output file: " + core::path_to_utf8(target));
                    existing = target;
                }
            }
            if (!options.overwrite && !existing.empty()) {
                std::println("Both {} and {} will be replaced.", core::path_to_utf8(options.output_path),
                             core::path_to_utf8(binary));
                if (ask_overwrite(existing, std::cin, std::cout) == OverwriteChoice::NO)
                    return 0;
            }
            TerminalProgressBar bar("Voxelize");
            const auto stop = run.stop_token();
            io::ExportProgressCallback progress = [&](float p, const std::string& stage) {
                bar.report(p, stage);
                return !stop.stop_requested();
            };
            auto source = io::loadCollisionSource(options.input_path, stop, io::scale_export_progress(progress, 0, .2f));
            auto volume = core::generateCollision(source, options.settings, stop, io::scale_export_progress(progress, .2f, .8f), {}, io::checkCollisionExportMixedBlocks);
            source = {};
            io::exportCollision(volume, options.output_path, glm::mat4(1), stop, io::scale_export_progress(progress, .8f, 1));
            bar.complete();
            std::println("Saved {} occupied voxels, cell size {} scene units: {}", volume.occupied_count,
                         volume.settings.voxel_size, core::path_to_utf8(options.output_path));
            return 0;
        } catch (const std::exception& error) {
            LOG_ERROR("Voxelization failed: {}", error.what());
            std::println(stderr, "Voxelization failed: {}", error.what());
            return run.interrupted() ? run.interrupted_exit_code() : 1;
        }
    }
} // namespace lfs::app
