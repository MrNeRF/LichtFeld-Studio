/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/path_utils.hpp"
#include "training/components/ppisp.hpp"
#include "training/components/ppisp_controller_pool.hpp"
#include "training/components/ppisp_file.hpp"

#include <expected>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

namespace lfs::test {

    // Test-only generator for import-compatibility fixtures. Product code has
    // no standalone .ppisp writer.
    [[nodiscard]] inline std::expected<void, std::string>
    write_ppisp_fixture(
        const std::filesystem::path& path,
        const lfs::training::PPISP& ppisp,
        const lfs::training::PPISPControllerPool* controller_pool = nullptr,
        const lfs::training::PPISPFileMetadata* metadata = nullptr) {
        std::error_code error;
        if (!path.parent_path().empty()) {
            std::filesystem::create_directories(
                path.parent_path(), error);
        }
        if (error) {
            return std::unexpected(
                "Cannot create PPISP fixture directory: " +
                error.message());
        }

        std::ofstream output(
            path,
            std::ios::binary |
                std::ios::trunc);
        if (!output) {
            return std::unexpected(
                "Cannot open PPISP fixture: " +
                lfs::core::path_to_utf8(path));
        }

        lfs::training::PPISPFileHeader header{};
        header.num_cameras =
            static_cast<std::uint32_t>(
                ppisp.num_cameras());
        header.num_frames =
            static_cast<std::uint32_t>(
                ppisp.num_frames());
        if (controller_pool) {
            header.flags |= static_cast<std::uint32_t>(
                lfs::training::PPISPFileFlags::
                    HAS_CONTROLLER);
        }
        if (metadata && !metadata->empty()) {
            header.flags |= static_cast<std::uint32_t>(
                lfs::training::PPISPFileFlags::
                    HAS_METADATA);
        }
        output.write(
            reinterpret_cast<const char*>(&header),
            sizeof(header));
        ppisp.serialize_inference(output);
        if (controller_pool) {
            controller_pool->serialize_inference(
                output);
        }
        if (metadata && !metadata->empty()) {
            const nlohmann::json json{
                {"dataset_path",
                 metadata->dataset_path_utf8},
                {"images_folder",
                 metadata->images_folder},
                {"frame_image_names",
                 metadata->frame_image_names},
                {"frame_camera_ids",
                 metadata->frame_camera_ids},
                {"camera_ids",
                 metadata->camera_ids},
            };
            const auto text = json.dump();
            const auto size =
                static_cast<std::uint64_t>(
                    text.size());
            output.write(
                reinterpret_cast<const char*>(&size),
                sizeof(size));
            output.write(
                text.data(),
                static_cast<std::streamsize>(
                    text.size()));
        }
        output.close();
        if (!output) {
            return std::unexpected(
                "Cannot finish PPISP fixture: " +
                lfs::core::path_to_utf8(path));
        }
        return {};
    }

} // namespace lfs::test
