/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "ppisp_file.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "ppisp.hpp"
#include "ppisp_controller_pool.hpp"
#include <fstream>
#include <nlohmann/json.hpp>

namespace lfs::training {

    namespace {
        nlohmann::json metadata_to_json(const PPISPFileMetadata& metadata) {
            nlohmann::json json;
            json["dataset_path"] = metadata.dataset_path_utf8;
            json["images_folder"] = metadata.images_folder;
            json["frame_image_names"] = metadata.frame_image_names;
            json["frame_camera_ids"] = metadata.frame_camera_ids;
            json["camera_ids"] = metadata.camera_ids;
            return json;
        }

        std::expected<PPISPFileMetadata, std::string> metadata_from_json(const nlohmann::json& json) {
            PPISPFileMetadata metadata;
            try {
                if (json.contains("dataset_path")) {
                    metadata.dataset_path_utf8 = json["dataset_path"].get<std::string>();
                }
                if (json.contains("images_folder")) {
                    metadata.images_folder = json["images_folder"].get<std::string>();
                }
                if (json.contains("frame_image_names")) {
                    metadata.frame_image_names = json["frame_image_names"].get<std::vector<std::string>>();
                }
                if (json.contains("frame_camera_ids")) {
                    metadata.frame_camera_ids = json["frame_camera_ids"].get<std::vector<int>>();
                }
                if (json.contains("camera_ids")) {
                    metadata.camera_ids = json["camera_ids"].get<std::vector<int>>();
                }
            } catch (const std::exception& e) {
                return std::unexpected(std::string("Failed to parse PPISP metadata: ") + e.what());
            }
            return metadata;
        }

        std::expected<void, std::string> write_metadata_block(std::ostream& file, const PPISPFileMetadata& metadata) {
            const std::string json = metadata_to_json(metadata).dump();
            const uint64_t size = static_cast<uint64_t>(json.size());
            file.write(reinterpret_cast<const char*>(&size), sizeof(size));
            file.write(json.data(), static_cast<std::streamsize>(json.size()));
            if (!file) {
                return std::unexpected("Failed to write PPISP metadata block");
            }
            return {};
        }

        std::expected<PPISPFileMetadata, std::string> read_metadata_block(std::istream& file) {
            uint64_t size = 0;
            file.read(reinterpret_cast<char*>(&size), sizeof(size));
            if (!file) {
                return std::unexpected("Failed to read PPISP metadata size");
            }

            std::string json(size, '\0');
            file.read(json.data(), static_cast<std::streamsize>(size));
            if (!file) {
                return std::unexpected("Failed to read PPISP metadata payload");
            }

            try {
                return metadata_from_json(nlohmann::json::parse(json));
            } catch (const std::exception& e) {
                return std::unexpected(std::string("Failed to parse PPISP metadata JSON: ") + e.what());
            }
        }
    } // namespace

    std::expected<void, std::string> save_ppisp_file(
        const std::filesystem::path& path,
        const PPISP& ppisp,
        const PPISPControllerPool* controller_pool,
        const PPISPFileMetadata* metadata) {

        try {
            std::ofstream file;
            if (!lfs::core::open_file_for_write(path, std::ios::binary, file)) {
                return std::unexpected("Failed to open file for writing: " + lfs::core::path_to_utf8(path));
            }

            PPISPFileHeader header{};
            header.num_cameras = static_cast<uint32_t>(ppisp.num_cameras());
            header.num_frames = static_cast<uint32_t>(ppisp.num_frames());
            header.flags = 0;
            if (controller_pool) {
                header.flags |= static_cast<uint32_t>(PPISPFileFlags::HAS_CONTROLLER);
            }
            if (metadata && !metadata->empty()) {
                header.flags |= static_cast<uint32_t>(PPISPFileFlags::HAS_METADATA);
            }

            file.write(reinterpret_cast<const char*>(&header), sizeof(header));

            ppisp.serialize_inference(file);

            // Save controller pool
            if (controller_pool) {
                controller_pool->serialize_inference(file);
            }

            if (metadata && !metadata->empty()) {
                if (auto result = write_metadata_block(file, *metadata); !result) {
                    return result;
                }
            }

            LOG_INFO("PPISP file saved: {} ({} cameras, {} frames{}{})",
                     lfs::core::path_to_utf8(path),
                     header.num_cameras,
                     header.num_frames,
                     controller_pool
                         ? ", +controller_pool(" + std::to_string(controller_pool->num_cameras()) + ")"
                         : "",
                     (metadata && !metadata->empty()) ? ", +metadata" : "");

            return {};

        } catch (const std::exception& e) {
            return std::unexpected(std::string("Failed to save PPISP file: ") + e.what());
        }
    }

    std::expected<void, std::string> load_ppisp_file(
        const std::filesystem::path& path,
        PPISP& ppisp,
        PPISPControllerPool* controller_pool,
        PPISPFileMetadata* metadata) {

        try {
            std::ifstream file;
            if (!lfs::core::open_file_for_read(path, std::ios::binary, file)) {
                return std::unexpected("Failed to open file for reading: " + lfs::core::path_to_utf8(path));
            }

            PPISPFileHeader header{};
            file.read(reinterpret_cast<char*>(&header), sizeof(header));

            if (header.magic != PPISP_FILE_MAGIC) {
                return std::unexpected("Invalid PPISP file: wrong magic number");
            }

            if (header.version > PPISP_FILE_VERSION) {
                return std::unexpected("Unsupported PPISP file version: " + std::to_string(header.version));
            }
            if (metadata) {
                *metadata = {};
            }

            const bool is_inference_load = ppisp.num_cameras() == 0 && ppisp.num_frames() == 0;
            if (!is_inference_load &&
                (static_cast<int>(header.num_cameras) != ppisp.num_cameras() ||
                 static_cast<int>(header.num_frames) != ppisp.num_frames())) {
                return std::unexpected(
                    "PPISP dimension mismatch: file has " +
                    std::to_string(header.num_cameras) + " cameras, " +
                    std::to_string(header.num_frames) + " frames; expected " +
                    std::to_string(ppisp.num_cameras()) + " cameras, " +
                    std::to_string(ppisp.num_frames()) + " frames");
            }

            ppisp.deserialize_inference(file);

            if (has_flag(header.flags, PPISPFileFlags::HAS_CONTROLLER)) {
                if (controller_pool) {
                    controller_pool->deserialize_inference(file);
                    LOG_INFO("PPISP file loaded: {} ({} cameras, {} frames, +controller_pool({}))",
                             lfs::core::path_to_utf8(path), header.num_cameras, header.num_frames,
                             controller_pool->num_cameras());
                } else {
                    LOG_DEBUG("PPISP file has controller pool but none provided - skipping controller data");
                    // Skip controller pool data by reading into a temporary
                    constexpr uint32_t INFERENCE_MAGIC = 0x4C464349;
                    uint32_t magic, version;
                    int num_cameras;
                    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
                    file.read(reinterpret_cast<char*>(&version), sizeof(version));
                    file.read(reinterpret_cast<char*>(&num_cameras), sizeof(num_cameras));
                    // Create temporary pool to skip data
                    PPISPControllerPool temp(num_cameras, 1);
                    file.seekg(-static_cast<std::streamoff>(sizeof(magic) + sizeof(version) + sizeof(num_cameras)),
                               std::ios::cur);
                    temp.deserialize_inference(file);
                    LOG_INFO("PPISP file loaded: {} ({} cameras, {} frames)",
                             lfs::core::path_to_utf8(path), header.num_cameras, header.num_frames);
                }
            } else {
                if (controller_pool) {
                    LOG_WARN("Controller pool requested but not present in PPISP file");
                }
                LOG_INFO("PPISP file loaded: {} ({} cameras, {} frames)",
                         lfs::core::path_to_utf8(path), header.num_cameras, header.num_frames);
            }

            if (header.version >= 2 && has_flag(header.flags, PPISPFileFlags::HAS_METADATA)) {
                auto metadata_result = read_metadata_block(file);
                if (!metadata_result) {
                    return std::unexpected(metadata_result.error());
                }
                if (metadata) {
                    *metadata = std::move(*metadata_result);
                }
            }

            return {};

        } catch (const std::exception& e) {
            return std::unexpected(std::string("Failed to load PPISP file: ") + e.what());
        }
    }

    std::filesystem::path find_ppisp_companion(const std::filesystem::path& splat_path) {
        auto companion = get_ppisp_companion_path(splat_path);
        if (std::filesystem::exists(companion)) {
            return companion;
        }
        return {};
    }

    std::filesystem::path get_ppisp_companion_path(const std::filesystem::path& splat_path) {
        auto path = splat_path;
        path.replace_extension(".ppisp");
        return path;
    }

} // namespace lfs::training
