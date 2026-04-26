/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/checkpoint_format.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include <fstream>
#include <nlohmann/json.hpp>

namespace lfs::core {

    namespace {

        std::expected<CheckpointHeader, std::string> open_and_validate(
            const std::filesystem::path& path, std::ifstream& file) {

            if (!open_file_for_read(path, std::ios::binary, file)) {
                return std::unexpected("Failed to open: " + path_to_utf8(path));
            }

            CheckpointHeader header{};
            file.read(reinterpret_cast<char*>(&header), sizeof(header));

            if (header.magic != CHECKPOINT_MAGIC) {
                return std::unexpected("Invalid checkpoint: wrong magic");
            }
            if (header.version > CHECKPOINT_VERSION) {
                return std::unexpected("Unsupported version: " + std::to_string(header.version));
            }
            return header;
        }

    } // namespace

    std::expected<CheckpointHeader, std::string> load_checkpoint_header(
        const std::filesystem::path& path) {

        try {
            std::ifstream file;
            return open_and_validate(path, file);
        } catch (const std::exception& e) {
            return std::unexpected(std::string("Read header failed: ") + e.what());
        }
    }

    std::expected<SplatData, std::string> load_checkpoint_splat_data(
        const std::filesystem::path& path) {

        try {
            std::ifstream file;
            auto header = open_and_validate(path, file);
            if (!header) {
                return std::unexpected(header.error());
            }

            uint32_t type_len = 0;
            file.read(reinterpret_cast<char*>(&type_len), sizeof(type_len));
            file.seekg(type_len, std::ios::cur);

            SplatData splat;
            splat.deserialize(file);

            LOG_DEBUG("SplatData loaded: {} Gaussians, iter {}", header->num_gaussians, header->iteration);
            return splat;

        } catch (const std::exception& e) {
            return std::unexpected(std::string("Load SplatData failed: ") + e.what());
        }
    }

    std::expected<param::TrainingParameters, std::string> load_checkpoint_params(
        const std::filesystem::path& path) {

        try {
            std::ifstream file;
            auto header = open_and_validate(path, file);
            if (!header) {
                return std::unexpected(header.error());
            }

            param::TrainingParameters params;
            if (header->params_json_size > 0) {
                file.seekg(static_cast<std::streamoff>(header->params_json_offset));
                std::string params_str(header->params_json_size, '\0');
                file.read(params_str.data(), static_cast<std::streamsize>(header->params_json_size));

                const auto params_json = nlohmann::json::parse(params_str);
                if (params_json.contains("optimization")) {
                    params.optimization = param::OptimizationParameters::from_json(params_json["optimization"]);
                    if (params_json.contains("dataset")) {
                        params.dataset = param::DatasetConfig::from_json(params_json["dataset"]);
                    }
                    if (params_json.contains("init_path")) {
                        params.init_path = params_json["init_path"].get<std::string>();
                    }
                } else {
                    params.optimization = param::OptimizationParameters::from_json(params_json);
                }
            }

            LOG_DEBUG("Params loaded from checkpoint: {}", path_to_utf8(params.dataset.data_path));
            return params;

        } catch (const std::exception& e) {
            return std::unexpected(std::string("Load params failed: ") + e.what());
        }
    }

} // namespace lfs::core
