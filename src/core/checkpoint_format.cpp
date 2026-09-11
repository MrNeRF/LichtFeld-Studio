/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/checkpoint_format.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <utility>

namespace lfs::core {

    namespace {

        std::expected<CheckpointHeader, std::string> open_and_validate(
            const std::filesystem::path& path, std::ifstream& file) {

            if (!open_file_for_read(path, std::ios::binary, file)) {
                return std::unexpected("Failed to open: " + path_to_utf8(path));
            }

            std::error_code size_error;
            const auto file_size = std::filesystem::file_size(path, size_error);
            if (size_error) {
                return std::unexpected("Failed to inspect checkpoint size: " + size_error.message());
            }

            CheckpointHeader header{};
            file.read(reinterpret_cast<char*>(&header), sizeof(header));
            if (!file)
                return std::unexpected("Invalid checkpoint: truncated header");
            if (auto validation = validate_checkpoint_header(header, file_size); !validation)
                return std::unexpected(validation.error());
            return header;
        }

        std::expected<void, std::string> skip_strategy_name(
            std::istream& file,
            const CheckpointHeader& header) {
            uint32_t type_len = 0;
            file.read(reinterpret_cast<char*>(&type_len), sizeof(type_len));
            if (!file)
                return std::unexpected("Invalid checkpoint: truncated strategy name length");
            if (type_len == 0 || type_len > MAX_CHECKPOINT_STRATEGY_NAME_BYTES)
                return std::unexpected("Invalid checkpoint: strategy name length is out of bounds");

            const auto name_offset = file.tellg();
            if (name_offset == std::streampos(-1))
                return std::unexpected("Invalid checkpoint: cannot locate strategy name");
            if (header.params_json_size > 0 &&
                (static_cast<uint64_t>(name_offset) > header.params_json_offset ||
                 type_len > header.params_json_offset - static_cast<uint64_t>(name_offset))) {
                return std::unexpected("Invalid checkpoint: strategy name overlaps parameter JSON");
            }
            file.seekg(static_cast<std::streamoff>(type_len), std::ios::cur);
            if (!file)
                return std::unexpected("Invalid checkpoint: truncated strategy name");
            return {};
        }

        CheckpointHeaderLoadResult read_and_validate(
            std::istream& file,
            const uint64_t file_size) {
            CheckpointHeader header{};
            file.read(reinterpret_cast<char*>(&header), sizeof(header));
            if (!file)
                return std::unexpected("Invalid checkpoint: truncated header");
            if (auto validation = validate_checkpoint_header(header, file_size);
                !validation) {
                return std::unexpected(validation.error());
            }
            return header;
        }

    } // namespace

    std::expected<void, std::string> validate_checkpoint_header(
        const CheckpointHeader& header,
        const uint64_t file_size) {
        if (file_size < sizeof(CheckpointHeader))
            return std::unexpected("Invalid checkpoint: truncated header");
        if (file_size > MAX_CHECKPOINT_FILE_BYTES)
            return std::unexpected("Invalid checkpoint: file exceeds byte budget");
        if (header.magic != CHECKPOINT_MAGIC)
            return std::unexpected("Invalid checkpoint: wrong magic");
        if (header.version < CHECKPOINT_MIN_SUPPORTED_VERSION ||
            header.version > CHECKPOINT_VERSION)
            return std::unexpected("Unsupported version: " + std::to_string(header.version));
        if (header.iteration < 0)
            return std::unexpected("Invalid checkpoint: iteration must be nonnegative");
        if (header.num_gaussians > MAX_CHECKPOINT_GAUSSIANS)
            return std::unexpected("Invalid checkpoint: Gaussian count exceeds budget");
        if (header.sh_degree < 0 || header.sh_degree > 3)
            return std::unexpected("Invalid checkpoint: unsupported SH degree");

        auto known_flags = static_cast<uint32_t>(CheckpointFlags::HAS_BILATERAL_GRID) |
                           static_cast<uint32_t>(CheckpointFlags::HAS_PPISP) |
                           static_cast<uint32_t>(CheckpointFlags::HAS_PPISP_CONTROLLER);
        if (header.version >= CHECKPOINT_VERSION_HAS_SPARSITY) {
            known_flags |= static_cast<uint32_t>(CheckpointFlags::HAS_SPARSITY);
        }
        if ((static_cast<uint32_t>(header.flags) & ~known_flags) != 0)
            return std::unexpected("Invalid checkpoint: unknown feature flags");

        if (header.params_json_size > MAX_CHECKPOINT_JSON_BYTES)
            return std::unexpected("Invalid checkpoint: parameter JSON exceeds byte budget");
        if (header.params_json_size > 0) {
            if (header.params_json_offset < sizeof(CheckpointHeader) ||
                header.params_json_offset > file_size ||
                header.params_json_size > file_size - header.params_json_offset) {
                return std::unexpected("Invalid checkpoint: parameter JSON range is outside the file");
            }
        }
        return {};
    }

    std::expected<CheckpointHeader, std::string> load_checkpoint_header(
        const std::filesystem::path& path) {

        try {
            std::ifstream file;
            return open_and_validate(path, file);
        } catch (const std::exception& e) {
            return std::unexpected(std::string("Read header failed: ") + e.what());
        }
    }

    CheckpointHeaderLoadResult load_checkpoint_header(
        std::istream& stream,
        const uint64_t file_size) {
        try {
            return read_and_validate(stream, file_size);
        } catch (const std::exception& e) {
            return std::unexpected(std::string("Read header failed: ") + e.what());
        }
    }

    std::expected<SplatData, std::string> load_checkpoint_splat_data(
        const std::filesystem::path& path,
        SplatTensorAllocator tensor_allocator) {

        try {
            std::ifstream file;
            auto header = open_and_validate(path, file);
            if (!header) {
                return std::unexpected(header.error());
            }

            if (auto skip_result = skip_strategy_name(file, *header); !skip_result)
                return std::unexpected(skip_result.error());

            SplatData splat;
            splat.deserialize(file, std::move(tensor_allocator));
            if (static_cast<uint64_t>(splat.size()) != header->num_gaussians)
                return std::unexpected("Invalid checkpoint: model count does not match header");
            if (splat.get_max_sh_degree() != header->sh_degree)
                return std::unexpected("Invalid checkpoint: model SH degree does not match header");

            LOG_DEBUG("SplatData loaded: {} Gaussians, iter {}", header->num_gaussians, header->iteration);
            return splat;

        } catch (const std::exception& e) {
            return std::unexpected(std::string("Load SplatData failed: ") + e.what());
        }
    }

    CheckpointSplatDataLoadResult load_checkpoint_splat_data(
        std::istream& stream,
        const uint64_t file_size,
        SplatTensorAllocator tensor_allocator) {
        try {
            auto header = read_and_validate(stream, file_size);
            if (!header) {
                return std::unexpected(header.error());
            }
            if (auto skip_result = skip_strategy_name(stream, *header);
                !skip_result) {
                return std::unexpected(skip_result.error());
            }

            SplatData splat;
            splat.deserialize(stream, std::move(tensor_allocator));
            if (static_cast<uint64_t>(splat.size()) != header->num_gaussians)
                return std::unexpected("Invalid checkpoint: model count does not match header");
            if (splat.get_max_sh_degree() != header->sh_degree)
                return std::unexpected("Invalid checkpoint: model SH degree does not match header");

            LOG_DEBUG("SplatData loaded from bounded CKPT: {} Gaussians, iter {}",
                      header->num_gaussians, header->iteration);
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
                if (!file)
                    return std::unexpected("Invalid checkpoint: truncated parameter JSON");

                params = parse_checkpoint_params_json(params_str);
            }

            if (params.optimization.max_cap < 0)
                return std::unexpected("Invalid checkpoint parameters: max_cap must be nonnegative");
            if (static_cast<uint64_t>(params.optimization.max_cap) > MAX_CHECKPOINT_GAUSSIANS)
                return std::unexpected("Invalid checkpoint parameters: max_cap exceeds checkpoint limit");
            if (const auto validation_error = params.optimization.validate(); !validation_error.empty())
                return std::unexpected("Invalid checkpoint parameters: " + validation_error);
            if (const auto validation_error = params.dataset.validate(); !validation_error.empty())
                return std::unexpected("Invalid checkpoint dataset parameters: " + validation_error);
            if (!(params.freeze_lr_scale >= 0.0f && params.freeze_lr_scale <= 1.0f))
                return std::unexpected("Invalid checkpoint parameters: freeze_lr_scale must be within [0, 1]");
            LOG_DEBUG("Params loaded from checkpoint: {}", path_to_utf8(params.dataset.data_path));
            return params;

        } catch (const std::exception& e) {
            return std::unexpected(std::string("Load params failed: ") + e.what());
        }
    }

    CheckpointParametersLoadResult load_checkpoint_params(
        std::istream& stream,
        const uint64_t file_size) {
        try {
            auto header = read_and_validate(stream, file_size);
            if (!header) {
                return std::unexpected(header.error());
            }

            param::TrainingParameters params;
            if (header->params_json_size > 0) {
                stream.seekg(static_cast<std::streamoff>(header->params_json_offset));
                std::string params_str(header->params_json_size, '\0');
                stream.read(params_str.data(),
                            static_cast<std::streamsize>(header->params_json_size));
                if (!stream)
                    return std::unexpected("Invalid checkpoint: truncated parameter JSON");
                params = parse_checkpoint_params_json(params_str);
            }

            if (params.optimization.max_cap < 0)
                return std::unexpected("Invalid checkpoint parameters: max_cap must be nonnegative");
            if (static_cast<uint64_t>(params.optimization.max_cap) > MAX_CHECKPOINT_GAUSSIANS)
                return std::unexpected("Invalid checkpoint parameters: max_cap exceeds checkpoint limit");
            if (const auto validation_error = params.optimization.validate(); !validation_error.empty())
                return std::unexpected("Invalid checkpoint parameters: " + validation_error);
            if (const auto validation_error = params.dataset.validate(); !validation_error.empty())
                return std::unexpected("Invalid checkpoint dataset parameters: " + validation_error);
            if (!(params.freeze_lr_scale >= 0.0f && params.freeze_lr_scale <= 1.0f))
                return std::unexpected("Invalid checkpoint parameters: freeze_lr_scale must be within [0, 1]");
            return params;
        } catch (const std::exception& e) {
            return std::unexpected(std::string("Load params failed: ") + e.what());
        }
    }

    param::TrainingParameters parse_checkpoint_params_json(
        const std::string_view json_text,
        param::TrainingParameters base_params) {
        const auto params_json = nlohmann::json::parse(json_text);
        const auto validate_splat_composition = [](const param::TrainingParameters& params) {
            if (!params.add_splat_paths.empty() && !params.add_splat_freeze.empty() &&
                params.add_splat_paths.size() != params.add_splat_freeze.size()) {
                throw std::runtime_error(
                    "Invalid checkpoint parameters: add_splat_freeze count does not match add_splat_paths");
            }
        };
        if (!params_json.contains("optimization")) {
            base_params.optimization = param::OptimizationParameters::from_json(params_json);
            validate_splat_composition(base_params);
            return base_params;
        }

        base_params.optimization = param::OptimizationParameters::from_json(params_json["optimization"]);
        if (params_json.contains("dataset")) {
            base_params.dataset = param::DatasetConfig::from_json(params_json["dataset"]);
        }
        if (params_json.contains("init_path")) {
            base_params.init_path = params_json["init_path"].get<std::string>();
        }
        if (params_json.contains("server")) {
            base_params.server = param::ServerConfig::from_json(params_json["server"]);
        }
        if (params_json.contains("exclude_frozen_add_splats_from_export")) {
            base_params.exclude_frozen_add_splats_from_export =
                params_json["exclude_frozen_add_splats_from_export"].get<bool>();
        }
        if (params_json.contains("freeze_lr_scale")) {
            base_params.freeze_lr_scale = params_json["freeze_lr_scale"].get<float>();
        }
        if (params_json.contains("disabled_camera_uids")) {
            base_params.disabled_camera_uids = params_json["disabled_camera_uids"].get<std::vector<int>>();
        }

        const auto utf8_to_paths = [](const nlohmann::json& array) {
            std::vector<std::filesystem::path> paths;
            paths.reserve(array.size());
            for (const auto& entry : array) {
                paths.push_back(utf8_to_path(entry.get<std::string>()));
            }
            return paths;
        };
        if (params_json.contains("view_paths")) {
            base_params.view_paths = utf8_to_paths(params_json["view_paths"]);
        }
        if (params_json.contains("import_cameras_path")) {
            base_params.import_cameras_path =
                utf8_to_path(params_json["import_cameras_path"].get<std::string>());
        }
        if (params_json.contains("add_splat_paths")) {
            base_params.add_splat_paths = utf8_to_paths(params_json["add_splat_paths"]);
        }
        if (params_json.contains("add_splat_freeze")) {
            base_params.add_splat_freeze = params_json["add_splat_freeze"].get<std::vector<bool>>();
        }
        validate_splat_composition(base_params);
        return base_params;
    }

} // namespace lfs::core
