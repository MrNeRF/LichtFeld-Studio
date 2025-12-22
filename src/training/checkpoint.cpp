/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "checkpoint.hpp"
#include "components/bilateral_grid.hpp"
#include "core/logger.hpp"
#include "strategies/istrategy.hpp"
#include <fstream>
#include <nlohmann/json.hpp>

namespace lfs::training {

    std::expected<void, std::string> save_checkpoint(
        const std::filesystem::path& path,
        const int iteration,
        const IStrategy& strategy,
        const lfs::core::param::TrainingParameters& params,
        const BilateralGrid* bilateral_grid) {

        try {
            const auto checkpoint_dir = path / "checkpoints";
            std::filesystem::create_directories(checkpoint_dir);
            const auto checkpoint_path = checkpoint_dir / ("checkpoint_" + std::to_string(iteration) + ".resume");

            std::ofstream file(checkpoint_path, std::ios::binary);
            if (!file) {
                return std::unexpected("Failed to open: " + checkpoint_path.string());
            }

            const auto& model = strategy.get_model();
            CheckpointHeader header{};
            header.iteration = iteration;
            header.num_gaussians = static_cast<uint32_t>(model.size());
            header.sh_degree = model.get_max_sh_degree();
            header.flags = bilateral_grid ? CheckpointFlags::HAS_BILATERAL_GRID : CheckpointFlags::NONE;

            const auto header_pos = file.tellp();
            file.write(reinterpret_cast<const char*>(&header), sizeof(header));

            // Strategy type
            const char* const strategy_type = strategy.strategy_type();
            const uint32_t type_len = static_cast<uint32_t>(std::strlen(strategy_type));
            file.write(reinterpret_cast<const char*>(&type_len), sizeof(type_len));
            file.write(strategy_type, type_len);

            // Model and strategy state
            model.serialize(file);
            strategy.serialize(file);

            // Bilateral grid (if present)
            if (bilateral_grid) {
                bilateral_grid->serialize(file);
                LOG_DEBUG("Bilateral grid state saved (step={}, lr={:.2e})",
                          bilateral_grid->get_step(), bilateral_grid->get_lr());
            }

            // Training parameters as JSON
            const auto params_pos = file.tellp();
            nlohmann::json params_json;
            params_json["optimization"] = params.optimization.to_json();
            params_json["dataset"] = params.dataset.to_json();
            const std::string params_str = params_json.dump();
            file.write(params_str.data(), static_cast<std::streamsize>(params_str.size()));
            const auto params_end = file.tellp();

            // Update header with JSON offset
            header.params_json_offset = static_cast<uint64_t>(params_pos);
            header.params_json_size = static_cast<uint64_t>(params_end - params_pos);
            file.seekp(header_pos);
            file.write(reinterpret_cast<const char*>(&header), sizeof(header));

            LOG_INFO("Checkpoint saved: {} ({} Gaussians, iter {}{})",
                     checkpoint_path.string(), header.num_gaussians, iteration,
                     bilateral_grid ? ", +bilateral" : "");
            return {};

        } catch (const std::exception& e) {
            return std::unexpected(std::string("Save checkpoint failed: ") + e.what());
        }
    }

    std::expected<CheckpointHeader, std::string> load_checkpoint_header(
        const std::filesystem::path& path) {

        try {
            std::ifstream file(path, std::ios::binary);
            if (!file) {
                return std::unexpected("Failed to open: " + path.string());
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

        } catch (const std::exception& e) {
            return std::unexpected(std::string("Read header failed: ") + e.what());
        }
    }

    std::expected<int, std::string> load_checkpoint(
        const std::filesystem::path& path,
        IStrategy& strategy,
        lfs::core::param::TrainingParameters& params,
        BilateralGrid* bilateral_grid) {

        try {
            std::ifstream file(path, std::ios::binary);
            if (!file) {
                return std::unexpected("Failed to open: " + path.string());
            }

            CheckpointHeader header{};
            file.read(reinterpret_cast<char*>(&header), sizeof(header));

            if (header.magic != CHECKPOINT_MAGIC) {
                return std::unexpected("Invalid checkpoint: wrong magic");
            }
            if (header.version > CHECKPOINT_VERSION) {
                return std::unexpected("Unsupported version: " + std::to_string(header.version));
            }

            // Verify strategy compatibility
            uint32_t type_len = 0;
            file.read(reinterpret_cast<char*>(&type_len), sizeof(type_len));
            std::string saved_type(type_len, '\0');
            file.read(saved_type.data(), type_len);

            if (saved_type != strategy.strategy_type()) {
                return std::unexpected("Strategy mismatch: '" + saved_type +
                                       "' vs '" + strategy.strategy_type() + "'");
            }

            // Model and strategy state
            strategy.get_model().deserialize(file);
            strategy.deserialize(file);

            // Bilateral grid (if present in checkpoint)
            if (has_flag(header.flags, CheckpointFlags::HAS_BILATERAL_GRID)) {
                if (bilateral_grid) {
                    bilateral_grid->deserialize(file);
                    LOG_INFO("Bilateral grid restored (step={}, lr={:.2e})",
                             bilateral_grid->get_step(), bilateral_grid->get_lr());
                } else {
                    LOG_WARN("Checkpoint has bilateral grid but none provided - skipping");
                    // Skip bilateral grid data by reading params offset
                }
            } else if (bilateral_grid) {
                LOG_WARN("Bilateral grid requested but not in checkpoint - using fresh state");
            }

            // Reserve capacity for MCMC densification
            const size_t max_cap = static_cast<size_t>(params.optimization.max_cap);
            if (max_cap > strategy.get_model().size()) {
                LOG_DEBUG("Reserving capacity: {} (current: {})", max_cap, strategy.get_model().size());
                strategy.get_model().reserve_capacity(max_cap);
                strategy.reserve_optimizer_capacity(max_cap);
            }

            // Load params from checkpoint, preserving CLI path overrides
            if (header.params_json_size > 0) {
                file.seekg(static_cast<std::streamoff>(header.params_json_offset));
                std::string params_str(header.params_json_size, '\0');
                file.read(params_str.data(), static_cast<std::streamsize>(header.params_json_size));

                const auto cli_data_path = params.dataset.data_path;
                const auto cli_output_path = params.dataset.output_path;

                const auto params_json = nlohmann::json::parse(params_str);
                if (params_json.contains("optimization")) {
                    params.optimization = lfs::core::param::OptimizationParameters::from_json(params_json["optimization"]);
                    if (params_json.contains("dataset")) {
                        params.dataset = lfs::core::param::DatasetConfig::from_json(params_json["dataset"]);
                    }
                } else {
                    params.optimization = lfs::core::param::OptimizationParameters::from_json(params_json);
                }

                if (!cli_data_path.empty())
                    params.dataset.data_path = cli_data_path;
                if (!cli_output_path.empty())
                    params.dataset.output_path = cli_output_path;
            }

            LOG_INFO("Checkpoint loaded: {} ({} Gaussians, iter {})",
                     path.string(), header.num_gaussians, header.iteration);
            return header.iteration;

        } catch (const std::exception& e) {
            return std::unexpected(std::string("Load checkpoint failed: ") + e.what());
        }
    }

    std::expected<lfs::core::SplatData, std::string> load_checkpoint_splat_data(
        const std::filesystem::path& path) {

        try {
            std::ifstream file(path, std::ios::binary);
            if (!file) {
                return std::unexpected("Failed to open: " + path.string());
            }

            CheckpointHeader header{};
            file.read(reinterpret_cast<char*>(&header), sizeof(header));

            if (header.magic != CHECKPOINT_MAGIC) {
                return std::unexpected("Invalid checkpoint: wrong magic");
            }
            if (header.version > CHECKPOINT_VERSION) {
                return std::unexpected("Unsupported version: " + std::to_string(header.version));
            }

            // Skip strategy type
            uint32_t type_len = 0;
            file.read(reinterpret_cast<char*>(&type_len), sizeof(type_len));
            file.seekg(type_len, std::ios::cur);

            lfs::core::SplatData splat;
            splat.deserialize(file);

            LOG_DEBUG("SplatData loaded: {} Gaussians, iter {}", header.num_gaussians, header.iteration);
            return splat;

        } catch (const std::exception& e) {
            return std::unexpected(std::string("Load SplatData failed: ") + e.what());
        }
    }

    std::expected<lfs::core::param::TrainingParameters, std::string> load_checkpoint_params(
        const std::filesystem::path& path) {

        try {
            std::ifstream file(path, std::ios::binary);
            if (!file) {
                return std::unexpected("Failed to open: " + path.string());
            }

            CheckpointHeader header{};
            file.read(reinterpret_cast<char*>(&header), sizeof(header));

            if (header.magic != CHECKPOINT_MAGIC) {
                return std::unexpected("Invalid checkpoint: wrong magic");
            }
            if (header.version > CHECKPOINT_VERSION) {
                return std::unexpected("Unsupported version: " + std::to_string(header.version));
            }

            lfs::core::param::TrainingParameters params;
            if (header.params_json_size > 0) {
                file.seekg(static_cast<std::streamoff>(header.params_json_offset));
                std::string params_str(header.params_json_size, '\0');
                file.read(params_str.data(), static_cast<std::streamsize>(header.params_json_size));

                const auto params_json = nlohmann::json::parse(params_str);
                if (params_json.contains("optimization")) {
                    params.optimization = lfs::core::param::OptimizationParameters::from_json(params_json["optimization"]);
                    if (params_json.contains("dataset")) {
                        params.dataset = lfs::core::param::DatasetConfig::from_json(params_json["dataset"]);
                    }
                } else {
                    params.optimization = lfs::core::param::OptimizationParameters::from_json(params_json);
                }
            }

            LOG_DEBUG("Params loaded from checkpoint: {}", params.dataset.data_path.string());
            return params;

        } catch (const std::exception& e) {
            return std::unexpected(std::string("Load params failed: ") + e.what());
        }
    }

} // namespace lfs::training
