/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "checkpoint.hpp"
#include "components/bilateral_grid.hpp"
#include "components/ppisp.hpp"
#include "components/ppisp_controller_pool.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "io/error.hpp"
#include "strategies/istrategy.hpp"
#include <fstream>
#include <nlohmann/json.hpp>

namespace lfs::training {

    namespace {
        constexpr char kCheckpointTempSuffix[] = ".tmp";

        std::filesystem::path checkpoint_temp_path(const std::filesystem::path& checkpoint_path) {
            auto temp_path = checkpoint_path;
            temp_path += kCheckpointTempSuffix;
            return temp_path;
        }

        std::expected<void, std::string> replace_checkpoint_file(
            const std::filesystem::path& checkpoint_path,
            const std::filesystem::path& temp_checkpoint_path) {

            std::error_code ec;
            std::filesystem::remove(checkpoint_path, ec);
            if (ec) {
                return std::unexpected("Failed to remove existing checkpoint file '" +
                                       lfs::core::path_to_utf8(checkpoint_path) + "': " + ec.message());
            }

            std::filesystem::rename(temp_checkpoint_path, checkpoint_path, ec);
            if (ec) {
                return std::unexpected("Failed to replace checkpoint file '" +
                                       lfs::core::path_to_utf8(checkpoint_path) + "': " + ec.message());
            }

            return {};
        }
    } // namespace

    using lfs::core::CHECKPOINT_MAGIC;
    using lfs::core::CHECKPOINT_VERSION;
    using lfs::core::CheckpointFlags;
    using lfs::core::CheckpointHeader;
    using lfs::core::has_flag;

    std::expected<void, std::string> save_checkpoint(
        const std::filesystem::path& path,
        const int iteration,
        const IStrategy& strategy,
        const lfs::core::param::TrainingParameters& params,
        const BilateralGrid* bilateral_grid,
        const PPISP* ppisp,
        const PPISPControllerPool* ppisp_controller_pool) {

        try {
            // Validate input path
            if (path.empty()) {
                return std::unexpected("Cannot save checkpoint: output path is empty");
            }

            const auto checkpoint_dir = checkpoint_directory(path);
            const auto checkpoint_path = checkpoint_output_path(path);
            const auto temp_checkpoint_path = checkpoint_temp_path(checkpoint_path);

            // Create checkpoint directory with error checking
            std::error_code ec;
            std::filesystem::create_directories(checkpoint_dir, ec);
            if (ec) {
                return std::unexpected("Failed to create checkpoint directory '" +
                                       lfs::core::path_to_utf8(checkpoint_dir) + "': " + ec.message());
            }

            const auto& model = strategy.get_model();

            // Model tensors
            size_t model_bytes = 0;
            model_bytes += model.means().bytes();
            model_bytes += model.sh0().bytes();
            model_bytes += model.scaling_raw().bytes();
            model_bytes += model.rotation_raw().bytes();
            model_bytes += model.opacity_raw().bytes();
            if (model.shN().is_valid()) {
                model_bytes += model.shN().bytes();
            }
            if (model.deleted().is_valid()) {
                model_bytes += model.deleted().bytes();
            }
            if (model._densification_info.is_valid()) {
                model_bytes += model._densification_info.bytes();
            }

            // Optimizer: 2x model (Adam m & v)
            const size_t optimizer_bytes = model_bytes * 2;

            // Bilateral grid: 3x (grids + Adam state)
            size_t bilateral_grid_bytes = 0;
            if (bilateral_grid) {
                bilateral_grid_bytes = bilateral_grid->grids().bytes() * 3;
            }

            // PPISP: estimate based on num_cameras and num_frames
            size_t ppisp_bytes = 0;
            if (ppisp) {
                // exposure + vignetting + color + crf, each with params + 2x Adam state
                const size_t exp_size = ppisp->num_frames() * sizeof(float) * 3;
                const size_t vig_size = ppisp->num_cameras() * 3 * 5 * sizeof(float) * 3;
                const size_t color_size = ppisp->num_frames() * 8 * sizeof(float) * 3;
                const size_t crf_size = ppisp->num_cameras() * 3 * 4 * sizeof(float) * 3;
                ppisp_bytes = exp_size + vig_size + color_size + crf_size;
            }

            constexpr size_t OVERHEAD_BYTES = 64 * 1024;

            const size_t estimated_size = sizeof(CheckpointHeader) +
                                          model_bytes +
                                          optimizer_bytes +
                                          bilateral_grid_bytes +
                                          ppisp_bytes +
                                          OVERHEAD_BYTES;

            if (auto space_check = lfs::io::check_disk_space(checkpoint_path, estimated_size, 1.1f);
                !space_check) {
                const auto& error = space_check.error();
                const bool is_disk_space = error.is(lfs::io::ErrorCode::INSUFFICIENT_DISK_SPACE);

                lfs::core::events::state::DiskSpaceSaveFailed{
                    .iteration = iteration,
                    .path = checkpoint_path,
                    .error = error.format(),
                    .required_bytes = estimated_size,
                    .available_bytes = error.available_bytes,
                    .is_disk_space_error = is_disk_space}
                    .emit();

                return std::unexpected(error.format());
            }

            std::ofstream file;
            if (!lfs::core::open_file_for_write(temp_checkpoint_path, std::ios::binary, file)) {
                return std::unexpected("Failed to open checkpoint file: " +
                                       lfs::core::path_to_utf8(temp_checkpoint_path));
            }

            CheckpointHeader header{};
            header.iteration = iteration;
            header.num_gaussians = static_cast<uint32_t>(model.size());
            header.sh_degree = model.get_max_sh_degree();
            header.flags = CheckpointFlags::NONE;
            if (bilateral_grid)
                header.flags = header.flags | CheckpointFlags::HAS_BILATERAL_GRID;
            if (ppisp)
                header.flags = header.flags | CheckpointFlags::HAS_PPISP;
            if (ppisp_controller_pool)
                header.flags = header.flags | CheckpointFlags::HAS_PPISP_CONTROLLER;

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

            // PPISP (if present)
            if (ppisp) {
                ppisp->serialize(file);
                LOG_DEBUG("PPISP state saved (step={}, lr={:.2e})",
                          ppisp->get_step(), ppisp->get_lr());
            }

            // PPISP controller pool (if present)
            if (ppisp_controller_pool) {
                ppisp_controller_pool->serialize(file);
                LOG_DEBUG("PPISP controller pool saved: {} cameras", ppisp_controller_pool->num_cameras());
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
            file.close();
            if (!file) {
                return std::unexpected("Failed to finalize checkpoint file: " +
                                       lfs::core::path_to_utf8(temp_checkpoint_path));
            }

            if (auto replace_result = replace_checkpoint_file(checkpoint_path, temp_checkpoint_path); !replace_result)
                return std::unexpected(replace_result.error());

            std::string extras;
            if (bilateral_grid)
                extras += ", +bilateral";
            if (ppisp)
                extras += ", +ppisp";
            if (ppisp_controller_pool)
                extras += ", +ppisp_ctrl(" + std::to_string(ppisp_controller_pool->num_cameras()) + ")";
            LOG_INFO("Checkpoint saved: {} ({} Gaussians, iter {}{})",
                     lfs::core::path_to_utf8(checkpoint_path), header.num_gaussians, iteration,
                     extras);
            return {};

        } catch (const std::exception& e) {
            return std::unexpected(std::string("Save checkpoint failed: ") + e.what());
        }
    }

    std::expected<int, std::string> load_checkpoint(
        const std::filesystem::path& path,
        IStrategy& strategy,
        lfs::core::param::TrainingParameters& params,
        BilateralGrid* bilateral_grid,
        PPISP* ppisp,
        PPISPControllerPool* ppisp_controller_pool) {

        try {
            std::ifstream file;
            if (!lfs::core::open_file_for_read(path, std::ios::binary, file)) {
                return std::unexpected("Failed to open: " + lfs::core::path_to_utf8(path));
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

            // Load params from checkpoint up front so strategy internals can be synced before deserialization.
            const auto strategy_state_pos = file.tellg();
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
            strategy.set_optimization_params(params.optimization);
            file.clear();
            file.seekg(strategy_state_pos);

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
                    LOG_WARN("Checkpoint has bilateral grid but none provided - skipping data");
                    BilateralGrid temp(1, 1, 1, 1, 1);
                    temp.deserialize(file);
                }
            } else if (bilateral_grid) {
                LOG_WARN("Bilateral grid requested but not in checkpoint - using fresh state");
            }

            // PPISP (if present in checkpoint)
            if (has_flag(header.flags, CheckpointFlags::HAS_PPISP)) {
                if (ppisp) {
                    ppisp->deserialize(file);
                    LOG_INFO("PPISP restored (step={}, lr={:.2e})", ppisp->get_step(), ppisp->get_lr());
                } else {
                    LOG_WARN("Checkpoint has PPISP but none provided - skipping data");
                    PPISP temp(1);
                    temp.deserialize(file);
                }
            } else if (ppisp) {
                LOG_WARN("PPISP requested but not in checkpoint - using fresh state");
            }

            // PPISP controller pool (if present in checkpoint)
            if (has_flag(header.flags, CheckpointFlags::HAS_PPISP_CONTROLLER)) {
                if (ppisp_controller_pool) {
                    ppisp_controller_pool->deserialize(file);
                    LOG_INFO("PPISP controller pool restored: {} cameras (step={}, lr={:.2e})",
                             ppisp_controller_pool->num_cameras(),
                             0, // step not easily accessible from pool
                             ppisp_controller_pool->get_learning_rate());
                } else {
                    LOG_WARN("Checkpoint has PPISP controller pool but none provided - skipping");
                    // Skip the pool data by reading into a temporary
                    // Pool format: magic + version + num_cameras + ... (variable size)
                    // We need to read it to advance the file position
                    uint32_t magic, version;
                    int num_cameras, total_iter;
                    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
                    file.read(reinterpret_cast<char*>(&version), sizeof(version));
                    file.read(reinterpret_cast<char*>(&num_cameras), sizeof(num_cameras));
                    // Create a temporary pool to skip the data
                    PPISPControllerPool temp(num_cameras, 1);
                    // Rewind and deserialize properly to skip
                    file.seekg(-static_cast<std::streamoff>(sizeof(magic) + sizeof(version) + sizeof(num_cameras)),
                               std::ios::cur);
                    temp.deserialize(file);
                }
            } else if (ppisp_controller_pool) {
                LOG_WARN("PPISP controller pool requested but not in checkpoint - using fresh state");
            }

            // Reserve capacity for densification after the checkpoint params are resolved.
            const size_t max_cap = static_cast<size_t>(params.optimization.max_cap);
            if (max_cap > strategy.get_model().size()) {
                LOG_DEBUG("Reserving capacity: {} (current: {})", max_cap, strategy.get_model().size());
                strategy.get_model().reserve_capacity(max_cap);
                strategy.reserve_optimizer_capacity(max_cap);
            }

            LOG_INFO("Checkpoint loaded: {} ({} Gaussians, iter {})",
                     lfs::core::path_to_utf8(path), header.num_gaussians, header.iteration);
            return header.iteration;

        } catch (const std::exception& e) {
            return std::unexpected(std::string("Load checkpoint failed: ") + e.what());
        }
    }

} // namespace lfs::training
