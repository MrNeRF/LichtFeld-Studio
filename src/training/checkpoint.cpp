/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "checkpoint.hpp"
#include "components/bilateral_grid.hpp"
#include "components/ppisp.hpp"
#include "components/ppisp_controller_pool.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "io/atomic_output.hpp"
#include "io/error.hpp"
#include "strategies/istrategy.hpp"
#include "strategies/strategy_factory.hpp"
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <type_traits>
#include <utility>

namespace lfs::training {

    namespace {
        [[nodiscard]] lfs::core::SplatTensorAllocator make_checkpoint_tensor_allocator(
            lfs::core::SplatTensorAllocator allocator,
            const std::size_t target_row_capacity) {
            if (!allocator) {
                return {};
            }
            return [allocator = std::move(allocator), target_row_capacity](
                       lfs::core::TensorShape shape,
                       std::size_t capacity,
                       lfs::core::DataType dtype,
                       std::string_view name) mutable -> lfs::core::Tensor {
                if (target_row_capacity > capacity && name != "SplatData.shN") {
                    capacity = target_row_capacity;
                }
                return allocator(std::move(shape), capacity, dtype, name);
            };
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
            lfs::io::ScopedAtomicOutputFile atomic_checkpoint(
                checkpoint_path,
                lfs::io::AtomicOutputTempName::AppendSuffix,
                lfs::io::AtomicOutputDurability::Durable);
            const auto& temp_checkpoint_path = atomic_checkpoint.temp_path();

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
            if (params.init_path.has_value()) {
                params_json["init_path"] = params.init_path.value();
            }
            if (params.exclude_frozen_add_splats_from_export) {
                params_json["exclude_frozen_add_splats_from_export"] = true;
            }
            if (!params.disabled_camera_uids.empty()) {
                params_json["disabled_camera_uids"] = params.disabled_camera_uids;
            }
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

            if (auto replace_result = atomic_checkpoint.commit(); !replace_result) {
                return std::unexpected(replace_result.error().format());
            }

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
            lfs::core::events::state::CheckpointSaved{
                .iteration = iteration,
                .path = checkpoint_path}
                .emit();
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
        PPISPControllerPool* ppisp_controller_pool,
        lfs::core::SplatTensorAllocator tensor_allocator) {
        try {
            std::ifstream file;
            if (!lfs::core::open_file_for_read(path, std::ios::binary, file)) {
                return std::unexpected("Failed to open: " + lfs::core::path_to_utf8(path));
            }

            std::error_code size_error;
            const auto file_size = std::filesystem::file_size(path, size_error);
            if (size_error)
                return std::unexpected("Failed to inspect checkpoint size: " + size_error.message());

            CheckpointHeader header{};
            file.read(reinterpret_cast<char*>(&header), sizeof(header));
            if (!file)
                return std::unexpected("Invalid checkpoint: truncated header");
            if (auto validation = lfs::core::validate_checkpoint_header(header, file_size); !validation)
                return std::unexpected(validation.error());

            // Verify strategy compatibility
            uint32_t type_len = 0;
            file.read(reinterpret_cast<char*>(&type_len), sizeof(type_len));
            if (!file)
                return std::unexpected("Invalid checkpoint: truncated strategy name length");
            if (type_len == 0 || type_len > lfs::core::MAX_CHECKPOINT_STRATEGY_NAME_BYTES)
                return std::unexpected("Invalid checkpoint: strategy name length is out of bounds");
            const auto type_offset = file.tellg();
            if (type_offset == std::streampos(-1) ||
                (header.params_json_size > 0 &&
                 (static_cast<uint64_t>(static_cast<std::streamoff>(type_offset)) > header.params_json_offset ||
                  type_len > header.params_json_offset -
                                 static_cast<uint64_t>(static_cast<std::streamoff>(type_offset))))) {
                return std::unexpected("Invalid checkpoint: strategy name overlaps parameter JSON");
            }
            std::string saved_type(type_len, '\0');
            file.read(saved_type.data(), type_len);
            if (!file)
                return std::unexpected("Invalid checkpoint: truncated strategy name");

            if (!lfs::core::param::strategy_names_match(saved_type, strategy.strategy_type())) {
                return std::unexpected("Strategy mismatch: '" + saved_type +
                                       "' vs '" + strategy.strategy_type() + "'");
            }

            // Load params from checkpoint up front so strategy internals can be synced before deserialization.
            const auto strategy_state_pos = file.tellg();
            auto loaded_params = params;
            if (header.params_json_size > 0) {
                file.seekg(static_cast<std::streamoff>(header.params_json_offset));
                std::string params_str(header.params_json_size, '\0');
                file.read(params_str.data(), static_cast<std::streamsize>(header.params_json_size));
                if (!file)
                    return std::unexpected("Invalid checkpoint: truncated parameter JSON");

                const auto cli_data_path = loaded_params.dataset.data_path;
                const auto cli_output_path = loaded_params.dataset.output_path;

                const auto params_json = nlohmann::json::parse(params_str);
                if (params_json.contains("optimization")) {
                    loaded_params.optimization = lfs::core::param::OptimizationParameters::from_json(params_json["optimization"]);
                    if (params_json.contains("dataset")) {
                        loaded_params.dataset = lfs::core::param::DatasetConfig::from_json(params_json["dataset"]);
                    }
                    if (params_json.contains("init_path")) {
                        loaded_params.init_path = params_json["init_path"].get<std::string>();
                    }
                    if (params_json.contains("exclude_frozen_add_splats_from_export")) {
                        loaded_params.exclude_frozen_add_splats_from_export =
                            params_json["exclude_frozen_add_splats_from_export"].get<bool>();
                    }
                    if (params_json.contains("disabled_camera_uids")) {
                        loaded_params.disabled_camera_uids = params_json["disabled_camera_uids"].get<std::vector<int>>();
                    }
                } else {
                    loaded_params.optimization = lfs::core::param::OptimizationParameters::from_json(params_json);
                }

                if (!cli_data_path.empty())
                    loaded_params.dataset.data_path = cli_data_path;
                if (!cli_output_path.empty())
                    loaded_params.dataset.output_path = cli_output_path;
            }
            if (loaded_params.optimization.max_cap < 0)
                return std::unexpected("Invalid checkpoint parameters: max_cap must be nonnegative");
            if (static_cast<uint64_t>(loaded_params.optimization.max_cap) >
                lfs::core::MAX_CHECKPOINT_GAUSSIANS) {
                return std::unexpected("Invalid checkpoint parameters: max_cap exceeds checkpoint limit");
            }
            if (const auto parameter_error = loaded_params.optimization.validate(); !parameter_error.empty())
                return std::unexpected("Invalid checkpoint parameters: " + parameter_error);
            if (const auto parameter_error = loaded_params.dataset.validate(); !parameter_error.empty())
                return std::unexpected("Invalid checkpoint dataset parameters: " + parameter_error);

            file.clear();
            file.seekg(strategy_state_pos);
            if (!file)
                throw std::runtime_error("Invalid checkpoint: cannot seek to model state");

            // Parse into an isolated state graph. Nothing reachable from the live
            // trainer is changed until every model, strategy, and component field
            // has passed its byte budget and logical schema checks.
            const size_t target_capacity =
                loaded_params.optimization.max_cap > 0
                    ? std::max<std::size_t>(static_cast<std::size_t>(loaded_params.optimization.max_cap),
                                            static_cast<std::size_t>(header.num_gaussians))
                    : 0;
            lfs::core::SplatData loaded_model;
            loaded_model.deserialize(
                file,
                make_checkpoint_tensor_allocator(std::move(tensor_allocator), target_capacity));
            if (static_cast<uint64_t>(loaded_model.size()) != header.num_gaussians)
                throw std::runtime_error("Invalid checkpoint: model count does not match header");
            if (loaded_model.get_max_sh_degree() != header.sh_degree)
                throw std::runtime_error("Invalid checkpoint: model SH degree does not match header");

            auto loaded_strategy_result = StrategyFactory::instance().create(saved_type, loaded_model);
            if (!loaded_strategy_result)
                throw std::runtime_error("Cannot construct checkpoint strategy: " + loaded_strategy_result.error());
            auto loaded_strategy = std::move(*loaded_strategy_result);
            auto* checkpoint_adopter = dynamic_cast<ICheckpointStateAdopter*>(&strategy);
            if (checkpoint_adopter && checkpoint_adopter->has_checkpoint_runtime_state())
                loaded_strategy->initialize(loaded_params.optimization);
            else
                loaded_strategy->set_optimization_params(loaded_params.optimization);
            loaded_strategy->deserialize(file);
            if (!checkpoint_adopter || !checkpoint_adopter->can_adopt_checkpoint_state(*loaded_strategy)) {
                throw std::runtime_error(
                    "Strategy does not support transactional checkpoint state adoption");
            }

            std::unique_ptr<BilateralGrid> loaded_bilateral_grid;
            std::unique_ptr<PPISP> loaded_ppisp;
            std::unique_ptr<PPISPControllerPool> loaded_ppisp_controller_pool;

            // Bilateral grid (if present in checkpoint)
            if (has_flag(header.flags, CheckpointFlags::HAS_BILATERAL_GRID)) {
                auto parsed = std::make_unique<BilateralGrid>(1, 1, 1, 1, 1);
                parsed->deserialize(file);
                if (bilateral_grid)
                    loaded_bilateral_grid = std::move(parsed);
                else
                    LOG_WARN("Checkpoint has bilateral grid but none provided - skipping data");
            } else if (bilateral_grid) {
                LOG_WARN("Bilateral grid requested but not in checkpoint - using fresh state");
            }

            // PPISP (if present in checkpoint)
            if (has_flag(header.flags, CheckpointFlags::HAS_PPISP)) {
                auto parsed = std::make_unique<PPISP>(1);
                parsed->deserialize(file);
                if (ppisp)
                    loaded_ppisp = std::move(parsed);
                else
                    LOG_WARN("Checkpoint has PPISP but none provided - skipping data");
            } else if (ppisp) {
                LOG_WARN("PPISP requested but not in checkpoint - using fresh state");
            }

            // PPISP controller pool (if present in checkpoint)
            if (has_flag(header.flags, CheckpointFlags::HAS_PPISP_CONTROLLER)) {
                if (ppisp_controller_pool) {
                    loaded_ppisp_controller_pool = std::make_unique<PPISPControllerPool>(
                        ppisp_controller_pool->num_cameras(), 1);
                    loaded_ppisp_controller_pool->deserialize(file);
                } else {
                    LOG_WARN("Checkpoint has PPISP controller pool but none provided - skipping");
                    PPISPControllerPool::consume_checkpoint(file);
                }
            } else if (ppisp_controller_pool) {
                LOG_WARN("PPISP controller pool requested but not in checkpoint - using fresh state");
            }

            // Reserve capacity for densification after the checkpoint params are resolved.
            if (header.params_json_size > 0) {
                const auto serialized_state_end = file.tellg();
                if (serialized_state_end == std::streampos(-1) ||
                    static_cast<uint64_t>(static_cast<std::streamoff>(serialized_state_end)) !=
                        header.params_json_offset) {
                    throw std::runtime_error("Invalid checkpoint: serialized state does not end at parameter JSON");
                }
            } else if (!file) {
                throw std::runtime_error("Invalid checkpoint: truncated serialized state");
            }

            const size_t max_cap = static_cast<size_t>(loaded_params.optimization.max_cap);
            if (max_cap > loaded_model.size()) {
                LOG_DEBUG("Reserving capacity: {} (current: {})", max_cap, loaded_model.size());
                loaded_model.reserve_capacity(max_cap);
                loaded_strategy->reserve_optimizer_capacity(max_cap);
            }

            static_assert(std::is_nothrow_swappable_v<lfs::core::param::TrainingParameters>);
            static_assert(std::is_nothrow_move_assignable_v<lfs::core::SplatData>);

            // All remaining operations transfer already-owned storage and cannot
            // allocate. The live state therefore changes as one commit boundary.
            std::swap(params, loaded_params);
            strategy.get_model() = std::move(loaded_model);
            checkpoint_adopter->adopt_checkpoint_state(*loaded_strategy);
            if (loaded_bilateral_grid) {
                bilateral_grid->adopt_checkpoint_state(*loaded_bilateral_grid);
                LOG_INFO("Bilateral grid restored (step={}, lr={:.2e})",
                         bilateral_grid->get_step(), bilateral_grid->get_lr());
            }
            if (loaded_ppisp) {
                ppisp->adopt_checkpoint_state(*loaded_ppisp);
                LOG_INFO("PPISP restored (step={}, lr={:.2e})", ppisp->get_step(), ppisp->get_lr());
            }
            if (loaded_ppisp_controller_pool) {
                ppisp_controller_pool->adopt_checkpoint_state(*loaded_ppisp_controller_pool);
                LOG_INFO("PPISP controller pool restored: {} cameras (lr={:.2e})",
                         ppisp_controller_pool->num_cameras(),
                         ppisp_controller_pool->get_learning_rate());
            }

            LOG_INFO("Checkpoint loaded: {} ({} Gaussians, iter {})",
                     lfs::core::path_to_utf8(path), header.num_gaussians, header.iteration);
            return header.iteration;

        } catch (const std::exception& e) {
            return std::unexpected(std::string("Load checkpoint failed: ") + e.what());
        } catch (...) {
            return std::unexpected("Load checkpoint failed: unknown exception");
        }
    }

} // namespace lfs::training
