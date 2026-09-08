/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "checkpoint.hpp"
#include "components/bilateral_grid.hpp"
#include "components/popspa_controller.hpp"
#include "components/ppisp.hpp"
#include "components/ppisp_controller_pool.hpp"
#include "components/sparsity_optimizer.hpp"
#include "core/logger.hpp"
#include "core/parameters.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "strategies/istrategy.hpp"
#include "strategies/strategy_factory.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

namespace lfs::training {

    namespace {
        void validate_popspa_checkpoint(
            const POPSpaController& controller,
            const AdamOptimizer& optimizer,
            const lfs::core::param::OptimizationParameters& params,
            const lfs::core::SplatData& model,
            const int iteration) {
            const size_t rows = model.size();
            if (const auto error = params.validate(); !error.empty())
                throw std::runtime_error("Invalid POPSpa checkpoint parameters: " + error);
            if (!params.enable_sparsity || params.sparsity_method != lfs::core::param::SparsityMethod::POPSpa ||
                !controller.is_initialized() || controller.state_size() != rows)
                throw std::runtime_error("Invalid POPSpa checkpoint: method or model row count disagrees");
            const auto& config = controller.config();
            const size_t expected_first_count = params.popspa_first_prune_count == 0
                                                    ? std::max(config.target_count, static_cast<size_t>(std::round(
                                                                                        std::sqrt(static_cast<long double>(controller.input_size()) * config.target_count))))
                                                    : static_cast<size_t>(params.popspa_first_prune_count);
            if (config.target_count != static_cast<size_t>(params.popspa_target_count) ||
                config.first_prune_count != expected_first_count ||
                config.sparsify_steps != static_cast<uint32_t>(params.popspa_sparsify_steps) ||
                config.refine_steps != static_cast<uint32_t>(params.popspa_refine_steps) ||
                config.projection_interval != static_cast<uint32_t>(params.popspa_projection_interval) ||
                config.rho != params.popspa_rho || config.erank_weight != params.popspa_erank_weight ||
                config.thin_scale_weight != params.popspa_thin_scale_weight || config.erank_epsilon != params.popspa_erank_epsilon)
                throw std::runtime_error("Invalid POPSpa checkpoint: controller config disagrees with saved parameters");

            uint64_t completed_steps = 0;
            uint64_t optimizer_age = 0;
            switch (controller.phase()) {
            case POPSpaPhase::FirstScore:
            case POPSpaPhase::FirstPrune:
                break;
            case POPSpaPhase::Sparsify:
                completed_steps = optimizer_age = controller.phase_step();
                break;
            case POPSpaPhase::SecondScore:
            case POPSpaPhase::FinalPrune:
                completed_steps = optimizer_age = config.sparsify_steps;
                break;
            case POPSpaPhase::Recover:
                optimizer_age = controller.phase_step();
                completed_steps = uint64_t(config.sparsify_steps) + optimizer_age;
                break;
            case POPSpaPhase::Complete:
                optimizer_age = config.refine_steps;
                completed_steps = uint64_t(config.sparsify_steps) + config.refine_steps;
                break;
            default:
                throw std::runtime_error("Invalid POPSpa checkpoint phase");
            }
            if (uint64_t(iteration) != uint64_t(params.iterations) + completed_steps)
                throw std::runtime_error("Invalid POPSpa checkpoint: phase and iteration disagree");
            const auto& adam = optimizer.get_config();
            if (adam.beta1 != 0.9 || adam.beta2 != 0.999 || adam.eps != 1e-15)
                throw std::runtime_error("Invalid POPSpa checkpoint: Adam profile disagrees");
            const std::pair<ParamType, float> learning_rates[] = {
                {ParamType::Means, params.popspa_means_lr},
                {ParamType::Scaling, params.popspa_scales_lr},
                {ParamType::Rotation, params.popspa_quaternions_lr},
                {ParamType::Opacity, params.popspa_opacities_lr},
                {ParamType::Sh0, params.popspa_sh0_lr},
                {ParamType::ShN, params.popspa_shn_lr}};
            for (const auto& [type, rate] : learning_rates) {
                const double expected_rate = type == ParamType::Means
                                                 ? rate * static_cast<double>(model.get_scene_scale())
                                                 : static_cast<double>(rate);
                if (!optimizer.has_param_lr(type) || optimizer.get_param_lr(type) != expected_rate)
                    throw std::runtime_error("Invalid POPSpa checkpoint: Adam learning rate disagrees");
                const auto step = optimizer.get_step_count(type);
                // Higher SH coefficients may be inactive during warmup.
                if (step < 0 || static_cast<uint64_t>(step) > optimizer_age ||
                    (type != ParamType::ShN && static_cast<uint64_t>(step) != optimizer_age))
                    throw std::runtime_error("Invalid POPSpa checkpoint: Adam step count disagrees with phase");
                if (const auto* state = optimizer.get_state(type);
                    state && type != ParamType::ShN && state->size != rows)
                    throw std::runtime_error("Invalid POPSpa checkpoint: Adam row count disagrees");
            }
        }

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

        [[nodiscard]] ADMMSparsityOptimizer::Config sparsity_config_from_params(
            const lfs::core::param::OptimizationParameters& params) {
            const int start_iteration = static_cast<int>(params.iterations);
            return {
                .sparsify_steps = params.enable_sparsity ? std::max(0, params.sparsify_steps) : 0,
                .init_rho = params.init_rho,
                .prune_ratio = params.prune_ratio,
                .update_every = 50,
                .start_iteration = start_iteration,
            };
        }

        [[nodiscard]] lfs::Error checkpoint_stream_error(
            const lfs::ErrorCode code,
            std::string detail,
            const lfs::core::SourceSite source) {
            return lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = lfs::ErrorDomain::Training,
                .user_message =
                    "The training checkpoint could not be serialized.",
                .detail = std::move(detail),
                .detection = source,
            });
        }
    } // namespace

    using lfs::core::CheckpointFlags;
    using lfs::core::CheckpointHeader;
    using lfs::core::has_flag;

    lfs::Result<CheckpointStreamResult>
    serialize_checkpoint(
        std::ostream& destination,
        const int iteration,
        const IStrategy& strategy,
        const lfs::core::param::TrainingParameters& params,
        const BilateralGrid* bilateral_grid,
        const PPISP* ppisp,
        const PPISPControllerPool* ppisp_controller_pool,
        const ADMMSparsityOptimizer* sparsity_optimizer,
        const POPSpaController* popspa_controller,
        const AdamOptimizer* popspa_optimizer) {
        try {
            if (iteration < 0) {
                return checkpoint_stream_error(
                    lfs::ErrorCode::InvalidArgument,
                    "Cannot serialize checkpoint: iteration is negative",
                    LFS_SOURCE_SITE_CURRENT());
            }
            if (!params.add_splat_paths.empty() &&
                !params.add_splat_freeze.empty() &&
                params.add_splat_paths.size() !=
                    params.add_splat_freeze.size()) {
                return checkpoint_stream_error(
                    lfs::ErrorCode::InvalidArgument,
                    "Cannot serialize checkpoint: add_splat_freeze count "
                    "does not match add_splat_paths",
                    LFS_SOURCE_SITE_CURRENT());
            }

            const auto& model = strategy.get_model();
            if (model.size() <= 0 ||
                static_cast<std::uint64_t>(model.size()) >
                    lfs::core::MAX_CHECKPOINT_GAUSSIANS) {
                return checkpoint_stream_error(
                    lfs::ErrorCode::InvalidArgument,
                    "Cannot serialize checkpoint: Gaussian count is out of bounds",
                    LFS_SOURCE_SITE_CURRENT());
            }
            const bool save_sparsity =
                sparsity_optimizer &&
                sparsity_optimizer->is_initialized();
            const bool save_popspa = popspa_controller && popspa_controller->is_initialized();
            if (save_popspa) {
                if (save_sparsity || !popspa_optimizer)
                    throw std::runtime_error("POPSpa checkpoint requires dedicated Adam and excludes legacy sparsity");
                validate_popspa_checkpoint(*popspa_controller, *popspa_optimizer, params.optimization, model, iteration);
            } else if (params.optimization.enable_sparsity &&
                       params.optimization.sparsity_method == lfs::core::param::SparsityMethod::POPSpa &&
                       (save_sparsity || uint64_t(iteration) > params.optimization.iterations)) {
                throw std::runtime_error("POPSpa checkpoint is missing initialized controller state");
            }
            if (save_sparsity) {
                // The legacy standalone writer used its byte estimate to
                // validate this invariant before serialization. CKPT is now
                // embedded directly, so the stream serializer must own the
                // validation itself.
                (void)sparsity_optimizer->checkpoint_size_bytes(
                    model.size());
            }

            CheckpointHeader header{};
            header.iteration = iteration;
            header.num_gaussians =
                static_cast<std::uint32_t>(model.size());
            header.sh_degree = model.get_max_sh_degree();
            header.flags = CheckpointFlags::NONE;
            if (bilateral_grid)
                header.flags =
                    header.flags |
                    CheckpointFlags::HAS_BILATERAL_GRID;
            if (ppisp)
                header.flags =
                    header.flags | CheckpointFlags::HAS_PPISP;
            if (ppisp_controller_pool)
                header.flags =
                    header.flags |
                    CheckpointFlags::HAS_PPISP_CONTROLLER;
            if (save_sparsity)
                header.flags =
                    header.flags | CheckpointFlags::HAS_SPARSITY;

            const auto header_pos = destination.tellp();
            if (save_popspa)
                header.flags = header.flags | CheckpointFlags::HAS_POPSPA;
            if (header_pos == std::streampos(-1)) {
                return checkpoint_stream_error(
                    lfs::ErrorCode::FailedPrecondition,
                    "Cannot serialize checkpoint: destination is not seekable",
                    LFS_SOURCE_SITE_CURRENT());
            }
            destination.write(
                reinterpret_cast<const char*>(&header),
                sizeof(header));

            const char* const strategy_type =
                strategy.strategy_type();
            const auto type_bytes = std::strlen(strategy_type);
            if (type_bytes == 0 ||
                type_bytes >
                    lfs::core::MAX_CHECKPOINT_STRATEGY_NAME_BYTES) {
                return checkpoint_stream_error(
                    lfs::ErrorCode::InvalidArgument,
                    "Cannot serialize checkpoint: strategy name is out of bounds",
                    LFS_SOURCE_SITE_CURRENT());
            }
            const auto type_len =
                static_cast<std::uint32_t>(type_bytes);
            destination.write(
                reinterpret_cast<const char*>(&type_len),
                sizeof(type_len));
            destination.write(strategy_type, type_len);

            model.serialize(destination);
            strategy.serialize(destination);

            if (bilateral_grid) {
                bilateral_grid->serialize(destination);
                LOG_DEBUG(
                    "Bilateral grid state staged (step={}, lr={:.2e})",
                    bilateral_grid->get_step(),
                    bilateral_grid->get_lr());
            }
            if (ppisp) {
                ppisp->serialize(destination);
                LOG_DEBUG(
                    "PPISP state staged (step={}, lr={:.2e})",
                    ppisp->get_step(), ppisp->get_lr());
            }
            if (ppisp_controller_pool) {
                ppisp_controller_pool->serialize(destination);
                LOG_DEBUG(
                    "PPISP controller pool staged: {} cameras",
                    ppisp_controller_pool->num_cameras());
            }
            if (save_sparsity) {
                sparsity_optimizer->serialize(destination);
                LOG_DEBUG(
                    "Sparsity ADMM state staged: {} rows",
                    sparsity_optimizer->state_size());
            }

            if (save_popspa) {
                auto serialized = popspa_controller->serialize(destination);
                if (!serialized)
                    return std::move(serialized).error();
                popspa_optimizer->serialize(destination);
            }

            const auto params_pos = destination.tellp();
            if (params_pos == std::streampos(-1)) {
                return checkpoint_stream_error(
                    lfs::ErrorCode::DataLoss,
                    "Cannot serialize checkpoint: cannot locate parameter JSON",
                    LFS_SOURCE_SITE_CURRENT());
            }
            nlohmann::json params_json;
            params_json["optimization"] =
                params.optimization.to_json();
            params_json["dataset"] = params.dataset.to_json();
            if (params.init_path) {
                params_json["init_path"] = *params.init_path;
            }
            if (params.exclude_frozen_add_splats_from_export) {
                params_json["exclude_frozen_add_splats_from_export"] =
                    true;
            }
            if (params.freeze_lr_scale != 0.0f) {
                params_json["freeze_lr_scale"] =
                    params.freeze_lr_scale;
            }
            if (!params.disabled_camera_uids.empty()) {
                params_json["disabled_camera_uids"] =
                    params.disabled_camera_uids;
            }
            const auto paths_to_utf8 =
                [](const std::vector<std::filesystem::path>& paths) {
                    auto array = nlohmann::json::array();
                    for (const auto& path : paths) {
                        array.push_back(
                            lfs::core::path_to_utf8(path));
                    }
                    return array;
                };
            if (!params.view_paths.empty()) {
                params_json["view_paths"] =
                    paths_to_utf8(params.view_paths);
            }
            if (params.import_cameras_path) {
                params_json["import_cameras_path"] =
                    lfs::core::path_to_utf8(
                        *params.import_cameras_path);
            }
            if (!params.add_splat_paths.empty()) {
                params_json["add_splat_paths"] =
                    paths_to_utf8(params.add_splat_paths);
            }
            if (!params.add_splat_freeze.empty()) {
                params_json["add_splat_freeze"] =
                    params.add_splat_freeze;
            }

            const std::string params_text = params_json.dump();
            destination.write(
                params_text.data(),
                static_cast<std::streamsize>(params_text.size()));
            const auto end_pos = destination.tellp();
            if (!destination ||
                end_pos == std::streampos(-1) ||
                end_pos < params_pos) {
                return checkpoint_stream_error(
                    lfs::ErrorCode::DataLoss,
                    "Cannot serialize checkpoint: destination write failed",
                    LFS_SOURCE_SITE_CURRENT());
            }

            header.params_json_offset =
                static_cast<std::uint64_t>(
                    static_cast<std::streamoff>(params_pos));
            header.params_json_size =
                static_cast<std::uint64_t>(end_pos - params_pos);
            destination.seekp(header_pos);
            destination.write(
                reinterpret_cast<const char*>(&header),
                sizeof(header));
            destination.seekp(end_pos);
            if (!destination) {
                return checkpoint_stream_error(
                    lfs::ErrorCode::DataLoss,
                    "Cannot serialize checkpoint: header finalization failed",
                    LFS_SOURCE_SITE_CURRENT());
            }

            return CheckpointStreamResult{
                .header = header,
                .bytes = static_cast<std::uint64_t>(
                    static_cast<std::streamoff>(end_pos)),
            };
        } catch (const std::exception& error) {
            // LFS-CENSUS-OK(empty-catch): normalize the exception into a typed checkpoint error.
            const bool layout_changed =
                std::string_view(error.what()).find("layout changed") !=
                std::string_view::npos;
            return checkpoint_stream_error(
                layout_changed
                    ? lfs::ErrorCode::FailedPrecondition
                    : lfs::ErrorCode::Internal,
                std::string("Serialize checkpoint failed: ") +
                    error.what(),
                LFS_SOURCE_SITE_CURRENT());
        }
    }

    std::expected<int, std::string> load_checkpoint(
        const std::filesystem::path& path,
        IStrategy& strategy,
        lfs::core::param::TrainingParameters& params,
        BilateralGrid* bilateral_grid,
        PPISP* ppisp,
        PPISPControllerPool* ppisp_controller_pool,
        ADMMSparsityOptimizer* sparsity_optimizer,
        lfs::core::SplatTensorAllocator tensor_allocator,
        POPSpaController* popspa_controller,
        AdamOptimizer* popspa_optimizer) {
        std::ifstream file;
        if (!lfs::core::open_file_for_read(
                path, std::ios::binary, file)) {
            return std::unexpected(
                "Failed to open: " +
                lfs::core::path_to_utf8(path));
        }

        std::error_code size_error;
        const auto file_size =
            std::filesystem::file_size(path, size_error);
        if (size_error) {
            return std::unexpected(
                "Failed to inspect checkpoint size: " +
                size_error.message());
        }
        return load_checkpoint(
            file, file_size, strategy, params, bilateral_grid,
            ppisp, ppisp_controller_pool, sparsity_optimizer,
            std::move(tensor_allocator),
            lfs::core::path_to_utf8(path), nullptr,
            popspa_controller, popspa_optimizer);
    }

    CheckpointLoadResult load_checkpoint(
        std::istream& file,
        const std::uint64_t file_size,
        IStrategy& strategy,
        lfs::core::param::TrainingParameters& params,
        BilateralGrid* bilateral_grid,
        PPISP* ppisp,
        PPISPControllerPool* ppisp_controller_pool,
        ADMMSparsityOptimizer* sparsity_optimizer,
        lfs::core::SplatTensorAllocator tensor_allocator,
        const std::string_view source_name,
        lfs::core::SplatData* preloaded_model,
        POPSpaController* popspa_controller,
        AdamOptimizer* popspa_optimizer) {
        try {
            const auto load_started = std::chrono::steady_clock::now();
            if (bool(popspa_controller) != bool(popspa_optimizer))
                return std::unexpected("POPSpa checkpoint load requires paired controller and Adam destinations");
            const auto milliseconds = [](const auto begin, const auto end) {
                return std::chrono::duration<double, std::milli>(end - begin).count();
            };
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
            auto saved_optimization = loaded_params.optimization;
            if (header.params_json_size > 0) {
                file.seekg(static_cast<std::streamoff>(header.params_json_offset));
                std::string params_str(header.params_json_size, '\0');
                file.read(params_str.data(), static_cast<std::streamsize>(header.params_json_size));
                if (!file)
                    return std::unexpected("Invalid checkpoint: truncated parameter JSON");

                const auto cli_data_path =
                    loaded_params.dataset.data_path;
                const auto cli_output_path =
                    loaded_params.dataset.output_path;
                const auto cli_output_name =
                    loaded_params.dataset.output_name;
                const auto runtime_headless =
                    loaded_params.optimization.headless;
                const auto runtime_auto_train =
                    loaded_params.optimization.auto_train;
                const auto runtime_no_splash =
                    loaded_params.optimization.no_splash;
                const auto runtime_debug_python =
                    loaded_params.optimization.debug_python;
                const auto runtime_debug_python_port =
                    loaded_params.optimization.debug_python_port;
                const auto runtime_config_file =
                    loaded_params.optimization.config_file;
                const auto cli_iterations_set =
                    loaded_params.cli_iterations_set;
                const auto cli_iterations =
                    loaded_params.optimization.iterations;
                const auto cli_bg_color_set =
                    loaded_params.cli_bg_color_set;
                const auto cli_bg_color =
                    loaded_params.optimization.bg_color;

                loaded_params = lfs::core::parse_checkpoint_params_json(params_str, std::move(loaded_params));
                saved_optimization = loaded_params.optimization;

                if (!cli_data_path.empty())
                    loaded_params.dataset.data_path = cli_data_path;
                if (!cli_output_path.empty())
                    loaded_params.dataset.output_path = cli_output_path;
                if (!cli_output_name.empty())
                    loaded_params.dataset.output_name = cli_output_name;
                loaded_params.optimization.headless =
                    runtime_headless;
                loaded_params.optimization.auto_train =
                    runtime_auto_train;
                loaded_params.optimization.no_splash =
                    runtime_no_splash;
                loaded_params.optimization.debug_python =
                    runtime_debug_python;
                loaded_params.optimization.debug_python_port =
                    runtime_debug_python_port;
                loaded_params.optimization.config_file =
                    runtime_config_file;
                if (cli_iterations_set)
                    loaded_params.optimization.iterations =
                        cli_iterations;
                loaded_params.cli_iterations_set =
                    cli_iterations_set;
                if (cli_bg_color_set)
                    loaded_params.optimization.bg_color =
                        cli_bg_color;
                loaded_params.cli_bg_color_set =
                    cli_bg_color_set;
                lfs::core::param::apply_explicit_training_overrides(
                    loaded_params, loaded_params.overrides);
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
            if (!(loaded_params.freeze_lr_scale >= 0.0f && loaded_params.freeze_lr_scale <= 1.0f)) {
                return std::unexpected("Invalid checkpoint parameters: freeze_lr_scale must be within [0, 1]");
            }
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
            const auto splat_started = std::chrono::steady_clock::now();
            lfs::core::SplatData loaded_model;
            bool reused_preloaded_model = false;
            if (preloaded_model) {
                if (static_cast<uint64_t>(preloaded_model->size()) != header.num_gaussians)
                    throw std::runtime_error("Invalid checkpoint: preloaded model count does not match header");
                if (preloaded_model->get_max_sh_degree() != header.sh_degree)
                    throw std::runtime_error("Invalid checkpoint: preloaded model SH degree does not match header");
                lfs::core::SplatData::skip_serialized(file);
                loaded_model = std::move(*preloaded_model);
                reused_preloaded_model = true;
            } else {
                loaded_model.deserialize(
                    file,
                    make_checkpoint_tensor_allocator(std::move(tensor_allocator), target_capacity));
            }
            if (static_cast<uint64_t>(loaded_model.size()) != header.num_gaussians)
                throw std::runtime_error("Invalid checkpoint: model count does not match header");
            if (loaded_model.get_max_sh_degree() != header.sh_degree)
                throw std::runtime_error("Invalid checkpoint: model SH degree does not match header");
            const auto splat_finished = std::chrono::steady_clock::now();

            auto loaded_strategy_result = StrategyFactory::instance().create(saved_type, loaded_model);
            if (!loaded_strategy_result)
                throw std::runtime_error("Cannot construct checkpoint strategy: " + loaded_strategy_result.error());
            auto loaded_strategy = std::move(*loaded_strategy_result);
            auto* checkpoint_adopter = dynamic_cast<ICheckpointStateAdopter*>(&strategy);
            if (checkpoint_adopter && checkpoint_adopter->has_checkpoint_runtime_state())
                loaded_strategy->initialize(loaded_params.optimization);
            else
                loaded_strategy->set_optimization_params(loaded_params.optimization);
            const auto strategy_initialized_at = std::chrono::steady_clock::now();
            loaded_strategy->deserialize(file);
            const auto strategy_deserialized_at = std::chrono::steady_clock::now();
            if (!checkpoint_adopter || !checkpoint_adopter->can_adopt_checkpoint_state(*loaded_strategy)) {
                throw std::runtime_error(
                    "Strategy does not support transactional checkpoint state adoption");
            }
            // Cold load (model-only, no optimizer constructed) is valid: the regression
            // allocator path and early resume scaffolding never call initialize(). Do not
            // dereference get_optimizer() unless the loaded strategy actually owns one.
            if (auto* loaded_adopter = dynamic_cast<ICheckpointStateAdopter*>(loaded_strategy.get());
                loaded_adopter && loaded_adopter->has_checkpoint_runtime_state()) {
                loaded_strategy->get_optimizer().set_frozen_lr_scale(loaded_params.freeze_lr_scale);
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

            // Sparsity ADMM state (if present in checkpoint)
            std::unique_ptr<ADMMSparsityOptimizer> loaded_sparsity;
            const bool has_sparsity =
                header.version >= lfs::core::CHECKPOINT_VERSION_HAS_SPARSITY &&
                has_flag(header.flags, CheckpointFlags::HAS_SPARSITY);
            if (has_sparsity) {
                if (sparsity_optimizer) {
                    auto parsed = std::make_unique<ADMMSparsityOptimizer>(
                        sparsity_config_from_params(loaded_params.optimization));
                    parsed->deserialize(file);
                    if (parsed->state_size() != static_cast<size_t>(loaded_model.size()))
                        throw std::runtime_error("Invalid checkpoint: sparsity ADMM state does not match model count");
                    loaded_sparsity = std::move(parsed);
                } else {
                    if (ADMMSparsityOptimizer::consume_checkpoint(file) !=
                        static_cast<size_t>(loaded_model.size())) {
                        throw std::runtime_error("Invalid checkpoint: sparsity ADMM state does not match model count");
                    }
                    LOG_WARN("Checkpoint has sparsity ADMM state but none provided - skipping data");
                }
            }

            const bool has_popspa = has_flag(header.flags, CheckpointFlags::HAS_POPSPA);
            std::unique_ptr<POPSpaController> loaded_popspa;
            std::unique_ptr<AdamOptimizer> loaded_popspa_optimizer;
            if (has_popspa) {
                if (header.params_json_size == 0)
                    throw std::runtime_error("Invalid POPSpa checkpoint: missing saved parameters");
                if (saved_optimization.popspa_seed != loaded_params.optimization.popspa_seed ||
                    saved_optimization.popspa_ssim_weight != loaded_params.optimization.popspa_ssim_weight)
                    throw std::runtime_error("Invalid POPSpa checkpoint: resume overrides change seed or photometric loss");
                loaded_popspa = std::make_unique<POPSpaController>();
                auto parsed = loaded_popspa->deserialize(file);
                if (!parsed)
                    throw std::runtime_error(lfs::format_for_developer(parsed.error()));
                loaded_popspa_optimizer = std::make_unique<AdamOptimizer>(loaded_model, AdamConfig{});
                loaded_popspa_optimizer->deserialize(file);
                // Gradients are transient and absent from checkpoint payloads.
                // Materialize every resident group before the commit so an empty
                // first view still advances the full Adam state after resume.
                for (const auto type : AdamOptimizer::all_param_types()) {
                    if (type == ParamType::ShN && loaded_model.max_sh_coeffs_rest() == 0)
                        continue;
                    (void)loaded_popspa_optimizer->get_grad(type);
                }
                validate_popspa_checkpoint(*loaded_popspa, *loaded_popspa_optimizer,
                                           saved_optimization, loaded_model, header.iteration);
                validate_popspa_checkpoint(*loaded_popspa, *loaded_popspa_optimizer,
                                           loaded_params.optimization, loaded_model, header.iteration);
            } else if (loaded_params.optimization.enable_sparsity &&
                       loaded_params.optimization.sparsity_method == lfs::core::param::SparsityMethod::POPSpa &&
                       (has_sparsity || uint64_t(header.iteration) > loaded_params.optimization.iterations)) {
                throw std::runtime_error("Invalid POPSpa checkpoint: missing controller state");
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

            if (bilateral_grid && loaded_bilateral_grid &&
                bilateral_grid->parameterization() != loaded_bilateral_grid->parameterization()) {
                throw std::runtime_error(
                    std::string("BilateralGrid parameterization mismatch: checkpoint is ") +
                    bilateral_grid_parameterization_name(loaded_bilateral_grid->parameterization()) +
                    ", current is " +
                    bilateral_grid_parameterization_name(bilateral_grid->parameterization()));
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
            if (loaded_sparsity) {
                sparsity_optimizer->adopt_checkpoint_state(*loaded_sparsity);
                LOG_INFO("Sparsity ADMM state restored: {} rows", sparsity_optimizer->state_size());
            } else if (sparsity_optimizer && !has_sparsity) {
                sparsity_optimizer->reset();
            }

            if (popspa_controller && loaded_popspa) {
                popspa_controller->adopt_checkpoint_state(*loaded_popspa);
                popspa_optimizer->adopt_checkpoint_state(*loaded_popspa_optimizer);
            } else if (popspa_controller) {
                popspa_controller->reset();
            }

            LOG_INFO("Checkpoint loaded: {} ({} Gaussians, iter {})",
                     source_name, header.num_gaussians, header.iteration);
            LOG_DEBUG(
                "Checkpoint load stages: source={} gaussians={} model_reuse={} splat={:.3f} ms strategy_init={:.3f} ms strategy_deserialize={:.3f} ms commit={:.3f} ms total={:.3f} ms",
                source_name,
                header.num_gaussians,
                reused_preloaded_model ? "yes" : "no",
                milliseconds(splat_started, splat_finished),
                milliseconds(splat_finished, strategy_initialized_at),
                milliseconds(strategy_initialized_at, strategy_deserialized_at),
                milliseconds(strategy_deserialized_at, std::chrono::steady_clock::now()),
                milliseconds(load_started, std::chrono::steady_clock::now()));
            return header.iteration;

        } catch (const std::exception& e) {
            return std::unexpected(std::string("Load checkpoint failed: ") + e.what());
        } catch (...) {
            return std::unexpected("Load checkpoint failed: unknown exception");
        }
    }

} // namespace lfs::training
