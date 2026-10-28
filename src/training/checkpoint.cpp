/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "checkpoint.hpp"
#include "components/bilateral_grid.hpp"
#include "components/popspa_controller.hpp"
#include "components/ppisp.hpp"
#include "components/ppisp_controller_pool.hpp"
#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "core/native_fs.hpp"
#include "core/parameters.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data.hpp"
#include "core/tensor/internal/tensor_serialization.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "strategies/istrategy.hpp"
#include "strategies/strategy_factory.hpp"
#include "training_setup.hpp"
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
        // Opacity ADMM sparsity was removed, but checkpoints that carry its state
        // (HAS_SPARSITY) still load. The payload is a magic, a version and the z,
        // u and sigmoid(opacity) tensors, each float32 [N, 1]; it is discarded.
        constexpr uint32_t LEGACY_ADMM_MAGIC = 0x4C464153; // "LFAS"
        constexpr uint32_t LEGACY_ADMM_VERSION = 1;

        size_t consume_legacy_admm_state(std::istream& is) {
            uint32_t magic = 0;
            uint32_t version = 0;
            lfs::core::serialization_detail::read_exact(is, &magic, sizeof(magic), "sparsity-ADMM magic");
            lfs::core::serialization_detail::read_exact(is, &version, sizeof(version), "sparsity-ADMM version");
            if (magic != LEGACY_ADMM_MAGIC)
                throw std::runtime_error("Invalid sparsity ADMM checkpoint");
            if (version != LEGACY_ADMM_VERSION)
                throw std::runtime_error("Unsupported sparsity ADMM checkpoint version");
            lfs::core::Tensor z;
            lfs::core::Tensor u;
            lfs::core::Tensor opa_sigmoid;
            is >> z >> u >> opa_sigmoid;
            const auto valid = [&z](const lfs::core::Tensor& tensor) {
                return tensor.is_valid() && tensor.dtype() == lfs::core::DataType::Float32 && tensor.ndim() == 2 &&
                       tensor.size(0) > 0 && tensor.size(1) == 1 && tensor.shape() == z.shape();
            };
            if (!valid(z) || !valid(u) || !valid(opa_sigmoid))
                throw std::runtime_error("Invalid sparsity ADMM checkpoint tensor schema");
            return z.size(0);
        }

        void validate_popspa_checkpoint(
            const POPSpaController& controller,
            const AdamOptimizer& optimizer,
            const lfs::core::param::OptimizationParameters& params,
            const lfs::core::SplatData& model,
            const int iteration) {
            const size_t rows = model.size();
            if (const auto error = params.validate(); !error.empty())
                throw std::runtime_error("Invalid POPSpa checkpoint parameters: " + error);
            if (!params.enable_sparsity || !controller.is_initialized() || controller.state_size() != rows)
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

    namespace {
        // This cursor visits existing checkpoint sections without materializing the
        // Gaussian model or its optimizer. The ordinary resume reader is unchanged.
        template <typename T>
        T appearance_read(std::istream& source) {
            T value{};
            lfs::core::serialization_detail::read_exact(source, &value, sizeof(value), "appearance checkpoint cursor");
            return value;
        }

        void appearance_skip(std::istream& source, const uint64_t bytes) {
            lfs::core::serialization_detail::require_remaining_bytes(source, bytes, "appearance checkpoint cursor");
            source.seekg(static_cast<std::streamoff>(bytes), std::ios::cur);
            if (!source)
                throw std::runtime_error("Cannot seek in appearance checkpoint");
        }

        void appearance_skip_name(std::istream& source) {
            const auto bytes = appearance_read<uint32_t>(source);
            if (bytes > lfs::core::MAX_CHECKPOINT_STRATEGY_NAME_BYTES)
                throw std::runtime_error("Appearance checkpoint name is too long");
            appearance_skip(source, bytes);
        }

        void appearance_skip_optimizer(std::istream& source) {
            if (appearance_read<uint32_t>(source) != 0x4C464144)
                throw std::runtime_error("Invalid appearance checkpoint optimizer");
            const auto version = appearance_read<uint32_t>(source);
            if (version < 1 || version > 3)
                throw std::runtime_error("Unsupported appearance checkpoint optimizer version");
            appearance_skip(source, sizeof(float) * 2 + sizeof(double) * 3 + sizeof(size_t));
            const auto lr_count = appearance_read<uint32_t>(source);
            if (lr_count > AdamOptimizer::all_param_types().size())
                throw std::runtime_error("Invalid appearance checkpoint learning-rate count");
            for (uint32_t i = 0; i < lr_count; ++i) {
                appearance_skip_name(source);
                appearance_skip(source, sizeof(double));
            }
            const auto states = appearance_read<uint32_t>(source);
            if (states > AdamOptimizer::all_param_types().size())
                throw std::runtime_error("Invalid appearance checkpoint optimizer state count");
            for (uint32_t i = 0; i < states; ++i) {
                appearance_skip_name(source);
                appearance_skip(source, sizeof(int64_t) + sizeof(size_t) * 2);
                const auto bits = version >= 3 ? appearance_read<int32_t>(source) : 0;
                if (bits != 0 && bits != 8 && bits != 16)
                    throw std::runtime_error("Invalid appearance checkpoint moment codec");
                const int tensors = version == 1 || bits != 0 ? 2 : 4;
                for (int j = 0; j < tensors; ++j)
                    lfs::core::serialization_detail::skip_serialized_tensor(source);
            }
        }

        void appearance_skip_strategy(std::istream& source, const std::string_view strategy) {
            const auto start = source.tellg();
            const auto magic = appearance_read<uint32_t>(source);
            const auto version = appearance_read<uint32_t>(source);
            if (strategy == "igs+" && magic != 0x4C464947) {
                // Legacy igs+ checkpoints contain no strategy section.
                source.seekg(start);
                return;
            }
            if (!((strategy == "mcmc" && magic == 0x4C464D43 && version == 1) ||
                  (strategy == "mrnf" && magic == 0x4C464252 && version >= 1 && version <= 3) ||
                  (strategy == "igs+" && magic == 0x4C464947 && version <= 1)))
                throw std::runtime_error("Unsupported appearance checkpoint strategy");
            if (appearance_read<uint8_t>(source))
                appearance_skip_optimizer(source);
            if (appearance_read<uint8_t>(source)) {
                if (appearance_read<uint32_t>(source) != 0x4C465345 || appearance_read<uint32_t>(source) != 1)
                    throw std::runtime_error("Unsupported appearance checkpoint scheduler");
                appearance_skip(source, sizeof(double));
                const auto count = appearance_read<uint32_t>(source);
                if (count > AdamOptimizer::all_param_types().size())
                    throw std::runtime_error("Invalid appearance checkpoint scheduler parameter count");
                appearance_skip(source, count);
            }
            if (strategy == "igs+") {
                appearance_skip(source, sizeof(int64_t) + sizeof(int) * 2);
                const auto count = appearance_read<uint32_t>(source);
                appearance_skip(source, uint64_t{count} * sizeof(int64_t));
            }
            if (strategy == "igs+" || (strategy == "mrnf" && version >= 2)) {
                if (appearance_read<uint8_t>(source))
                    lfs::core::serialization_detail::skip_serialized_tensor(source);
            }
            if (strategy == "mrnf" && version >= 3)
                appearance_skip(source, sizeof(double) * 2);
        }

        void appearance_skip_grid(std::istream& source) {
            if (appearance_read<uint32_t>(source) != 0x4C464247)
                throw std::runtime_error("Invalid appearance checkpoint grid");
            const auto version = appearance_read<uint32_t>(source);
            if (version < 1 || version > 3)
                throw std::runtime_error("Unsupported appearance checkpoint grid version");
            const auto images = appearance_read<int>(source);
            if (images <= 0)
                throw std::runtime_error("Invalid appearance checkpoint grid image count");
            appearance_skip(source, sizeof(int) * 3);
            if (version == 1) {
                struct LegacyConfig {
                    double lr, beta1, beta2, eps;
                    int warmup_steps;
                    double warmup_start_factor, final_lr_factor;
                };
                appearance_skip(source, sizeof(LegacyConfig));
            } else {
                (void)appearance_read<uint32_t>(source);
                const auto bytes = appearance_read<uint32_t>(source);
                appearance_skip(source, bytes);
            }
            appearance_skip(source, sizeof(int64_t) + sizeof(double) * 2 + sizeof(int));
            for (int i = 0; i < 3; ++i)
                lfs::core::serialization_detail::skip_serialized_tensor(source);
            if (version >= 3)
                appearance_skip(source, uint64_t{static_cast<uint32_t>(images)} * sizeof(int64_t));
        }
    } // namespace

    lfs::Result<CheckpointAppearance> load_checkpoint_appearance(
        std::istream& source, const uint64_t source_bytes) {
        const auto failure = [](std::string detail) {
            return lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::DataLoss,
                .domain = lfs::ErrorDomain::Training,
                .user_message = "Could not restore appearance correction.",
                .detail = std::move(detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        };
        try {
            auto header = lfs::core::load_checkpoint_header(source, source_bytes);
            if (!header)
                return failure(header.error());
            CheckpointAppearance result;
            if (!has_flag(header->flags, CheckpointFlags::HAS_PPISP))
                return result;
            source.seekg(sizeof(CheckpointHeader));
            const auto name_bytes = appearance_read<uint32_t>(source);
            if (name_bytes == 0 || name_bytes > lfs::core::MAX_CHECKPOINT_STRATEGY_NAME_BYTES)
                return failure("Invalid appearance checkpoint strategy name");
            std::string strategy(name_bytes, '\0');
            lfs::core::serialization_detail::read_exact(source, strategy.data(), name_bytes, "appearance strategy name");
            lfs::core::SplatData::skip_serialized(source);
            appearance_skip_strategy(source, lfs::core::param::canonical_strategy_name(strategy));
            if (has_flag(header->flags, CheckpointFlags::HAS_BILATERAL_GRID))
                appearance_skip_grid(source);
            result.ppisp = std::make_unique<PPISP>(1);
            result.ppisp->deserialize(source);
            if (has_flag(header->flags, CheckpointFlags::HAS_PPISP_CONTROLLER)) {
                result.controller = std::make_unique<PPISPControllerPool>(result.ppisp->num_cameras(), 1);
                result.controller->deserialize(source);
            }
            // Publish the restored tensors to the viewport thread only after upload.
            LFS_CUDA_CHECK(cudaDeviceSynchronize());
            return result;
        } catch (const std::exception& error) {
            return lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::DataLoss,
                .domain = lfs::ErrorDomain::Training,
                .user_message = "Could not restore appearance correction.",
                .detail = error.what(),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }
    }

    lfs::Result<CheckpointStreamResult>
    serialize_checkpoint(
        std::ostream& destination,
        const int iteration,
        const IStrategy& strategy,
        const lfs::core::param::TrainingParameters& params,
        const BilateralGrid* bilateral_grid,
        const PPISP* ppisp,
        const PPISPControllerPool* ppisp_controller_pool,
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
            const bool save_popspa = popspa_controller && popspa_controller->is_initialized();
            if (save_popspa) {
                if (!popspa_optimizer)
                    throw std::runtime_error("POPSpa checkpoint requires its dedicated Adam");
                validate_popspa_checkpoint(*popspa_controller, *popspa_optimizer, params.optimization, model, iteration);
            } else if (params.optimization.enable_sparsity &&
                       uint64_t(iteration) > params.optimization.iterations) {
                throw std::runtime_error("POPSpa checkpoint is missing initialized controller state");
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

            const auto header_pos = destination.tellp();
            if (save_popspa)
                header.flags = header.flags | CheckpointFlags::HAS_POPSPA;
            // Only POPSpa state needs the newer layout; keep other checkpoints
            // readable by builds that predate it.
            header.version = save_popspa ? lfs::core::CHECKPOINT_VERSION_HAS_POPSPA
                                         : lfs::core::CHECKPOINT_VERSION_FIELDWISE_CONFIGS;
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
            lfs::core::native_fs::file_size(path, size_error);
        if (size_error) {
            return std::unexpected(
                "Failed to inspect checkpoint size: " +
                size_error.message());
        }
        return load_checkpoint(
            file, file_size, strategy, params, bilateral_grid,
            ppisp, ppisp_controller_pool,
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
            // Builds with both methods saved "opacity_admm" or "popspa". Builds
            // before POPSpa saved neither that key nor popspa_* settings; current
            // builds save only the popspa_* settings.
            bool saved_admm_sparsity = false;
            if (header.params_json_size > 0) {
                file.seekg(static_cast<std::streamoff>(header.params_json_offset));
                std::string params_str(header.params_json_size, '\0');
                file.read(params_str.data(), static_cast<std::streamsize>(header.params_json_size));
                if (!file)
                    return std::unexpected("Invalid checkpoint: truncated parameter JSON");
                if (const auto saved = nlohmann::json::parse(params_str, nullptr, false);
                    saved.is_object() && saved.contains("optimization") && saved.at("optimization").is_object()) {
                    const auto& optimization = saved.at("optimization");
                    const auto method = optimization.find("sparsity_method");
                    saved_admm_sparsity = method != optimization.end() ? *method != "popspa"
                                                                       : !optimization.contains("popspa_target_count");
                }

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
                // A preloaded model is a plain copy of the scene's, and the commit below swaps it into the scene
                // while the viewer may draw; the viewer renders only tensors in the caller's storage.
                if (auto migrated = migrateTrainingModelToAllocator(loaded_params, loaded_model, tensor_allocator);
                    !migrated) {
                    throw std::runtime_error(migrated.error());
                }
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
            // Dataset-derived state belongs to the candidate too: MRNF builds
            // its camera hull and far-field mask during initialization. Adopting
            // a candidate without cameras would disable those protections.
            loaded_strategy->set_training_dataset(strategy.get_training_dataset());
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

            // Opacity ADMM state written by older builds is consumed and dropped.
            const bool has_sparsity =
                header.version >= lfs::core::CHECKPOINT_VERSION_HAS_SPARSITY &&
                has_flag(header.flags, CheckpointFlags::HAS_SPARSITY);
            if (has_sparsity) {
                if (consume_legacy_admm_state(file) != static_cast<size_t>(loaded_model.size()))
                    throw std::runtime_error("Invalid checkpoint: sparsity ADMM state does not match model count");
                LOG_WARN("Checkpoint carries opacity ADMM state from an older version; it is no longer used");
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
                       (has_sparsity || uint64_t(header.iteration) > loaded_params.optimization.iterations)) {
                if (!saved_admm_sparsity)
                    throw std::runtime_error("Invalid POPSpa checkpoint: missing controller state");
                // Saved during or after opacity ADMM sparsification. That phase no
                // longer exists, so resume the model without sparsification.
                LOG_WARN("Checkpoint was saved during opacity ADMM sparsification, which is no longer supported; "
                         "resuming without sparsification");
                loaded_params.optimization.enable_sparsity = false;
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
