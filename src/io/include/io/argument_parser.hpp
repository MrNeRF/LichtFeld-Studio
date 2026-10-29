/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/gpu_backend_fwd.hpp"

#include "core/collision.hpp"
#include "core/parameters.hpp"
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace lfs::io::args {

    enum class OptimizationCliParseType {
        Bool,
        Integer,
        Float,
        String,
        Enum,
    };

    struct OptimizationCliBinding {
        std::string_view flag;
        std::string_view property_id;
        OptimizationCliParseType parse_type;
        bool inverted = false;
        std::string_view help_suffix;
        std::string_view registry_default_alias;
        std::string_view cli_default_alias;
    };

    LFS_IO_API std::span<const OptimizationCliBinding> optimization_cli_bindings();
    LFS_IO_API std::string optimization_cli_help(std::string_view flag);

    struct TrainingFlag {
        std::string name; // "--iter" or "-i"
        bool takes_value = false;
        std::string canonical_name; // Shared by aliases of the same parser option.
    };

    // Every option of the training command line, read from the parser itself.
    LFS_IO_API std::vector<TrainingFlag> training_flags();
    LFS_IO_API const TrainingFlag* training_flag(std::string_view argument);

    // Parsed argument modes
    struct TrainingMode {
        std::unique_ptr<core::param::TrainingParameters> params;
    };
    struct ConvertMode {
        core::param::ConvertParameters params;
    };
    struct VoxelizeMode {
        std::filesystem::path input_path, output_path;
        core::CollisionSettings settings;
        bool overwrite = false;
    };
    struct Mesh2SplatMode {
        core::param::Mesh2SplatParameters params;
    };
    struct PreprocessMode {
        core::param::PreprocessParameters params;
    };
    struct VideoMaskMode {
        core::param::VideoMaskParameters params;
    };
    struct CompareMode {
        core::param::CompareParameters params;
    };
    struct HelpMode {};
    struct VersionMode {};
    struct WarmupMode {}; // JIT compile PTX kernels and exit
    struct TensorBackendSelftestMode {
        core::GpuBackend backend;
    };
    struct PluginMode {
        enum class Command { CREATE,
                             CHECK,
                             LIST };
        Command command;
        std::string name;
    };
    struct LichtMode {
        enum class Action { EMBED,
                            LIST_CHECKPOINTS,
                            PRUNE_CHECKPOINTS,
                            REMOVE_CHECKPOINTS };
        std::filesystem::path project_path;
        std::optional<std::filesystem::path> dataset_path;
        Action action = Action::EMBED;
        std::size_t keep_checkpoints = 0;
        std::vector<std::int32_t> checkpoint_iterations;
    };
    struct BatchTrainingMode {
        core::param::BatchTrainingParameters params;
    };

    using ParsedArgs = std::variant<TrainingMode, ConvertMode, Mesh2SplatMode, PreprocessMode, VideoMaskMode, CompareMode,
                                    HelpMode, VersionMode, WarmupMode, TensorBackendSelftestMode, PluginMode, LichtMode,
                                    VoxelizeMode, BatchTrainingMode>;

    LFS_IO_API std::expected<ParsedArgs, std::string> parse_args(int argc, const char* const argv[]);

    // Legacy interface - prefer parse_args()
    LFS_IO_API std::expected<std::unique_ptr<core::param::TrainingParameters>, std::string>
    parse_args_and_params(int argc, const char* const argv[], bool create_output_directory = true, bool validate_only = false);

} // namespace lfs::io::args
