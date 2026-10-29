/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include "core/collision.hpp"
#include "core/parameters.hpp"
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace lfs::core::args {

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

    LFS_CORE_API std::span<const OptimizationCliBinding> optimization_cli_bindings();
    LFS_CORE_API std::string optimization_cli_help(std::string_view flag);

    struct TrainingFlag {
        std::string name; // "--iter" or "-i"
        bool takes_value = false;
        std::string canonical_name; // Shared by aliases of the same parser option.
    };

    // Every option of the training command line, read from the parser itself.
    LFS_CORE_API std::vector<TrainingFlag> training_flags();
    LFS_CORE_API const TrainingFlag* training_flag(std::string_view argument);

    // Parsed argument modes
    struct TrainingMode {
        std::unique_ptr<param::TrainingParameters> params;
    };
    struct ConvertMode {
        param::ConvertParameters params;
    };
    struct VoxelizeMode {
        std::filesystem::path input_path, output_path;
        CollisionSettings settings;
        bool overwrite = false;
    };
    struct Mesh2SplatMode {
        param::Mesh2SplatParameters params;
    };
    struct PreprocessMode {
        param::PreprocessParameters params;
    };
    struct VideoMaskMode {
        param::VideoMaskParameters params;
    };
    struct CompareMode {
        param::CompareParameters params;
    };
    struct HelpMode {};
    struct VersionMode {};
    struct WarmupMode {}; // JIT compile PTX kernels and exit
    struct PluginMode {
        enum class Command { CREATE,
                             CHECK,
                             LIST };
        Command command;
        std::string name;
    };
    struct LichtMode {
        std::filesystem::path project_path;
        std::optional<std::filesystem::path> dataset_path;
    };
    struct BatchTrainingMode {
        param::BatchTrainingParameters params;
    };

    using ParsedArgs = std::variant<TrainingMode, ConvertMode, Mesh2SplatMode, PreprocessMode, VideoMaskMode, CompareMode,
                                    HelpMode, VersionMode, WarmupMode, PluginMode, LichtMode, VoxelizeMode,
                                    BatchTrainingMode>;

    LFS_CORE_API std::expected<ParsedArgs, std::string> parse_args(int argc, const char* const argv[]);

    // Legacy interface - prefer parse_args()
    LFS_CORE_API std::expected<std::unique_ptr<param::TrainingParameters>, std::string>
    parse_args_and_params(int argc, const char* const argv[], bool create_output_directory = true, bool validate_only = false);

} // namespace lfs::core::args
