/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "core/argument_parser.hpp"
#include "core/optimization_properties.hpp"
#include "core/parameter_manager.hpp"
#include "core/parameters.hpp"
#include "core/path_utils.hpp"
#include "core/property_registry.hpp"
#include "io/project_path.hpp"

using lfs::core::param::apply_explicit_training_overrides;

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

    std::string make_test_path(const char* name) {
        const auto path = std::filesystem::temp_directory_path() / name;
        std::filesystem::create_directories(path);
        return path.string();
    }

} // namespace

// An untrained .licht on --data-path is its own destination, so
// --output-path is optional and the project stays bound for the run.
TEST(ArgumentParserTest, DataPathLichtWithoutOutputPathBindsProject) {
    const auto directory =
        make_test_path("lfs_arg_parser_dataset_project");
    const auto project =
        std::filesystem::path(directory) / "project.licht";
    std::ofstream(project).put('\n');
    const auto project_text = project.string();

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "-d",
        project_text.c_str(),
    };
    auto parsed = lfs::core::args::parse_args_and_params(
        static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed) << parsed.error();
    EXPECT_EQ((*parsed)->dataset_project, project);
    EXPECT_TRUE((*parsed)->dataset.data_path.empty());
    EXPECT_TRUE((*parsed)->dataset.output_path.empty());
    EXPECT_FALSE((*parsed)->dataset.output_path_explicit);
}

TEST(ArgumentParserTest, RejectsUnknownLogLevels) {
    const auto data_path = make_test_path("lfs_arg_parser_log_level_data");
    const auto output_path = make_test_path("lfs_arg_parser_log_level_output");
    for (const std::string level : {"nonsense", "5", "", "fatal"}) {
        const char* argv[] = {
            "LichtFeld-Studio", "--headless", "--data-path", data_path.c_str(),
            "--output-path", output_path.c_str(), "--log-level", level.c_str()};
        const auto parsed = lfs::core::args::parse_args_and_params(
            static_cast<int>(std::size(argv)), argv);
        ASSERT_FALSE(parsed) << level;
        EXPECT_NE(parsed.error().find("log level"), std::string::npos) << parsed.error();
        EXPECT_NE(parsed.error().find(level), std::string::npos) << parsed.error();
    }
}

TEST(ArgumentParserTest, LogLevelsAreCaseInsensitiveAndKeepAliases) {
    const auto data_path = make_test_path("lfs_arg_parser_log_alias_data");
    const auto output_path = make_test_path("lfs_arg_parser_log_alias_output");
    for (const std::string level : {"TRACE", "performance", "WARNING"}) {
        const char* argv[] = {
            "LichtFeld-Studio", "--headless", "--data-path", data_path.c_str(),
            "--output-path", output_path.c_str(), "--log-level", level.c_str()};
        const auto parsed = lfs::core::args::parse_args_and_params(
            static_cast<int>(std::size(argv)), argv);
        ASSERT_TRUE(parsed) << parsed.error();
    }
}

TEST(ArgumentParserTest,
     GuiProjectAndResumeLichtSelectProjectOpenFlow) {
    const auto directory =
        make_test_path("lfs_arg_parser_project");
    const auto project =
        std::filesystem::path(directory) /
        "session.licht";
    std::ofstream(project).put('\n');
    const auto project_text = project.string();

    const char* project_argv[] = {
        "LichtFeld-Studio",
        "-v",
        project_text.c_str(),
    };
    auto project_parsed =
        lfs::core::args::parse_args_and_params(
            static_cast<int>(
                std::size(project_argv)),
            project_argv);
    ASSERT_TRUE(project_parsed)
        << project_parsed.error();
    EXPECT_EQ(
        (*project_parsed)->project_path,
        project);
    EXPECT_FALSE(
        (*project_parsed)->resume_project);

    const char* resume_argv[] = {
        "LichtFeld-Studio",
        "--resume",
        project_text.c_str(),
    };
    auto resume_parsed =
        lfs::core::args::parse_args_and_params(
            static_cast<int>(
                std::size(resume_argv)),
            resume_argv);
    ASSERT_TRUE(resume_parsed)
        << resume_parsed.error();
    EXPECT_EQ(
        (*resume_parsed)->resume_project,
        project);
    EXPECT_FALSE(
        (*resume_parsed)->resume_checkpoint);
}

TEST(ArgumentParserTest, HeadlessResumeSelectsEmbeddedCheckpointFlow) {
    const auto directory =
        make_test_path(
            "lfs_arg_parser_headless_project");
    const auto project =
        std::filesystem::path(directory) /
        "session.licht";
    std::ofstream(project).put('\n');
    const auto project_text = project.string();
    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--resume",
        project_text.c_str(),
    };

    auto parsed =
        lfs::core::args::parse_args_and_params(
            static_cast<int>(std::size(argv)),
            argv);
    ASSERT_TRUE(parsed)
        << parsed.error();
    EXPECT_EQ((*parsed)->resume_project, project);
    EXPECT_FALSE((*parsed)->project_path);
}

TEST(ArgumentParserTest, RejectsOutputNamePathComponents) {
    const auto data_path = make_test_path("lfs_arg_parser_output_name_data");
    const auto output_path = make_test_path("lfs_arg_parser_output_name_output");
    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--train",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--output-name",
        "../outside",
    };

    auto parsed = lfs::core::args::parse_args_and_params(
        static_cast<int>(std::size(argv)), argv);
    ASSERT_FALSE(parsed);
    EXPECT_NE(parsed.error().find("output-name"), std::string::npos);
}

TEST(ArgumentParserTest,
     TrainingSaveProjectAtIterLeavesPathEmptyWithoutSaveProjectPath) {
    const auto data_path =
        make_test_path("lfs_arg_parser_save_project_data");
    const auto output_path =
        make_test_path("lfs_arg_parser_save_project_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "-d",
        data_path.c_str(),
        "-o",
        output_path.c_str(),
        "--save-project-at-iter",
        "7000",
    };
    auto parsed = lfs::core::args::parse_args_and_params(
        static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    EXPECT_EQ(
        (*parsed)->save_project_at_iteration,
        std::optional<size_t>(7000));
    EXPECT_TRUE((*parsed)->save_project_path.empty());
}

TEST(ArgumentParserTest,
     ResumeSaveProjectAtIterLeavesPathEmptyWithoutOutput) {
    const auto directory = make_test_path(
        "lfs_arg_parser_resume_save_project");
    const auto project =
        std::filesystem::path(directory) / "x.licht";
    std::ofstream(project).put('\n');
    const auto project_text = project.string();

    const char* argv[] = {
        "LichtFeld-Studio",
        "--resume",
        project_text.c_str(),
        "--headless",
        "--save-project-at-iter",
        "7000",
    };
    auto parsed = lfs::core::args::parse_args_and_params(
        static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    EXPECT_EQ(
        (*parsed)->save_project_at_iteration,
        std::optional<size_t>(7000));
    EXPECT_TRUE((*parsed)->save_project_path.empty());
    EXPECT_NE(
        (*parsed)->save_project_path,
        std::filesystem::path("project.licht"));
}

TEST(ArgumentParserTest, RemovedProjectAndRecoverFlagsAreUnknown) {
    const auto directory =
        make_test_path(
            "lfs_arg_parser_recover_project");
    const auto project =
        std::filesystem::path(directory) /
        "session.licht";
    std::ofstream(project).put('\n');
    const auto project_text = project.string();
    const char* project_flag[] = {
        "LichtFeld-Studio",
        "--project",
        project_text.c_str(),
    };
    auto project_parsed =
        lfs::core::args::parse_args_and_params(
            static_cast<int>(
                std::size(project_flag)),
            project_flag);
    EXPECT_FALSE(project_parsed);

    const char* recover_flag[] = {
        "LichtFeld-Studio",
        "--recover",
    };
    auto recover_parsed =
        lfs::core::args::parse_args_and_params(
            static_cast<int>(
                std::size(recover_flag)),
            recover_flag);
    EXPECT_FALSE(recover_parsed);
}

TEST(ArgumentParserTest, BarePositionalPlyAndLichtFollowViewFlag) {
    const auto directory =
        make_test_path("lfs_arg_parser_bare_positional");
    const auto ply =
        std::filesystem::path(directory) / "model.ply";
    const auto project =
        std::filesystem::path(directory) / "session.licht";
    std::ofstream(ply).put('\n');
    std::ofstream(project).put('\n');
    const auto ply_text = ply.string();
    const auto project_text = project.string();

    const char* ply_argv[] = {
        "LichtFeld-Studio",
        ply_text.c_str(),
    };
    auto ply_parsed = lfs::core::args::parse_args_and_params(
        static_cast<int>(std::size(ply_argv)), ply_argv);
    ASSERT_TRUE(ply_parsed) << ply_parsed.error();
    ASSERT_EQ((*ply_parsed)->view_paths.size(), 1u);
    EXPECT_EQ((*ply_parsed)->view_paths.front(), ply);
    EXPECT_FALSE((*ply_parsed)->project_path);

    const char* project_argv[] = {
        "LichtFeld-Studio",
        project_text.c_str(),
    };
    auto project_parsed = lfs::core::args::parse_args_and_params(
        static_cast<int>(std::size(project_argv)), project_argv);
    ASSERT_TRUE(project_parsed) << project_parsed.error();
    EXPECT_EQ((*project_parsed)->project_path, project);
    EXPECT_TRUE((*project_parsed)->view_paths.empty());

    const auto unpublished =
        std::filesystem::path(directory) /
        "session.project-write.1.tmp.licht";
    std::ofstream(unpublished).put('\n');
    const auto unpublished_text = unpublished.string();
    const char* unpublished_argv[] = {
        "LichtFeld-Studio",
        unpublished_text.c_str(),
    };
    auto unpublished_parsed = lfs::core::args::parse_args_and_params(
        static_cast<int>(std::size(unpublished_argv)), unpublished_argv);
    ASSERT_FALSE(unpublished_parsed);
    EXPECT_EQ(
        unpublished_parsed.error(),
        lfs::io::project::unpublishedLichtUserMessage(unpublished));
}

TEST(ArgumentParserTest, UsdFilesAcceptExplicitAndBareViewPaths) {
    const auto directory = std::filesystem::temp_directory_path() / "lfs_arg_parser_usd" /
                           std::filesystem::path(u8"splat \u00e8 \u6d4b\u8bd5");
    std::filesystem::create_directories(directory);

    // Parsing only routes paths; USD decoding is covered by UsdFormatTest.
    for (const auto* extension : {".usd", ".usda", ".usdc", ".usdz",
                                  ".USD", ".USDA", ".USDC", ".USDZ",
                                  ".Usd", ".UsdA", ".UsdC", ".UsdZ"}) {
        const auto path = directory / (std::string("model with spaces") + extension);
        std::ofstream file(path);
        file.put('\n');
        file.close();
        ASSERT_TRUE(file.good());
        const auto path_text = lfs::core::path_to_utf8(path);

        for (const auto* flag : {"-v", "--view", ""}) {
            SCOPED_TRACE(path_text + " via " + (flag[0] ? flag : "bare path"));
            std::vector<const char*> argv = {"LichtFeld-Studio"};
            if (flag[0])
                argv.push_back(flag);
            argv.push_back(path_text.c_str());

            auto parsed = lfs::core::args::parse_args_and_params(
                static_cast<int>(argv.size()), argv.data());
            ASSERT_TRUE(parsed) << parsed.error();
            EXPECT_EQ((*parsed)->view_paths, std::vector<std::filesystem::path>{path});
            EXPECT_FALSE((*parsed)->project_path);
            EXPECT_FALSE((*parsed)->resume_checkpoint);
            EXPECT_FALSE((*parsed)->resume_project);
        }
    }
}

TEST(ArgumentParserTest, ViewDirectoryIncludesUsdAndPreservesFiltering) {
    const auto directory = std::filesystem::temp_directory_path() / "lfs_arg_parser_usd_directory";
    std::filesystem::create_directories(directory);
    std::vector<std::filesystem::path> expected;
    for (const auto* name : {"d.usdz", "b.USDA", "c.UsdC", "a.usd", "f.ply", "e.obj"}) {
        const auto path = directory / name;
        std::ofstream file(path);
        file.put('\n');
        file.close();
        ASSERT_TRUE(file.good());
        expected.push_back(path);
    }
    const auto ignored_directory = directory / "nested.usd";
    std::filesystem::create_directories(ignored_directory);
    for (const auto& path : {directory / "notes.txt", directory / "project.licht",
                             ignored_directory / "nested.usda"}) {
        std::ofstream file(path);
        file.put('\n');
        file.close();
        ASSERT_TRUE(file.good());
    }
    std::sort(expected.begin(), expected.end());
    const auto path_text = lfs::core::path_to_utf8(directory);

    for (const auto* flag : {"-v", "--view", ""}) {
        SCOPED_TRACE(flag[0] ? flag : "bare directory");
        std::vector<const char*> argv = {"LichtFeld-Studio"};
        if (flag[0])
            argv.push_back(flag);
        argv.push_back(path_text.c_str());
        auto parsed = lfs::core::args::parse_args_and_params(
            static_cast<int>(argv.size()), argv.data());
        ASSERT_TRUE(parsed) << parsed.error();
        EXPECT_EQ((*parsed)->view_paths, expected);
        EXPECT_FALSE((*parsed)->project_path);
    }
}

TEST(ArgumentParserTest, GuiViewProjectExtensionIsCaseInsensitive) {
    const auto directory = make_test_path("lfs_arg_parser_view_project");
    const auto project = std::filesystem::path(directory) / "session.LICHT";
    std::ofstream(project).put('\n');
    const auto project_text = project.string();
    const char* argv[] = {"LichtFeld-Studio", "-v", project_text.c_str()};
    auto parsed = lfs::core::args::parse_args_and_params(
        static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed) << parsed.error();
    EXPECT_EQ((*parsed)->project_path, project);
    EXPECT_TRUE((*parsed)->view_paths.empty());
}

TEST(ArgumentParserTest,
     ViewAndResumeRejectUnpublishedWriteTempLicht) {
    // Would fail if -v / --resume still accepted
    // *.project-write.*.tmp.licht as a published project.
    const auto directory =
        make_test_path("lfs_arg_parser_unpublished");
    const auto temp_project =
        std::filesystem::path(directory) /
        "project.project-write.1.2.3.tmp.licht";
    std::ofstream(temp_project).put('\n');
    const auto temp_text = temp_project.string();

    const char* view_argv[] = {
        "LichtFeld-Studio",
        "-v",
        temp_text.c_str(),
    };
    auto view_parsed =
        lfs::core::args::parse_args_and_params(
            static_cast<int>(std::size(view_argv)),
            view_argv);
    ASSERT_FALSE(view_parsed);
    EXPECT_NE(
        view_parsed.error().find("project.licht"),
        std::string::npos);

    const char* resume_argv[] = {
        "LichtFeld-Studio",
        "--resume",
        temp_text.c_str(),
    };
    auto resume_parsed =
        lfs::core::args::parse_args_and_params(
            static_cast<int>(std::size(resume_argv)),
            resume_argv);
    ASSERT_FALSE(resume_parsed);
    EXPECT_NE(
        resume_parsed.error().find("project.licht"),
        std::string::npos);
}

TEST(ArgumentParserMetadataTest, OptimizationFlagBindingsResolveWithCompatibleTypes) {
    using lfs::core::args::OptimizationCliParseType;
    using lfs::core::prop::PropType;

    lfs::core::param::ensure_optimization_properties_registered();
    std::set<std::string_view> flags;
    for (const auto& binding : lfs::core::args::optimization_cli_bindings()) {
        SCOPED_TRACE(binding.flag);
        EXPECT_TRUE(flags.insert(binding.flag).second);

        const auto meta = lfs::core::prop::PropertyRegistry::instance().get_property(
            "optimization", std::string(binding.property_id));
        ASSERT_TRUE(meta.has_value());

        switch (binding.parse_type) {
        case OptimizationCliParseType::Bool:
            EXPECT_EQ(meta->type, PropType::Bool);
            break;
        case OptimizationCliParseType::Integer:
            EXPECT_TRUE(meta->type == PropType::Int || meta->type == PropType::SizeT);
            break;
        case OptimizationCliParseType::Float:
            EXPECT_EQ(meta->type, PropType::Float);
            break;
        case OptimizationCliParseType::String:
            EXPECT_EQ(meta->type, PropType::String);
            break;
        case OptimizationCliParseType::Enum:
            EXPECT_EQ(meta->type, PropType::Enum);
            break;
        }
    }
}

TEST(ArgumentParserMetadataTest, BuiltHelpContainsRegistryDescriptionsAndDefaults) {
    lfs::core::param::ensure_optimization_properties_registered();
    for (const std::string_view flag : {"--iter", "--strategy", "--depth-loss-weight"}) {
        SCOPED_TRACE(flag);
        const auto binding = std::ranges::find(
            lfs::core::args::optimization_cli_bindings(), flag,
            &lfs::core::args::OptimizationCliBinding::flag);
        ASSERT_NE(binding, lfs::core::args::optimization_cli_bindings().end());
        const auto meta = lfs::core::prop::PropertyRegistry::instance().get_property(
            "optimization", std::string(binding->property_id));
        ASSERT_TRUE(meta.has_value());

        const auto help = lfs::core::args::optimization_cli_help(flag);
        EXPECT_NE(help.find(meta->description), std::string::npos);
        EXPECT_NE(help.find("(default: "), std::string::npos);
    }
}

TEST(ArgumentParserTest, CliOutputPathSetsExplicitAndSurvivesCreateForDataset) {
    const auto data_path = make_test_path("lfs_arg_parser_cli_output_data");
    const auto output_path = make_test_path("lfs_arg_parser_cli_output_out");

    const char* argv[] = {
        "LichtFeld-Studio",
        "-d",
        data_path.c_str(),
        "-o",
        output_path.c_str(),
        "--export",
        "ply",
    };
    auto parsed = lfs::core::args::parse_args_and_params(
        static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();
    EXPECT_TRUE((*parsed)->dataset.output_path_explicit);
    EXPECT_EQ((*parsed)->dataset.output_path, std::filesystem::path(output_path));
    ASSERT_EQ((*parsed)->export_formats.size(), 1u);
    EXPECT_EQ((*parsed)->export_formats.front(), lfs::core::param::OutputFormat::PLY);

    lfs::vis::ParameterManager manager;
    const auto load_result = manager.ensureLoaded();
    ASSERT_TRUE(load_result.has_value()) << load_result.error();
    manager.setSessionDefaults(**parsed);
    const auto recreated = manager.createForDataset(
        "/tmp/override_dataset", "/tmp/override_output");
    EXPECT_TRUE(recreated.dataset.output_path_explicit);
    EXPECT_EQ(recreated.dataset.output_path, std::filesystem::path("/tmp/override_output"));
    ASSERT_EQ(recreated.export_formats.size(), 1u);
    EXPECT_EQ(recreated.export_formats.front(), lfs::core::param::OutputFormat::PLY);
}

TEST(ArgumentParserTest, TrainingDefaultsApplyMaxWidthCap) {
    const auto data_path = make_test_path("lfs_arg_parser_default_data");
    const auto output_path = make_test_path("lfs_arg_parser_default_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str()};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    EXPECT_FALSE((*parsed)->cli_bg_color_set);
    EXPECT_EQ((*parsed)->dataset.max_width, 3840);
    EXPECT_EQ((*parsed)->dataset.resize_factor, 1);
    EXPECT_EQ((*parsed)->optimization.depth_loss_mode, "ssi");
    EXPECT_FLOAT_EQ((*parsed)->optimization.cropbox_lr_scale, 0.1f);
    EXPECT_FLOAT_EQ((*parsed)->optimization.cropbox_loss_weight, 0.1f);
    EXPECT_FLOAT_EQ((*parsed)->freeze_lr_scale, 0.0f);
    EXPECT_EQ((*parsed)->optimization.morton_reorder_interval, 5000u);
}

TEST(ArgumentParserTest, ExposureCorrectionDefaultsOnAndCanBeDisabled) {
    const auto data_path = make_test_path("lfs_arg_parser_exposure_correction_data");
    const auto output_path = make_test_path("lfs_arg_parser_exposure_correction_output");

    const char* default_argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str()};
    auto default_parsed = lfs::core::args::parse_args_and_params(
        static_cast<int>(std::size(default_argv)), default_argv);
    ASSERT_TRUE(default_parsed.has_value()) << default_parsed.error();
    EXPECT_TRUE((*default_parsed)->optimization.use_exposure_correction);
    EXPECT_FALSE((*default_parsed)->optimization.use_bilateral_grid);
    EXPECT_FALSE((*default_parsed)->optimization.use_ppisp);

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--no-exposure-correction"};
    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();
    EXPECT_FALSE((*parsed)->optimization.use_exposure_correction);
    EXPECT_FALSE((*parsed)->optimization.use_bilateral_grid);
    EXPECT_FALSE((*parsed)->optimization.use_ppisp);
    EXPECT_EQ((*parsed)->optimization.exposure_correction_grid_start_iter, 1000);
}

TEST(ArgumentParserTest, NoPpispExifExposureDisablesSeed) {
    const auto data_path = make_test_path("lfs_arg_parser_ppisp_exif_data");
    const auto output_path = make_test_path("lfs_arg_parser_ppisp_exif_output");

    const char* default_argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str()};
    auto default_parsed =
        lfs::core::args::parse_args_and_params(static_cast<int>(std::size(default_argv)), default_argv);
    ASSERT_TRUE(default_parsed.has_value()) << default_parsed.error();
    EXPECT_TRUE((*default_parsed)->optimization.ppisp_exposure_from_exif);

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--no-ppisp-exif-exposure"};
    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();
    EXPECT_FALSE((*parsed)->optimization.ppisp_exposure_from_exif);
}

TEST(ArgumentParserTest, MortonReorderIntervalFlag) {
    const auto data_path = make_test_path("lfs_arg_parser_morton_data");
    const auto output_path = make_test_path("lfs_arg_parser_morton_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--morton-reorder-interval",
        "0"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();
    EXPECT_EQ((*parsed)->optimization.morton_reorder_interval, 0u);

    const char* argv_1000[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--morton-reorder-interval=1000"};
    auto parsed_1000 = lfs::core::args::parse_args_and_params(
        static_cast<int>(std::size(argv_1000)), argv_1000);
    ASSERT_TRUE(parsed_1000.has_value()) << parsed_1000.error();
    EXPECT_EQ((*parsed_1000)->optimization.morton_reorder_interval, 1000u);
}

TEST(ArgumentParserTest, MrnfCapacityDefaultsResolveAfterCliOverrides) {
    const auto data_path = make_test_path("lfs_arg_parser_mrnf_capacity_data");
    const auto output_path = make_test_path("lfs_arg_parser_mrnf_capacity_output");
    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--strategy",
        "mrnf",
        "--max-cap",
        "1000000"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();
    EXPECT_FLOAT_EQ((*parsed)->optimization.grow_fraction, -1.0f);
    EXPECT_FLOAT_EQ((*parsed)->optimization.shs_lr, -1.0f);

    (*parsed)->optimization.resolve_mrnf_capacity_defaults();
    EXPECT_NEAR((*parsed)->optimization.grow_fraction, 0.0758f, 1.0e-7f);
    EXPECT_FLOAT_EQ((*parsed)->optimization.shs_lr, 0.005f);
}

TEST(ArgumentParserTest, SafeModeIsProcessLocalAndNotATrainingConfigurationOption) {
    const auto data_path = make_test_path("lfs_arg_parser_safe_mode_data");
    const auto output_path = make_test_path("lfs_arg_parser_safe_mode_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--safe-mode",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str()};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();
    EXPECT_TRUE((*parsed)->safe_mode);
}

TEST(ArgumentParserTest, ResetAllSettingsIsProcessLocalAndExplicit) {
    const auto data_path = make_test_path("lfs_arg_parser_reset_all_data");
    const auto output_path = make_test_path("lfs_arg_parser_reset_all_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--reset-all-settings",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str()};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();
    EXPECT_TRUE((*parsed)->reset_all_settings);
    EXPECT_FALSE((*parsed)->reset_preferences);
    EXPECT_FALSE((*parsed)->reset_layout);
}

TEST(ArgumentParserTest, MaxWidthCanBeExplicitlySet) {
    const auto data_path = make_test_path("lfs_arg_parser_explicit_data");
    const auto output_path = make_test_path("lfs_arg_parser_explicit_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--max-width",
        "8192"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    EXPECT_EQ((*parsed)->dataset.max_width, 8192);
}

TEST(ArgumentParserTest, MaxWidthZeroDisablesCapExplicitly) {
    const auto data_path = make_test_path("lfs_arg_parser_zero_data");
    const auto output_path = make_test_path("lfs_arg_parser_zero_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--max-width",
        "0"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    EXPECT_EQ((*parsed)->dataset.max_width, 0);
}

TEST(ArgumentParserTest, CommandLineOverridesConfigAfterLoading) {
    const auto dir = std::filesystem::path(make_test_path("lfs_arg_parser_config_precedence"));
    const auto data_path = dir / "data";
    const auto output_path = dir / "output";
    const auto config_path = dir / "optimization.json";
    std::filesystem::create_directories(data_path);
    std::filesystem::create_directories(output_path);

    auto config = lfs::core::param::OptimizationParameters::mcmc_defaults().to_json();
    config["iterations"] = 12'345;
    config["opacity_lr"] = 0.0375f;
    std::ofstream(config_path) << config.dump(2);

    const auto data_str = data_path.string();
    const auto output_str = output_path.string();
    const auto config_str = config_path.string();
    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_str.c_str(),
        "--output-path",
        output_str.c_str(),
        "--config",
        config_str.c_str(),
        "-i",
        "777"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    std::error_code ec;
    std::filesystem::remove(config_path, ec);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    EXPECT_EQ((*parsed)->optimization.iterations, 777u);
    EXPECT_FLOAT_EQ((*parsed)->optimization.opacity_lr, 0.0375f);
}

TEST(ArgumentParserTest, Mesh2SplatParsesOutputPathAndOptions) {
    const auto dir = make_test_path("lfs_mesh2splat_arg_parser");
    const auto input = std::filesystem::path(dir) / "input.obj";
    const auto output = std::filesystem::path(dir) / "output.spz";
    std::ofstream(input).put('\n');

    const std::string input_str = input.string();
    const std::string output_str = output.string();
    const char* argv[] = {
        "LichtFeld-Studio",
        "mesh2splat",
        input_str.c_str(),
        "--output",
        output_str.c_str(),
        "--resolution",
        "512",
        "--sigma",
        "0.5"};

    auto parsed = lfs::core::args::parse_args(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    auto* mode = std::get_if<lfs::core::args::Mesh2SplatMode>(&*parsed);
    ASSERT_NE(mode, nullptr);
    EXPECT_EQ(mode->params.input_path, input);
    EXPECT_EQ(mode->params.output_path, output);
    EXPECT_EQ(mode->params.format, lfs::core::param::OutputFormat::SPZ);
    ASSERT_EQ(mode->params.formats.size(), 1u);
    EXPECT_EQ(mode->params.formats[0], lfs::core::param::OutputFormat::SPZ);
    EXPECT_EQ(mode->params.options.resolution_target, 512);
    EXPECT_FLOAT_EQ(mode->params.options.sigma, 0.5f);
}

TEST(ArgumentParserTest, ConvertRejectsOutputSuffixThatConflictsWithFormat) {
    const auto directory = std::filesystem::path(
        make_test_path("lfs_arg_parser_convert_suffix"));
    const auto input = directory / "input.ply";
    std::ofstream(input).put('\n');
    const auto output = (directory / "output.ply").string();
    const auto input_text = input.string();
    const char* argv[] = {
        "LichtFeld-Studio", "convert", input_text.c_str(),
        "--format", "spz", "--output", output.c_str()};

    const auto parsed = lfs::core::args::parse_args(
        static_cast<int>(std::size(argv)), argv);
    ASSERT_FALSE(parsed);
    EXPECT_NE(parsed.error().find("extension"), std::string::npos);
    EXPECT_NE(parsed.error().find(".ply"), std::string::npos);
}

TEST(ArgumentParserTest, Mesh2SplatRejectsOutputSuffixThatConflictsWithFormat) {
    const auto directory = std::filesystem::path(
        make_test_path("lfs_arg_parser_mesh2splat_suffix"));
    const auto input = directory / "input.obj";
    std::ofstream(input) << "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
    const auto output = (directory / "output.ply").string();
    const auto input_text = input.string();
    const char* argv[] = {
        "LichtFeld-Studio", "mesh2splat", input_text.c_str(),
        "--format", "spz", "--output", output.c_str()};

    const auto parsed = lfs::core::args::parse_args(
        static_cast<int>(std::size(argv)), argv);
    ASSERT_FALSE(parsed);
    EXPECT_NE(parsed.error().find("extension"), std::string::npos);
    EXPECT_NE(parsed.error().find(".ply"), std::string::npos);
}

TEST(ArgumentParserTest, Mesh2SplatParsesMultipleOutputFormats) {
    const auto dir = make_test_path("lfs_mesh2splat_multi_format_arg_parser");
    const auto input = std::filesystem::path(dir) / "input.obj";
    const auto output = std::filesystem::path(dir) / "output";
    std::ofstream(input).put('\n');

    const std::string input_str = input.string();
    const std::string output_str = output.string();
    const char* argv[] = {
        "LichtFeld-Studio",
        "mesh2splat",
        input_str.c_str(),
        "--output",
        output_str.c_str(),
        "--format",
        ".ply,.spz,.html,.ssog"};

    auto parsed = lfs::core::args::parse_args(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    auto* mode = std::get_if<lfs::core::args::Mesh2SplatMode>(&*parsed);
    ASSERT_NE(mode, nullptr);
    ASSERT_EQ(mode->params.formats.size(), 4u);
    EXPECT_EQ(mode->params.formats[0], lfs::core::param::OutputFormat::PLY);
    EXPECT_EQ(mode->params.formats[1], lfs::core::param::OutputFormat::SPZ);
    EXPECT_EQ(mode->params.formats[2], lfs::core::param::OutputFormat::HTML);
    EXPECT_EQ(mode->params.formats[3], lfs::core::param::OutputFormat::SSOG);
}

TEST(ArgumentParserTest, ConvertDefaultsIncludeProvenance) {
    const auto dir = make_test_path("lfs_convert_arg_parser_provenance_default");
    const auto input = std::filesystem::path(dir) / "input.ply";
    std::ofstream(input).put('\n');

    const std::string input_str = input.string();
    const char* argv[] = {
        "LichtFeld-Studio",
        "convert",
        input_str.c_str(),
        "-f",
        "ply"};

    auto parsed = lfs::core::args::parse_args(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    auto* mode = std::get_if<lfs::core::args::ConvertMode>(&*parsed);
    ASSERT_NE(mode, nullptr);
    EXPECT_TRUE(mode->params.include_provenance);
}

TEST(ArgumentParserTest, ConvertNoProvenanceDisablesStamp) {
    const auto dir = make_test_path("lfs_convert_arg_parser_no_provenance");
    const auto input = std::filesystem::path(dir) / "input.ply";
    std::ofstream(input).put('\n');

    const std::string input_str = input.string();
    const char* argv[] = {
        "LichtFeld-Studio",
        "convert",
        input_str.c_str(),
        "-f",
        "ply",
        "--no-provenance"};

    auto parsed = lfs::core::args::parse_args(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    auto* mode = std::get_if<lfs::core::args::ConvertMode>(&*parsed);
    ASSERT_NE(mode, nullptr);
    EXPECT_FALSE(mode->params.include_provenance);
}

TEST(ArgumentParserTest, ConvertHelpListsLichtProjectInput) {
    const char* argv[] = {"LichtFeld-Studio", "convert", "--help"};
    testing::internal::CaptureStdout();
    auto parsed = lfs::core::args::parse_args(static_cast<int>(std::size(argv)), argv);
    const auto help = testing::internal::GetCapturedStdout();

    ASSERT_TRUE(parsed.has_value()) << parsed.error();
    EXPECT_TRUE(std::holds_alternative<lfs::core::args::HelpMode>(*parsed));
    EXPECT_NE(help.find(".licht (project)"), std::string::npos);
    EXPECT_NE(help.find("LichtFeld-Studio convert project.licht output.ply"), std::string::npos);
}

TEST(ArgumentParserTest, TrainingParsesAddSplats) {
    const auto dir = make_test_path("lfs_arg_parser_add_splat");
    const auto data_path = std::filesystem::path(dir) / "data";
    const auto output_path = std::filesystem::path(dir) / "output";
    const auto splat_a = std::filesystem::path(dir) / "background.ply";
    const auto splat_b = std::filesystem::path(dir) / "sky.sog";
    std::filesystem::create_directories(data_path);
    std::filesystem::create_directories(output_path);
    std::ofstream(splat_a).put('\n');
    std::ofstream(splat_b).put('\n');

    const std::string data_str = data_path.string();
    const std::string output_str = output_path.string();
    const std::string splat_a_str = splat_a.string();
    const std::string splat_b_str = splat_b.string();
    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_str.c_str(),
        "--output-path",
        output_str.c_str(),
        "--add-splat",
        splat_a_str.c_str(),
        "--add-splat",
        splat_b_str.c_str()};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    ASSERT_EQ((*parsed)->add_splat_paths.size(), 2u);
    EXPECT_EQ((*parsed)->add_splat_paths[0], splat_a);
    EXPECT_EQ((*parsed)->add_splat_paths[1], splat_b);
    EXPECT_EQ((*parsed)->add_splat_freeze, (std::vector<bool>{false, false}));
    EXPECT_FALSE((*parsed)->exclude_frozen_add_splats_from_export);
}

TEST(ArgumentParserTest, TrainingParsesFrozenAddSplatExcludeExport) {
    const auto dir = make_test_path("lfs_arg_parser_add_splat_exclude");
    const auto data_path = std::filesystem::path(dir) / "data";
    const auto output_path = std::filesystem::path(dir) / "output";
    const auto splat = std::filesystem::path(dir) / "background.ply";
    std::filesystem::create_directories(data_path);
    std::filesystem::create_directories(output_path);
    std::ofstream(splat).put('\n');

    const std::string data_str = data_path.string();
    const std::string output_str = output_path.string();
    const std::string splat_str = splat.string();
    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_str.c_str(),
        "--output-path",
        output_str.c_str(),
        "--add-splat",
        splat_str.c_str(),
        "--freeze",
        "--exclude-export"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    ASSERT_EQ((*parsed)->add_splat_paths.size(), 1u);
    EXPECT_EQ((*parsed)->add_splat_paths[0], splat);
    EXPECT_EQ((*parsed)->add_splat_freeze, (std::vector<bool>{true}));
    EXPECT_TRUE((*parsed)->exclude_frozen_add_splats_from_export);
}

TEST(ArgumentParserTest, TrainingParsesFrozenLrScale) {
    const auto data_path = make_test_path("lfs_arg_parser_freeze_lr_scale_data");
    const auto output_path = make_test_path("lfs_arg_parser_freeze_lr_scale_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--freeze-lr-scale",
        "0.05"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();
    EXPECT_FLOAT_EQ((*parsed)->freeze_lr_scale, 0.05f);
}

TEST(ArgumentParserTest, TrainingRejectsFrozenLrScaleOutsideUnitInterval) {
    const auto data_path = make_test_path("lfs_arg_parser_invalid_freeze_lr_scale_data");
    const auto output_path = make_test_path("lfs_arg_parser_invalid_freeze_lr_scale_output");

    for (const char* scale : {"1.5", "-0.1"}) {
        SCOPED_TRACE(scale);
        const char* argv[] = {
            "LichtFeld-Studio",
            "--headless",
            "--data-path",
            data_path.c_str(),
            "--output-path",
            output_path.c_str(),
            "--freeze-lr-scale",
            scale};

        auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
        ASSERT_FALSE(parsed.has_value());
        EXPECT_NE(parsed.error().find("freeze_lr_scale must be within [0, 1]"), std::string::npos)
            << parsed.error();
    }
}

TEST(ArgumentParserTest, TrainingParsesCropBoxLrScale) {
    const auto data_path = make_test_path("lfs_arg_parser_cropbox_lr_scale_data");
    const auto output_path = make_test_path("lfs_arg_parser_cropbox_lr_scale_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--cropbox-lr-scale",
        "0.25"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();
    EXPECT_FLOAT_EQ((*parsed)->optimization.cropbox_lr_scale, 0.25f);
}

TEST(ArgumentParserTest, TrainingRejectsCropBoxLrScaleOutsideUnitInterval) {
    const auto data_path = make_test_path("lfs_arg_parser_invalid_cropbox_lr_scale_data");
    const auto output_path = make_test_path("lfs_arg_parser_invalid_cropbox_lr_scale_output");

    for (const char* scale : {"1.5", "-0.1"}) {
        SCOPED_TRACE(scale);
        const char* argv[] = {
            "LichtFeld-Studio",
            "--headless",
            "--data-path",
            data_path.c_str(),
            "--output-path",
            output_path.c_str(),
            "--cropbox-lr-scale",
            scale};

        auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
        ASSERT_FALSE(parsed.has_value());
        EXPECT_NE(parsed.error().find("cropbox_lr_scale must be finite and within [0, 1]"), std::string::npos)
            << parsed.error();
    }
}

TEST(ArgumentParserTest, TrainingParsesCropBoxLossWeight) {
    const auto data_path = make_test_path("lfs_arg_parser_cropbox_loss_weight_data");
    const auto output_path = make_test_path("lfs_arg_parser_cropbox_loss_weight_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--cropbox-loss-weight",
        "0.4"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();
    EXPECT_FLOAT_EQ((*parsed)->optimization.cropbox_loss_weight, 0.4f);
}

TEST(ArgumentParserTest, TrainingRejectsCropBoxLossWeightOutsideUnitInterval) {
    const auto data_path = make_test_path("lfs_arg_parser_invalid_cropbox_loss_weight_data");
    const auto output_path = make_test_path("lfs_arg_parser_invalid_cropbox_loss_weight_output");

    for (const char* weight : {"1.5", "-0.1"}) {
        SCOPED_TRACE(weight);
        const char* argv[] = {
            "LichtFeld-Studio",
            "--headless",
            "--data-path",
            data_path.c_str(),
            "--output-path",
            output_path.c_str(),
            "--cropbox-loss-weight",
            weight};

        auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
        ASSERT_FALSE(parsed.has_value());
        EXPECT_NE(parsed.error().find("cropbox_loss_weight must be finite and within [0, 1]"), std::string::npos)
            << parsed.error();
    }
}

TEST(ArgumentParserTest, FreezeMustImmediatelyFollowAddedSplat) {
    const auto dir = make_test_path("lfs_arg_parser_freeze_order");
    const auto data_path = std::filesystem::path(dir) / "data";
    const auto output_path = std::filesystem::path(dir) / "output";
    const auto splat = std::filesystem::path(dir) / "background.ply";
    std::filesystem::create_directories(data_path);
    std::filesystem::create_directories(output_path);
    std::ofstream(splat).put('\n');

    const std::string data_str = data_path.string();
    const std::string output_str = output_path.string();
    const std::string splat_str = splat.string();
    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_str.c_str(),
        "--output-path",
        output_str.c_str(),
        "--add-splat",
        splat_str.c_str(),
        "--freeze-lr-scale",
        "0.05",
        "--freeze"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_FALSE(parsed.has_value());
    EXPECT_NE(parsed.error().find("--freeze must immediately follow --add-splat <path>"),
              std::string::npos)
        << parsed.error();
}

TEST(ArgumentParserTest, TrainingParsesExplicitDepthLossOptions) {
    const auto data_path = make_test_path("lfs_arg_parser_depth_loss_data");
    const auto output_path = make_test_path("lfs_arg_parser_depth_loss_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--use-depth-loss",
        "--depth-loss-weight",
        "3.25",
        "--depth-loss-mode",
        "ssi-disparity"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    EXPECT_TRUE((*parsed)->optimization.use_depth_loss);
    EXPECT_FLOAT_EQ((*parsed)->optimization.depth_loss_weight, 3.25f);
    EXPECT_EQ((*parsed)->optimization.depth_loss_mode, "ssi-disparity");
}

TEST(ArgumentParserTest, TrainingRejectsLegacyDepthLossAlias) {
    const auto data_path = make_test_path("lfs_arg_parser_depth_invalid_data");
    const auto output_path = make_test_path("lfs_arg_parser_depth_invalid_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--use-depth-loss",
        "--depth-loss-mode",
        "lod"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_FALSE(parsed.has_value());
    EXPECT_NE(parsed.error().find("depth_loss_mode must be 'ssi', 'ssi-disparity', or 'ssi-depth'"), std::string::npos);
}

TEST(ArgumentParserTest, TrainingParsesExplicitNormalLossOptions) {
    const auto data_path = make_test_path("lfs_arg_parser_normal_loss_data");
    const auto output_path = make_test_path("lfs_arg_parser_normal_loss_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--use-normal-loss",
        "--normal-loss-weight",
        "0.75",
        "--normal-consistency-weight",
        "0.25",
        "--normal-flatten-weight",
        "5.0",
        "--normal-start-fraction",
        "0.3",
        "--normal-end-fraction",
        "0.9",
        "--normal-loss-space",
        "world"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    EXPECT_TRUE((*parsed)->optimization.use_normal_loss);
    EXPECT_TRUE((*parsed)->optimization.normal_auto_generate);
    EXPECT_FLOAT_EQ((*parsed)->optimization.normal_loss_weight, 0.75f);
    EXPECT_FLOAT_EQ((*parsed)->optimization.normal_consistency_weight, 0.25f);
    EXPECT_FLOAT_EQ((*parsed)->optimization.normal_flatten_weight, 5.0f);
    EXPECT_FLOAT_EQ((*parsed)->optimization.normal_start_fraction, 0.3f);
    EXPECT_FLOAT_EQ((*parsed)->optimization.normal_end_fraction, 0.9f);
    EXPECT_EQ((*parsed)->optimization.normal_loss_space, lfs::core::param::NormalLossSpace::World);
}

TEST(ArgumentParserTest, TrainingRejectsOutOfRangeNumericFlags) {
    const auto data_path = make_test_path("lfs_arg_parser_numeric_data");
    const auto output_path = make_test_path("lfs_arg_parser_numeric_output");
    const std::vector<std::pair<std::vector<std::string>, std::string>> cases{
        {{"--normal-loss-weight", "-1"}, "normal_loss_weight"},
        {{"--normal-consistency-weight", "-1"}, "normal_consistency_weight"},
        {{"--normal-flatten-weight", "-1"}, "normal_flatten_weight"},
        {{"--perf-bench-warmup", "0"}, "perf_bench_warmup"},
        {{"--max-screen-share", "-1"}, "max_screen_share"},
        {{"--max-screen-share", "2"}, "max_screen_share"},
    };

    for (const auto& [extra, expected_error] : cases) {
        std::vector<std::string> args{
            "LichtFeld-Studio", "--headless", "--data-path", data_path,
            "--output-path", output_path};
        args.insert(args.end(), extra.begin(), extra.end());
        std::vector<const char*> argv;
        argv.reserve(args.size());
        for (const auto& arg : args)
            argv.push_back(arg.c_str());

        auto parsed = lfs::core::args::parse_args_and_params(
            static_cast<int>(argv.size()), argv.data());
        ASSERT_FALSE(parsed.has_value()) << extra.front();
        EXPECT_NE(parsed.error().find(expected_error), std::string::npos)
            << extra.front() << " " << extra.back();
    }
}

TEST(ArgumentParserTest, NegativeShDegreeIntervalReportsSuppliedValue) {
    const auto data_path = make_test_path("lfs_arg_parser_sh_interval_data");
    const auto output_path = make_test_path("lfs_arg_parser_sh_interval_output");
    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--sh-degree-interval",
        "-1",
    };

    auto parsed = lfs::core::args::parse_args_and_params(
        static_cast<int>(std::size(argv)), argv);
    ASSERT_FALSE(parsed.has_value());
    EXPECT_NE(parsed.error().find("-1"), std::string::npos);
    EXPECT_EQ(parsed.error().find("18446744073709551615"), std::string::npos);
}

TEST(ArgumentParserTest, TrainingRejectsInvalidNormalLossSpace) {
    const auto data_path = make_test_path("lfs_arg_parser_invalid_normal_space_data");
    const auto output_path = make_test_path("lfs_arg_parser_invalid_normal_space_output");

    for (const auto& value : {"bogus", "WORLD", "", "42"}) {
        SCOPED_TRACE(std::string("--normal-loss-space=") + value);
        const char* argv[] = {
            "LichtFeld-Studio",
            "--headless",
            "--data-path",
            data_path.c_str(),
            "--output-path",
            output_path.c_str(),
            "--normal-loss-space",
            value};

        auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
        ASSERT_FALSE(parsed.has_value());
        EXPECT_NE(parsed.error().find("--normal-loss-space must be one of"), std::string::npos) << parsed.error();
    }
}

TEST(ArgumentParserTest, TrainingConfigRejectsUnknownNormalLossSpace) {
    const auto config_path = std::filesystem::path(make_test_path("lfs_invalid_normal_space_config")) / "config.json";
    {
        auto config_json = lfs::core::param::OptimizationParameters::mrnf_defaults().to_json();
        config_json["normal_loss_space"] = "bogus";
        std::ofstream config_file(config_path);
        config_file << config_json.dump(2);
    }

    auto parsed = lfs::core::param::read_optim_params_from_json(config_path);
    ASSERT_FALSE(parsed.has_value());
    EXPECT_NE(parsed.error().find("normal_loss_space"), std::string::npos) << parsed.error();
    EXPECT_NE(parsed.error().find("bogus"), std::string::npos) << parsed.error();
    EXPECT_NE(parsed.error().find("camera-opencv"), std::string::npos) << parsed.error();
    EXPECT_NE(parsed.error().find("world"), std::string::npos) << parsed.error();
}

TEST(ArgumentParserTest, TrainingConfigRejectsUnknownBackgroundMode) {
    const auto config_path = std::filesystem::path(make_test_path("lfs_invalid_bg_mode_config")) / "config.json";
    {
        auto config_json = lfs::core::param::OptimizationParameters::mrnf_defaults().to_json();
        config_json["bg_mode"] = "future-mode";
        std::ofstream config_file(config_path);
        config_file << config_json.dump(2);
    }

    auto parsed = lfs::core::param::read_optim_params_from_json(config_path);
    ASSERT_FALSE(parsed.has_value());
    EXPECT_NE(parsed.error().find("bg_mode"), std::string::npos) << parsed.error();
    EXPECT_NE(parsed.error().find("future-mode"), std::string::npos) << parsed.error();
    EXPECT_NE(parsed.error().find("solid_color"), std::string::npos) << parsed.error();
    EXPECT_NE(parsed.error().find("modulation"), std::string::npos) << parsed.error();
}

TEST(ArgumentParserTest, TrainingParsesNoNormalAutoGenerate) {
    const auto data_path = make_test_path("lfs_arg_parser_no_normal_auto_data");
    const auto output_path = make_test_path("lfs_arg_parser_no_normal_auto_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--use-normal-loss",
        "--no-normal-auto-generate"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();
    EXPECT_TRUE((*parsed)->optimization.use_normal_loss);
    EXPECT_FALSE((*parsed)->optimization.normal_auto_generate);
}

TEST(ArgumentParserTest, TrainingRejectsNormalStartAfterEnd) {
    const auto data_path = make_test_path("lfs_arg_parser_normal_schedule_order_data");
    const auto output_path = make_test_path("lfs_arg_parser_normal_schedule_order_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--use-normal-loss",
        "--normal-start-fraction",
        "0.8",
        "--normal-end-fraction",
        "0.4"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_FALSE(parsed.has_value());
    EXPECT_NE(parsed.error().find("normal_start_fraction must not exceed normal_end_fraction"),
              std::string::npos)
        << parsed.error();
}

TEST(ArgumentParserTest, TrainingRejectsNormalScheduleOutsideUnitInterval) {
    const auto data_path = make_test_path("lfs_arg_parser_normal_schedule_range_data");
    const auto output_path = make_test_path("lfs_arg_parser_normal_schedule_range_output");

    for (const auto& [flag, value] : {
             std::pair{"--normal-start-fraction", "1.5"},
             std::pair{"--normal-start-fraction", "-0.1"},
             std::pair{"--normal-end-fraction", "1.5"},
             std::pair{"--normal-end-fraction", "-0.1"}}) {
        SCOPED_TRACE(std::string(flag) + "=" + value);
        const char* argv[] = {
            "LichtFeld-Studio",
            "--headless",
            "--data-path",
            data_path.c_str(),
            "--output-path",
            output_path.c_str(),
            flag,
            value};

        auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
        ASSERT_FALSE(parsed.has_value());
        EXPECT_NE(parsed.error().find("must be finite and within [0, 1]"), std::string::npos)
            << parsed.error();
    }
}

TEST(ArgumentParserTest, TrainingParsesBackgroundModeModulation) {
    const auto data_path = make_test_path("lfs_arg_parser_bg_mode_modulation_data");
    const auto output_path = make_test_path("lfs_arg_parser_bg_mode_modulation_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--bg-mode",
        "modulation"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    EXPECT_EQ((*parsed)->optimization.bg_mode, lfs::core::param::BackgroundMode::Modulation);
    EXPECT_TRUE((*parsed)->optimization.bg_modulation);
}

TEST(ArgumentParserTest, TrainingRejectsFloatBackgroundColor) {
    const auto data_path = make_test_path("lfs_arg_parser_bg_color_data");
    const auto output_path = make_test_path("lfs_arg_parser_bg_color_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--bg-mode",
        "solidcolor",
        "--bg-color",
        "0.1,0.2,0.3"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_FALSE(parsed.has_value());
    EXPECT_NE(parsed.error().find("--bg-color must be #RRGGBB"), std::string::npos);
}

TEST(ArgumentParserTest, TrainingParsesHexBackgroundColor) {
    const auto data_path = make_test_path("lfs_arg_parser_bg_hex_color_data");
    const auto output_path = make_test_path("lfs_arg_parser_bg_hex_color_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--bg-mode",
        "solidcolor",
        "--bg-color",
        "#FF8040"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    EXPECT_TRUE((*parsed)->cli_bg_color_set);
    EXPECT_FLOAT_EQ((*parsed)->optimization.bg_color[0], 1.0f);
    EXPECT_FLOAT_EQ((*parsed)->optimization.bg_color[1], 128.0f / 255.0f);
    EXPECT_FLOAT_EQ((*parsed)->optimization.bg_color[2], 64.0f / 255.0f);
}

TEST(ArgumentParserTest, TrainingParsesLowercaseHexBackgroundColor) {
    const auto data_path = make_test_path("lfs_arg_parser_bg_lower_hex_color_data");
    const auto output_path = make_test_path("lfs_arg_parser_bg_lower_hex_color_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--bg-mode",
        "solidcolor",
        "--bg-color",
        "#ff8040"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    EXPECT_TRUE((*parsed)->cli_bg_color_set);
    EXPECT_FLOAT_EQ((*parsed)->optimization.bg_color[0], 1.0f);
    EXPECT_FLOAT_EQ((*parsed)->optimization.bg_color[1], 128.0f / 255.0f);
    EXPECT_FLOAT_EQ((*parsed)->optimization.bg_color[2], 64.0f / 255.0f);
}

TEST(ArgumentParserTest, TrainingParsesIntegerRgbBackgroundColorWithSpaces) {
    const auto data_path = make_test_path("lfs_arg_parser_bg_rgb_color_data");
    const auto output_path = make_test_path("lfs_arg_parser_bg_rgb_color_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--bg-mode",
        "solidcolor",
        "--bg-color",
        "(255, 64, 32)"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    EXPECT_TRUE((*parsed)->cli_bg_color_set);
    EXPECT_FLOAT_EQ((*parsed)->optimization.bg_color[0], 1.0f);
    EXPECT_FLOAT_EQ((*parsed)->optimization.bg_color[1], 64.0f / 255.0f);
    EXPECT_FLOAT_EQ((*parsed)->optimization.bg_color[2], 32.0f / 255.0f);
}

TEST(ArgumentParserTest, TrainingParsesIntegerRgbBackgroundColorWithoutSpaces) {
    const auto data_path = make_test_path("lfs_arg_parser_bg_rgb_compact_color_data");
    const auto output_path = make_test_path("lfs_arg_parser_bg_rgb_compact_color_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--bg-mode",
        "solidcolor",
        "--bg-color",
        "(255,64,32)"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    EXPECT_TRUE((*parsed)->cli_bg_color_set);
    EXPECT_FLOAT_EQ((*parsed)->optimization.bg_color[0], 1.0f);
    EXPECT_FLOAT_EQ((*parsed)->optimization.bg_color[1], 64.0f / 255.0f);
    EXPECT_FLOAT_EQ((*parsed)->optimization.bg_color[2], 32.0f / 255.0f);
}

TEST(ArgumentParserTest, TrainingRejectsHexBackgroundColorWithoutHash) {
    const auto data_path = make_test_path("lfs_arg_parser_bg_hex_no_hash_data");
    const auto output_path = make_test_path("lfs_arg_parser_bg_hex_no_hash_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--bg-mode",
        "solidcolor",
        "--bg-color",
        "FF8040"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_FALSE(parsed.has_value());
    EXPECT_NE(parsed.error().find("--bg-color must be #RRGGBB"), std::string::npos);
}

TEST(ArgumentParserTest, TrainingRejectsRgbBackgroundColorOutOfRange) {
    const auto data_path = make_test_path("lfs_arg_parser_bg_rgb_out_of_range_data");
    const auto output_path = make_test_path("lfs_arg_parser_bg_rgb_out_of_range_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--bg-mode",
        "solidcolor",
        "--bg-color",
        "(256,64,32)"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_FALSE(parsed.has_value());
    EXPECT_NE(parsed.error().find("--bg-color must be #RRGGBB"), std::string::npos);
}

TEST(ArgumentParserTest, TrainingParsesImageBackgroundPath) {
    const auto dir = make_test_path("lfs_arg_parser_bg_image");
    const auto data_path = std::filesystem::path(dir) / "data";
    const auto output_path = std::filesystem::path(dir) / "output";
    const auto image_path = std::filesystem::path(dir) / "bg.png";
    std::filesystem::create_directories(data_path);
    std::filesystem::create_directories(output_path);
    std::ofstream(image_path).put('\n');

    const std::string data_str = data_path.string();
    const std::string output_str = output_path.string();
    const std::string image_str = image_path.string();
    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_str.c_str(),
        "--output-path",
        output_str.c_str(),
        "--bg-mode",
        "image",
        "--bg-image-path",
        image_str.c_str()};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    EXPECT_EQ((*parsed)->optimization.bg_mode, lfs::core::param::BackgroundMode::Image);
    EXPECT_EQ((*parsed)->optimization.bg_image_path, image_path);
}

TEST(ArgumentParserTest, TrainingRejectsImageBackgroundWithoutPath) {
    const auto data_path = make_test_path("lfs_arg_parser_bg_image_missing_data");
    const auto output_path = make_test_path("lfs_arg_parser_bg_image_missing_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--bg-mode",
        "image"};

    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_FALSE(parsed.has_value());
    EXPECT_NE(parsed.error().find("--bg-image-path is required"), std::string::npos);
}

TEST(ArgumentParserTest, TrainingParsesMissingImageBackgroundForRuntimeValidation) {
    const auto data_path = make_test_path("lfs_arg_parser_bg_image_missing_file_data");
    const auto output_path = make_test_path("lfs_arg_parser_bg_image_missing_file_output");
    const auto image_path =
        std::filesystem::path(output_path) / "missing-background.png";
    const auto image_text = image_path.string();
    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--bg-mode",
        "image",
        "--bg-image-path",
        image_text.c_str(),
    };

    auto parsed = lfs::core::args::parse_args_and_params(
        static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();
    EXPECT_EQ((*parsed)->optimization.bg_image_path, image_path);
}

TEST(ArgumentParserTest, ResumeCliFlagsPopulateExplicitOverrides) {
    const auto directory = make_test_path("lfs_arg_parser_resume_overrides");
    const auto project = std::filesystem::path(directory) / "session.licht";
    std::ofstream(project).put('\n');
    const auto project_text = project.string();
    const auto output_path = make_test_path("lfs_arg_parser_resume_overrides_out");
    const auto output_text = output_path;

    const char* argv[] = {
        "LichtFeld-Studio",
        "--resume",
        project_text.c_str(),
        "--headless",
        "--train",
        "--eval",
        "--eval-steps",
        "30100",
        "-i",
        "30100",
        "--test-every",
        "64",
        "-o",
        output_text.c_str(),
    };
    auto parsed = lfs::core::args::parse_args_and_params(
        static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    EXPECT_TRUE((*parsed)->cli_iterations_set);
    EXPECT_EQ((*parsed)->optimization.iterations, 30100u);
    EXPECT_TRUE((*parsed)->optimization.enable_eval);
    EXPECT_EQ((*parsed)->optimization.eval_steps, std::vector<size_t>({30100}));
    EXPECT_EQ((*parsed)->dataset.test_every, 64);
    EXPECT_TRUE((*parsed)->overrides.has_optimization_key("iterations"));
    EXPECT_TRUE((*parsed)->overrides.has_optimization_key("enable_eval"));
    EXPECT_TRUE((*parsed)->overrides.has_optimization_key("eval_steps"));
    EXPECT_TRUE((*parsed)->overrides.has_dataset_key("test_every"));
    EXPECT_FALSE((*parsed)->overrides.has_optimization_key("max_cap"));

    lfs::core::param::TrainingParameters restored;
    restored.optimization.iterations = 30'000;
    restored.optimization.enable_eval = false;
    restored.optimization.eval_steps = {7'000, 30'000};
    restored.optimization.save_steps = {7'000, 30'000};
    restored.dataset.test_every = 8;
    restored.optimization.max_cap = 42;
    apply_explicit_training_overrides(restored, (*parsed)->overrides);
    EXPECT_EQ(restored.optimization.iterations, 30100u);
    EXPECT_TRUE(restored.optimization.enable_eval);
    EXPECT_EQ(restored.optimization.eval_steps, std::vector<size_t>({30100}));
    EXPECT_EQ(restored.dataset.test_every, 64);
    EXPECT_EQ(restored.optimization.max_cap, 42);
}

TEST(ArgumentParserTest, ResumeRejectsExplicitInitFile) {
    const auto directory = make_test_path("lfs_arg_parser_resume_init_conflict");
    const auto project = std::filesystem::path(directory) / "session.licht";
    const auto init = std::filesystem::path(directory) / "init.ply";
    std::ofstream(project).put('\n');
    std::ofstream(init).put('\n');
    const auto project_text = project.string();
    const auto init_text = init.string();

    const char* argv[] = {
        "LichtFeld-Studio",
        "--resume",
        project_text.c_str(),
        "--headless",
        "--train",
        "--init",
        init_text.c_str(),
    };
    auto parsed = lfs::core::args::parse_args_and_params(
        static_cast<int>(std::size(argv)), argv);

    ASSERT_FALSE(parsed.has_value());
    EXPECT_NE(parsed.error().find("--init"), std::string::npos);
    EXPECT_NE(parsed.error().find("--resume"), std::string::npos);
}

TEST(ArgumentParserTest, ResumeConfigKeysPopulateExplicitOverrides) {
    const auto directory = make_test_path("lfs_arg_parser_resume_config_overrides");
    const auto project = std::filesystem::path(directory) / "session.licht";
    std::ofstream(project).put('\n');
    const auto config_path = std::filesystem::path(directory) / "resume.json";
    auto config = lfs::core::param::OptimizationParameters::mrnf_defaults().to_json();
    config["iterations"] = 30100;
    config["enable_eval"] = true;
    config["enable_save_eval_images"] = true;
    config["eval_steps"] = {30100};
    config["save_steps"] = {30100};
    std::ofstream(config_path) << config.dump(2);
    const auto project_text = project.string();
    const auto config_text = config_path.string();
    const auto output_path = make_test_path("lfs_arg_parser_resume_config_overrides_out");

    const char* argv[] = {
        "LichtFeld-Studio",
        "--resume",
        project_text.c_str(),
        "--headless",
        "--config",
        config_text.c_str(),
        "--test-every",
        "64",
        "-o",
        output_path.c_str(),
    };
    auto parsed = lfs::core::args::parse_args_and_params(
        static_cast<int>(std::size(argv)), argv);
    std::error_code ec;
    std::filesystem::remove(config_path, ec);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    EXPECT_TRUE((*parsed)->overrides.has_optimization_key("iterations"));
    EXPECT_TRUE((*parsed)->overrides.has_optimization_key("eval_steps"));
    EXPECT_TRUE((*parsed)->overrides.has_optimization_key("save_steps"));
    EXPECT_TRUE((*parsed)->overrides.has_optimization_key("enable_eval"));
    EXPECT_TRUE((*parsed)->overrides.has_optimization_key("enable_save_eval_images"));
    EXPECT_TRUE((*parsed)->overrides.has_dataset_key("test_every"));

    lfs::core::param::TrainingParameters restored;
    restored.optimization.iterations = 30'000;
    restored.optimization.enable_eval = false;
    restored.optimization.enable_save_eval_images = false;
    restored.optimization.eval_steps = {7'000, 30'000};
    restored.optimization.save_steps = {7'000, 30'000};
    restored.dataset.test_every = 8;
    apply_explicit_training_overrides(restored, (*parsed)->overrides);
    EXPECT_EQ(restored.optimization.iterations, 30100u);
    EXPECT_TRUE(restored.optimization.enable_eval);
    EXPECT_TRUE(restored.optimization.enable_save_eval_images);
    EXPECT_EQ(restored.optimization.eval_steps, std::vector<size_t>({30100}));
    EXPECT_EQ(restored.optimization.save_steps, std::vector<size_t>({30100}));
    EXPECT_EQ(restored.dataset.test_every, 64);
}

TEST(ArgumentParserTest, InitFileConflictsWithRandomInitialization) {
    const auto data_path = make_test_path("lfs_arg_parser_random_init_data");
    const auto output_path = make_test_path("lfs_arg_parser_random_init_output");
    const auto init_path = std::filesystem::path(
                               make_test_path("lfs_arg_parser_random_init_input")) /
                           "init.ply";
    std::ofstream(init_path).put('\n');
    const auto init_text = init_path.string();

    const char* argv[] = {
        "LichtFeld-Studio",
        "--headless",
        "--data-path",
        data_path.c_str(),
        "--output-path",
        output_path.c_str(),
        "--random",
        "--init",
        init_text.c_str(),
    };
    auto parsed = lfs::core::args::parse_args_and_params(
        static_cast<int>(std::size(argv)), argv);

    ASSERT_FALSE(parsed.has_value());
    EXPECT_NE(parsed.error().find("--init"), std::string::npos);
    EXPECT_NE(parsed.error().find("--random"), std::string::npos);
}

TEST(ArgumentParserTest, ViewModeHonorsMcpPortOverride) {
    const auto directory = make_test_path("lfs_arg_parser_view_mcp_port");
    const auto ply = std::filesystem::path(directory) / "some.ply";
    std::ofstream(ply).put('\n');
    const auto ply_text = ply.string();

    const char* argv[] = {
        "LichtFeld-Studio",
        "-v",
        ply_text.c_str(),
        "--mcp-port",
        "45690",
    };
    auto parsed = lfs::core::args::parse_args_and_params(
        static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();
    EXPECT_EQ((*parsed)->mcp_port, std::optional<int>(45690));
}

#ifndef LFS_BUILD_PORTABLE
TEST(ArgumentParserTest, ViewModeHonorsNoSplash) {
    const auto directory = make_test_path("lfs_arg_parser_view_no_splash");
    const auto ply = std::filesystem::path(directory) / "some.ply";
    std::ofstream(ply).put('\n');
    const auto ply_text = ply.string();

    const char* argv[] = {
        "LichtFeld-Studio",
        "-v",
        ply_text.c_str(),
        "--no-splash",
    };
    auto parsed = lfs::core::args::parse_args_and_params(
        static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();
    EXPECT_TRUE((*parsed)->optimization.no_splash);
}
#endif

TEST(ArgumentParserTest, ViewModeRejectsOutOfRangeMcpPort) {
    const auto directory = make_test_path("lfs_arg_parser_view_mcp_port_invalid");
    const auto ply = std::filesystem::path(directory) / "some.ply";
    std::ofstream(ply).put('\n');
    const auto ply_text = ply.string();

    const char* argv[] = {
        "LichtFeld-Studio",
        "-v",
        ply_text.c_str(),
        "--mcp-port",
        "70000",
    };
    auto parsed = lfs::core::args::parse_args_and_params(
        static_cast<int>(std::size(argv)), argv);
    ASSERT_FALSE(parsed.has_value());
    EXPECT_NE(parsed.error().find("must be between 1 and 65535"),
              std::string::npos);
}

TEST(ArgumentParserTest, EvalStepsAcceptCommaListsAndRepeats) {
    const auto data_path = make_test_path("lfs_arg_parser_eval_steps_data");
    const auto output_path = make_test_path("lfs_arg_parser_eval_steps_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "-d",
        data_path.c_str(),
        "-o",
        output_path.c_str(),
        "--eval",
        "--eval-steps",
        "7000, 1000,7000",
        "--eval-steps",
        "30000",
    };
    auto parsed = lfs::core::args::parse_args_and_params(
        static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();

    EXPECT_TRUE((*parsed)->optimization.enable_eval);
    EXPECT_EQ((*parsed)->optimization.eval_steps, (std::vector<size_t>{1000, 7000, 30000}));
}

// Catches a parser that accepts only one spelling or fails to store the selected enum value.
TEST(ArgumentParserTest, EvalSpaceAcceptsBothRegisteredValues) {
    const auto data_path = make_test_path("lfs_arg_parser_eval_space_data");
    const auto output_path = make_test_path("lfs_arg_parser_eval_space_output");
    const std::vector<std::pair<const char*, lfs::core::param::EvalSpace>> cases = {
        {"distorted", lfs::core::param::EvalSpace::Distorted},
        {"undistorted", lfs::core::param::EvalSpace::Undistorted},
    };

    for (const auto& [value, expected] : cases) {
        const char* argv[] = {
            "LichtFeld-Studio",
            "-d",
            data_path.c_str(),
            "-o",
            output_path.c_str(),
            "--undistort",
            "--eval-space",
            value,
        };
        auto parsed = lfs::core::args::parse_args_and_params(
            static_cast<int>(std::size(argv)), argv);
        ASSERT_TRUE(parsed.has_value()) << value << ": " << parsed.error();
        EXPECT_EQ((*parsed)->optimization.eval_space, expected) << value;
        EXPECT_TRUE((*parsed)->overrides.has_optimization_key("eval_space")) << value;
    }
}

// Catches an unvalidated CLI map or JSON enum that silently falls back to distorted evaluation.
TEST(ArgumentParserTest, EvalSpaceRejectsInvalidCliAndConfigValues) {
    const auto data_path = make_test_path("lfs_arg_parser_bad_eval_space_data");
    const auto output_path = make_test_path("lfs_arg_parser_bad_eval_space_output");

    const char* cli_argv[] = {
        "LichtFeld-Studio",
        "-d",
        data_path.c_str(),
        "-o",
        output_path.c_str(),
        "--undistort",
        "--eval-space",
        "source",
    };
    auto from_cli = lfs::core::args::parse_args_and_params(
        static_cast<int>(std::size(cli_argv)), cli_argv);
    ASSERT_FALSE(from_cli.has_value());
    EXPECT_NE(from_cli.error().find("eval_space"), std::string::npos) << from_cli.error();

    const auto config_path =
        std::filesystem::path(make_test_path("lfs_arg_parser_bad_eval_space_config")) /
        "config.json";
    auto optimization = lfs::core::param::OptimizationParameters::mrnf_defaults().to_json();
    optimization["undistort"] = true;
    optimization["eval_space"] = "source";
    std::ofstream(config_path) << nlohmann::json{{"optimization", optimization}}.dump();
    const auto config_text = config_path.string();
    const char* config_argv[] = {
        "LichtFeld-Studio",
        "-d",
        data_path.c_str(),
        "-o",
        output_path.c_str(),
        "--config",
        config_text.c_str(),
    };
    auto from_config = lfs::core::args::parse_args_and_params(
        static_cast<int>(std::size(config_argv)), config_argv);
    ASSERT_FALSE(from_config.has_value());
    EXPECT_NE(from_config.error().find("eval_space"), std::string::npos)
        << from_config.error();
}

// Catches an explicit CLI selection being accepted when there is no undistorted render space.
TEST(ArgumentParserTest, EvalSpaceCliWithoutUndistortStopsTheRun) {
    const auto data_path = make_test_path("lfs_arg_parser_eval_space_no_undistort_data");
    const auto output_path = make_test_path("lfs_arg_parser_eval_space_no_undistort_output");

    for (const auto& args : std::vector<std::vector<const char*>>{
             {"LichtFeld-Studio", "-d", data_path.c_str(), "-o", output_path.c_str(),
              "--eval-space", "distorted"},
             {"LichtFeld-Studio", "-d", data_path.c_str(), "-o", output_path.c_str(),
              "--eval-space=undistorted"}}) {
        auto parsed = lfs::core::args::parse_args_and_params(
            static_cast<int>(args.size()), args.data());
        ASSERT_FALSE(parsed.has_value());
        EXPECT_NE(parsed.error().find("--eval-space needs --undistort"), std::string::npos)
            << parsed.error();
    }
}

// Every saved training config carries eval_space; reusing one without --undistort must still
// load, the value simply has no effect there.
TEST(ArgumentParserTest, EvalSpaceConfigWithoutUndistortIsAccepted) {
    const auto directory = make_test_path("lfs_arg_parser_eval_space_config_rule");
    const auto data_path = make_test_path("lfs_arg_parser_eval_space_config_rule_data");
    const auto output_path = make_test_path("lfs_arg_parser_eval_space_config_rule_output");
    const auto config_path = std::filesystem::path(directory) / "config.json";
    auto optimization = lfs::core::param::OptimizationParameters::mrnf_defaults().to_json();
    optimization["undistort"] = false;
    const auto config_text = config_path.string();
    const char* argv[] = {
        "LichtFeld-Studio",
        "-d",
        data_path.c_str(),
        "-o",
        output_path.c_str(),
        "--config",
        config_text.c_str(),
    };

    for (const char* value : {"distorted", "undistorted"}) {
        optimization["eval_space"] = value;
        std::ofstream(config_path) << nlohmann::json{{"optimization", optimization}}.dump();
        auto parsed = lfs::core::args::parse_args_and_params(
            static_cast<int>(std::size(argv)), argv);
        EXPECT_TRUE(parsed.has_value()) << value << ": " << parsed.error();
    }
}

// Catches missing serialization, config loading, or explicit override replay during resume.
TEST(ArgumentParserTest, EvalSpaceConfigRoundTripAndResumeKeepTheValue) {
    const auto directory = make_test_path("lfs_arg_parser_eval_space_round_trip");
    const auto data_path = make_test_path("lfs_arg_parser_eval_space_round_trip_data");
    const auto output_path = make_test_path("lfs_arg_parser_eval_space_round_trip_output");
    const auto config_path = std::filesystem::path(directory) / "training_config.json";
    lfs::core::param::TrainingParameters written;
    written.optimization.undistort = true;
    written.optimization.eval_space = lfs::core::param::EvalSpace::Undistorted;
    const auto saved = lfs::core::param::save_training_parameters_to_json(written, config_path);
    ASSERT_TRUE(saved.has_value()) << saved.error();

    const auto config_text = config_path.string();
    const char* argv[] = {
        "LichtFeld-Studio",
        "-d",
        data_path.c_str(),
        "-o",
        output_path.c_str(),
        "--config",
        config_text.c_str(),
    };
    auto parsed = lfs::core::args::parse_args_and_params(
        static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();
    EXPECT_TRUE((*parsed)->optimization.undistort);
    EXPECT_EQ((*parsed)->optimization.eval_space,
              lfs::core::param::EvalSpace::Undistorted);
    EXPECT_EQ((*parsed)->optimization.to_json().at("eval_space"), "undistorted");
    EXPECT_TRUE((*parsed)->overrides.has_optimization_key("eval_space"));

    lfs::core::param::TrainingParameters restored;
    restored.optimization.undistort = false;
    restored.optimization.eval_space = lfs::core::param::EvalSpace::Distorted;
    apply_explicit_training_overrides(restored, (*parsed)->overrides);
    EXPECT_TRUE(restored.optimization.undistort);
    EXPECT_EQ(restored.optimization.eval_space,
              lfs::core::param::EvalSpace::Undistorted);
}

TEST(ArgumentParserTest, EvalStepsRejectNonPositiveOrMalformedValues) {
    const auto data_path = make_test_path("lfs_arg_parser_bad_eval_steps_data");
    const auto output_path = make_test_path("lfs_arg_parser_bad_eval_steps_output");

    for (const char* value : {"1000,abc", "0", "1000,,2000", "-5", "7000x"}) {
        const char* argv[] = {
            "LichtFeld-Studio",
            "-d",
            data_path.c_str(),
            "-o",
            output_path.c_str(),
            "--eval-steps",
            value,
        };
        auto parsed = lfs::core::args::parse_args_and_params(
            static_cast<int>(std::size(argv)), argv);
        ASSERT_FALSE(parsed.has_value()) << value;
        EXPECT_NE(parsed.error().find("--eval-steps"), std::string::npos) << parsed.error();
    }
}

TEST(ArgumentParserTest, EvaluationFlagsOverrideTheConfigFile) {
    const auto data_path = make_test_path("lfs_arg_parser_eval_config_data");
    const auto output_path = make_test_path("lfs_arg_parser_eval_config_output");
    const auto config_path = std::filesystem::path(make_test_path("lfs_arg_parser_eval_config")) / "config.json";
    auto optimization = lfs::core::param::OptimizationParameters::mrnf_defaults().to_json();
    optimization["enable_eval"] = false;
    optimization["eval_steps"] = {100};
    std::ofstream(config_path) << nlohmann::json{{"dataset", {{"test_every", 4}}}, {"optimization", optimization}}.dump();
    const auto config_text = config_path.string();

    const char* config_only[] = {
        "LichtFeld-Studio",
        "-d",
        data_path.c_str(),
        "-o",
        output_path.c_str(),
        "--config",
        config_text.c_str(),
    };
    auto from_config = lfs::core::args::parse_args_and_params(
        static_cast<int>(std::size(config_only)), config_only);
    ASSERT_TRUE(from_config.has_value()) << from_config.error();
    EXPECT_FALSE((*from_config)->optimization.enable_eval);
    EXPECT_EQ((*from_config)->optimization.eval_steps, (std::vector<size_t>{100}));

    const char* with_cli[] = {
        "LichtFeld-Studio",
        "-d",
        data_path.c_str(),
        "-o",
        output_path.c_str(),
        "--config",
        config_text.c_str(),
        "--eval",
        "--eval-steps",
        "5,10",
        "--test-every",
        "2",
    };
    auto parsed = lfs::core::args::parse_args_and_params(
        static_cast<int>(std::size(with_cli)), with_cli);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();
    const auto& params = **parsed;
    EXPECT_TRUE(params.optimization.enable_eval);
    EXPECT_EQ(params.optimization.eval_steps, (std::vector<size_t>{5, 10}));
    EXPECT_EQ(params.dataset.test_every, 2);

    // Resume and project flows re-apply the recorded overrides; the CLI still wins there.
    lfs::core::param::TrainingParameters restored;
    apply_explicit_training_overrides(restored, params.overrides);
    EXPECT_TRUE(restored.optimization.enable_eval);
    EXPECT_EQ(restored.optimization.eval_steps, (std::vector<size_t>{5, 10}));
    EXPECT_EQ(restored.dataset.test_every, 2);
}

TEST(ArgumentParserTest, EvalStepsWithoutEvaluationStopTheRun) {
    const auto data_path = make_test_path("lfs_arg_parser_eval_steps_no_eval_data");
    const auto output_path = make_test_path("lfs_arg_parser_eval_steps_no_eval_output");

    for (const char* flag : {"--eval-steps", "--eval-steps=7000,20000"}) {
        std::vector<const char*> argv = {"LichtFeld-Studio", "-d", data_path.c_str(), "-o", output_path.c_str(), flag};
        if (std::string_view(flag) == "--eval-steps")
            argv.push_back("7000,20000");
        auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(argv.size()), argv.data());
        ASSERT_FALSE(parsed.has_value()) << flag;
        EXPECT_NE(parsed.error().find("--eval-steps needs --eval"), std::string::npos) << parsed.error();
    }

    const char* with_eval[] = {
        "LichtFeld-Studio",
        "-d",
        data_path.c_str(),
        "-o",
        output_path.c_str(),
        "--eval-steps",
        "7000,20000",
        "--eval",
    };
    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(with_eval)), with_eval);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();
    EXPECT_EQ((*parsed)->optimization.eval_steps, (std::vector<size_t>{7000, 20000}));
}

TEST(ArgumentParserTest, EvalStepsAcceptEvaluationEnabledByTheConfigFile) {
    const auto data_path = make_test_path("lfs_arg_parser_eval_steps_config_data");
    const auto output_path = make_test_path("lfs_arg_parser_eval_steps_config_output");
    const auto config_path =
        std::filesystem::path(make_test_path("lfs_arg_parser_eval_steps_config")) / "config.json";
    auto optimization = lfs::core::param::OptimizationParameters::mrnf_defaults().to_json();
    optimization["enable_eval"] = true;
    std::ofstream(config_path) << nlohmann::json{{"optimization", optimization}}.dump();
    const auto config_text = config_path.string();

    const char* argv[] = {
        "LichtFeld-Studio",
        "-d",
        data_path.c_str(),
        "-o",
        output_path.c_str(),
        "--config",
        config_text.c_str(),
        "--eval-steps",
        "500",
    };
    auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();
    EXPECT_TRUE((*parsed)->optimization.enable_eval);
    EXPECT_EQ((*parsed)->optimization.eval_steps, (std::vector<size_t>{500}));
}

// Catches working-directory-dependent persistence or lost explicit resume overrides.
TEST(ArgumentParserTest, EvalMeshMaskStoresAbsolutePathAndSurvivesResume) {
    const auto directory = std::filesystem::path(
        make_test_path("lfs_arg_parser_eval_mesh"));
    const auto mesh = directory / "mask.obj";
    std::ofstream(mesh) << "v 0 0 1\nv 1 0 1\nv 0 1 1\nf 1 2 3\n";
    const auto spec = "mesh:" + mesh.string();
    const char* argv[] = {
        "LichtFeld-Studio",
        "--eval",
        "--eval-mask",
        spec.c_str(),
        "--eval-mask-invert",
    };
    const auto parsed = lfs::core::args::parse_args_and_params(
        static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed) << parsed.error();
    const auto expected = lfs::core::path_to_utf8(
        std::filesystem::weakly_canonical(mesh));
    EXPECT_EQ((*parsed)->optimization.eval_mask, expected);
    EXPECT_TRUE((*parsed)->optimization.eval_mask_invert);
    EXPECT_TRUE((*parsed)->overrides.has_optimization_key("eval_mask"));
    EXPECT_TRUE((*parsed)->overrides.has_optimization_key("eval_mask_invert"));

    const auto round_trip = lfs::core::param::OptimizationParameters::from_json(
        (*parsed)->optimization.to_json());
    EXPECT_EQ(round_trip.eval_mask, expected);
    EXPECT_TRUE(round_trip.eval_mask_invert);

    lfs::core::param::TrainingParameters restored;
    restored.optimization.eval_mask.clear();
    restored.optimization.eval_mask_invert = false;
    apply_explicit_training_overrides(restored, (*parsed)->overrides);
    EXPECT_EQ(restored.optimization.eval_mask, expected);
    EXPECT_TRUE(restored.optimization.eval_mask_invert);
}

// Catches a box spec that is stored unnormalized, lost on resume, or accepted when malformed.
TEST(ArgumentParserTest, EvalBoxMaskNormalizesSurvivesResumeAndRejectsMalformedBoxes) {
    const char* argv[] = {"LichtFeld-Studio", "--eval", "--eval-mask", "bbox: -1, -2,0.5 ,1,2,3"};
    const auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed) << parsed.error();
    EXPECT_EQ((*parsed)->optimization.eval_mask, "bbox:-1,-2,0.5,1,2,3");
    EXPECT_EQ(lfs::core::param::OptimizationParameters::from_json((*parsed)->optimization.to_json()).eval_mask,
              "bbox:-1,-2,0.5,1,2,3");
    lfs::core::param::TrainingParameters restored;
    apply_explicit_training_overrides(restored, (*parsed)->overrides);
    EXPECT_EQ(restored.optimization.eval_mask, "bbox:-1,-2,0.5,1,2,3");

    for (const char* spec : {"bbox:1,0,0,0,1,1", "bbox:0,0,0,1,1", "bbox:0,0,0,1,1,1,1", "bbox:0,0,0,1,x,1", "bbox:"}) {
        const char* bad[] = {"LichtFeld-Studio", "--eval", "--eval-mask", spec};
        EXPECT_FALSE(lfs::core::args::parse_args_and_params(static_cast<int>(std::size(bad)), bad)) << spec;
        lfs::core::param::OptimizationParameters params;
        params.enable_eval = true;
        params.eval_mask = spec;
        EXPECT_FALSE(params.validate().empty()) << spec;
    }
}

// Catches a folder, depth or crop box source that is stored unnormalized or accepted when malformed, and
// "none" failing to clear the mask a resumed project carries.
TEST(ArgumentParserTest, EvalFolderDepthAndCropBoxMasksParseAndValidate) {
    const auto directory = std::filesystem::path(make_test_path("lfs_arg_parser_eval_folder"));
    std::filesystem::create_directories(directory / "masks");
    const auto spec = "masks:" + (directory / "masks" / "..").string() + "/masks";
    const char* argv[] = {"LichtFeld-Studio", "--eval", "--eval-mask", spec.c_str()};
    const auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed) << parsed.error();
    const auto expected = "masks:" + lfs::core::path_to_utf8(std::filesystem::weakly_canonical(directory / "masks"));
    EXPECT_EQ((*parsed)->optimization.eval_mask, expected);
    EXPECT_EQ(lfs::core::param::OptimizationParameters::from_json((*parsed)->optimization.to_json()).eval_mask, expected);

    const char* depth[] = {"LichtFeld-Studio", "--eval", "--eval-mask", "depth: 0.5 ,12"};
    const auto parsed_depth = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(depth)), depth);
    ASSERT_TRUE(parsed_depth) << parsed_depth.error();
    EXPECT_EQ((*parsed_depth)->optimization.eval_mask, "depth:0.5,12");
    for (const char* spec : {"depth:3,2", "depth:-1,2", "depth:1,2,3", "depth:"}) {
        const char* bad_depth[] = {"LichtFeld-Studio", "--eval", "--eval-mask", spec};
        EXPECT_FALSE(lfs::core::args::parse_args_and_params(static_cast<int>(std::size(bad_depth)), bad_depth)) << spec;
    }

    const char* none[] = {"LichtFeld-Studio", "--eval", "--eval-mask", "none"};
    const auto parsed_none = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(none)), none);
    ASSERT_TRUE(parsed_none) << parsed_none.error();
    lfs::core::param::TrainingParameters resumed;
    resumed.optimization.eval_mask = "points:2,3";
    resumed.optimization.eval_mask_invert = true;
    apply_explicit_training_overrides(resumed, (*parsed_none)->overrides);
    EXPECT_TRUE(resumed.optimization.eval_mask.empty());
    EXPECT_FALSE(resumed.optimization.eval_mask_invert);

    const char* cropbox[] = {"LichtFeld-Studio", "--eval", "--eval-mask", "cropbox"};
    const auto parsed_cropbox = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(cropbox)), cropbox);
    ASSERT_TRUE(parsed_cropbox) << parsed_cropbox.error();
    EXPECT_EQ((*parsed_cropbox)->optimization.eval_mask, "cropbox");

    // A missing folder stops a command line, but stored settings stay valid so the project still opens;
    // training reports the folder when it sets up evaluation.
    const auto missing = "masks:" + (directory / "absent").string();
    const char* bad[] = {"LichtFeld-Studio", "--eval", "--eval-mask", missing.c_str()};
    EXPECT_FALSE(lfs::core::args::parse_args_and_params(static_cast<int>(std::size(bad)), bad));
    lfs::core::param::OptimizationParameters params;
    params.enable_eval = true;
    params.eval_mask = missing;
    EXPECT_TRUE(params.validate().empty()) << params.validate();
    params.eval_mask = "masks:relative/masks";
    EXPECT_FALSE(params.validate().empty());
}

// Catches a points file mistaken for radius,close, stored relative, accepted when missing, or lost on resume.
TEST(ArgumentParserTest, EvalPointsFileMaskParsesValidatesAndSurvivesResume) {
    const auto directory = std::filesystem::path(make_test_path("lfs_arg_parser_eval_points"));
    std::filesystem::create_directories(directory);
    std::ofstream(directory / "subject.ply") << "ply\n";
    const auto spec = "points:" + (directory / "." / "subject.ply").string();
    const char* argv[] = {"LichtFeld-Studio", "--eval", "--eval-mask", spec.c_str()};
    const auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed) << parsed.error();
    const auto expected = "points:" + lfs::core::path_to_utf8(std::filesystem::weakly_canonical(directory / "subject.ply"));
    EXPECT_EQ((*parsed)->optimization.eval_mask, expected);
    EXPECT_EQ(lfs::core::param::eval_mask_points_file(expected), expected.substr(7));
    EXPECT_EQ(lfs::core::param::OptimizationParameters::from_json((*parsed)->optimization.to_json()).eval_mask, expected);
    EXPECT_TRUE((*parsed)->optimization.validate().empty());

    EXPECT_FALSE(lfs::core::param::eval_mask_points_file("points:3,4"));
    EXPECT_FALSE(lfs::core::param::eval_mask_splat_file(spec));

    const auto splat_spec = "splat:" + (directory / "." / "subject.ply").string();
    const char* splat_argv[] = {"LichtFeld-Studio", "--eval", "--eval-mask", splat_spec.c_str()};
    const auto parsed_splat = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(splat_argv)), splat_argv);
    ASSERT_TRUE(parsed_splat) << parsed_splat.error();
    const auto splat_expected =
        "splat:" + lfs::core::path_to_utf8(std::filesystem::weakly_canonical(directory / "subject.ply"));
    EXPECT_EQ((*parsed_splat)->optimization.eval_mask, splat_expected);
    EXPECT_EQ(lfs::core::param::eval_mask_splat_file(splat_expected), splat_expected.substr(6));
    EXPECT_FALSE(lfs::core::param::eval_mask_points_file(splat_expected));
    const auto absent_splat = "splat:" + (directory / "absent.ply").string();
    const char* missing_splat[] = {"LichtFeld-Studio", "--eval", "--eval-mask", absent_splat.c_str()};
    EXPECT_FALSE(lfs::core::args::parse_args_and_params(static_cast<int>(std::size(missing_splat)), missing_splat));
    EXPECT_FLOAT_EQ((*parsed_splat)->optimization.eval_mask_opacity, 0.85f);
    const char* opacity_argv[] = {"LichtFeld-Studio", "--eval", "--eval-mask", splat_spec.c_str(), "--eval-mask-opacity", "0.6"};
    const auto parsed_opacity = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(opacity_argv)), opacity_argv);
    ASSERT_TRUE(parsed_opacity) << parsed_opacity.error();
    EXPECT_FLOAT_EQ((*parsed_opacity)->optimization.eval_mask_opacity, 0.6f);
    EXPECT_FLOAT_EQ(lfs::core::param::OptimizationParameters::from_json((*parsed_opacity)->optimization.to_json()).eval_mask_opacity, 0.6f);
    for (const char* bad : {"0", "1.5"}) {
        const char* bad_argv[] = {"LichtFeld-Studio", "--eval", "--eval-mask", splat_spec.c_str(), "--eval-mask-opacity", bad};
        EXPECT_FALSE(lfs::core::args::parse_args_and_params(static_cast<int>(std::size(bad_argv)), bad_argv)) << bad;
    }
    lfs::core::param::OptimizationParameters relative_splat;
    relative_splat.enable_eval = true;
    relative_splat.eval_mask = "splat:subject.ply";
    EXPECT_FALSE(relative_splat.validate().empty());
    EXPECT_FALSE(lfs::core::param::eval_mask_points_file("points"));

    const auto absent = "points:" + (directory / "absent.ply").string();
    const char* missing[] = {"LichtFeld-Studio", "--eval", "--eval-mask", absent.c_str()};
    EXPECT_FALSE(lfs::core::args::parse_args_and_params(static_cast<int>(std::size(missing)), missing));

    lfs::core::param::OptimizationParameters relative;
    relative.enable_eval = true;
    relative.eval_mask = "points:subject.ply";
    EXPECT_FALSE(relative.validate().empty());
}

// Catches accepting an unknown source, missing file, or ineffective evaluation flags.
TEST(ArgumentParserTest, EvalMeshMaskRejectsInvalidSourceFileAndFlagCombinations) {
    const auto directory = std::filesystem::path(
        make_test_path("lfs_arg_parser_bad_eval_mesh"));
    const auto mesh = directory / "mask.obj";
    std::ofstream(mesh).put('\n');
    const auto mesh_spec = "mesh:" + mesh.string();
    const auto missing_spec = "mesh:" + (directory / "missing.obj").string();
    const auto unknown_spec = "image:" + mesh.string();

    const std::vector<std::pair<std::vector<const char*>, std::string_view>> cases{
        {{"LichtFeld-Studio", "--eval", "--eval-mask", unknown_spec.c_str()},
         "Invalid --eval-mask source"},
        {{"LichtFeld-Studio", "--eval", "--eval-mask", missing_spec.c_str()},
         "does not exist"},
        {{"LichtFeld-Studio", "--eval-mask", mesh_spec.c_str()},
         "need --eval"},
        {{"LichtFeld-Studio", "--eval", "--eval-mask-invert"},
         "needs --eval-mask"},
    };
    for (const auto& [arguments, expected] : cases) {
        const auto parsed = lfs::core::args::parse_args_and_params(
            static_cast<int>(arguments.size()), arguments.data());
        ASSERT_FALSE(parsed) << expected;
        EXPECT_NE(parsed.error().find(expected), std::string::npos)
            << parsed.error();
    }
}

TEST(ArgumentParserTest, EvalAllTrainsOnEveryImageAndEnablesEvaluation) {
    const auto data_path = make_test_path("lfs_arg_parser_eval_all_data");
    const auto output_path = make_test_path("lfs_arg_parser_eval_all_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "-d",
        data_path.c_str(),
        "-o",
        output_path.c_str(),
        "--eval-all",
        "--eval-steps",
        "500,1000",
    };
    auto parsed = lfs::core::args::parse_args_and_params(
        static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed.has_value()) << parsed.error();
    EXPECT_TRUE((*parsed)->optimization.enable_eval);
    EXPECT_TRUE((*parsed)->optimization.eval_all);
    EXPECT_FALSE((*parsed)->optimization.holds_out_eval_images());
    EXPECT_EQ((*parsed)->optimization.eval_steps, (std::vector<size_t>{500, 1000}));
}

TEST(ArgumentParserTest, EvalAllRejectsTestEvery) {
    const auto data_path = make_test_path("lfs_arg_parser_eval_all_bad_data");
    const auto output_path = make_test_path("lfs_arg_parser_eval_all_bad_output");

    const char* argv[] = {
        "LichtFeld-Studio",
        "-d",
        data_path.c_str(),
        "-o",
        output_path.c_str(),
        "--eval-all",
        "--test-every",
        "4",
    };
    auto parsed = lfs::core::args::parse_args_and_params(
        static_cast<int>(std::size(argv)), argv);
    ASSERT_FALSE(parsed.has_value());
    EXPECT_NE(parsed.error().find("--test-every"), std::string::npos) << parsed.error();
}

// Catches a conflict check placed after a mode's early return or blind to -i, joined or = spellings.
TEST(ArgumentParserTest, IterationsAndStepsScalerConflictBeforeModeChecks) {
    const std::vector<std::vector<std::string>> cases{
        {"--iter", "2500", "--steps-scaler", "0.5"},
        {"-i", "2500", "--steps-scaler", "0.5"},
        {"--steps-scaler", "0.5", "--iter", "2500"},
        {"--iter=2500", "--steps-scaler=0.5"},
        {"-i2500", "--steps-scaler", "0.5"},
        {"--headless", "--iter", "2500", "--steps-scaler", "0.5"},
        {"--resume", "missing.resume", "-i", "2500", "--steps-scaler", "2"},
        {"-v", "missing.ply", "--iter", "2500", "--steps-scaler", "0.5"},
        {"--render-camera-path", "missing.json", "--iter", "2500", "--steps-scaler", "0.5"},
        {"--import-cameras", "missing", "--iter", "2500", "--steps-scaler", "0.5"},
    };
    for (const auto& values : cases) {
        SCOPED_TRACE(nlohmann::json(values).dump());
        std::vector<const char*> argv{"LichtFeld-Studio"};
        for (const auto& value : values)
            argv.push_back(value.c_str());
        const auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(argv.size()), argv.data());
        ASSERT_FALSE(parsed);
        EXPECT_EQ(parsed.error(), "--iter and --steps-scaler are mutually exclusive: --iter sets the iteration count exactly, --steps-scaler rescales the default schedule");
    }
}

// Catches --iter starting to rescale the timetable.
TEST(ArgumentParserTest, IterationsAlonePreserveDefaultTimetable) {
    const char* argv[]{"LichtFeld-Studio", "--iter", "7000"};
    const auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed) << parsed.error();
    const auto& opt = (*parsed)->optimization;
    const auto defaults = lfs::core::param::OptimizationParameters::mrnf_defaults();
    EXPECT_EQ(opt.iterations, 7000u);
    EXPECT_FLOAT_EQ(opt.steps_scaler, 1.f);
    EXPECT_EQ(opt.stop_refine, defaults.stop_refine);
    EXPECT_EQ(opt.refine_every, defaults.refine_every);
    EXPECT_EQ(opt.sh_degree_interval, defaults.sh_degree_interval);
    EXPECT_EQ(opt.eval_steps, defaults.eval_steps);
    EXPECT_EQ(opt.save_steps, defaults.save_steps);
}

// Catches step scaling applied after explicit CLI step values.
TEST(ArgumentParserTest, ScalingPrecedesAbsoluteStepOverrides) {
    const char* argv[]{"LichtFeld-Studio", "--steps-scaler", "0.5", "--sh-degree-interval", "1000",
                       "--morton-reorder-interval", "3000", "--eval", "--eval-steps", "1000"};
    const auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed) << parsed.error();
    const auto& opt = (*parsed)->optimization;
    EXPECT_EQ(opt.sh_degree_interval, 1000u);
    EXPECT_EQ(opt.morton_reorder_interval, 3000u);
    EXPECT_EQ(opt.eval_steps, std::vector<size_t>{1000});
    EXPECT_EQ(opt.iterations, 15000u);
    EXPECT_EQ(opt.stop_refine, 14250u);
    EXPECT_EQ(opt.refine_every, 82u);
    EXPECT_TRUE((*parsed)->overrides.has_optimization_key("morton_reorder_interval"));
    lfs::core::param::TrainingParameters restored;
    apply_explicit_training_overrides(restored, (*parsed)->overrides);
    EXPECT_EQ(restored.optimization.morton_reorder_interval, 3000u);
    EXPECT_EQ(restored.optimization.eval_steps, std::vector<size_t>{1000});
}

// Catches a config steps_scaler multiplying an explicit --iter.
TEST(ArgumentParserTest, ConfigScalingPrecedesExplicitIterations) {
    const auto path = std::filesystem::path(make_test_path("lfs_arg_parser_step_scaling")) / "config.json";
    auto json = lfs::core::param::OptimizationParameters::mrnf_defaults().to_json();
    json["steps_scaler"] = 0.5f;
    std::ofstream(path) << json.dump();
    const auto path_text = path.string();
    const char* argv[]{"LichtFeld-Studio", "--config", path_text.c_str(), "--iter", "2500"};
    const auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    std::filesystem::remove(path);
    ASSERT_TRUE(parsed) << parsed.error();
    EXPECT_EQ((*parsed)->optimization.iterations, 2500u);
    EXPECT_EQ((*parsed)->optimization.stop_refine, 14250u);
    EXPECT_FLOAT_EQ((*parsed)->optimization.steps_scaler, 0.5f);
}

// Catches help text that hides the --iter / --steps-scaler conflict.
TEST(ArgumentParserTest, StepScalingHelpShowsMutualExclusion) {
    EXPECT_NE(lfs::core::args::optimization_cli_help("--iter").find("; cannot be combined with --steps-scaler"), std::string::npos);
    EXPECT_NE(lfs::core::args::optimization_cli_help("--steps-scaler").find("; cannot be combined with --iter"), std::string::npos);
}

// Catches a joined short value such as -i2500 being parsed but then ignored.
TEST(ArgumentParserTest, JoinedShortIterationValueIsUsed) {
    const char* argv[]{"LichtFeld-Studio", "-i2500"};
    const auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
    ASSERT_TRUE(parsed) << parsed.error();
    EXPECT_EQ((*parsed)->optimization.iterations, 2500u);
    EXPECT_TRUE((*parsed)->cli_iterations_set);
}
