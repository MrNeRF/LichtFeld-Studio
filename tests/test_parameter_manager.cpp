/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "core/parameter_manager.hpp"
#include "core/parameters.hpp"
#include "core/services.hpp"
#include "core/training_manager.hpp"
#include "io/argument_parser.hpp"
#include "io/project_chapters.hpp"

#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <random>

namespace {

    std::filesystem::path unique_temp_config_path() {
        const auto* test = ::testing::UnitTest::GetInstance()->current_test_info();
        return std::filesystem::temp_directory_path() /
               std::format("lfs_{}_{}.json", test->name(), std::random_device{}());
    }

    TEST(ParameterManagerTest, EditableDatasetUsesConfigurationAuthorityWithHeadlessFallback) {
        lfs::vis::ParameterManager manager;
        ASSERT_TRUE(manager.ensureLoaded());
        lfs::vis::TrainerManager trainer_manager;
        auto* previous = lfs::vis::services().paramsOrNull();
        struct RestoreService {
            lfs::vis::ParameterManager* previous;
            ~RestoreService() { lfs::vis::services().set(previous); }
        } restore{previous};
        lfs::vis::services().set(&manager);
        trainer_manager.getEditableDatasetParams().max_width = 800;
        EXPECT_EQ(manager.getDatasetConfig().max_width, 800);
        manager.getDatasetConfig().max_width = 1600;
        EXPECT_EQ(std::as_const(trainer_manager).getEditableDatasetParams().max_width, 1600);
        EXPECT_EQ(&trainer_manager.getEditableDatasetParams(), &manager.getDatasetConfig());
        EXPECT_EQ(trainer_manager.getEditableTrainingParams(manager).dataset.max_width, 1600);
        manager.modifyActiveParams([](auto& params) { params.iterations = 1234; });
        EXPECT_EQ(trainer_manager.getEditableTrainingParams(manager).optimization.iterations, 1234u);

        lfs::vis::services().set(static_cast<lfs::vis::ParameterManager*>(nullptr));
        trainer_manager.getEditableDatasetParams().max_width = 1024;
        EXPECT_EQ(std::as_const(trainer_manager).getEditableDatasetParams().max_width, 1024);
        EXPECT_EQ(manager.getDatasetConfig().max_width, 1600);
    }

    TEST(ParameterManagerTest, ConfigRoundTripRestoresEditedDatasetAndOptimization) {
        lfs::vis::ParameterManager source;
        ASSERT_TRUE(source.ensureLoaded());
        source.getDatasetConfig().max_width = 800;
        source.getDatasetConfig().resize_factor = 2;
        source.getDatasetConfig().invert_masks = true;
        source.getDatasetConfig().loading_params.use_cpu_memory = false;
        source.modifyActiveParams([](auto& params) { params.steps_scaler = 2.f; params.iterations = 1200; });
        const auto path = unique_temp_config_path();
        const auto exported = source.createForDataset("dataset", "output");
        ASSERT_TRUE(lfs::core::param::save_training_parameters_to_json(exported, path));

        lfs::vis::ParameterManager target;
        ASSERT_TRUE(target.ensureLoaded());
        ASSERT_TRUE(target.importConfigFile(path));
        EXPECT_EQ(target.getDatasetConfig().max_width, 800);
        EXPECT_EQ(target.getDatasetConfig().resize_factor, 2);
        EXPECT_TRUE(target.getDatasetConfig().invert_masks);
        EXPECT_FALSE(target.getDatasetConfig().loading_params.use_cpu_memory);
        EXPECT_EQ(target.copyActiveParams().iterations, 1200);
        EXPECT_TRUE(target.isDirty());
        std::filesystem::remove(path);
    }

    TEST(ParameterManagerTest, PartialConfigPreservesOmittedDatasetFields) {
        lfs::vis::ParameterManager manager;
        ASSERT_TRUE(manager.ensureLoaded());
        manager.getDatasetConfig().images = "images_4";
        manager.getDatasetConfig().loading_params.use_cpu_memory = false;
        const auto path = unique_temp_config_path();
        nlohmann::json config{
            {"optimization", lfs::core::param::OptimizationParameters::mrnf_defaults().to_json()},
            {"dataset", {{"max_width", 800}}}};
        std::ofstream(path) << config.dump();
        ASSERT_TRUE(manager.importConfigFile(path));
        EXPECT_EQ(manager.getDatasetConfig().max_width, 800);
        EXPECT_EQ(manager.getDatasetConfig().images, "images_4");
        EXPECT_FALSE(manager.getDatasetConfig().loading_params.use_cpu_memory);
        std::ofstream(path) << lfs::core::param::OptimizationParameters::mcmc_defaults().to_json().dump();
        ASSERT_TRUE(manager.importConfigFile(path));
        EXPECT_EQ(manager.getDatasetConfig().max_width, 800);
        EXPECT_EQ(manager.getActiveStrategy(), "mcmc");
        std::filesystem::remove(path);
    }

    TEST(ParameterManagerTest, ConfigImportFromAnotherDatasetPreservesLoadedPaths) {
        lfs::vis::ParameterManager source;
        ASSERT_TRUE(source.ensureLoaded());
        source.getDatasetConfig().max_width = 800;
        source.modifyActiveParams([](auto& params) { params.iterations = 1200; });
        const auto path = unique_temp_config_path();
        const auto exported = source.createForDataset("dataset_a", "output_a");
        ASSERT_TRUE(lfs::core::param::save_training_parameters_to_json(exported, path));

        for (const bool import_dataset : {false, true}) {
            for (const bool output_explicit : {false, true}) {
                SCOPED_TRACE(std::format("import_dataset={}, output_explicit={}", import_dataset, output_explicit));
                lfs::vis::ParameterManager target;
                ASSERT_TRUE(target.ensureLoaded());
                auto* previous = lfs::vis::services().paramsOrNull();
                struct RestoreService {
                    lfs::vis::ParameterManager* previous;
                    ~RestoreService() { lfs::vis::services().set(previous); }
                } restore{previous};
                lfs::vis::services().set(&target);
                auto& dataset = target.getDatasetConfig();
                dataset.data_path = "dataset_b";
                dataset.output_path = "output_b";
                dataset.output_path_explicit = output_explicit;
                dataset.max_width = 1600;
                ASSERT_TRUE(target.importConfigFile(path, import_dataset));
                EXPECT_EQ(dataset.data_path, std::filesystem::path("dataset_b"));
                EXPECT_EQ(dataset.output_path, std::filesystem::path("output_b"));
                EXPECT_EQ(dataset.output_path_explicit, output_explicit);
                EXPECT_EQ(dataset.max_width, import_dataset ? 800 : 1600);
                EXPECT_EQ(target.copyActiveParams().iterations, 1200u);

                lfs::vis::TrainerManager trainer_manager;
                const auto training_params = trainer_manager.getEditableTrainingParams(target);
                EXPECT_EQ(training_params.dataset.data_path, dataset.data_path);
                EXPECT_EQ(training_params.dataset.output_path, dataset.output_path);
                EXPECT_EQ(training_params.dataset.output_path_explicit, output_explicit);
                EXPECT_EQ(training_params.dataset.max_width, dataset.max_width);
                const auto saved_dataset = training_params.dataset.to_json();
                EXPECT_EQ(saved_dataset.at("data_path"), "dataset_b");
                EXPECT_EQ(saved_dataset.at("output_folder"), "output_b");
            }
        }
        std::filesystem::remove(path);
    }

    TEST(ParameterManagerTest, InvalidDatasetImportDoesNotPartiallyChangeParameters) {
        lfs::vis::ParameterManager manager;
        ASSERT_TRUE(manager.ensureLoaded());
        manager.getDatasetConfig().max_width = 800;
        const auto path = unique_temp_config_path();
        for (const auto* value : {"-1", "\"invalid\""}) {
            nlohmann::json config{
                {"optimization", lfs::core::param::OptimizationParameters::mcmc_defaults().to_json()},
                {"dataset", {{"max_width", nlohmann::json::parse(value)}}}};
            std::ofstream(path) << config.dump();
            const auto result = manager.importConfigFile(path);
            ASSERT_FALSE(result);
            EXPECT_EQ(result.error().code(), lfs::ErrorCode::InvalidArgument);
            EXPECT_EQ(result.error().domain(), lfs::ErrorDomain::IO);
            EXPECT_FALSE(result.error().detail().empty());
            EXPECT_EQ(manager.getDatasetConfig().max_width, 800);
            EXPECT_EQ(manager.getActiveStrategy(), "mrnf");
        }
        std::filesystem::remove(path);
    }

    TEST(ParameterManagerTest, ConfigImportCanKeepActiveDatasetLocked) {
        lfs::vis::ParameterManager manager;
        ASSERT_TRUE(manager.ensureLoaded());
        manager.getDatasetConfig().max_width = 800;
        const auto path = unique_temp_config_path();
        nlohmann::json config{
            {"optimization", lfs::core::param::OptimizationParameters::mcmc_defaults().to_json()},
            {"dataset", {{"max_width", 1600}}}};
        std::ofstream(path) << config.dump();
        ASSERT_TRUE(manager.importConfigFile(path, false));
        EXPECT_EQ(manager.getDatasetConfig().max_width, 800);
        EXPECT_EQ(manager.getActiveStrategy(), "mcmc");
        std::filesystem::remove(path);
    }

    TEST(ParameterManagerTest, DefaultStrategyIsMrnf) {
        lfs::vis::ParameterManager manager;
        const auto load_result = manager.ensureLoaded();
        ASSERT_TRUE(load_result.has_value()) << load_result.error();

        EXPECT_EQ(manager.getActiveStrategy(), "mrnf");
        EXPECT_EQ(manager.getActiveParams().strategy, "mrnf");
        EXPECT_EQ(lfs::core::param::OptimizationParameters{}.strategy, "mrnf");
        EXPECT_EQ(lfs::core::param::OptimizationParameters::mcmc_defaults().strategy, "mcmc");
    }

    TEST(ParameterManagerTest, SessionCopyTracksExplicitSourcesAndResetBaseline) {
        lfs::vis::ParameterManager manager;
        const auto load_result = manager.ensureLoaded();
        ASSERT_TRUE(load_result.has_value()) << load_result.error();

        const auto factory_defaults = lfs::core::param::OptimizationParameters::mrnf_defaults();
        const auto initial_session = manager.copySessionParams();
        EXPECT_EQ(initial_session.strategy, factory_defaults.strategy);
        EXPECT_FLOAT_EQ(initial_session.opacity_lr, factory_defaults.opacity_lr);
        EXPECT_EQ(initial_session.max_cap, factory_defaults.max_cap);

        lfs::core::param::TrainingParameters cli_params;
        cli_params.optimization = factory_defaults;
        cli_params.optimization.opacity_lr = 0.123f;
        cli_params.optimization.max_cap = 1'234'567;
        manager.setSessionDefaults(cli_params);

        const auto cli_session = manager.copySessionParams();
        EXPECT_FLOAT_EQ(cli_session.opacity_lr, 0.123f);
        EXPECT_EQ(cli_session.max_cap, 1'234'567);

        manager.modifyActiveParams([](auto& params) {
            params.opacity_lr = 0.75f;
            params.max_cap = 42;
        });
        manager.resetToDefaults("mrnf");
        const auto reset_current = manager.copyActiveParams();
        EXPECT_FLOAT_EQ(reset_current.opacity_lr, 0.123f);
        EXPECT_EQ(reset_current.max_cap, 1'234'567);

        lfs::core::param::TrainingParameters checkpoint_params;
        checkpoint_params.optimization = lfs::core::param::OptimizationParameters::igs_plus_defaults();
        checkpoint_params.optimization.opacity_lr = 0.321f;
        checkpoint_params.optimization.max_cap = 765'432;
        manager.importTrainingParams(checkpoint_params);

        const auto checkpoint_session = manager.copySessionParams();
        EXPECT_EQ(checkpoint_session.strategy, "igs+");
        EXPECT_FLOAT_EQ(checkpoint_session.opacity_lr, 0.321f);
        EXPECT_EQ(checkpoint_session.max_cap, 765'432);
    }

    TEST(ParameterManagerTest, ImportTrainingParamsRestoresResolvedCheckpointState) {
        lfs::vis::ParameterManager manager;
        const auto load_result = manager.ensureLoaded();
        ASSERT_TRUE(load_result.has_value()) << load_result.error();

        lfs::core::param::TrainingParameters startup_params;
        startup_params.optimization.strategy = "mcmc";
        startup_params.optimization.iterations = 1000;
        startup_params.dataset.data_path = "/tmp/startup_dataset";
        startup_params.dataset.output_path = "/tmp/startup_output";
        startup_params.dataset.images = "images";
        startup_params.dataset.resize_factor = 2;
        startup_params.dataset.max_width = 2048;
        manager.setSessionDefaults(startup_params);

        lfs::core::param::TrainingParameters checkpoint_params;
        checkpoint_params.optimization = lfs::core::param::OptimizationParameters::igs_plus_defaults();
        checkpoint_params.optimization.strategy = "igs+";
        checkpoint_params.optimization.iterations = 600;
        checkpoint_params.optimization.max_cap = 123456;
        checkpoint_params.optimization.save_steps = {500};
        checkpoint_params.dataset.data_path = "/tmp/checkpoint_dataset";
        checkpoint_params.dataset.output_path = "/tmp/checkpoint_output";
        checkpoint_params.dataset.images = "images_4";
        checkpoint_params.dataset.resize_factor = -1;
        checkpoint_params.dataset.max_width = 1536;
        checkpoint_params.dataset.test_every = 4;
        checkpoint_params.dataset.loading_params.use_cpu_memory = false;
        checkpoint_params.dataset.invert_masks = true;
        checkpoint_params.dataset.mask_threshold = 0.75f;

        manager.importTrainingParams(checkpoint_params);

        EXPECT_EQ(manager.getActiveStrategy(), "igs+");
        EXPECT_FALSE(manager.consumeDirty());

        const auto& active = manager.getActiveParams();
        EXPECT_EQ(active.strategy, "igs+");
        EXPECT_EQ(active.iterations, 600u);
        EXPECT_EQ(active.max_cap, 123456);
        EXPECT_EQ(active.save_steps, std::vector<size_t>({500}));

        const auto& igs_params = manager.getCurrentParams("igs+");
        EXPECT_EQ(igs_params.iterations, 600u);
        EXPECT_EQ(manager.getCurrentParams("mcmc").iterations, 1000u);

        const auto& dataset = manager.getDatasetConfig();
        EXPECT_EQ(dataset.data_path, checkpoint_params.dataset.data_path);
        EXPECT_EQ(dataset.output_path, checkpoint_params.dataset.output_path);
        EXPECT_EQ(dataset.images, "images_4");
        EXPECT_EQ(dataset.resize_factor, -1);
        EXPECT_EQ(dataset.max_width, 1536);
        EXPECT_EQ(dataset.test_every, 4);
        EXPECT_FALSE(dataset.loading_params.use_cpu_memory);
        EXPECT_TRUE(dataset.invert_masks);
        EXPECT_FLOAT_EQ(dataset.mask_threshold, 0.75f);

        const auto recreated = manager.createForDataset("/tmp/override_dataset", "/tmp/override_output");
        EXPECT_EQ(recreated.optimization.strategy, "igs+");
        EXPECT_EQ(recreated.optimization.iterations, 600u);
        EXPECT_EQ(recreated.dataset.data_path, "/tmp/override_dataset");
        EXPECT_EQ(recreated.dataset.output_path, "/tmp/override_output");
        EXPECT_EQ(recreated.dataset.images, "images_4");
    }

    TEST(ParameterManagerTest, TrainingConfigRoundTripsDatasetAndTrainingValues) {
        const auto config_path =
            std::filesystem::temp_directory_path() / "lfs_training_config_roundtrip_test.json";
        lfs::core::param::TrainingParameters source;
        source.dataset.data_path = "/tmp/config_dataset";
        source.dataset.output_path = "/tmp/config_output";
        source.dataset.images = "images_2";
        source.dataset.resize_factor = 2;
        source.dataset.max_width = 800;
        source.dataset.test_every = 5;
        source.dataset.invert_masks = true;
        source.dataset.mask_threshold = 0.7f;
        source.dataset.centralize_dataset = "by_cameras";
        source.dataset.loading_params.use_cpu_memory = false;
        source.dataset.loading_params.use_16bit_color = false;
        source.optimization = lfs::core::param::OptimizationParameters::mrnf_defaults();
        source.optimization.iterations = 1234;
        source.optimization.undistort = true;

        lfs::vis::ParameterManager parameter_manager;
        ASSERT_TRUE(parameter_manager.ensureLoaded());
        lfs::vis::TrainerManager trainer_manager;
        trainer_manager.getEditableDatasetParams() = source.dataset;
        lfs::core::param::TrainingParameters session_defaults;
        session_defaults.optimization.iterations = 1234;
        session_defaults.optimization.undistort = true;
        session_defaults.server.tcp_connection = true;
        session_defaults.server.tcp_server_connection_port = 12345;
        parameter_manager.setSessionDefaults(session_defaults);
        const auto edited_params = trainer_manager.getEditableTrainingParams(parameter_manager);
        ASSERT_EQ(parameter_manager.getDatasetConfig().max_width, 3840);
        ASSERT_EQ(edited_params.dataset.max_width, 800);
        ASSERT_TRUE(edited_params.server.tcp_connection);
        ASSERT_EQ(edited_params.server.tcp_server_connection_port, 12345);
        ASSERT_TRUE(lfs::core::param::save_training_parameters_to_json(edited_params, config_path));
        const auto imported = lfs::core::param::read_training_parameters_from_json(config_path);
        std::error_code ec;
        std::filesystem::remove(config_path, ec);
        ASSERT_TRUE(imported) << imported.error().detail();

        EXPECT_EQ(imported->dataset.data_path, source.dataset.data_path);
        EXPECT_EQ(imported->dataset.output_path, source.dataset.output_path);
        EXPECT_EQ(imported->dataset.images, "images_2");
        EXPECT_EQ(imported->dataset.resize_factor, 2);
        EXPECT_EQ(imported->dataset.max_width, 800);
        EXPECT_EQ(imported->dataset.test_every, 5);
        EXPECT_TRUE(imported->dataset.invert_masks);
        EXPECT_FLOAT_EQ(imported->dataset.mask_threshold, 0.7f);
        EXPECT_EQ(imported->dataset.centralize_dataset, "by_cameras");
        EXPECT_FALSE(imported->dataset.loading_params.use_cpu_memory);
        EXPECT_FALSE(imported->dataset.loading_params.use_16bit_color);
        EXPECT_EQ(imported->optimization.iterations, 1234u);
        EXPECT_TRUE(imported->optimization.undistort);
        EXPECT_TRUE(imported->server.tcp_connection);
        EXPECT_EQ(imported->server.tcp_server_connection_port, 12345);

        parameter_manager.importTrainingParams(*imported);
        const auto training_params = trainer_manager.getEditableTrainingParams(parameter_manager);
        EXPECT_EQ(training_params.dataset.max_width, 800);
        EXPECT_EQ(training_params.dataset.resize_factor, 2);
        EXPECT_EQ(training_params.dataset.images, "images_2");
        EXPECT_EQ(training_params.dataset.test_every, 5);
        EXPECT_EQ(training_params.optimization.iterations, 1234u);
        EXPECT_TRUE(training_params.optimization.undistort);
        EXPECT_TRUE(training_params.server.tcp_connection);
        EXPECT_EQ(training_params.server.tcp_server_connection_port, 12345);

        const auto partial_path =
            std::filesystem::temp_directory_path() / "lfs_training_config_partial_test.json";
        std::ofstream(partial_path)
            << nlohmann::json{{"optimization", {{"iterations", 4321}}}}.dump();
        auto defaults = training_params;
        defaults.dataset.max_width = 640;
        defaults.server.tcp_server_connection_port = 23456;
        const auto partial = lfs::core::param::read_training_parameters_from_json(partial_path, defaults);
        std::filesystem::remove(partial_path, ec);
        ASSERT_TRUE(partial) << partial.error().detail();
        EXPECT_EQ(partial->dataset.max_width, 640);
        EXPECT_EQ(partial->server.tcp_server_connection_port, 23456);
        EXPECT_EQ(partial->optimization.iterations, 4321u);
    }

    TEST(ParameterManagerTest, SessionDefaultsCanReplaceCheckpointImportState) {
        lfs::vis::ParameterManager manager;
        const auto load_result = manager.ensureLoaded();
        ASSERT_TRUE(load_result.has_value()) << load_result.error();

        lfs::core::param::TrainingParameters checkpoint_params;
        checkpoint_params.optimization = lfs::core::param::OptimizationParameters::igs_plus_defaults();
        checkpoint_params.optimization.strategy = "igs+";
        checkpoint_params.optimization.iterations = 600;
        checkpoint_params.dataset.images = "images_4";
        checkpoint_params.dataset.data_path = "/tmp/checkpoint_dataset";
        checkpoint_params.dataset.output_path = "/tmp/checkpoint_output";
        checkpoint_params.dataset.max_width = 1536;

        manager.importTrainingParams(checkpoint_params);

        lfs::core::param::TrainingParameters dataset_params;
        dataset_params.optimization = lfs::core::param::OptimizationParameters::mcmc_defaults();
        dataset_params.optimization.strategy = "mcmc";
        dataset_params.optimization.iterations = 900;
        dataset_params.dataset.images = "images_8";
        dataset_params.dataset.resize_factor = 4;
        dataset_params.dataset.max_width = 0;
        dataset_params.dataset.data_path = "/tmp/new_dataset";
        dataset_params.dataset.output_path = "/tmp/new_output";

        manager.setSessionDefaults(dataset_params);

        EXPECT_EQ(manager.getActiveStrategy(), "mcmc");
        EXPECT_EQ(manager.getActiveParams().strategy, "mcmc");
        EXPECT_EQ(manager.getActiveParams().iterations, 900u);

        const auto& dataset = manager.getDatasetConfig();
        EXPECT_EQ(dataset.images, "images_8");
        EXPECT_EQ(dataset.resize_factor, 4);
        EXPECT_EQ(dataset.max_width, 0);
        const auto recreated = manager.createForDataset("/tmp/override_dataset", "/tmp/override_output");
        EXPECT_EQ(recreated.optimization.strategy, "mcmc");
        EXPECT_EQ(recreated.optimization.iterations, 900u);
        EXPECT_EQ(recreated.dataset.images, "images_8");
        EXPECT_EQ(recreated.dataset.data_path, "/tmp/override_dataset");
        EXPECT_EQ(recreated.dataset.output_path, "/tmp/override_output");
    }

    TEST(ParameterManagerTest, ClearSessionRestoresBuiltinsAndClearsDatasetConfig) {
        lfs::vis::ParameterManager manager;
        const auto load_result = manager.ensureLoaded();
        ASSERT_TRUE(load_result.has_value()) << load_result.error();

        lfs::core::param::TrainingParameters params;
        params.optimization = lfs::core::param::OptimizationParameters::mcmc_defaults();
        params.optimization.strategy = "mcmc";
        params.optimization.iterations = 900;
        params.dataset.data_path = "/tmp/dataset";
        params.dataset.output_path = "/tmp/output";
        params.dataset.images = "images_8";
        params.dataset.resize_factor = 4;
        params.dataset.loading_params.use_cpu_memory = false;

        manager.setSessionDefaults(params);
        manager.clearSession();

        EXPECT_EQ(manager.getActiveStrategy(), "mrnf");
        EXPECT_EQ(manager.getActiveParams().strategy, "mrnf");
        EXPECT_EQ(manager.getActiveParams().iterations,
                  lfs::core::param::OptimizationParameters::mrnf_defaults().iterations);

        const auto& dataset = manager.getDatasetConfig();
        EXPECT_TRUE(dataset.data_path.empty());
        EXPECT_TRUE(dataset.output_path.empty());
        EXPECT_EQ(dataset.images, "images");
        EXPECT_EQ(dataset.resize_factor, -1);
        EXPECT_EQ(dataset.max_width, 3840);
        EXPECT_TRUE(dataset.loading_params.use_cpu_memory);
    }

    TEST(ParameterManagerTest, PpispAutoControllerUsesPlannedTotalIterations) {
        auto params = lfs::core::param::OptimizationParameters::mrnf_defaults();
        params.ppisp_use_controller = true;
        params.iterations = 10'000;
        params.enable_sparsity = true;
        params.sparsify_steps = 15'000;

        EXPECT_EQ(params.resolved_total_iterations(), 25'000);
        EXPECT_EQ(params.resolved_ppisp_controller_activation_step(params.resolved_total_iterations()), 20'000);
    }

    TEST(ParameterManagerTest,
         PendingProjectRestoreChangesOnlyRoleQualifiedManagerState) {
        lfs::vis::ParameterManager source;
        ASSERT_TRUE(source.ensureLoaded());
        auto captured = source.capturePendingProjectState();
        ASSERT_TRUE(captured) << captured.error().user_message();
        captured->active_strategy = "igs+";
        captured->mcmc_current.iterations = 101;
        captured->mrnf_current.iterations = 202;
        captured->igs_current.iterations = 303;
        captured->dataset.images = "images_project";
        captured->dataset.centralize_dataset = "pointcloud";

        lfs::vis::ParameterManager target;
        ASSERT_TRUE(target.ensureLoaded());
        target.markDirty();
        auto restored = target.restorePendingProjectState(*captured);
        ASSERT_TRUE(restored) << restored.error().user_message();
        EXPECT_EQ(target.getActiveStrategy(), "igs+");
        EXPECT_EQ(target.getCurrentParams("mcmc").iterations, 101u);
        EXPECT_EQ(target.getCurrentParams("mrnf").iterations, 202u);
        EXPECT_EQ(target.getCurrentParams("igs+").iterations, 303u);
        EXPECT_EQ(target.getDatasetConfig().images, "images_project");
        EXPECT_FALSE(target.consumeDirty());

        auto invalid = *captured;
        invalid.mcmc_current =
            lfs::core::param::OptimizationParameters::mrnf_defaults();
        auto rejected = target.restorePendingProjectState(invalid);
        EXPECT_FALSE(rejected);
        EXPECT_EQ(target.getCurrentParams("mcmc").iterations, 101u);
    }

    TEST(ParameterValidationTest, RejectsCrashProneIterationAndNumericValues) {
        lfs::core::param::OptimizationParameters params;
        EXPECT_TRUE(params.validate().empty());
        EXPECT_TRUE(lfs::core::param::OptimizationParameters::mcmc_defaults().validate().empty());
        EXPECT_TRUE(lfs::core::param::OptimizationParameters::mrnf_defaults().validate().empty());
        EXPECT_TRUE(lfs::core::param::OptimizationParameters::igs_plus_defaults().validate().empty());

        params.refine_every = 0;
        EXPECT_NE(params.validate().find("refine_every"), std::string::npos);
        params = {};
        params.reset_every = 0;
        EXPECT_NE(params.validate().find("reset_every"), std::string::npos);
        params = {};
        params.sh_degree_interval = 0;
        EXPECT_NE(params.validate().find("sh_degree_interval"), std::string::npos);
        params = {};
        params.morton_reorder_interval = 0;
        EXPECT_TRUE(params.validate().empty());
        params.morton_reorder_interval = 5000;
        EXPECT_TRUE(params.validate().empty());
        params = {};
        params.start_refine = 10;
        params.stop_refine = 9;
        EXPECT_NE(params.validate().find("start_refine"), std::string::npos);
        params = {};
        params.bounds_percentile = std::numeric_limits<float>::quiet_NaN();
        EXPECT_NE(params.validate().find("bounds_percentile"), std::string::npos);
        params = {};
        params.means_lr = std::numeric_limits<float>::infinity();
        EXPECT_NE(params.validate().find("means_lr"), std::string::npos);
        params = {};
        params.cropbox_lr_scale = std::numeric_limits<float>::quiet_NaN();
        EXPECT_NE(params.validate().find("cropbox_lr_scale"), std::string::npos);
        params.cropbox_lr_scale = -0.1f;
        EXPECT_NE(params.validate().find("cropbox_lr_scale"), std::string::npos);
        params.cropbox_lr_scale = 1.1f;
        EXPECT_NE(params.validate().find("cropbox_lr_scale"), std::string::npos);
        params = {};
        params.cropbox_loss_weight = std::numeric_limits<float>::quiet_NaN();
        EXPECT_NE(params.validate().find("cropbox_loss_weight"), std::string::npos);
        params.cropbox_loss_weight = -0.1f;
        EXPECT_NE(params.validate().find("cropbox_loss_weight"), std::string::npos);
        params.cropbox_loss_weight = 1.1f;
        EXPECT_NE(params.validate().find("cropbox_loss_weight"), std::string::npos);
    }

    TEST(ParameterValidationTest, CropBoxLrScaleJsonIsBackwardCompatible) {
        lfs::core::param::OptimizationParameters params;
        params.cropbox_lr_scale = 0.35f;
        auto json = params.to_json();

        EXPECT_FLOAT_EQ(json.at("cropbox_lr_scale").get<float>(), 0.35f);
        EXPECT_FLOAT_EQ(
            lfs::core::param::OptimizationParameters::from_json(json).cropbox_lr_scale,
            0.35f);

        json.erase("cropbox_lr_scale");
        EXPECT_FLOAT_EQ(
            lfs::core::param::OptimizationParameters::from_json(json).cropbox_lr_scale,
            0.1f);
    }

    TEST(ParameterValidationTest, CropBoxLossWeightJsonIsBackwardCompatible) {
        lfs::core::param::OptimizationParameters params;
        params.cropbox_loss_weight = 0.45f;
        auto json = params.to_json();

        EXPECT_FLOAT_EQ(json.at("cropbox_loss_weight").get<float>(), 0.45f);
        EXPECT_FLOAT_EQ(
            lfs::core::param::OptimizationParameters::from_json(json).cropbox_loss_weight,
            0.45f);

        json.erase("cropbox_loss_weight");
        EXPECT_FLOAT_EQ(
            lfs::core::param::OptimizationParameters::from_json(json).cropbox_loss_weight,
            0.1f);
    }

    TEST(ParameterValidationTest, RejectsDatasetCadenceAndOddVideoDimensions) {
        lfs::core::param::TrainingParameters params;
        params.dataset.test_every = 0;
        EXPECT_NE(params.validate().find("test_every"), std::string::npos);

        params.dataset.test_every = 8;
        params.dataset.timelapse_every = 0;
        EXPECT_NE(params.validate().find("timelapse_every"), std::string::npos);

        params.dataset.timelapse_every = 50;
        params.render_path = lfs::core::param::RenderPathConfig{
            .width = 1919,
            .height = 1080};
        EXPECT_NE(params.validate().find("render dimensions"), std::string::npos);
    }

    // Catches auto-scale replacing the user's factor instead of multiplying it.
    TEST(ParameterManagerTest, ImageScalingComposesWithUserScaling) {
        lfs::vis::ParameterManager manager;
        lfs::core::param::TrainingParameters params;
        params.optimization = lfs::core::param::OptimizationParameters::mrnf_defaults();
        params.optimization.steps_scaler = 0.5f;
        params.optimization.apply_step_scaling();
        manager.setSessionDefaults(params);
        manager.autoScaleSteps(600);
        EXPECT_EQ(manager.getActiveParams().iterations, 30000u);
        EXPECT_FLOAT_EQ(manager.getActiveParams().steps_scaler, 1.f);
        EXPECT_FLOAT_EQ(manager.getActiveParams().image_count_scaler, 2.f);
        const auto scaled = manager.getActiveParams().to_json();
        manager.autoScaleSteps(600);
        EXPECT_EQ(manager.getActiveParams().to_json(), scaled);
        manager.autoScaleSteps(300);
        EXPECT_EQ(manager.getActiveParams().iterations, 15000u);
        EXPECT_FLOAT_EQ(manager.getActiveParams().steps_scaler, 0.5f);
        EXPECT_FLOAT_EQ(manager.getActiveParams().image_count_scaler, 1.f);
    }

    // Catches the command-line lock being lost on reset or replace-load, or never released.
    TEST(ParameterManagerTest, CliIterationLockSurvivesResetAndReplaceLoad) {
        lfs::vis::ParameterManager manager;
        lfs::core::param::TrainingParameters params;
        params.optimization = lfs::core::param::OptimizationParameters::mrnf_defaults();
        params.optimization.strategy = "mnrf";
        params.optimization.iterations = 2500;
        params.cli_iterations_set = true;
        manager.setSessionDefaults(params);
        const auto seeded = manager.getActiveParams().to_json();
        manager.autoScaleSteps(600);
        EXPECT_EQ(manager.getActiveParams().to_json(), seeded);
        EXPECT_EQ(manager.getCurrentParams("mcmc").iterations,
                  2 * lfs::core::param::OptimizationParameters::mcmc_defaults().iterations);
        manager.modifyActiveParams([](auto& opt) { opt.iterations = 999; });
        manager.resetToDefaults();
        manager.autoScaleSteps(600);
        EXPECT_EQ(manager.getActiveParams().to_json(), seeded);
        const auto captured = manager.capturePendingProjectState();
        ASSERT_TRUE(captured);
        manager.clearSession();
        manager.installValidatedPendingProjectState(*captured);
        manager.autoScaleSteps(600);
        EXPECT_EQ(manager.getActiveParams().to_json(), seeded);
        manager.clearSession();
        manager.autoScaleSteps(600);
        EXPECT_EQ(manager.getActiveParams().iterations,
                  2 * lfs::core::param::OptimizationParameters::mrnf_defaults().iterations);
    }

    // Catches a lock that outlives imports or is read back from a project file.
    TEST(ParameterManagerTest, ImportsAndParsedProjectClearCliStepLock) {
        lfs::vis::ParameterManager manager;
        lfs::core::param::TrainingParameters params;
        params.optimization = lfs::core::param::OptimizationParameters::mrnf_defaults();
        params.optimization.iterations = 2500;
        params.cli_iterations_set = true;
        manager.setSessionDefaults(params);
        manager.importParams(params.optimization);
        manager.autoScaleSteps(600);
        EXPECT_EQ(manager.getActiveParams().iterations, 5000u);
        manager.setSessionDefaults(params);
        manager.importTrainingParams(params);
        manager.autoScaleSteps(600);
        EXPECT_EQ(manager.getActiveParams().iterations, 5000u);
        manager.setSessionDefaults(params);
        const auto captured = manager.capturePendingProjectState();
        ASSERT_TRUE(captured);
        ASSERT_TRUE(captured->cli_step_locked_strategy);
        lfs::io::project::ParametersChapter chapter;
        ASSERT_TRUE(chapter.set_snapshot(*captured));
        const auto parsed = lfs::io::project::ParametersChapter::from_bytes(chapter.to_bytes());
        ASSERT_TRUE(parsed);
        const auto snapshot = parsed->snapshot();
        ASSERT_TRUE(snapshot);
        EXPECT_FALSE(snapshot->cli_step_locked_strategy);
        manager.installValidatedPendingProjectState(*snapshot);
        manager.autoScaleSteps(600);
        EXPECT_EQ(manager.getActiveParams().iterations, 5000u);
    }

    // Catches a disabled factor (<= 0) diverging from the previous auto-scale outcome.
    TEST(ParameterManagerTest, DisabledScalingUsesImageFactorRegardlessOfPreviousImageScaling) {
        for (const float disabled : {0.f, -1.f}) {
            lfs::vis::ParameterManager manager;
            lfs::core::param::TrainingParameters params;
            params.optimization = lfs::core::param::OptimizationParameters::mrnf_defaults();
            params.optimization.steps_scaler = disabled;
            params.optimization.image_count_scaler = 2.f;
            manager.setSessionDefaults(params);
            manager.autoScaleSteps(600);
            EXPECT_EQ(manager.getActiveParams().iterations, 60000u);
            EXPECT_FLOAT_EQ(manager.getActiveParams().steps_scaler, 2.f);
            EXPECT_FLOAT_EQ(manager.getActiveParams().image_count_scaler, 2.f);
        }
    }

    // Catches old projects being auto-scaled a second time on reopen.
    TEST(ParameterManagerTest, LegacyProjectImageFactorIsNotAppliedTwice) {
        lfs::vis::ParameterManager manager;
        ASSERT_TRUE(manager.ensureLoaded());
        auto captured = manager.capturePendingProjectState();
        ASSERT_TRUE(captured);
        captured->mrnf_current.scale_steps(2.f);
        captured->mrnf_current.steps_scaler = 2.f;
        lfs::io::project::ParametersChapter chapter;
        ASSERT_TRUE(chapter.set_snapshot(*captured));
        auto bytes = nlohmann::json::parse(chapter.dom().dump());
        for (auto& strategy : bytes["presets"].items()) {
            for (auto& role : strategy.value().items()) {
                role.value().erase("image_count_scaler");
                role.value().erase("image_count_scaler_total");
            }
        }
        const auto parsed = lfs::io::project::ParametersChapter::parse(bytes.dump());
        ASSERT_TRUE(parsed);
        const auto snapshot = parsed->snapshot();
        ASSERT_TRUE(snapshot);
        manager.installValidatedPendingProjectState(*snapshot);
        EXPECT_FLOAT_EQ(manager.getActiveParams().image_count_scaler, 2.f);
        const auto before = manager.getActiveParams().to_json();
        manager.autoScaleSteps(600);
        EXPECT_EQ(manager.getActiveParams().to_json(), before);
    }

    // Catches an absolute CLI step flag that auto-scale still rescales.
    TEST(ParameterManagerTest, EveryAbsoluteCliStepFlagLocksSeededStrategy) {
        const std::vector<std::vector<std::string>> cases{
            {"--sh-degree-interval", "1000"},
            {"--morton-reorder-interval", "3000"},
            {"--eval", "--eval-steps", "1000"},
        };
        for (const auto& flags : cases) {
            SCOPED_TRACE(nlohmann::json(flags).dump());
            for (const char* strategy : {"mrnf", "mcmc", "igs+"}) {
                SCOPED_TRACE(strategy);
                std::vector<const char*> argv{"LichtFeld-Studio", "--strategy", strategy, "--steps-scaler", "0.5"};
                for (const auto& flag : flags)
                    argv.push_back(flag.c_str());
                const auto parsed = lfs::io::args::parse_args_and_params(static_cast<int>(argv.size()), argv.data());
                ASSERT_TRUE(parsed) << parsed.error();
                ASSERT_TRUE((*parsed)->cli_step_values_set);
                lfs::vis::ParameterManager manager;
                manager.setSessionDefaults(**parsed);
                const auto before = manager.getActiveParams().to_json();
                manager.autoScaleSteps(600);
                EXPECT_EQ(manager.getActiveParams().to_json(), before);
                manager.setActiveStrategy(strategy == std::string_view("mrnf") ? "mcmc" : "mrnf");
                EXPECT_FLOAT_EQ(manager.getActiveParams().steps_scaler, 2.f);
            }
        }
    }

    // Catches config keys being mistaken for typed CLI values.
    TEST(ParameterManagerTest, ConfigStepKeysDoNotCountAsAbsoluteCliOverrides) {
        const auto path = unique_temp_config_path();
        auto json = lfs::core::param::OptimizationParameters::mrnf_defaults().to_json();
        json["steps_scaler"] = 0.5f;
        json["image_count_scaler_total"] = 0.5f;
        const auto path_text = path.string();
        for (const bool legacy : {false, true}) {
            SCOPED_TRACE(legacy);
            if (legacy) {
                json.erase("image_count_scaler");
                json.erase("image_count_scaler_total");
            }
            std::ofstream(path) << json.dump();
            const char* argv[]{"LichtFeld-Studio", "--config", path_text.c_str()};
            const auto parsed = lfs::io::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
            ASSERT_TRUE(parsed) << parsed.error();
            EXPECT_FALSE((*parsed)->cli_step_values_set);
            EXPECT_TRUE((*parsed)->overrides.has_optimization_key("iterations"));
            lfs::vis::ParameterManager manager;
            manager.setSessionDefaults(**parsed);
            manager.autoScaleSteps(600);
            EXPECT_EQ(manager.getActiveParams().iterations, legacy ? 60000u : 30000u);
            EXPECT_FLOAT_EQ(manager.getActiveParams().steps_scaler, legacy ? 2.f : 1.f);
        }
        const char* argv[]{"LichtFeld-Studio", "--config", path_text.c_str(), "--sh-degree-interval", "1000"};
        const auto parsed = lfs::io::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
        ASSERT_TRUE(parsed) << parsed.error();
        lfs::vis::ParameterManager manager;
        manager.setSessionDefaults(**parsed);
        const auto before = manager.getActiveParams().to_json();
        manager.autoScaleSteps(600);
        EXPECT_EQ(manager.getActiveParams().to_json(), before);
        std::filesystem::remove(path);
    }

    // Catches --steps-scaler keeping a config's image share that its timetable no longer contains.
    TEST(ParameterManagerTest, CliStepsScalerDropsConfigImageFactor) {
        const auto path = unique_temp_config_path();
        auto json = lfs::core::param::OptimizationParameters::mrnf_defaults().to_json();
        json["steps_scaler"] = 2.f;
        json["image_count_scaler"] = 2.f;
        json["image_count_scaler_total"] = 2.f;
        std::ofstream(path) << json.dump();
        const auto path_text = path.string();
        const char* argv[]{"LichtFeld-Studio", "--config", path_text.c_str(), "--steps-scaler", "0.5"};
        const auto parsed = lfs::io::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
        std::filesystem::remove(path);
        ASSERT_TRUE(parsed) << parsed.error();
        EXPECT_EQ((*parsed)->optimization.iterations, 15000u);
        EXPECT_FLOAT_EQ((*parsed)->optimization.image_count_scaler, 1.f);

        lfs::vis::ParameterManager manager;
        manager.setSessionDefaults(**parsed);
        manager.autoScaleSteps(600);
        EXPECT_EQ(manager.getActiveParams().iterations, 30000u);
        EXPECT_FLOAT_EQ(manager.getActiveParams().steps_scaler, 1.f);
    }

} // namespace
