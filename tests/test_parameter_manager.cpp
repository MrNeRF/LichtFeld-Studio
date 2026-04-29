/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "core/parameter_manager.hpp"
#include "core/parameters.hpp"

namespace {

    TEST(ParameterManagerTest, DefaultStrategyIsMrnf) {
        lfs::vis::ParameterManager manager;
        const auto load_result = manager.ensureLoaded();
        ASSERT_TRUE(load_result.has_value()) << load_result.error();

        EXPECT_EQ(manager.getActiveStrategy(), "mrnf");
        EXPECT_EQ(manager.getActiveParams().strategy, "mrnf");
        EXPECT_EQ(lfs::core::param::OptimizationParameters{}.strategy, "mrnf");
        EXPECT_EQ(lfs::core::param::OptimizationParameters::mcmc_defaults().strategy, "mcmc");
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
        checkpoint_params.dataset.loading_params.use_fs_cache = false;
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
        EXPECT_FALSE(dataset.loading_params.use_fs_cache);
        EXPECT_TRUE(dataset.invert_masks);
        EXPECT_FLOAT_EQ(dataset.mask_threshold, 0.75f);

        const auto recreated = manager.createForDataset("/tmp/override_dataset", "/tmp/override_output");
        EXPECT_EQ(recreated.optimization.strategy, "igs+");
        EXPECT_EQ(recreated.optimization.iterations, 600u);
        EXPECT_EQ(recreated.dataset.data_path, "/tmp/override_dataset");
        EXPECT_EQ(recreated.dataset.output_path, "/tmp/override_output");
        EXPECT_EQ(recreated.dataset.images, "images_4");
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
        EXPECT_EQ(dataset.max_width, 0);
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

} // namespace
