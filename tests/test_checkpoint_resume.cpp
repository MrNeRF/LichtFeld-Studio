/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <utility>
#include <vector>

#include "core/camera.hpp"
#include "core/checkpoint_format.hpp"
#include "core/logger.hpp"
#include "core/parameters.hpp"
#include "core/path_utils.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "io/loader.hpp"
#include "training/checkpoint.hpp"
#include "training/strategies/mcmc.hpp"
#include "training/trainer.hpp"
#include "training/training_setup.hpp"

namespace {

    constexpr const char* TEST_IMAGES = "images_4";
    constexpr int CHECKPOINT_ITER = 1200;
    constexpr int TOTAL_ITER = 2100;

    std::unique_ptr<lfs::core::SplatData> make_checkpoint_test_splat(const size_t count) {
        std::vector<float> means(count * 3, 0.0f);
        std::vector<float> rotations(count * 4, 0.0f);
        for (size_t i = 0; i < count; ++i) {
            means[i * 3] = static_cast<float>(i);
            rotations[i * 4] = 1.0f;
        }

        return std::make_unique<lfs::core::SplatData>(
            0,
            lfs::core::Tensor::from_vector(means, {count, size_t{3}}, lfs::core::Device::CPU),
            lfs::core::Tensor::zeros({count, size_t{1}, size_t{3}}, lfs::core::Device::CPU, lfs::core::DataType::Float32),
            lfs::core::Tensor::zeros({size_t{0}}, lfs::core::Device::CPU, lfs::core::DataType::Float32),
            lfs::core::Tensor::zeros({count, size_t{3}}, lfs::core::Device::CPU, lfs::core::DataType::Float32),
            lfs::core::Tensor::from_vector(rotations, {count, size_t{4}}, lfs::core::Device::CPU),
            lfs::core::Tensor::zeros({count, size_t{1}}, lfs::core::Device::CPU, lfs::core::DataType::Float32),
            1.0f);
    }

    std::streamoff first_model_tensor_header_offset(const std::filesystem::path& checkpoint) {
        std::ifstream file(checkpoint, std::ios::binary);
        if (!file)
            return -1;
        file.seekg(static_cast<std::streamoff>(sizeof(lfs::core::CheckpointHeader)));
        uint32_t strategy_name_size = 0;
        file.read(reinterpret_cast<char*>(&strategy_name_size), sizeof(strategy_name_size));
        file.seekg(static_cast<std::streamoff>(strategy_name_size), std::ios::cur);
        constexpr std::streamoff splat_prefix_bytes =
            sizeof(uint32_t) * 2 + sizeof(int32_t) * 2 + sizeof(float);
        file.seekg(splat_prefix_bytes, std::ios::cur);
        return file ? static_cast<std::streamoff>(file.tellg()) : -1;
    }

    template <typename T>
    bool overwrite_checkpoint_field(const std::filesystem::path& checkpoint,
                                    const std::streamoff offset,
                                    const T& value) {
        std::fstream file(checkpoint, std::ios::binary | std::ios::in | std::ios::out);
        if (!file)
            return false;
        file.seekp(offset);
        file.write(reinterpret_cast<const char*>(&value), sizeof(value));
        file.flush();
        return file.good();
    }

    TEST(TrainingSetupRegressionTest, ApplyLoadedDatasetKeepsFullInitPointCloudUntilTrainingStarts) {
        constexpr size_t initial_points = 12;
        constexpr int target_splats = 5;

        const auto temp_dir = std::filesystem::temp_directory_path() / "lfs_training_setup_full_init_regression";
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
        std::filesystem::create_directories(temp_dir);

        const auto init_path = temp_dir / "init_points.ply";
        {
            std::ofstream ply(init_path);
            ASSERT_TRUE(ply.is_open());
            ply << "ply\n"
                   "format ascii 1.0\n"
                   "element vertex "
                << initial_points
                << "\n"
                   "property float x\n"
                   "property float y\n"
                   "property float z\n"
                   "property uchar red\n"
                   "property uchar green\n"
                   "property uchar blue\n"
                   "end_header\n";
            for (size_t i = 0; i < initial_points; ++i) {
                ply << static_cast<float>(i) << ' '
                    << static_cast<float>(i % 3) << ' '
                    << static_cast<float>(i % 5) << ' '
                    << static_cast<int>(10 + i) << ' '
                    << static_cast<int>(20 + i) << ' '
                    << static_cast<int>(30 + i) << '\n';
            }
        }

        lfs::core::param::TrainingParameters params;
        params.dataset.data_path = temp_dir / "dataset";
        params.init_path = lfs::core::path_to_utf8(init_path);
        params.optimization.max_cap = target_splats;

        lfs::io::LoadedScene loaded_scene;
        loaded_scene.cameras.push_back(std::make_shared<lfs::core::Camera>());

        lfs::io::LoadResult load_result;
        load_result.data = std::move(loaded_scene);
        load_result.scene_center = lfs::core::Tensor::from_vector(
            std::vector<float>{0.0f, 0.0f, 0.0f},
            {size_t{3}},
            lfs::core::Device::CPU);
        load_result.loader_used = "test";

        lfs::core::Scene scene;
        auto apply_result = lfs::training::applyLoadResultToScene(params, scene, std::move(load_result));
        ASSERT_TRUE(apply_result.has_value()) << apply_result.error();

        const auto* model = scene.getTrainingModel();
        ASSERT_NE(model, nullptr);
        EXPECT_EQ(static_cast<size_t>(model->size()), initial_points);
        EXPECT_EQ(scene.getTrainingModelGaussianCount(), initial_points);

        auto trainer = std::make_unique<lfs::training::Trainer>(scene);
        auto init_result = trainer->initialize(params);
        ASSERT_TRUE(init_result.has_value()) << init_result.error();

        EXPECT_EQ(static_cast<size_t>(trainer->get_strategy().get_model().size()), static_cast<size_t>(target_splats));
        EXPECT_EQ(scene.getTrainingModelGaussianCount(), static_cast<size_t>(target_splats));

        trainer->shutdown();
        std::filesystem::remove_all(temp_dir, ec);
    }

    TEST(CheckpointAllocatorRegressionTest, LoadCheckpointUsesAllocatorWithMaxCapacity) {
        constexpr size_t count = 4;
        constexpr size_t max_cap = 16;

        const auto temp_dir = std::filesystem::temp_directory_path() / "lfs_checkpoint_allocator_regression";
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
        std::filesystem::create_directories(temp_dir / "checkpoints");

        lfs::core::param::TrainingParameters params;
        params.dataset.output_path = temp_dir;
        params.optimization.strategy = "mcmc";
        params.optimization.max_cap = max_cap;

        auto source_model = make_checkpoint_test_splat(count);
        lfs::training::MCMC source_strategy(*source_model);
        auto save_result = lfs::training::save_checkpoint(temp_dir, 7, source_strategy, params);
        ASSERT_TRUE(save_result.has_value()) << save_result.error();

        struct AllocationCall {
            std::string name;
            size_t capacity = 0;
        };
        std::vector<AllocationCall> calls;
        lfs::core::SplatTensorAllocator allocator =
            [&calls](lfs::core::TensorShape shape,
                     const size_t capacity,
                     const lfs::core::DataType dtype,
                     const std::string_view name) -> lfs::core::Tensor {
            calls.push_back({std::string{name}, capacity});
            EXPECT_EQ(dtype, lfs::core::DataType::Float32);
            auto tensor = lfs::core::Tensor::zeros_direct(std::move(shape), capacity, lfs::core::Device::CUDA);
            tensor.set_name(std::string{name});
            return tensor;
        };

        auto target_model = make_checkpoint_test_splat(1);
        lfs::training::MCMC target_strategy(*target_model);
        auto load_params = params;
        const auto checkpoint_path = lfs::training::checkpoint_output_path(temp_dir);
        auto load_result = lfs::training::load_checkpoint(
            checkpoint_path, target_strategy, load_params, nullptr, nullptr, nullptr, allocator);
        ASSERT_TRUE(load_result.has_value()) << load_result.error();

        EXPECT_EQ(*load_result, 7);
        EXPECT_EQ(static_cast<size_t>(target_strategy.get_model().size()), count);
        EXPECT_GE(target_strategy.get_model().means_raw().capacity(), max_cap);
        EXPECT_EQ(calls.size(), 5u);
        for (const auto& call : calls) {
            EXPECT_GE(call.capacity, max_cap) << call.name;
        }

        std::filesystem::remove_all(temp_dir, ec);
    }

    TEST(CheckpointInputValidationTest, RejectsInvalidTensorDtypeAndPreservesLiveModel) {
        const auto temp_dir = std::filesystem::temp_directory_path() / "lfs_checkpoint_invalid_tensor_dtype";
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
        std::filesystem::create_directories(temp_dir / "checkpoints");

        lfs::core::param::TrainingParameters params;
        params.dataset.output_path = temp_dir;
        params.optimization.strategy = "mcmc";
        params.optimization.max_cap = 16;

        auto source_model = make_checkpoint_test_splat(4);
        lfs::training::MCMC source_strategy(*source_model);
        ASSERT_TRUE(lfs::training::save_checkpoint(temp_dir, 7, source_strategy, params).has_value());

        const auto checkpoint = lfs::training::checkpoint_output_path(temp_dir);
        const auto tensor_offset = first_model_tensor_header_offset(checkpoint);
        ASSERT_GE(tensor_offset, 0);
        constexpr uint8_t invalid_dtype = 0xff;
        ASSERT_TRUE(overwrite_checkpoint_field(
            checkpoint,
            tensor_offset + static_cast<std::streamoff>(offsetof(lfs::core::TensorFileHeader, dtype)),
            invalid_dtype));

        auto target_model = make_checkpoint_test_splat(2);
        lfs::training::MCMC target_strategy(*target_model);
        auto loaded_params = params;
        const auto result = lfs::training::load_checkpoint(
            checkpoint, target_strategy, loaded_params, nullptr, nullptr, nullptr);

        ASSERT_FALSE(result.has_value());
        EXPECT_NE(result.error().find("unsupported dtype"), std::string::npos);
        ASSERT_EQ(target_strategy.get_model().size(), 2);
        const auto means = target_strategy.get_model().means().cpu().to_vector();
        ASSERT_EQ(means.size(), 6u);
        EXPECT_FLOAT_EQ(means[3], 1.0f);

        std::filesystem::remove_all(temp_dir, ec);
    }

    TEST(CheckpointInputValidationTest, RejectsLateStrategyCorruptionWithoutPartialCommit) {
        const auto temp_dir = std::filesystem::temp_directory_path() / "lfs_checkpoint_late_corruption";
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
        std::filesystem::create_directories(temp_dir / "checkpoints");

        lfs::core::param::TrainingParameters params;
        params.dataset.output_path = temp_dir;
        params.optimization.strategy = "mcmc";
        params.optimization.max_cap = 16;

        auto source_model = make_checkpoint_test_splat(4);
        lfs::training::MCMC source_strategy(*source_model);
        source_strategy.initialize(params.optimization);
        ASSERT_TRUE(lfs::training::save_checkpoint(temp_dir, 9, source_strategy, params).has_value());

        const auto checkpoint = lfs::training::checkpoint_output_path(temp_dir);
        const auto header = lfs::core::load_checkpoint_header(checkpoint);
        ASSERT_TRUE(header.has_value()) << header.error();
        ASSERT_GT(header->params_json_offset, 0u);
        constexpr uint8_t invalid_scheduler_parameter = 0xff;
        ASSERT_TRUE(overwrite_checkpoint_field(
            checkpoint,
            static_cast<std::streamoff>(header->params_json_offset - 1),
            invalid_scheduler_parameter));

        auto target_model = make_checkpoint_test_splat(2);
        lfs::training::MCMC target_strategy(*target_model);
        target_strategy.initialize(params.optimization);
        target_strategy.get_optimizer().set_lr(0.123f);
        auto loaded_params = params;

        const auto result = lfs::training::load_checkpoint(
            checkpoint, target_strategy, loaded_params, nullptr, nullptr, nullptr);

        ASSERT_FALSE(result.has_value());
        EXPECT_NE(result.error().find("invalid parameter id"), std::string::npos);
        EXPECT_EQ(target_strategy.get_model().size(), 2);
        EXPECT_FLOAT_EQ(target_strategy.get_optimizer().get_lr(), 0.123f);
        EXPECT_EQ(loaded_params.optimization.max_cap, params.optimization.max_cap);

        std::filesystem::remove_all(temp_dir, ec);
    }

    TEST(CheckpointInputValidationTest, RejectsJsonRangeOutsideFileBeforeStateMutation) {
        const auto temp_dir = std::filesystem::temp_directory_path() / "lfs_checkpoint_invalid_json_range";
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
        std::filesystem::create_directories(temp_dir / "checkpoints");

        lfs::core::param::TrainingParameters params;
        params.dataset.output_path = temp_dir;
        params.optimization.strategy = "mcmc";
        auto source_model = make_checkpoint_test_splat(2);
        lfs::training::MCMC source_strategy(*source_model);
        ASSERT_TRUE(lfs::training::save_checkpoint(temp_dir, 3, source_strategy, params).has_value());

        const auto checkpoint = lfs::training::checkpoint_output_path(temp_dir);
        constexpr uint64_t oversized_json = lfs::core::MAX_CHECKPOINT_JSON_BYTES + 1;
        ASSERT_TRUE(overwrite_checkpoint_field(
            checkpoint,
            static_cast<std::streamoff>(offsetof(lfs::core::CheckpointHeader, params_json_size)),
            oversized_json));

        const auto header = lfs::core::load_checkpoint_header(checkpoint);
        ASSERT_FALSE(header.has_value());
        EXPECT_NE(header.error().find("JSON exceeds byte budget"), std::string::npos);

        std::filesystem::remove_all(temp_dir, ec);
    }

    class CheckpointResumeTest : public ::testing::TestWithParam<std::tuple<std::string, int>> {
    protected:
        void SetUp() override {
            auto [strategy, sh_degree] = GetParam();
            strategy_ = strategy;
            sh_degree_ = sh_degree;

            // Create unique output directory for this test
            output_path_ = std::filesystem::temp_directory_path() /
                           std::format("lfs_test_checkpoint_{}_{}", strategy_, sh_degree_);
            std::filesystem::create_directories(output_path_);
            std::filesystem::create_directories(output_path_ / "checkpoints");
        }

        void TearDown() override {
            std::error_code ec;
            std::filesystem::remove_all(output_path_, ec);
        }

        lfs::core::param::TrainingParameters createParams(int iterations) {
            lfs::core::param::TrainingParameters params;
            params.dataset.data_path = std::filesystem::path(TEST_DATA_DIR) / "bicycle";
            params.dataset.images = TEST_IMAGES;
            params.dataset.output_path = output_path_;
            params.optimization.iterations = iterations;
            params.optimization.strategy = strategy_;
            params.optimization.sh_degree = sh_degree_;
            params.optimization.headless = true;
            params.optimization.max_cap = 100000;
            params.optimization.refine_every = 100;
            params.optimization.start_refine = 500;
            params.optimization.stop_refine = iterations;
            return params;
        }

        std::string strategy_;
        int sh_degree_;
        std::filesystem::path output_path_;
    };

    TEST_P(CheckpointResumeTest, TrainSaveLoadResume) {
        auto [strategy, sh_degree] = GetParam();
        LOG_INFO("Testing checkpoint resume: strategy={}, sh_degree={}", strategy, sh_degree);
        const bool fixed_horizon_resume = strategy == "igs+";
        const int phase_one_iterations = fixed_horizon_resume ? TOTAL_ITER : CHECKPOINT_ITER + 1;
        // Phase 1 always leaves the rotating checkpoint at the completed iteration because the
        // final save path writes a .resume alongside the final PLY.
        const int checkpoint_iteration = phase_one_iterations;

        // Phase 1: Write multiple checkpoints and verify the latest save is the only one retained.
        {
            auto params = createParams(phase_one_iterations);
            params.optimization.save_steps = fixed_horizon_resume
                                                 ? std::vector<size_t>{static_cast<size_t>(CHECKPOINT_ITER)}
                                                 : std::vector<size_t>{static_cast<size_t>(CHECKPOINT_ITER / 2),
                                                                       static_cast<size_t>(CHECKPOINT_ITER)};
            lfs::core::Scene scene;

            auto load_result = lfs::training::loadTrainingDataIntoScene(params, scene);
            ASSERT_TRUE(load_result.has_value()) << "Failed to load training data: " << load_result.error();

            auto model_result = lfs::training::initializeTrainingModel(params, scene);
            ASSERT_TRUE(model_result.has_value()) << "Failed to init model: " << model_result.error();

            auto trainer = std::make_unique<lfs::training::Trainer>(scene);
            auto init_result = trainer->initialize(params);
            ASSERT_TRUE(init_result.has_value()) << "Failed to init trainer: " << init_result.error();

            auto train_result = trainer->train();
            ASSERT_TRUE(train_result.has_value()) << "Training failed: " << train_result.error();

            EXPECT_EQ(trainer->get_current_iteration(), phase_one_iterations);

            trainer->shutdown();
        }

        // Verify the rotating checkpoint exists and is the only checkpoint file.
        auto checkpoint_path = lfs::training::checkpoint_output_path(output_path_);
        ASSERT_TRUE(std::filesystem::exists(checkpoint_path))
            << "Checkpoint file not found: " << checkpoint_path;
        EXPECT_EQ(checkpoint_path.filename(), "checkpoint.resume");

        size_t resume_file_count = 0;
        for (const auto& entry : std::filesystem::directory_iterator(output_path_ / "checkpoints")) {
            if (entry.path().extension() == ".resume") {
                ++resume_file_count;
            }
            EXPECT_EQ(entry.path().filename(), checkpoint_path.filename())
                << "Unexpected stale checkpoint file left behind: " << entry.path();
        }
        EXPECT_EQ(resume_file_count, 1u);

        // Phase 2: Load checkpoint and resume to final iteration
        {
            auto checkpoint_params_result = lfs::core::load_checkpoint_params(checkpoint_path);
            ASSERT_TRUE(checkpoint_params_result.has_value())
                << "Failed to load checkpoint params: " << checkpoint_params_result.error();

            auto params = std::move(*checkpoint_params_result);
            params.resume_checkpoint = checkpoint_path;
            params.dataset.data_path = std::filesystem::path(TEST_DATA_DIR) / "bicycle";
            params.dataset.output_path = output_path_;
            auto resumed_params = params;
            if (!fixed_horizon_resume) {
                resumed_params.optimization.iterations = TOTAL_ITER;
                resumed_params.optimization.stop_refine = TOTAL_ITER;
            }

            lfs::core::Scene scene;

            auto load_result = lfs::training::loadTrainingDataIntoScene(params, scene);
            ASSERT_TRUE(load_result.has_value()) << "Failed to load training data: " << load_result.error();

            auto model_result = lfs::training::initializeTrainingModel(params, scene);
            ASSERT_TRUE(model_result.has_value()) << "Failed to init model: " << model_result.error();

            auto trainer = std::make_unique<lfs::training::Trainer>(scene);
            auto init_result = trainer->initialize(params);
            ASSERT_TRUE(init_result.has_value()) << "Failed to init trainer: " << init_result.error();
            if (!fixed_horizon_resume) {
                trainer->get_strategy_mutable().set_optimization_params(resumed_params.optimization);
                trainer->setParams(resumed_params);
            }

            // After loading checkpoint, iteration should be at checkpoint point
            EXPECT_EQ(trainer->get_current_iteration(), checkpoint_iteration);
            EXPECT_EQ(trainer->getParams().optimization.iterations, static_cast<size_t>(TOTAL_ITER));
            EXPECT_EQ(trainer->getParams().optimization.refine_every, static_cast<size_t>(100));
            EXPECT_EQ(trainer->getParams().optimization.stop_refine, static_cast<size_t>(TOTAL_ITER));
            EXPECT_TRUE(trainer->getParams().optimization.headless);

            auto train_result = trainer->train();
            ASSERT_TRUE(train_result.has_value()) << "Resume training failed: " << train_result.error();

            EXPECT_EQ(trainer->get_current_iteration(), TOTAL_ITER);

            trainer->shutdown();
        }

        LOG_INFO("Checkpoint resume test passed: strategy={}, sh_degree={}", strategy, sh_degree);
    }

    std::string TestName(const ::testing::TestParamInfo<CheckpointResumeTest::ParamType>& info) {
        auto name = std::format("{}_{}", std::get<0>(info.param), std::get<1>(info.param));
        std::replace_if(
            name.begin(), name.end(), [](const unsigned char c) { return !std::isalnum(c); }, '_');
        return name;
    }

    INSTANTIATE_TEST_SUITE_P(
        CheckpointStrategies,
        CheckpointResumeTest,
        ::testing::Values(
            std::make_tuple("mcmc", 0),
            std::make_tuple("mcmc", 1),
            std::make_tuple("mcmc", 2),
            std::make_tuple("mcmc", 3),
            std::make_tuple("mrnf", 3),
            std::make_tuple("igs+", 3)),
        TestName);

} // namespace
