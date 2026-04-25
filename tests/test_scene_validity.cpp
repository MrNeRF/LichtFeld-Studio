// SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <atomic>
#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <vector>

#include "core/parameters.hpp"
#include "core/point_cloud.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "python/python_runtime.hpp"
#include "training/training_setup.hpp"
#include "visualizer/scene/scene_manager.hpp"

namespace lfs::python {

    namespace {
        size_t sh_rest_coefficients_for_degree(const int degree) {
            if (degree <= 0) {
                return 0;
            }
            const auto d = static_cast<size_t>(degree);
            return (d + 1) * (d + 1) - 1;
        }

        std::unique_ptr<core::SplatData> make_test_splat(size_t count, const int sh_degree = 0) {
            std::vector<float> means(count * 3, 0.0f);
            std::vector<float> rotations(count * 4, 0.0f);
            for (size_t i = 0; i < count; ++i) {
                means[i * 3] = static_cast<float>(i);
                rotations[i * 4] = 1.0f;
            }

            return std::make_unique<core::SplatData>(
                sh_degree,
                core::Tensor::from_vector(means, {count, size_t{3}}, core::Device::CPU),
                core::Tensor::zeros({count, size_t{1}, size_t{3}}, core::Device::CPU, core::DataType::Float32),
                core::Tensor::zeros({count, sh_rest_coefficients_for_degree(sh_degree), size_t{3}}, core::Device::CPU, core::DataType::Float32),
                core::Tensor::zeros({count, size_t{3}}, core::Device::CPU, core::DataType::Float32),
                core::Tensor::from_vector(rotations, {count, size_t{4}}, core::Device::CPU),
                core::Tensor::zeros({count, size_t{1}}, core::Device::CPU, core::DataType::Float32),
                1.0f);
        }

        void expect_sh_degree(const core::SplatData& splat, const int sh_degree, const size_t count) {
            const size_t expected_rest_coeffs = sh_rest_coefficients_for_degree(sh_degree);

            EXPECT_EQ(splat.get_max_sh_degree(), sh_degree);
            EXPECT_EQ(splat.get_active_sh_degree(), sh_degree);
            ASSERT_TRUE(splat.shN().is_valid());
            ASSERT_EQ(splat.shN().ndim(), 3);
            EXPECT_EQ(splat.shN().shape()[0], count);
            EXPECT_EQ(splat.shN().shape()[1], expected_rest_coeffs);
            EXPECT_EQ(splat.shN().shape()[2], size_t{3});
            EXPECT_EQ(splat.get_shs().shape()[1], expected_rest_coeffs + 1);
        }
    } // namespace

    class SceneValidityTest : public ::testing::Test {
    protected:
        void SetUp() override {
            set_application_scene(nullptr);
        }

        void TearDown() override {
            set_application_scene(nullptr);
        }

        core::Scene dummy_scene_;
    };

    TEST_F(SceneValidityTest, GenerationNonNegative) {
        auto gen = get_scene_generation();
        EXPECT_GE(gen, 0u);
    }

    TEST_F(SceneValidityTest, GenerationIncrementsOnSet) {
        auto gen1 = get_scene_generation();
        set_application_scene(&dummy_scene_);
        auto gen2 = get_scene_generation();
        EXPECT_GT(gen2, gen1);
    }

    TEST_F(SceneValidityTest, GenerationIncrementsOnClear) {
        set_application_scene(&dummy_scene_);
        auto gen1 = get_scene_generation();
        set_application_scene(nullptr);
        auto gen2 = get_scene_generation();
        EXPECT_GT(gen2, gen1);
    }

    TEST_F(SceneValidityTest, GetApplicationSceneReturnsCorrectPointer) {
        EXPECT_EQ(get_application_scene(), nullptr);
        set_application_scene(&dummy_scene_);
        EXPECT_EQ(get_application_scene(), &dummy_scene_);
        set_application_scene(nullptr);
        EXPECT_EQ(get_application_scene(), nullptr);
    }

    TEST_F(SceneValidityTest, ConcurrentReadsAreSafe) {
        set_application_scene(&dummy_scene_);
        std::atomic<int> success_count{0};
        std::vector<std::thread> threads;

        for (int i = 0; i < 10; ++i) {
            threads.emplace_back([&]() {
                for (int j = 0; j < 1000; ++j) {
                    auto gen = get_scene_generation();
                    auto* scene = get_application_scene();
                    EXPECT_GE(gen, 0u);
                    EXPECT_EQ(scene, &dummy_scene_);
                }
                success_count++;
            });
        }

        for (auto& t : threads) {
            t.join();
        }

        EXPECT_EQ(success_count.load(), 10);
    }

    TEST_F(SceneValidityTest, GenerationIsMonotonic) {
        std::vector<uint64_t> generations;
        generations.push_back(get_scene_generation());

        for (int i = 0; i < 10; ++i) {
            set_application_scene(&dummy_scene_);
            generations.push_back(get_scene_generation());
            set_application_scene(nullptr);
            generations.push_back(get_scene_generation());
        }

        for (size_t i = 1; i < generations.size(); ++i) {
            EXPECT_GT(generations[i], generations[i - 1]);
        }
    }

    TEST_F(SceneValidityTest, MutationFlagsAccumulateUntilConsumed) {
        set_application_scene(&dummy_scene_);

        constexpr uint32_t node_added = 1u << 0;
        constexpr uint32_t transform_changed = 1u << 4;
        constexpr uint32_t combined = node_added | transform_changed;

        set_scene_mutation_flags(node_added);
        set_scene_mutation_flags(transform_changed);

        EXPECT_EQ(get_scene_mutation_flags(), combined);
        EXPECT_EQ(consume_scene_mutation_flags(), combined);
        EXPECT_EQ(get_scene_mutation_flags(), 0u);
        EXPECT_EQ(consume_scene_mutation_flags(), 0u);
    }

    TEST_F(SceneValidityTest, ClearResetsDatasetMetadata) {
        auto means = core::Tensor::from_vector({0.0f, 0.0f, 0.0f}, {size_t{1}, size_t{3}}, core::Device::CPU);
        auto colors = core::Tensor::from_vector({1.0f, 1.0f, 1.0f}, {size_t{1}, size_t{3}}, core::Device::CPU);

        dummy_scene_.setInitialPointCloud(std::make_shared<core::PointCloud>(std::move(means), std::move(colors)));
        dummy_scene_.setSceneCenter(core::Tensor::from_vector({1.0f, 2.0f, 3.0f}, {size_t{3}}, core::Device::CPU));
        dummy_scene_.setImagesHaveAlpha(true);
        dummy_scene_.setTrainingModelNode("Model");
        const auto dataset_id = dummy_scene_.addDataset("Dataset");
        const auto cameras_group_id = dummy_scene_.addGroup("Cameras", dataset_id);
        const auto train_group_id = dummy_scene_.addCameraGroup("Training (1)", cameras_group_id, 1);
        dummy_scene_.addCamera("cam_0001.png", train_group_id, std::make_shared<core::Camera>());

        ASSERT_TRUE(dummy_scene_.getInitialPointCloud());
        ASSERT_TRUE(dummy_scene_.getSceneCenter().is_valid());
        ASSERT_TRUE(dummy_scene_.imagesHaveAlpha());
        ASSERT_EQ(dummy_scene_.getTrainingModelNodeName(), "Model");
        ASSERT_EQ(dummy_scene_.getAllCameras().size(), 1u);
        ASSERT_GT(dummy_scene_.getNodeCount(), 0u);

        dummy_scene_.clear();

        EXPECT_FALSE(dummy_scene_.getInitialPointCloud());
        EXPECT_FALSE(dummy_scene_.getSceneCenter().is_valid());
        EXPECT_FALSE(dummy_scene_.imagesHaveAlpha());
        EXPECT_TRUE(dummy_scene_.getTrainingModelNodeName().empty());
        EXPECT_TRUE(dummy_scene_.getAllCameras().empty());
        EXPECT_EQ(dummy_scene_.getNodeCount(), 0u);
    }

    TEST_F(SceneValidityTest, TrainingModelActiveCountUsesSyncedTopologyCount) {
        dummy_scene_.addNode("Model", make_test_splat(2));
        dummy_scene_.setTrainingModelNode("Model");

        const auto model_id = dummy_scene_.getNodeIdByName("Model");
        ASSERT_NE(model_id, core::NULL_NODE);

        dummy_scene_.syncTrainingModelTopology(6);

        const auto counts = dummy_scene_.getActiveGaussianCountsByNode();
        const auto count_it = counts.find(model_id);
        ASSERT_NE(count_it, counts.end());
        EXPECT_EQ(count_it->second, 6u);
        EXPECT_EQ(dummy_scene_.getTrainingModelGaussianCount(), 6u);
    }

    TEST_F(SceneValidityTest, SplatDataSetSHDegreeSupportsAllDegrees) {
        constexpr size_t count = 4;

        for (const int target_degree : {0, 1, 2, 3}) {
            auto splat = make_test_splat(count, 3);

            splat->set_sh_degree(target_degree);

            expect_sh_degree(*splat, target_degree, count);
        }
    }

    TEST_F(SceneValidityTest, SplatDataSetSHDegreeCanExpandMissingCoefficients) {
        constexpr size_t count = 4;
        auto splat = make_test_splat(count, 0);

        EXPECT_TRUE(splat->set_sh_degree(2));

        expect_sh_degree(*splat, 2, count);
    }

    TEST_F(SceneValidityTest, InitializeTrainingModelAdjustsExistingTrainingModelSHDegree) {
        constexpr size_t count = 4;
        dummy_scene_.addNode("Model", make_test_splat(count, 3));
        dummy_scene_.setTrainingModelNode("Model");

        core::param::TrainingParameters params;
        params.optimization.sh_degree = 1;

        const auto result = lfs::training::initializeTrainingModel(params, dummy_scene_);

        ASSERT_TRUE(result.has_value()) << result.error();
        const auto* model = dummy_scene_.getTrainingModel();
        ASSERT_NE(model, nullptr);
        expect_sh_degree(*model, 1, count);
    }

    TEST_F(SceneValidityTest, SceneManagerEmptyStateKeepsApplicationSceneContext) {
        lfs::vis::SceneManager scene_manager;
        EXPECT_EQ(get_application_scene(), &scene_manager.getScene());

        scene_manager.addGroupNode("Bootstrap");
        ASSERT_GT(scene_manager.getScene().getNodeCount(), 0u);

        ASSERT_TRUE(scene_manager.clear());

        EXPECT_EQ(get_application_scene(), &scene_manager.getScene());
        EXPECT_EQ(scene_manager.getContentType(), lfs::vis::SceneManager::ContentType::Empty);
        EXPECT_EQ(scene_manager.getScene().getNodeCount(), 0u);
    }

} // namespace lfs::python
