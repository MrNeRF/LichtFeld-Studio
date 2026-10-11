/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/image_io.hpp"
#include "core/parameters.hpp"
#include "core/scene.hpp"
#include "io/loader.hpp"
#include "training/trainer.hpp"
#include "training/training_setup.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

namespace {
    using Cameras = std::vector<std::shared_ptr<lfs::core::Camera>>;

    std::vector<std::string> eval_names(const Cameras& cameras) {
        std::vector<std::string> names;
        for (const auto& camera : cameras) {
            if (camera->split() == lfs::core::CameraSplit::Eval)
                names.push_back(camera->image_name());
        }
        std::ranges::sort(names);
        return names;
    }

    // Image records are written out of name order: a holdout taken by record or uid index selects frame_1/3/5
    // instead of frame_0/2/4.
    class EvaluationSplitTest : public ::testing::Test {
    protected:
        void SetUp() override {
            params.dataset.data_path = std::filesystem::temp_directory_path() / "lfs_eval_split_test";
            std::filesystem::create_directories(params.dataset.data_path / "images");
            std::ofstream cameras(params.dataset.data_path / "cameras.txt");
            cameras << "1 PINHOLE 8 6 8 8 4 3\n";
            cameras.close();
            ASSERT_TRUE(cameras.good());

            const std::vector<unsigned char> pixels(8 * 6 * 3, 128);
            std::ofstream images(params.dataset.data_path / "images.txt");
            int id = 1;
            for (const int index : {3, 0, 5, 2, 1, 4}) {
                const auto name = "frame_" + std::to_string(index) + ".png";
                ASSERT_TRUE(lfs::core::save_png(params.dataset.data_path / "images" / name,
                                                pixels.data(), 8, 6, 3, 8, 6));
                images << id << " 1 0 0 0 " << id << " 0 0 1 " << name << "\n\n";
                ++id;
            }
            images.close();
            ASSERT_TRUE(images.good());
            std::ofstream points(params.dataset.data_path / "points3D.txt");
            for (int i = 0; i < 8; ++i)
                points << i + 1 << ' ' << (i & 1) << ' ' << ((i >> 1) & 1) << ' ' << ((i >> 2) & 1)
                       << " 128 128 128 0\n";
            points.close();
            ASSERT_TRUE(points.good());

            params.dataset.test_every = 2;
            params.dataset.output_path = params.dataset.data_path / "output";
            params.optimization.enable_eval = true;
            params.optimization.headless = true;
            params.optimization.max_cap = 64;
            params.no_download = true;
            auto loaded = lfs::io::Loader::create()->load(params.dataset.data_path);
            ASSERT_TRUE(loaded) << loaded.error().format();
            load_result = std::move(*loaded);
        }

        void TearDown() override {
            std::error_code ec;
            std::filesystem::remove_all(params.dataset.data_path, ec);
        }

        lfs::core::param::TrainingParameters params;
        lfs::io::LoadResult load_result;
        const std::vector<std::string> sorted_names{"frame_0.png", "frame_2.png", "frame_4.png"};
    };

    TEST_F(EvaluationSplitTest, SceneLoadPathsHoldOutInImageNameOrder) {
        lfs::core::Scene direct;
        const auto loaded = lfs::training::loadTrainingDataIntoScene(params, direct);
        ASSERT_TRUE(loaded) << loaded.error();
        EXPECT_EQ(eval_names(direct.getAllCameras()), sorted_names);

        lfs::core::Scene applied;
        const auto installed = lfs::training::applyLoadResultToScene(params, applied, std::move(load_result));
        ASSERT_TRUE(installed) << installed.error();
        EXPECT_EQ(eval_names(applied.getAllCameras()), sorted_names);
    }

    TEST_F(EvaluationSplitTest, ExplicitEvalOverrideHoldsOutInImageNameOrder) {
        params.optimization.enable_eval = false;
        lfs::core::Scene scene;
        ASSERT_TRUE(lfs::training::applyLoadResultToScene(params, scene, std::move(load_result)));
        params.optimization.enable_eval = true;
        params.overrides.optimization_json = R"({"enable_eval":true})";
        const auto model = lfs::training::initializeTrainingModel(params, scene);
        ASSERT_TRUE(model) << model.error();
        lfs::training::Trainer trainer(scene);
        const auto initialized = trainer.initialize(params);
        ASSERT_TRUE(initialized) << initialized.error();
        trainer.shutdown();
        EXPECT_EQ(eval_names(scene.getAllCameras()), sorted_names);
    }

} // namespace
