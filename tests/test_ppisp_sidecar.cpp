/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "components/ppisp.hpp"
#include "components/ppisp_controller_pool.hpp"
#include "components/ppisp_file.hpp"
#include "core/parameters.hpp"
#include "core/tensor.hpp"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <gtest/gtest.h>
#include <string_view>

namespace {

    using lfs::core::Device;
    using lfs::core::Tensor;
    using lfs::training::PPISP;
    using lfs::training::PPISPControllerPool;
    using lfs::training::PPISPConfig;
    using lfs::training::PPISPFileMetadata;

    Tensor make_input(float base) {
        std::vector<float> values(3 * 4 * 4);
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = base + 0.01f * static_cast<float>(i);
        }
        return Tensor::from_vector(values, {3, 4, 4}, Device::CUDA);
    }

    Tensor make_grad(float value) {
        std::vector<float> values(3 * 4 * 4, value);
        return Tensor::from_vector(values, {3, 4, 4}, Device::CUDA);
    }

    Tensor make_controller_input(float base) {
        std::vector<float> values(3 * 48 * 48);
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = base + 0.001f * static_cast<float>(i);
        }
        return Tensor::from_vector(values, {1, 3, 48, 48}, Device::CUDA);
    }

    Tensor make_controller_grad(float value) {
        std::vector<float> values(9, value);
        return Tensor::from_vector(values, {1, 9}, Device::CUDA);
    }

    void update_ppisp(PPISP& ppisp, int camera_id, int uid, float input_base, float grad_value, int steps) {
        for (int i = 0; i < steps; ++i) {
            const auto input = make_input(input_base + static_cast<float>(i) * 0.05f);
            const auto grad = make_grad(grad_value + static_cast<float>(i) * 0.01f);
            (void)ppisp.apply(input, camera_id, uid);
            (void)ppisp.backward(input, grad, camera_id, uid);
            ppisp.optimizer_step();
            ppisp.zero_grad();
            ppisp.scheduler_step();
        }
    }

    void update_controller(PPISPControllerPool& controller_pool, int camera_idx, float input_base,
                           float grad_value, int steps) {
        controller_pool.allocate_buffers(48, 48);
        for (int i = 0; i < steps; ++i) {
            const auto input = make_controller_input(input_base + static_cast<float>(i) * 0.03f);
            const auto grad = make_controller_grad(grad_value + static_cast<float>(i) * 0.02f);
            (void)controller_pool.predict(camera_idx, input, 1.0f);
            controller_pool.backward(camera_idx, grad);
            controller_pool.optimizer_step(camera_idx);
            controller_pool.zero_grad();
            controller_pool.scheduler_step(camera_idx);
        }
    }

    bool vectors_close(const std::vector<float>& expected, const std::vector<float>& actual, float tol = 1e-5f) {
        if (expected.size() != actual.size()) {
            return false;
        }
        for (size_t i = 0; i < expected.size(); ++i) {
            if (std::abs(expected[i] - actual[i]) > tol) {
                return false;
            }
        }
        return true;
    }

    class PPISPSidecarTest : public ::testing::Test {
    protected:
        std::filesystem::path make_temp_path(std::string_view stem) {
            const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
            const auto path = std::filesystem::temp_directory_path() /
                              std::format("{}_{}.ppisp", stem, now);
            temp_paths_.push_back(path);
            return path;
        }

        void TearDown() override {
            std::error_code ec;
            for (const auto& path : temp_paths_) {
                std::filesystem::remove(path, ec);
            }
        }

    private:
        std::vector<std::filesystem::path> temp_paths_;
    };

    TEST_F(PPISPSidecarTest, SaveLoadWithMetadataSupportsRemappedImport) {
        PPISPConfig config;
        config.lr = 1e-2;
        config.warmup_steps = 1;

        PPISP source(128, config);
        source.register_frame(101, 20);
        source.register_frame(100, 10);
        source.register_frame(102, 20);
        source.finalize();

        update_ppisp(source, 20, 101, 0.10f, 0.03f, 2);
        update_ppisp(source, 10, 100, 0.25f, 0.05f, 1);
        update_ppisp(source, 20, 102, 0.40f, 0.07f, 3);
        EXPECT_GT(source.get_step(), 0);

        PPISPControllerPool source_controller(2, 64);
        update_controller(source_controller, 0, 0.10f, 0.03f, 2);
        update_controller(source_controller, 1, 0.25f, 0.05f, 3);

        PPISPFileMetadata metadata;
        metadata.dataset_path_utf8 = "/tmp/source_dataset";
        metadata.images_folder = "images";
        metadata.frame_image_names = {"frame_b.png", "frame_a.png", "frame_c.png"};
        metadata.frame_camera_ids = {20, 10, 20};
        metadata.camera_ids = {20, 10};

        const auto path = make_temp_path("ppisp_sidecar_metadata");
        auto save_result = lfs::training::save_ppisp_file(path, source, &source_controller, &metadata);
        ASSERT_TRUE(save_result) << save_result.error();

        PPISP loaded(1);
        PPISPControllerPool loaded_controller(2, 1);
        loaded_controller.allocate_buffers(48, 48);
        PPISPFileMetadata loaded_metadata;
        auto load_result = lfs::training::load_ppisp_file(path, loaded, &loaded_controller, &loaded_metadata);
        ASSERT_TRUE(load_result) << load_result.error();

        EXPECT_EQ(loaded_metadata.dataset_path_utf8, metadata.dataset_path_utf8);
        EXPECT_EQ(loaded_metadata.images_folder, metadata.images_folder);
        EXPECT_EQ(loaded_metadata.frame_image_names, metadata.frame_image_names);
        EXPECT_EQ(loaded_metadata.frame_camera_ids, metadata.frame_camera_ids);
        EXPECT_EQ(loaded_metadata.camera_ids, metadata.camera_ids);

        PPISP target(128, config);
        target.register_frame(201, 10); // frame_a.png
        target.register_frame(202, 20); // frame_c.png
        target.register_frame(203, 20); // frame_b.png
        target.finalize();

        const std::vector<int> frame_mapping = {1, 2, 0};
        const std::vector<int> camera_mapping = {1, 0};
        auto import_result = target.copy_inference_weights_from(loaded, frame_mapping, camera_mapping);
        ASSERT_TRUE(import_result) << import_result.error();

        PPISPControllerPool target_controller(2, 64);
        target_controller.allocate_buffers(48, 48);
        const auto controller_import_error = target_controller.copy_inference_weights_from(loaded_controller, camera_mapping);
        ASSERT_TRUE(controller_import_error.empty()) << controller_import_error;

        EXPECT_EQ(target.get_step(), 0);
        EXPECT_DOUBLE_EQ(target.get_lr(), target.get_config().lr);

        struct ComparisonCase {
            int target_camera_id;
            int target_uid;
            int source_camera_id;
            int source_uid;
            float input_base;
        };

        const std::vector<ComparisonCase> cases = {
            {.target_camera_id = 10, .target_uid = 201, .source_camera_id = 10, .source_uid = 100, .input_base = 0.15f},
            {.target_camera_id = 20, .target_uid = 202, .source_camera_id = 20, .source_uid = 102, .input_base = 0.35f},
            {.target_camera_id = 20, .target_uid = 203, .source_camera_id = 20, .source_uid = 101, .input_base = 0.55f},
        };

        for (const auto& test_case : cases) {
            const auto input = make_input(test_case.input_base);
            const auto expected = source.apply(input, test_case.source_camera_id, test_case.source_uid).cpu().to_vector();
            const auto actual = target.apply(input, test_case.target_camera_id, test_case.target_uid).cpu().to_vector();
            EXPECT_TRUE(vectors_close(expected, actual, 1e-5f))
                << "Mapped PPISP output mismatch for target uid " << test_case.target_uid;
        }

        const auto controller_input_a = make_controller_input(0.20f);
        const auto controller_input_b = make_controller_input(0.45f);
        const auto expected_controller_a = source_controller.predict(1, controller_input_a, 1.0f).cpu().to_vector();
        const auto actual_controller_a = target_controller.predict(0, controller_input_a, 1.0f).cpu().to_vector();
        EXPECT_TRUE(vectors_close(expected_controller_a, actual_controller_a, 1e-5f));
        const auto expected_controller_b = source_controller.predict(0, controller_input_b, 1.0f).cpu().to_vector();
        const auto actual_controller_b = target_controller.predict(1, controller_input_b, 1.0f).cpu().to_vector();
        EXPECT_TRUE(vectors_close(expected_controller_b, actual_controller_b, 1e-5f));
    }

    TEST_F(PPISPSidecarTest, SaveLoadWithoutMetadataSupportsIdentityImport) {
        PPISPConfig config;
        config.lr = 5e-3;
        config.warmup_steps = 1;

        PPISP source(64, config);
        source.register_frame(11, 3);
        source.register_frame(12, 5);
        source.finalize();

        update_ppisp(source, 3, 11, 0.20f, 0.04f, 2);
        update_ppisp(source, 5, 12, 0.50f, 0.06f, 2);

        const auto path = make_temp_path("ppisp_sidecar_identity");
        auto save_result = lfs::training::save_ppisp_file(path, source);
        ASSERT_TRUE(save_result) << save_result.error();

        PPISP loaded(1);
        PPISPFileMetadata metadata;
        auto load_result = lfs::training::load_ppisp_file(path, loaded, nullptr, &metadata);
        ASSERT_TRUE(load_result) << load_result.error();
        EXPECT_TRUE(metadata.empty());

        PPISP target(64, config);
        target.register_frame(1011, 3);
        target.register_frame(1012, 5);
        target.finalize();

        const std::vector<int> frame_mapping = {0, 1};
        const std::vector<int> camera_mapping = {0, 1};
        auto import_result = target.copy_inference_weights_from(loaded, frame_mapping, camera_mapping);
        ASSERT_TRUE(import_result) << import_result.error();

        EXPECT_EQ(target.get_step(), 0);
        EXPECT_DOUBLE_EQ(target.get_lr(), target.get_config().lr);

        const auto input_a = make_input(0.30f);
        const auto input_b = make_input(0.60f);

        const auto expected_a = source.apply(input_a, 3, 11).cpu().to_vector();
        const auto actual_a = target.apply(input_a, 3, 1011).cpu().to_vector();
        EXPECT_TRUE(vectors_close(expected_a, actual_a, 1e-5f));

        const auto expected_b = source.apply(input_b, 5, 12).cpu().to_vector();
        const auto actual_b = target.apply(input_b, 5, 1012).cpu().to_vector();
        EXPECT_TRUE(vectors_close(expected_b, actual_b, 1e-5f));
    }

    TEST_F(PPISPSidecarTest, TrainingParamsValidationSkipsSidecarExistenceOnResume) {
        lfs::core::param::TrainingParameters params;
        params.optimization.use_ppisp = true;
        params.optimization.ppisp_freeze_from_sidecar = true;
        params.optimization.ppisp_sidecar_path = "/definitely/missing.ppisp";

        EXPECT_EQ(params.validate(), "PPISP sidecar does not exist: '/definitely/missing.ppisp'");

        params.resume_checkpoint = std::filesystem::path("/tmp/checkpoint.resume");
        EXPECT_TRUE(params.validate().empty());
    }

} // namespace
