/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include <sstream>

#include "components/ppisp_controller.hpp"
#include "core/tensor.hpp"

namespace {

    class PPISPControllerTest : public ::testing::Test {
    protected:
        void SetUp() override {
            // Preallocate shared buffers for the largest test image size
            lfs::training::PPISPController::preallocate_shared_buffers(256, 256);
        }
    };

    TEST_F(PPISPControllerTest, ConstructorInitializesWeights) {
        lfs::training::PPISPController controller(10000);

        EXPECT_EQ(controller.get_step(), 0);
        EXPECT_GT(controller.get_lr(), 0.0);
    }

    TEST_F(PPISPControllerTest, ForwardPassShape) {
        lfs::training::PPISPController controller(10000);

        auto input = lfs::core::Tensor::randn({1, 3, 99, 99}, lfs::core::Device::CUDA);

        auto output = controller.predict(input, 1.0f);

        ASSERT_EQ(output.shape().rank(), 2);
        EXPECT_EQ(output.shape()[0], 1);
        EXPECT_EQ(output.shape()[1], 9);
    }

    TEST_F(PPISPControllerTest, ForwardPassDifferentInputSizes) {
        lfs::training::PPISPController controller(10000);

        auto input_small = lfs::core::Tensor::randn({1, 3, 30, 30}, lfs::core::Device::CUDA);
        auto output_small = controller.predict(input_small, 1.0f);
        EXPECT_EQ(output_small.shape()[1], 9);

        auto input_large = lfs::core::Tensor::randn({1, 3, 256, 256}, lfs::core::Device::CUDA);
        auto output_large = controller.predict(input_large, 1.0f);
        EXPECT_EQ(output_large.shape()[1], 9);
    }

    TEST_F(PPISPControllerTest, BackwardPassRuns) {
        lfs::training::PPISPController controller(10000);

        auto input = lfs::core::Tensor::randn({1, 3, 64, 64}, lfs::core::Device::CUDA);
        auto output = controller.predict(input, 1.0f);

        auto grad_output = lfs::core::Tensor::randn({1, 9}, lfs::core::Device::CUDA);
        controller.backward(grad_output);
    }

    TEST_F(PPISPControllerTest, OptimizerStepChangesWeights) {
        lfs::training::PPISPController controller(10000);

        auto input = lfs::core::Tensor::randn({1, 3, 64, 64}, lfs::core::Device::CUDA);

        auto output1 = controller.predict(input, 1.0f);
        auto output1_cpu = output1.cpu().to_vector();

        auto grad_output = lfs::core::Tensor::ones({1, 9}, lfs::core::Device::CUDA);
        controller.backward(grad_output);
        controller.optimizer_step();

        auto output2 = controller.predict(input, 1.0f);
        auto output2_cpu = output2.cpu().to_vector();

        bool changed = false;
        for (size_t i = 0; i < 9; ++i) {
            if (std::abs(output1_cpu[i] - output2_cpu[i]) > 1e-7f) {
                changed = true;
                break;
            }
        }
        EXPECT_TRUE(changed);
    }

    TEST_F(PPISPControllerTest, SchedulerStepUpdatesLR) {
        lfs::training::PPISPControllerConfig config;
        config.lr = 0.002;
        config.warmup_steps = 100;
        config.warmup_start_factor = 0.01;
        lfs::training::PPISPController controller(10000, config);

        double initial_lr = controller.get_lr();
        EXPECT_NEAR(initial_lr, config.lr * config.warmup_start_factor, 1e-6);

        // After warmup completes, LR should be at initial_lr
        for (int i = 0; i < 100; ++i) {
            controller.scheduler_step();
        }
        EXPECT_NEAR(controller.get_lr(), config.lr, 1e-6);

        // After more steps, LR should decay
        for (int i = 0; i < 1000; ++i) {
            controller.scheduler_step();
        }
        EXPECT_LT(controller.get_lr(), config.lr);
    }

    TEST_F(PPISPControllerTest, ZeroGradClearsGradients) {
        lfs::training::PPISPController controller(10000);

        auto input = lfs::core::Tensor::randn({1, 3, 64, 64}, lfs::core::Device::CUDA);
        controller.predict(input, 1.0f);

        auto grad = lfs::core::Tensor::ones({1, 9}, lfs::core::Device::CUDA);
        controller.backward(grad);
        controller.zero_grad();

        auto input2 = lfs::core::Tensor::randn({1, 3, 64, 64}, lfs::core::Device::CUDA);
        auto output1 = controller.predict(input2, 1.0f);
        auto output1_cpu = output1.cpu().to_vector();

        controller.optimizer_step();

        auto output2 = controller.predict(input2, 1.0f);
        auto output2_cpu = output2.cpu().to_vector();

        bool same = true;
        for (size_t i = 0; i < 9; ++i) {
            if (std::abs(output1_cpu[i] - output2_cpu[i]) > 1e-7f) {
                same = false;
                break;
            }
        }
        EXPECT_TRUE(same);
    }

    TEST_F(PPISPControllerTest, SerializeDeserializePreservesState) {
        lfs::training::PPISPControllerConfig config;
        config.lr = 0.003;
        lfs::training::PPISPController controller1(10000, config);

        auto input = lfs::core::Tensor::randn({1, 3, 64, 64}, lfs::core::Device::CUDA);
        auto output1 = controller1.predict(input, 1.0f);

        for (int i = 0; i < 10; ++i) {
            auto grad = lfs::core::Tensor::randn({1, 9}, lfs::core::Device::CUDA);
            controller1.backward(grad);
            controller1.optimizer_step();
            controller1.zero_grad();
            controller1.scheduler_step();
        }

        auto output_after_train = controller1.predict(input, 1.0f);
        auto output_after_cpu = output_after_train.cpu().to_vector();

        std::stringstream ss;
        controller1.serialize(ss);

        lfs::training::PPISPController controller2(10000);
        ss.seekg(0);
        controller2.deserialize(ss);

        EXPECT_EQ(controller2.get_step(), controller1.get_step());
        EXPECT_NEAR(controller2.get_lr(), controller1.get_lr(), 1e-10);

        auto output_loaded = controller2.predict(input, 1.0f);
        auto output_loaded_cpu = output_loaded.cpu().to_vector();

        for (size_t i = 0; i < 9; ++i) {
            EXPECT_NEAR(output_after_cpu[i], output_loaded_cpu[i], 1e-5f);
        }
    }

    TEST_F(PPISPControllerTest, TrainingLoopConverges) {
        lfs::training::PPISPControllerConfig config;
        config.lr = 0.01;
        config.warmup_steps = 10;
        lfs::training::PPISPController controller(1000, config);

        auto input = lfs::core::Tensor::randn({1, 3, 48, 48}, lfs::core::Device::CUDA);
        auto target = lfs::core::Tensor::randn({1, 9}, lfs::core::Device::CUDA);

        float initial_loss = 0.0f;
        float final_loss = 0.0f;

        for (int iter = 0; iter < 100; ++iter) {
            auto pred = controller.predict(input, 1.0f);
            auto loss = pred.sub(target).square().mean();

            if (iter == 0) {
                initial_loss = loss.cpu().item<float>();
            }
            if (iter == 99) {
                final_loss = loss.cpu().item<float>();
            }

            auto grad = (pred - target).mul(2.0f / 9.0f);
            controller.backward(grad);
            controller.optimizer_step();
            controller.zero_grad();
            controller.scheduler_step();
        }

        EXPECT_LT(final_loss, initial_loss);
    }

} // namespace
