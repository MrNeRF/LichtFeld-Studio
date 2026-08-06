/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "components/ppisp.hpp"
#include "core/tensor.hpp"
#include "tensor_hardening_test_utils.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <sstream>
#include <vector>

namespace {

    using lfs::core::Device;
    using lfs::core::Tensor;
    using lfs::training::PPISP;
    using lfs::training::PPISPConfig;

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

    void register_test_frames(PPISP& ppisp) {
        ppisp.register_frame(101, 20);
        ppisp.register_frame(102, 20);
        ppisp.register_frame(201, 30);
        ppisp.finalize();
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

    void train_nontrivial_state(PPISP& ppisp) {
        update_ppisp(ppisp, 20, 101, 0.10f, 0.03f, 2);
        update_ppisp(ppisp, 20, 102, 0.20f, 0.05f, 2);
        update_ppisp(ppisp, 30, 201, 0.30f, 0.07f, 2);
    }

    float expect_loss_contract(PPISP& ppisp) {
        const auto loss = ppisp.reg_loss_gpu();
        EXPECT_TRUE(loss.is_valid());
        EXPECT_EQ(loss.device(), Device::CUDA);
        EXPECT_EQ(loss.ndim(), 1u);
        EXPECT_EQ(loss.numel(), 1u);
        return loss.cpu().item<float>();
    }

    class PPISPCheckpointRoundtripTest : public tensor_hardening::CudaTest {};

    TEST_F(PPISPCheckpointRoundtripTest, AdoptCheckpointStatePreservesColorMeanRegularizer) {
        PPISPConfig config;
        config.lr = 1e-2;
        config.warmup_steps = 0;
        config.color_mean = 1.0f;

        PPISP source(1000, config);
        register_test_frames(source);
        train_nontrivial_state(source);
        const float expected = expect_loss_contract(source);
        ASSERT_TRUE(std::isfinite(expected));

        std::stringstream checkpoint;
        source.serialize(checkpoint);
        checkpoint.seekg(0);

        PPISP live(1000, config);
        register_test_frames(live);

        PPISP loaded(1);
        loaded.deserialize(checkpoint);
        live.adopt_checkpoint_state(loaded);

        EXPECT_TRUE(live.isFinalized());
        EXPECT_EQ(live.get_step(), source.get_step());

        float restored = 0.0f;
        EXPECT_NO_THROW(restored = expect_loss_contract(live));
        EXPECT_NEAR(restored, expected, 1e-5f);

        EXPECT_NO_THROW(live.reg_backward());
        live.optimizer_step();
        const float after_step = expect_loss_contract(live);
        EXPECT_TRUE(std::isfinite(after_step));
        EXPECT_GT(std::abs(after_step - restored), 1e-9f);
    }

    TEST_F(PPISPCheckpointRoundtripTest, DeserializeInferencePreservesColorMeanRegularizer) {
        PPISPConfig config;
        config.lr = 1e-2;
        config.warmup_steps = 0;
        config.color_mean = 1.0f;

        PPISP source(1000, config);
        register_test_frames(source);
        train_nontrivial_state(source);

        std::stringstream inference_weights;
        source.serialize_inference(inference_weights);
        inference_weights.seekg(0);

        PPISP restored(1);
        restored.deserialize_inference(inference_weights);

        float loss = 0.0f;
        EXPECT_NO_THROW(loss = expect_loss_contract(restored));
        EXPECT_TRUE(std::isfinite(loss));
    }

} // namespace
