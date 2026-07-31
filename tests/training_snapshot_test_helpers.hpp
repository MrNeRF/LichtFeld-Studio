/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "core/tensor.hpp"
#include "training/optimizer/adam_optimizer.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace lfs::training::test {

    struct TensorByteSnapshot {
        bool valid = false;
        lfs::core::TensorShape shape;
        lfs::core::DataType dtype =
            lfs::core::DataType::Float32;
        std::vector<std::byte> bytes;
    };

    struct AdamStateByteSnapshot {
        bool present = false;
        std::int64_t step_count = 0;
        std::size_t capacity = 0;
        std::size_t size = 0;
        TensorByteSnapshot exp_avg;
        TensorByteSnapshot exp_avg_sq;
        TensorByteSnapshot exp_avg_scale;
        TensorByteSnapshot exp_avg_sq_scale;
    };

    struct AdamMomentByteSnapshot {
        float lr = 0.0f;
        double beta1 = 0.0;
        double beta2 = 0.0;
        double eps = 0.0;
        float growth_factor = 0.0f;
        std::size_t initial_capacity = 0;
        std::unordered_map<std::string, double>
            param_lrs;
        std::array<
            AdamStateByteSnapshot,
            AdamOptimizer::all_param_types().size()>
            states;
    };

    inline TensorByteSnapshot capture_tensor_bytes(
        const lfs::core::Tensor& tensor) {
        TensorByteSnapshot result;
        result.valid = tensor.is_valid();
        if (!result.valid) {
            return result;
        }
        const auto cpu = tensor.cpu().contiguous();
        result.shape = cpu.shape();
        result.dtype = cpu.dtype();
        result.bytes.resize(cpu.bytes());
        if (!result.bytes.empty()) {
            std::memcpy(
                result.bytes.data(), cpu.data_ptr(),
                result.bytes.size());
        }
        return result;
    }

    inline AdamMomentByteSnapshot
    capture_optimizer_moment_bytes(
        const AdamOptimizer& optimizer) {
        AdamMomentByteSnapshot result;
        const auto& config = optimizer.get_config();
        result.lr = config.lr;
        result.beta1 = config.beta1;
        result.beta2 = config.beta2;
        result.eps = config.eps;
        result.growth_factor = config.growth_factor;
        result.initial_capacity =
            config.initial_capacity;
        result.param_lrs = config.param_lrs;
        const auto types =
            AdamOptimizer::all_param_types();
        for (std::size_t index = 0;
             index < types.size(); ++index) {
            const auto* state =
                optimizer.get_state(types[index]);
            if (!state) {
                continue;
            }
            const bool serialized =
                state->exp_avg.is_valid() &&
                state->exp_avg_sq.is_valid() &&
                state->exp_avg_scale.is_valid() &&
                state->exp_avg_sq_scale.is_valid();
            if (!serialized) {
                continue;
            }
            auto& captured = result.states[index];
            captured.present = true;
            captured.step_count = state->step_count;
            captured.capacity = state->capacity;
            captured.size = state->size;
            captured.exp_avg =
                capture_tensor_bytes(state->exp_avg);
            captured.exp_avg_sq =
                capture_tensor_bytes(state->exp_avg_sq);
            captured.exp_avg_scale =
                capture_tensor_bytes(
                    state->exp_avg_scale);
            captured.exp_avg_sq_scale =
                capture_tensor_bytes(
                    state->exp_avg_sq_scale);
        }
        return result;
    }

    inline void expect_tensor_bytes_equal(
        const TensorByteSnapshot& expected,
        const lfs::core::Tensor& actual,
        const char* label) {
        ASSERT_EQ(expected.valid, actual.is_valid())
            << label;
        if (!expected.valid) {
            return;
        }
        const auto cpu = actual.cpu().contiguous();
        EXPECT_EQ(expected.shape, cpu.shape())
            << label;
        EXPECT_EQ(expected.dtype, cpu.dtype())
            << label;
        ASSERT_EQ(expected.bytes.size(), cpu.bytes())
            << label;
        if (!expected.bytes.empty()) {
            EXPECT_EQ(
                std::memcmp(
                    expected.bytes.data(),
                    cpu.data_ptr(),
                    expected.bytes.size()),
                0)
                << label;
        }
    }

    inline void expect_optimizer_moment_bytes_equal(
        const AdamMomentByteSnapshot& expected,
        const AdamOptimizer& actual) {
        const auto& config = actual.get_config();
        EXPECT_FLOAT_EQ(expected.lr, config.lr);
        EXPECT_DOUBLE_EQ(expected.beta1, config.beta1);
        EXPECT_DOUBLE_EQ(expected.beta2, config.beta2);
        EXPECT_DOUBLE_EQ(expected.eps, config.eps);
        EXPECT_FLOAT_EQ(
            expected.growth_factor,
            config.growth_factor);
        EXPECT_EQ(
            expected.initial_capacity,
            config.initial_capacity);
        EXPECT_EQ(expected.param_lrs, config.param_lrs);

        const auto types =
            AdamOptimizer::all_param_types();
        for (std::size_t index = 0;
             index < types.size(); ++index) {
            const auto* state =
                actual.get_state(types[index]);
            const bool actual_serialized =
                state &&
                state->exp_avg.is_valid() &&
                state->exp_avg_sq.is_valid() &&
                state->exp_avg_scale.is_valid() &&
                state->exp_avg_sq_scale.is_valid();
            ASSERT_EQ(
                expected.states[index].present,
                actual_serialized)
                << "Adam parameter state index " << index;
            if (!actual_serialized) {
                continue;
            }
            const auto& captured =
                expected.states[index];
            EXPECT_EQ(
                captured.step_count,
                state->step_count)
                << "Adam state index " << index;
            EXPECT_EQ(captured.capacity, state->capacity)
                << "Adam state index " << index;
            EXPECT_EQ(captured.size, state->size)
                << "Adam state index " << index;
            expect_tensor_bytes_equal(
                captured.exp_avg, state->exp_avg,
                "exp_avg");
            expect_tensor_bytes_equal(
                captured.exp_avg_sq,
                state->exp_avg_sq, "exp_avg_sq");
            expect_tensor_bytes_equal(
                captured.exp_avg_scale,
                state->exp_avg_scale,
                "exp_avg_scale");
            expect_tensor_bytes_equal(
                captured.exp_avg_sq_scale,
                state->exp_avg_sq_scale,
                "exp_avg_sq_scale");
        }
    }

} // namespace lfs::training::test
