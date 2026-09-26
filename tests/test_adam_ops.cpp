/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "adam_api.h"
#include "core/sh_layout.hpp"
#include "core/sh_value_quant.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "cuda_backend_test.hpp"
#include "lfs/training/joint_adam_codec.hpp"
#include "lfs/training/ops/adam_cuda.hpp"
#include "lfs/training/ops/registry.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "lfs/training/sh_value_storage.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;
    namespace ops = lfs::gpu_ops;
    namespace joint_adam = lfs::training::joint_adam;

    constexpr float kBeta1 = 0.9f;
    constexpr float kBeta2 = 0.999f;
    constexpr float kEps = 1e-15f;

    Tensor pattern(const lfs::core::TensorShape& shape, const float scale, const int seed) {
        const size_t n = shape.elements();
        std::vector<float> values(n);
        for (size_t i = 0; i < n; ++i) {
            const auto k = static_cast<int>((i * 7919u + static_cast<size_t>(seed) * 104729u) % 2003u);
            values[i] = scale * (static_cast<float>(k) / 1001.0f - 1.0f);
        }
        return Tensor::from_vector(values, shape, Device::GPU);
    }

    Tensor bool_mask(const size_t n, const size_t period) {
        std::vector<bool> values(n);
        for (size_t i = 0; i < n; ++i) {
            values[i] = i % period == 0;
        }
        return Tensor::from_vector(values, {n}, Device::GPU);
    }

    Tensor rows(const std::vector<int64_t>& values) {
        auto cpu = Tensor::empty({values.size()}, Device::CPU, DataType::Int64);
        for (size_t i = 0; i < values.size(); ++i) {
            cpu.ptr<int64_t>()[i] = values[i];
        }
        return cpu.gpu();
    }

    std::vector<uint8_t> bytes(const Tensor& tensor) {
        const auto cpu = tensor.cpu().contiguous();
        const auto* data = static_cast<const uint8_t*>(cpu.data_ptr());
        return {data, data + cpu.bytes()};
    }

    void expect_same_bytes(const Tensor& actual, const Tensor& expected, const std::string& what) {
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(bytes(actual), bytes(expected)) << what;
    }

    // Guards against a comparison of two untouched buffers.
    void expect_changed(const Tensor& tensor, const std::vector<uint8_t>& before, const std::string& what) {
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_NE(bytes(tensor), before) << what;
    }

    template <typename T>
    const T* optional_ptr(const Tensor& tensor) {
        return tensor.is_valid() && tensor.numel() > 0 ? tensor.ptr<T>() : nullptr;
    }

    // Parameter, packed joint moments and block bounds for one [n, attrs] group.
    struct Group {
        Tensor parameter, packed, bounds, gradient;

        static Group make(const size_t n, const size_t attrs, const int bits, const int seed) {
            Group group;
            group.parameter = pattern({n, attrs}, 0.5f, seed);
            group.packed = Tensor::zeros(
                {n, attrs * static_cast<size_t>(joint_adam::bytes_per_cell(bits))},
                Device::GPU, DataType::UInt8);
            group.bounds = Tensor::zeros({joint_adam::n_bounds_for_prims(n), size_t{4}}, Device::GPU);
            group.gradient = pattern({n, attrs}, 1e-3f, seed + 1);
            return group;
        }

        [[nodiscard]] Group clone() const {
            return {parameter.clone(), packed.clone(), bounds.clone(), gradient.clone()};
        }

        void expect_same(const Group& expected, const std::string& what) const {
            expect_same_bytes(parameter, expected.parameter, what + " parameter");
            expect_same_bytes(packed, expected.packed, what + " packed");
            expect_same_bytes(bounds, expected.bounds, what + " bounds");
        }
    };

    struct Masks {
        Tensor frozen, crop, raw_scales, far, share;

        static Masks make(const size_t n, const bool enabled) {
            if (!enabled) {
                return {};
            }
            return {
                bool_mask(n, 5),
                bool_mask(n, 7),
                pattern({n, size_t{3}}, 2.0f, 17),
                bool_mask(n, 3),
                pattern({n}, 0.4f, 23).abs(),
            };
        }

        [[nodiscard]] ops::AdamMasks view() const {
            return {frozen, crop, raw_scales, far, share};
        }
    };

    constexpr ops::AdamModifiers kModifiers{
        .frozen_lr_scale = 0.25f,
        .cropbox_lr_scale = 0.5f,
        .median_extent = 1.5f,
        .r_min = 1.0f,
        .r_max = 300.0f,
        .screen_share_limit = 0.3f,
        .screen_share_penalty = 0.05f,
    };

    constexpr ops::AdamHyper kHyper{.beta1 = kBeta1, .beta2 = kBeta2, .eps = kEps};

    float bc1(const int step) {
        return static_cast<float>(1.0 / (1.0 - std::pow(0.9, step)));
    }

    float bc2(const int step) {
        return static_cast<float>(1.0 / std::sqrt(1.0 - std::pow(0.999, step)));
    }

    ops::JointStep joint_step(Group& g, const int bits, const int step,
                              const bool mean_step, const bool share) {
        return {
            .parameter = g.parameter,
            .packed = g.packed,
            .bounds = g.bounds,
            .gradient = g.gradient,
            .primitives = static_cast<int>(g.parameter.shape()[0]),
            .attributes = static_cast<int>(g.parameter.shape()[1]),
            .bits = bits,
            .lr = 0.01f,
            .bc1_rcp = bc1(step),
            .bc2_sqrt_rcp = bc2(step),
            .apply_mean_step = mean_step,
            .apply_screen_share = share,
        };
    }

    void advance_gradient(Group& a, Group& b, const int step) {
        a.gradient = pattern(a.gradient.shape(), 1e-3f * static_cast<float>(step), 31 + step);
        b.gradient = a.gradient.clone();
    }

    // Swizzled SH values, packed 8-bit joint moments and block bounds.
    struct ShGroup {
        Tensor parameter, packed, bounds, value_bounds, gradient;
        int value_bits = 0;
        int value_cells = 0;

        [[nodiscard]] ShGroup clone() const {
            return {parameter.clone(), packed.clone(), bounds.clone(),
                    value_bounds.is_valid() ? value_bounds.clone() : Tensor{},
                    gradient.clone(), value_bits, value_cells};
        }

        void expect_same(const ShGroup& expected, const std::string& what) const {
            expect_same_bytes(parameter, expected.parameter, what + " parameter");
            expect_same_bytes(packed, expected.packed, what + " packed");
            expect_same_bytes(bounds, expected.bounds, what + " bounds");
            if (value_bounds.is_valid()) {
                expect_same_bytes(value_bounds, expected.value_bounds, what + " value bounds");
            }
        }
    };

    enum class ShValues { Float32,
                          Half,
                          Q16 };

    lfs::core::SplatData make_splat(const size_t n) {
        const size_t rest = 15;
        return lfs::core::SplatData(
            3,
            pattern({n, size_t{3}}, 1.0f, 1),
            pattern({n, size_t{1}, size_t{3}}, 0.5f, 2),
            pattern({n, rest, size_t{3}}, 0.2f, 3),
            pattern({n, size_t{3}}, 1.0f, 4),
            pattern({n, size_t{4}}, 1.0f, 5),
            pattern({n, size_t{1}}, 1.0f, 6),
            1.0f);
    }

    ShGroup make_sh_group(const size_t n, const ShValues values) {
        constexpr uint32_t rest = 15;
        const size_t floats = lfs::core::sh_swizzled_float_count(n, rest);
        ShGroup group;
        group.packed = Tensor::zeros(
            {floats * static_cast<size_t>(joint_adam::bytes_per_cell(8))}, Device::GPU, DataType::UInt8);
        group.bounds = Tensor::zeros({joint_adam::n_bounds_for_prims(n), size_t{4}}, Device::GPU);
        group.gradient = pattern({floats}, 1e-3f, 41);
        switch (values) {
        case ShValues::Float32:
            group.parameter = pattern({floats}, 0.2f, 43);
            break;
        case ShValues::Half:
            group.parameter = pattern({floats}, 0.2f, 43).to(DataType::Float16);
            group.value_bits = 16;
            break;
        case ShValues::Q16: {
            lfs::training::sh_value::set_sh_value_quant_enabled_for_testing(true);
            auto splat = make_splat(n);
            const bool quantized = lfs::training::sh_value::apply_shN_value_quant(splat);
            lfs::training::sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
            EXPECT_TRUE(quantized);
            group.parameter = splat.shN().clone();
            group.value_bounds = splat.shN_value_bounds().clone();
            group.value_bits = 16;
            group.value_cells = static_cast<int>(lfs::core::sh_value_quant::n_value_cells_per_prim(rest));
            break;
        }
        }
        return group;
    }

    void launch_sh_step(ShGroup& g, const Masks& m, const size_t n, const int active_bases, const int step) {
        fast_lfs::optimizer::adam_step_shN_joint_from_grad(
            static_cast<float*>(g.parameter.data_ptr()), g.packed.ptr<uint8_t>(), g.bounds.ptr<float>(),
            g.value_bounds.is_valid() ? g.value_bounds.ptr<float>() : nullptr, g.gradient.ptr<float>(),
            optional_ptr<bool>(m.frozen), static_cast<int>(m.frozen.numel()), kModifiers.frozen_lr_scale,
            optional_ptr<bool>(m.crop), static_cast<int>(m.crop.numel()), kModifiers.cropbox_lr_scale,
            static_cast<int>(n), static_cast<int>(lfs::core::sh_float4_slots_for_rest(15)), active_bases,
            g.value_bits, g.value_cells, 0.005f * bc1(step), kBeta1, kBeta2, kEps, bc2(step),
            lfs::core::getCurrentCUDAStream());
    }

    void op_sh_step(ShGroup& g, const Masks& m, const size_t n, const int active_bases, const int step) {
        Tensor absent;
        lfs::training::cuda_adam_ops().step_sh(
            g.parameter, g.packed, g.bounds, g.value_bounds.is_valid() ? g.value_bounds : absent,
            g.gradient, m.view(), kHyper, kModifiers,
            {
                .primitives = static_cast<int>(n),
                .layout_slots = static_cast<int>(lfs::core::sh_float4_slots_for_rest(15)),
                .active_bases = active_bases,
                .value_bits = g.value_bits,
                .value_cells = g.value_cells,
                .step_size = 0.005f * bc1(step),
                .bc2_sqrt_rcp = bc2(step),
            });
    }

    const std::vector<int64_t> kResetRows{0, 5, 5, 255, 256, 300, 511, 512, 699, 3};

} // namespace

class AdamOpsBytes : public lfs::test::CudaBackendTest {};

TEST_F(AdamOpsBytes, StepBatchMatchesLauncherAndSkipsAbsentSteps) {
    constexpr size_t n = 700;
    constexpr std::array<size_t, 5> attrs{3, 3, 3, 4, 1};
    for (const bool with_absent : {false, true}) {
        SCOPED_TRACE(with_absent ? "absent sh0" : "all present");
        const Masks masks = Masks::make(n, true);
        std::vector<Group> expected;
        std::vector<Group> actual;
        std::vector<std::vector<uint8_t>> parameters_before;
        for (size_t i = 0; i < attrs.size(); ++i) {
            expected.push_back(Group::make(n - 40 * i, attrs[i], 16, static_cast<int>(10 * i)));
            actual.push_back(expected.back().clone());
            parameters_before.push_back(bytes(actual.back().parameter));
        }
        Tensor absent;
        for (int step = 1; step <= 3; ++step) {
            std::vector<fast_lfs::optimizer::JointContiguousBatchEntry> entries;
            for (size_t i = 0; i < attrs.size(); ++i) {
                if (with_absent && i == 1) {
                    continue;
                }
                auto& g = expected[i];
                entries.push_back({
                    .param = g.parameter.ptr<float>(),
                    .packed = g.packed.ptr<uint8_t>(),
                    .bounds = g.bounds.ptr<float>(),
                    .grad = g.gradient.ptr<float>(),
                    .n_prims = static_cast<int>(g.parameter.shape()[0]),
                    .n_attr = static_cast<int>(attrs[i]),
                    .lr = 0.01f * static_cast<float>(i + 1),
                    .bias_correction1_rcp = bc1(step),
                    .bias_correction2_sqrt_rcp = bc2(step),
                    .apply_mean_step = i == 0 ? 1 : 0,
                    .apply_screen_share = i == 2 ? 1 : 0,
                });
            }
            fast_lfs::optimizer::adam_step_joint_contiguous_batched(
                entries.data(), static_cast<int>(entries.size()),
                masks.frozen.ptr<bool>(), static_cast<int>(masks.frozen.numel()), kModifiers.frozen_lr_scale,
                masks.crop.ptr<bool>(), static_cast<int>(masks.crop.numel()), kModifiers.cropbox_lr_scale,
                kBeta1, kBeta2, kEps, lfs::core::getCurrentCUDAStream(),
                masks.raw_scales.ptr<float>(), static_cast<int>(masks.raw_scales.numel()),
                kModifiers.median_extent, kModifiers.r_min, kModifiers.r_max,
                masks.far.ptr<bool>(), static_cast<int>(masks.far.numel()),
                masks.share.ptr<float>(), static_cast<int>(masks.share.numel()),
                kModifiers.screen_share_limit, kModifiers.screen_share_penalty);

            std::vector<ops::JointStep> steps;
            for (size_t i = 0; i < attrs.size(); ++i) {
                auto step_i = joint_step(actual[i], 16, step, i == 0, i == 2);
                step_i.lr = 0.01f * static_cast<float>(i + 1);
                if (with_absent && i == 1) {
                    steps.push_back({.parameter = absent, .packed = absent, .bounds = absent, .gradient = absent});
                } else {
                    steps.push_back(step_i);
                }
            }
            lfs::training::cuda_adam_ops().step_batch(steps, masks.view(), kHyper, kModifiers);
            for (size_t i = 0; i < attrs.size(); ++i) {
                advance_gradient(expected[i], actual[i], step);
            }
        }
        for (size_t i = 0; i < attrs.size(); ++i) {
            const std::string what = "group " + std::to_string(i);
            if (with_absent && i == 1) {
                EXPECT_EQ(bytes(actual[i].parameter), parameters_before[i]) << what;
            } else {
                expect_changed(actual[i].parameter, parameters_before[i], what);
            }
            actual[i].expect_same(expected[i], what);
        }
    }
}

TEST_F(AdamOpsBytes, StepShMatchesLauncher) {
    constexpr size_t n = 700;
    for (const auto values : {ShValues::Float32, ShValues::Half, ShValues::Q16}) {
        for (const int active_bases : {4, 9, 16}) {
            SCOPED_TRACE(std::to_string(static_cast<int>(values)) + " values, " +
                         std::to_string(active_bases) + " bases");
            const Masks masks = Masks::make(n, active_bases == 9);
            ShGroup expected = make_sh_group(n, values);
            ShGroup actual = expected.clone();
            const auto parameter_before = bytes(actual.parameter);
            for (int step = 1; step <= 3; ++step) {
                launch_sh_step(expected, masks, n, active_bases, step);
                op_sh_step(actual, masks, n, active_bases, step);
                expected.gradient = pattern({expected.gradient.numel()}, 1e-3f * static_cast<float>(step), 50 + step);
                actual.gradient = expected.gradient.clone();
            }
            expect_changed(actual.parameter, parameter_before, "step_sh parameter");
            actual.expect_same(expected, "step_sh");
        }
    }
}

TEST_F(AdamOpsBytes, EncodeZeroMatchesLauncher) {
    constexpr size_t n = 700;
    const Tensor indices = rows(kResetRows);
    const Masks masks = Masks::make(n, false);
    for (const int bits : {8, 16}) {
        SCOPED_TRACE(std::to_string(bits) + " bit rows");
        Group expected = Group::make(n, 4, bits, 3);
        expected.packed.copy_from(pattern(expected.packed.shape(), 100.0f, 63).abs().to(DataType::UInt8));
        expected.bounds.copy_from(pattern(expected.bounds.shape(), 0.5f, 64));
        Group actual = expected.clone();
        const auto packed_before = bytes(actual.packed);
        fast_lfs::optimizer::joint_encode_zero_rows_at_indices(
            expected.packed.ptr<uint8_t>(), expected.bounds.ptr<float>(), indices.ptr<int64_t>(),
            static_cast<int>(indices.numel()), 4, bits, static_cast<int>(n),
            lfs::core::getCurrentCUDAStream());
        lfs::training::cuda_adam_ops().encode_zero(
            actual.packed, actual.bounds, indices,
            {.layout = ops::JointLayout::Rows, .primitives = static_cast<int>(n), .attributes_or_slots = 4, .bits = bits});
        expect_changed(actual.packed, packed_before, "encode_zero rows");
        actual.expect_same(expected, "encode_zero rows");
    }

    const int slots = static_cast<int>(lfs::core::sh_float4_slots_for_rest(15));
    for (const int bits : {8, 16}) {
        SCOPED_TRACE(std::to_string(bits) + " bit SH");
        ShGroup expected = make_sh_group(n, ShValues::Float32);
        if (bits == 16) {
            expected.packed = Tensor::zeros(
                {expected.gradient.numel() * static_cast<size_t>(joint_adam::bytes_per_cell(16))},
                Device::GPU, DataType::UInt8);
            expected.packed.copy_from(pattern({expected.packed.numel()}, 100.0f, 61).abs().to(DataType::UInt8));
            expected.bounds.copy_from(pattern({expected.bounds.shape()[0], size_t{4}}, 0.5f, 62));
        } else {
            for (int step = 1; step <= 2; ++step) {
                launch_sh_step(expected, masks, n, 16, step);
            }
        }
        ShGroup actual = expected.clone();
        const auto packed_before = bytes(actual.packed);
        fast_lfs::optimizer::joint_encode_zero_shN_at_indices(
            expected.packed.ptr<uint8_t>(), expected.bounds.ptr<float>(), indices.ptr<int64_t>(),
            static_cast<int>(indices.numel()), slots, bits, static_cast<int>(n),
            lfs::core::getCurrentCUDAStream());
        lfs::training::cuda_adam_ops().encode_zero(
            actual.packed, actual.bounds, indices,
            {.layout = ops::JointLayout::SwizzledSH, .primitives = static_cast<int>(n), .attributes_or_slots = slots, .bits = bits});
        expect_changed(actual.packed, packed_before, "encode_zero SH");
        actual.expect_same(expected, "encode_zero SH");
    }
}

TEST(AdamOpsCapability, OnlyCudaCarriesAdam) {
    EXPECT_EQ(lfs::training::training_ops(lfs::core::GpuBackend::CUDA).adam,
              &lfs::training::cuda_adam_ops());
    EXPECT_EQ(lfs::training::training_ops(lfs::core::GpuBackend::Vulkan).adam, nullptr);
    EXPECT_EQ(lfs::training::training_ops(lfs::core::GpuBackend::Metal).adam, nullptr);
    EXPECT_FALSE(lfs::training::unavailable_training_family(
        lfs::core::GpuBackend::CUDA, lfs::training::Family::Adam));
    EXPECT_EQ(
        lfs::training::unavailable_training_family(lfs::core::GpuBackend::Vulkan, lfs::training::Family::Adam),
        "Vulkan training is unavailable for this configuration.\nMissing families: Adam.");
}
