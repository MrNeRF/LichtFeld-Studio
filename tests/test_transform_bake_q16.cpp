/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/error.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "visualizer/gui_capabilities.hpp"

#include <cmath>
#include <cstring>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace lfs::core;
using namespace lfs::training;

namespace {

    constexpr size_t kN = 1000; // not divisible by 256 — exercises pad-dropped block tails
    constexpr int kShDegree = 3;

    struct ShValueQuantGuard {
        explicit ShValueQuantGuard(const bool enabled) {
            sh_value::set_sh_value_quant_enabled_for_testing(enabled);
        }
        ~ShValueQuantGuard() {
            sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
        }
    };

    SplatData make_random_sh3(const size_t n, const uint32_t seed = 42) {
        auto means = Tensor::zeros({n, size_t{3}}, Device::CUDA, DataType::Float32);
        auto sh0 = Tensor::zeros({n, size_t{1}, size_t{3}}, Device::CUDA, DataType::Float32);
        auto shN_can = Tensor::zeros({n, size_t{15}, size_t{3}}, Device::CUDA, DataType::Float32);
        auto scaling = Tensor::zeros({n, size_t{3}}, Device::CUDA, DataType::Float32);
        auto rotation = Tensor::zeros({n, size_t{4}}, Device::CUDA, DataType::Float32);
        auto opacity = Tensor::zeros({n, size_t{1}}, Device::CUDA, DataType::Float32);

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> sh_dist(-1.0f, 1.0f);
        std::uniform_real_distribution<float> mean_dist(-2.0f, 2.0f);
        std::uniform_real_distribution<float> scale_dist(-1.0f, 0.0f);

        {
            auto cpu = means.cpu();
            auto* p = cpu.ptr<float>();
            for (size_t i = 0; i < n * 3; ++i)
                p[i] = mean_dist(rng);
            means = cpu.to(Device::CUDA);
        }
        {
            auto cpu = shN_can.cpu();
            auto* p = cpu.ptr<float>();
            for (size_t i = 0; i < n * 15 * 3; ++i)
                p[i] = sh_dist(rng);
            shN_can = cpu.to(Device::CUDA);
        }
        {
            auto cpu = scaling.cpu();
            auto* p = cpu.ptr<float>();
            for (size_t i = 0; i < n * 3; ++i)
                p[i] = scale_dist(rng);
            scaling = cpu.to(Device::CUDA);
        }
        {
            auto cpu = rotation.cpu();
            auto* r = cpu.ptr<float>();
            for (size_t i = 0; i < n; ++i)
                r[i * 4] = 1.0f;
            rotation = cpu.to(Device::CUDA);
        }

        return SplatData(kShDegree, means, sh0, shN_can, scaling, rotation, opacity, 1.0f);
    }

    [[nodiscard]] float max_abs_diff(const Tensor& a, const Tensor& b) {
        EXPECT_EQ(a.numel(), b.numel());
        EXPECT_EQ(a.dtype(), DataType::Float32);
        EXPECT_EQ(b.dtype(), DataType::Float32);
        auto ac = a.cpu().contiguous();
        auto bc = b.cpu().contiguous();
        const auto* pa = ac.ptr<float>();
        const auto* pb = bc.ptr<float>();
        float max_err = 0.0f;
        for (size_t i = 0; i < ac.numel(); ++i)
            max_err = std::max(max_err, std::abs(pa[i] - pb[i]));
        return max_err;
    }

    [[nodiscard]] glm::mat4 rotation_and_translation() {
        const glm::vec3 axis = glm::normalize(glm::vec3(1.0f, 2.0f, 3.0f));
        return glm::translate(glm::mat4(1.0f), glm::vec3(1.25f, -0.5f, 0.75f)) *
               glm::rotate(glm::mat4(1.0f), glm::radians(30.0f), axis);
    }

    [[nodiscard]] glm::mat4 translation_only() {
        return glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, -1.0f, 0.5f));
    }

} // namespace

// Catches the pre-#1620 bake, which cloned raw q16 codes without bounds and
// failed the shN copy-back on shape/dtype (and silently corrupted SH2 colors).
TEST(TransformBakeQ16, RotationBakeRoundTripsQ16) {
    const ShValueQuantGuard quant_guard{true};
    auto reference = make_random_sh3(kN, 42);
    auto quantized = reference.clone();
    ASSERT_TRUE(sh_value::apply_shN_value_quant(quantized));
    ASSERT_TRUE(quantized.shN_value_quantized());
    ASSERT_TRUE(quantized.shN_value_bounds().is_valid());

    const glm::mat4 transform = rotation_and_translation();
    const auto ref_result = lfs::vis::cap::bakeSplatTransformPreservingStorage(reference, transform);
    ASSERT_TRUE(ref_result.has_value()) << lfs::format_for_developer(ref_result.error());
    const auto q16_result = lfs::vis::cap::bakeSplatTransformPreservingStorage(quantized, transform);
    ASSERT_TRUE(q16_result.has_value()) << lfs::format_for_developer(q16_result.error());

    EXPECT_LE(max_abs_diff(quantized.shN_canonical(), reference.shN_canonical()), 1e-2f);
    EXPECT_LE(max_abs_diff(quantized.means_raw(), reference.means_raw()), 1e-5f);
    EXPECT_LE(max_abs_diff(quantized.rotation_raw(), reference.rotation_raw()), 1e-5f);
    EXPECT_LE(max_abs_diff(quantized.scaling_raw(), reference.scaling_raw()), 1e-5f);

    ASSERT_TRUE(quantized.shN_value_quantized());
    ASSERT_TRUE(quantized.shN_value_bounds().is_valid());
    EXPECT_GT(quantized.shN_value_bounds().numel(), 0u);
}

// Catches a bake that decodes+re-encodes SH on non-rotating transforms, which
// would drift codes lossily on every translate/scale bake.
TEST(TransformBakeQ16, TranslationOnlyBakeKeepsQ16CodesBitExact) {
    const ShValueQuantGuard quant_guard{true};
    auto model = make_random_sh3(kN, 7);
    ASSERT_TRUE(sh_value::apply_shN_value_quant(model));
    ASSERT_TRUE(model.shN_value_quantized());

    const auto codes_before = model.shN_raw().cpu().contiguous();
    const auto result =
        lfs::vis::cap::bakeSplatTransformPreservingStorage(model, translation_only());
    ASSERT_TRUE(result.has_value()) << lfs::format_for_developer(result.error());

    ASSERT_TRUE(model.shN_value_quantized());
    const auto codes_after = model.shN_raw().cpu().contiguous();
    ASSERT_EQ(codes_before.dtype(), codes_after.dtype());
    ASSERT_EQ(codes_before.shape(), codes_after.shape());
    ASSERT_EQ(codes_before.bytes(), codes_after.bytes());
    EXPECT_EQ(std::memcmp(codes_before.data_ptr(), codes_after.data_ptr(), codes_before.bytes()), 0);
}

// Catches the q16 path leaking into fp32 models (e.g. an unconditional
// canonical round-trip that would quantize a previously-fp32 model).
TEST(TransformBakeQ16, Fp32BakeUnchanged) {
    auto model = make_random_sh3(kN, 99);
    ASSERT_EQ(model.shN_raw().dtype(), DataType::Float32);

    const auto result =
        lfs::vis::cap::bakeSplatTransformPreservingStorage(model, rotation_and_translation());
    ASSERT_TRUE(result.has_value()) << lfs::format_for_developer(result.error());
    EXPECT_EQ(model.shN_raw().dtype(), DataType::Float32);
    EXPECT_FALSE(model.shN_value_quantized());
}
