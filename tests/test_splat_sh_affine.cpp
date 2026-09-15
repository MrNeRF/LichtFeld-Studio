/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/scene.hpp"
#include "core/splat_data.hpp"
#include "core/splat_data_transform.hpp"
#include "core/tensor.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "visualizer/gui_capabilities.hpp"
#include <array>
#include <cstring>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <random>

using namespace lfs::core;

namespace {
    // The renderer evaluates these solid-harmonic polynomials even when the
    // node's column-normalized affine matrix produces a non-unit direction.
    std::array<double, 16> basis(glm::dvec3 d) {
        const double x = d.x, y = d.y, z = d.z, xx = x * x, yy = y * y, zz = z * z;
        return {0.28209479177387814, -0.4886025119029199 * y, 0.4886025119029199 * z, -0.4886025119029199 * x,
                1.0925484305920792 * x * y, -1.0925484305920792 * y * z,
                0.31539156525252005 * (2 * zz - xx - yy), -1.0925484305920792 * x * z, 0.5462742152960396 * (xx - yy),
                -0.5900435899266435 * y * (3 * xx - yy), 2.890611442640554 * x * y * z,
                -0.4570457994644658 * y * (4 * zz - xx - yy), 0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy),
                -0.4570457994644658 * x * (4 * zz - xx - yy), 1.445305721320277 * z * (xx - yy),
                -0.5900435899266435 * x * (xx - 3 * yy)};
    }

    SplatData fixture() {
        constexpr size_t n = 37;
        std::vector<float> coefficients(n * 15 * 3), rotations(n * 4);
        std::mt19937 random(1981);
        std::uniform_real_distribution<float> coefficient(-0.4f, 0.4f);
        for (float& value : coefficients)
            value = coefficient(random);
        for (size_t i = 0; i < n; ++i)
            rotations[4 * i] = 1;
        return SplatData(3, Tensor::zeros({n, 3}, Device::CUDA), Tensor::zeros({n, 1, 3}, Device::CUDA),
                         Tensor::from_vector(coefficients, {n, 15, 3}, Device::CUDA),
                         Tensor::zeros({n, 3}, Device::CUDA),
                         Tensor::from_vector(rotations, {n, 4}, Device::CUDA),
                         Tensor::zeros({n, 1}, Device::CUDA), 1.0f);
    }

    glm::mat4 shear() {
        glm::mat4 matrix(1);
        matrix[1][0] = 0.7f;
        matrix[2][1] = -0.4f;
        return matrix;
    }
} // namespace

TEST(SplatAffineSH, BakedColorsMatchNativeNonNormalizedDirection) {
    auto data = fixture();
    const auto before = data.shN_canonical().cpu().contiguous();
    const auto matrix = shear();
    glm::dmat3 direction(matrix);
    for (int i = 0; i < 3; ++i)
        direction[i] = glm::normalize(direction[i]);
    transform(data, matrix);
    const auto dc = data.sh0_raw().cpu().contiguous();
    const auto after = data.shN_canonical().cpu().contiguous();
    for (int sample = 0; sample < 101; ++sample) {
        const double z = 1 - 2 * (sample + 0.5) / 101;
        const double angle = sample * 2.399963229728653;
        const glm::dvec3 world(std::sqrt(1 - z * z) * std::cos(angle), std::sqrt(1 - z * z) * std::sin(angle), z);
        const auto native = basis(glm::transpose(direction) * world), baked = basis(world);
        for (size_t i = 0; i < data.size(); ++i)
            for (int c = 0; c < 3; ++c) {
                double expected = 0, actual = baked[0] * dc.ptr<float>()[3 * i + c];
                for (int k = 1; k < 16; ++k) {
                    expected += native[k] * before.ptr<float>()[i * 45 + (k - 1) * 3 + c];
                    actual += baked[k] * after.ptr<float>()[i * 45 + (k - 1) * 3 + c];
                }
                ASSERT_NEAR(actual, expected, 2e-6) << "sample=" << sample << " splat=" << i;
            }
    }
}

TEST(SplatAffineSH, ExportDegreeLimitPrecedesMixingAndPreservesSource) {
    const auto source = fixture();
    const auto original_dc = source.sh0_raw().cpu().contiguous();
    const auto original_sh = source.shN_canonical().cpu().contiguous();
    const auto matrix = shear();
    for (int degree = 0; degree <= 3; ++degree) {
        auto expected = source.clone();
        expected.set_sh_degree(degree);
        transform(expected, matrix);
        auto exported = Scene::mergeSplatsWithTransforms({{&source, matrix}}, Scene::MergeStorageMode::Clone, degree);
        ASSERT_TRUE(exported);
        EXPECT_EQ(exported->get_max_sh_degree(), degree);
        const auto expected_dc = expected.sh0_raw().cpu().contiguous();
        const auto exported_dc = exported->sh0_raw().cpu().contiguous();
        for (size_t i = 0; i < exported_dc.numel(); ++i)
            EXPECT_NEAR(exported_dc.ptr<float>()[i], expected_dc.ptr<float>()[i], 1e-6);
        if (degree == 0)
            for (size_t i = 0; i < exported_dc.numel(); ++i)
                EXPECT_EQ(exported_dc.ptr<float>()[i], original_dc.ptr<float>()[i]);
        if (degree > 0) {
            const auto expected_sh = expected.shN_canonical().cpu().contiguous();
            const auto exported_sh = exported->shN_canonical().cpu().contiguous();
            ASSERT_EQ(exported_sh.numel(), expected_sh.numel());
            for (size_t i = 0; i < exported_sh.numel(); ++i)
                EXPECT_NEAR(exported_sh.ptr<float>()[i], expected_sh.ptr<float>()[i], 1e-6);
        }
    }
    EXPECT_EQ(source.get_max_sh_degree(), 3);
    const auto final_dc = source.sh0_raw().cpu().contiguous();
    const auto final_sh = source.shN_canonical().cpu().contiguous();
    for (size_t i = 0; i < final_dc.numel(); ++i)
        EXPECT_EQ(final_dc.ptr<float>()[i], original_dc.ptr<float>()[i]);
    for (size_t i = 0; i < final_sh.numel(); ++i)
        EXPECT_EQ(final_sh.ptr<float>()[i], original_sh.ptr<float>()[i]);
}

TEST(SplatAffineSH, IdentityBorrowDegreeLimitLeavesSourceStorageUnchanged) {
    const auto source = fixture();
    const auto original = source.shN_canonical().cpu().contiguous();
    auto exported = Scene::mergeSplatsWithTransforms({{&source, glm::mat4(1)}}, Scene::MergeStorageMode::BorrowSingleIdentity, 1);
    ASSERT_TRUE(exported);
    EXPECT_EQ(exported->get_max_sh_degree(), 1);
    EXPECT_EQ(source.get_max_sh_degree(), 3);
    const auto final = source.shN_canonical().cpu().contiguous();
    for (size_t i = 0; i < final.numel(); ++i)
        EXPECT_EQ(final.ptr<float>()[i], original.ptr<float>()[i]);
}

TEST(SplatAffineSH, InactiveStoredBandsCannotLeakIntoExportColors) {
    auto source = fixture();
    source.set_active_sh_degree(1);
    auto exported = Scene::mergeSplatsWithTransforms({{&source, shear()}}, Scene::MergeStorageMode::Clone, 3);
    ASSERT_TRUE(exported);
    EXPECT_EQ(exported->get_max_sh_degree(), 1);
    const auto dc = exported->sh0_raw().cpu().contiguous();
    for (size_t i = 0; i < dc.numel(); ++i)
        EXPECT_NEAR(dc.ptr<float>()[i], 0, 1e-7);
    EXPECT_EQ(source.get_max_sh_degree(), 3);
    EXPECT_EQ(source.get_active_sh_degree(), 1);
}

TEST(SplatAffineSH, Q16BorrowDegreeLimitPreservesSourceCodesAndBounds) {
    using namespace lfs::training;
    struct QuantGuard {
        QuantGuard() { sh_value::set_sh_value_quant_enabled_for_testing(true); }
        ~QuantGuard() { sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt); }
    } guard;
    auto source = fixture();
    ASSERT_TRUE(sh_value::apply_shN_value_quant(source));
    const auto codes = source.shN_raw().cpu().contiguous();
    const auto bounds = source.shN_value_bounds().cpu().contiguous();
    auto exported = Scene::mergeSplatsWithTransforms({{&source, glm::mat4(1)}}, Scene::MergeStorageMode::BorrowSingleIdentity, 1);
    ASSERT_TRUE(exported);
    EXPECT_EQ(exported->get_max_sh_degree(), 1);
    EXPECT_EQ(source.get_max_sh_degree(), 3);
    EXPECT_TRUE(source.shN_value_quantized());
    const auto final_codes = source.shN_raw().cpu().contiguous();
    const auto final_bounds = source.shN_value_bounds().cpu().contiguous();
    ASSERT_EQ(codes.bytes(), final_codes.bytes());
    ASSERT_EQ(bounds.bytes(), final_bounds.bytes());
    EXPECT_EQ(std::memcmp(codes.data_ptr(), final_codes.data_ptr(), codes.bytes()), 0);
    EXPECT_EQ(std::memcmp(bounds.data_ptr(), final_bounds.data_ptr(), bounds.bytes()), 0);
}

TEST(SplatAffineSH, MixedActiveDegreesMergeWithoutRevivingDormantBands) {
    auto first = fixture(), second = fixture();
    first.set_active_sh_degree(1);
    for (const auto matrix : {glm::mat4(1), shear()}) {
        auto merged = Scene::mergeSplatsWithTransforms({{&first, matrix}, {&second, matrix}}, Scene::MergeStorageMode::Clone, 2);
        ASSERT_TRUE(merged);
        EXPECT_EQ(merged->get_max_sh_degree(), 2);
        const auto dc = merged->sh0_raw().cpu().contiguous();
        const auto sh = merged->shN_canonical().cpu().contiguous();
        size_t offset = 0;
        for (const auto* source : {&first, &second}) {
            auto expected = source->clone();
            const int degree = std::min(2, source->get_active_sh_degree());
            expected.set_sh_degree(degree);
            transform(expected, matrix);
            const auto expected_dc = expected.sh0_raw().cpu().contiguous();
            const auto expected_sh = expected.shN_canonical().cpu().contiguous();
            const size_t coefficients = (degree + 1) * (degree + 1) - 1;
            for (size_t i = 0; i < expected.size(); ++i)
                for (size_t c = 0; c < 3; ++c) {
                    EXPECT_NEAR(dc.ptr<float>()[(offset + i) * 3 + c], expected_dc.ptr<float>()[i * 3 + c], 1e-6);
                    for (size_t k = 0; k < 8; ++k) {
                        const float value = k < coefficients ? expected_sh.ptr<float>()[(i * coefficients + k) * 3 + c] : 0;
                        EXPECT_NEAR(sh.ptr<float>()[((offset + i) * 8 + k) * 3 + c], value, 1e-6);
                    }
                }
            offset += expected.size();
        }
    }
    EXPECT_EQ(first.get_max_sh_degree(), 3);
    EXPECT_EQ(first.get_active_sh_degree(), 1);
    EXPECT_EQ(second.get_max_sh_degree(), 3);
}

TEST(SplatAffineSH, BakeTransformCopiesMixedBaseColorAndPreservesStorageKind) {
    using namespace lfs::training;
    struct QuantGuard {
        QuantGuard() { sh_value::set_sh_value_quant_enabled_for_testing(true); }
        ~QuantGuard() { sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt); }
    } guard;
    for (bool quantized : {false, true}) {
        auto source = fixture();
        if (quantized)
            ASSERT_TRUE(sh_value::apply_shN_value_quant(source));
        for (const auto matrix : {shear(), glm::scale(glm::mat4(1), glm::vec3(-1, 1, 1))}) {
            auto expected = source.clone();
            transform(expected, matrix);
            auto baked = source.clone();
            const auto result = lfs::vis::cap::bakeSplatTransformPreservingStorage(baked, matrix);
            ASSERT_TRUE(result.has_value());
            EXPECT_EQ(baked.shN_value_quantized(), quantized);
            const auto expected_dc = expected.sh0_raw().cpu().contiguous(), dc = baked.sh0_raw().cpu().contiguous();
            const auto expected_sh = expected.shN_canonical().cpu().contiguous(), sh = baked.shN_canonical().cpu().contiguous();
            for (size_t i = 0; i < dc.numel(); ++i)
                ASSERT_NEAR(dc.ptr<float>()[i], expected_dc.ptr<float>()[i], 1e-6);
            ASSERT_EQ(sh.numel(), expected_sh.numel());
            for (size_t i = 0; i < sh.numel(); ++i)
                ASSERT_NEAR(sh.ptr<float>()[i], expected_sh.ptr<float>()[i], quantized ? 1e-2 : 1e-6);
        }
    }
}
