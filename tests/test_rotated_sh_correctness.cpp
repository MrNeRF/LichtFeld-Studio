/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "core/scene.hpp"
#include "core/splat_data.hpp"
#include "core/splat_data_transform.hpp"
#include "io/exporter.hpp"
#include "io/formats/ply.hpp"

namespace fs = std::filesystem;

namespace {

    constexpr float SH_C0 = 0.28209479177387814f;
    constexpr float SH_C1 = 0.48860251190291987f;
    constexpr float SH_C2_0 = 1.0925484305920792f;
    constexpr float SH_C2_1 = 0.94617469575755997f;
    constexpr float SH_C2_2 = 0.31539156525251999f;
    constexpr float SH_C2_3 = 0.54627421529603959f;

    constexpr float SH_C3_0 = 0.59004358992664352f;
    constexpr float SH_C3_1 = 2.8906114426405538f;
    constexpr float SH_C3_2 = 0.45704579946446572f;
    constexpr float SH_C3_3 = 0.3731763325901154f;
    constexpr float SH_C3_4 = 1.4453057213202769f;

    struct CpuShData {
        lfs::core::Tensor sh0;
        lfs::core::Tensor shN;
        const float* sh0_ptr = nullptr;
        const float* shN_ptr = nullptr;
        int rest_coeffs = 0;
        int degree = 0;
    };

    [[nodiscard]] glm::mat3 extract_rotation(const glm::mat4& transform) {
        glm::mat3 rot(transform);
        for (int i = 0; i < 3; ++i) {
            const float s = glm::length(rot[i]);
            if (s > 0.0f) {
                rot[i] /= s;
            }
        }
        return rot;
    }

    [[nodiscard]] CpuShData to_cpu_sh(const lfs::core::SplatData& data) {
        CpuShData out;
        out.degree = std::min(data.get_max_sh_degree(), 3);
        out.sh0 = data.sh0().contiguous().to(lfs::core::Device::CPU);
        out.sh0_ptr = out.sh0.ptr<float>();

        if (data.shN().is_valid()) {
            out.shN = data.shN().contiguous().to(lfs::core::Device::CPU);
            out.rest_coeffs = out.shN.ndim() >= 2 ? static_cast<int>(out.shN.size(1)) : 0;
            if (out.rest_coeffs > 0) {
                out.shN_ptr = out.shN.ptr<float>();
            }
        }
        return out;
    }

    [[nodiscard]] float eval_sh_channel(const CpuShData& sh, const size_t idx, const glm::vec3& dir, const int ch) {
        const glm::vec3 dir_n = glm::normalize(dir);
        const float x = dir_n.x;
        const float y = dir_n.y;
        const float z = dir_n.z;
        const float xx = x * x;
        const float yy = y * y;
        const float zz = z * z;
        const float xy = x * y;
        const float xz = x * z;
        const float yz = y * z;

        const float* const dc = sh.sh0_ptr + idx * 3;
        const float* const rest = (sh.shN_ptr != nullptr)
                                      ? sh.shN_ptr + idx * static_cast<size_t>(sh.rest_coeffs) * 3
                                      : nullptr;

        const auto coeff = [&](const int i) -> float {
            if (rest == nullptr || i < 0 || i >= sh.rest_coeffs) {
                return 0.0f;
            }
            return rest[i * 3 + ch];
        };

        float result = 0.5f + SH_C0 * dc[ch];

        if (sh.degree >= 1 && sh.rest_coeffs >= 3) {
            result += (-SH_C1 * y) * coeff(0) + (SH_C1 * z) * coeff(1) + (-SH_C1 * x) * coeff(2);
        }
        if (sh.degree >= 2 && sh.rest_coeffs >= 8) {
            result += (SH_C2_0 * xy) * coeff(3) + (-SH_C2_0 * yz) * coeff(4) +
                      (SH_C2_1 * zz - SH_C2_2) * coeff(5) + (-SH_C2_0 * xz) * coeff(6) +
                      (SH_C2_3 * (xx - yy)) * coeff(7);
        }
        if (sh.degree >= 3 && sh.rest_coeffs >= 15) {
            result += (SH_C3_0 * y * (-3.0f * xx + yy)) * coeff(8) +
                      (SH_C3_1 * xy * z) * coeff(9) +
                      (SH_C3_2 * y * (1.0f - 5.0f * zz)) * coeff(10) +
                      (SH_C3_3 * z * (5.0f * zz - 3.0f)) * coeff(11) +
                      (SH_C3_2 * x * (1.0f - 5.0f * zz)) * coeff(12) +
                      (SH_C3_4 * z * (xx - yy)) * coeff(13) +
                      (SH_C3_0 * x * (-xx + 3.0f * yy)) * coeff(14);
        }

        return result;
    }

    [[nodiscard]] glm::vec3 eval_sh_color(const CpuShData& sh, const size_t idx, const glm::vec3& dir) {
        return {
            eval_sh_channel(sh, idx, dir, 0),
            eval_sh_channel(sh, idx, dir, 1),
            eval_sh_channel(sh, idx, dir, 2)};
    }

    struct CoeffError {
        float mean = 0.0f;
        float max = 0.0f;
    };

    [[nodiscard]] CoeffError compare_sh_coefficients(const CpuShData& a, const CpuShData& b,
                                                     const size_t count, const size_t stride) {
        assert(a.shN_ptr && b.shN_ptr);
        assert(a.rest_coeffs == b.rest_coeffs);

        const int coeffs_per_gaussian = a.rest_coeffs * 3;
        double sum_abs = 0.0;
        float max_abs = 0.0f;
        size_t n = 0;

        for (size_t i = 0, used = 0; used < count; i += stride, ++used) {
            const float* pa = a.shN_ptr + i * coeffs_per_gaussian;
            const float* pb = b.shN_ptr + i * coeffs_per_gaussian;
            for (int j = 0; j < coeffs_per_gaussian; ++j) {
                const float diff = std::abs(pa[j] - pb[j]);
                sum_abs += static_cast<double>(diff);
                max_abs = std::max(max_abs, diff);
                ++n;
            }
        }
        return {n > 0 ? static_cast<float>(sum_abs / static_cast<double>(n)) : 0.0f, max_abs};
    }

    [[nodiscard]] lfs::core::SplatData clone_splat_data(const lfs::core::SplatData& src) {
        lfs::core::SplatData dst(
            src.get_max_sh_degree(),
            src.means_raw().clone(),
            src.sh0_raw().clone(),
            src.shN_raw().is_valid() ? src.shN_raw().clone() : lfs::core::Tensor(),
            src.scaling_raw().clone(),
            src.rotation_raw().clone(),
            src.opacity_raw().clone(),
            src.get_scene_scale());
        dst.set_active_sh_degree(src.get_active_sh_degree());
        return dst;
    }

} // namespace

class RotatedShCorrectnessTest : public ::testing::Test {
protected:
    fs::path bike_path = fs::path(PROJECT_ROOT_PATH) / "tests" / "data" / "bike.ply";
    fs::path temp_dir = fs::temp_directory_path() / "lfs_rotated_sh_export";

    void SetUp() override {
        fs::create_directories(temp_dir);
    }

    void TearDown() override {
        fs::remove_all(temp_dir);
    }
};

TEST_F(RotatedShCorrectnessTest, ExportedPlyPreservesRotatedShAppearance) {
    if (!fs::exists(bike_path)) {
        GTEST_SKIP() << "Missing test asset: " << bike_path;
    }

    auto loaded = lfs::io::load_ply(bike_path);
    ASSERT_TRUE(loaded.has_value()) << "Failed to load bike PLY: " << loaded.error();

    lfs::core::SplatData original = std::move(loaded.value());
    ASSERT_GT(original.size(), 0UL);
    ASSERT_TRUE(original.shN().is_valid());
    ASSERT_GE(original.get_max_sh_degree(), 1);

    const glm::mat4 rotation = glm::rotate(
        glm::mat4(1.0f), glm::radians(53.0f), glm::normalize(glm::vec3(0.37f, 0.82f, -0.44f)));
    const glm::mat4 translation = glm::translate(glm::mat4(1.0f), glm::vec3(0.4f, -0.3f, 1.2f));
    const glm::mat4 world_transform = translation * rotation;

    std::vector<std::pair<const lfs::core::SplatData*, glm::mat4>> splats;
    splats.emplace_back(&original, world_transform);

    auto merged = lfs::core::Scene::mergeSplatsWithTransforms(splats);
    ASSERT_NE(merged, nullptr);
    ASSERT_EQ(merged->size(), original.size());

    const fs::path out_path = temp_dir / "rotated_bike_export.ply";
    const auto save_result = lfs::io::save_ply(*merged, lfs::io::PlySaveOptions{
                                                            .output_path = out_path,
                                                            .binary = true,
                                                            .async = false});
    ASSERT_TRUE(save_result.has_value()) << save_result.error().message;

    auto exported_load = lfs::io::load_ply(out_path);
    ASSERT_TRUE(exported_load.has_value()) << "Failed to reload exported PLY: " << exported_load.error();
    lfs::core::SplatData exported = std::move(exported_load.value());
    ASSERT_EQ(exported.size(), original.size());

    const CpuShData original_sh = to_cpu_sh(original);
    const CpuShData exported_sh = to_cpu_sh(exported);
    ASSERT_GE(exported_sh.degree, 1);
    ASSERT_GE(exported_sh.rest_coeffs, 3);

    const glm::mat3 rot = extract_rotation(world_transform);
    const glm::mat3 rot_inv = glm::inverse(rot);

    const std::array<glm::vec3, 12> dirs_world = {
        glm::normalize(glm::vec3(1.0f, 0.0f, 0.0f)),
        glm::normalize(glm::vec3(-1.0f, 0.0f, 0.0f)),
        glm::normalize(glm::vec3(0.0f, 1.0f, 0.0f)),
        glm::normalize(glm::vec3(0.0f, -1.0f, 0.0f)),
        glm::normalize(glm::vec3(0.0f, 0.0f, 1.0f)),
        glm::normalize(glm::vec3(0.0f, 0.0f, -1.0f)),
        glm::normalize(glm::vec3(1.0f, 1.0f, 1.0f)),
        glm::normalize(glm::vec3(-1.0f, 1.0f, 1.0f)),
        glm::normalize(glm::vec3(1.0f, -1.0f, 1.0f)),
        glm::normalize(glm::vec3(1.0f, 1.0f, -1.0f)),
        glm::normalize(glm::vec3(0.2f, 0.7f, -0.6f)),
        glm::normalize(glm::vec3(-0.8f, 0.1f, 0.5f))};

    const size_t sample_count = std::min<size_t>(64, original.size());
    const size_t stride = std::max<size_t>(1, original.size() / sample_count);

    float max_abs_error = 0.0f;
    double sum_abs_error = 0.0;
    size_t n_compared = 0;

    for (size_t i = 0, used = 0; i < original.size() && used < sample_count; i += stride, ++used) {
        for (const auto& dir_world : dirs_world) {
            const glm::vec3 dir_local = glm::normalize(rot_inv * dir_world);
            const glm::vec3 ref_color = eval_sh_color(original_sh, i, dir_local);
            const glm::vec3 exported_color = eval_sh_color(exported_sh, i, dir_world);
            const glm::vec3 diff = glm::abs(ref_color - exported_color);

            max_abs_error = std::max(max_abs_error, std::max({diff.x, diff.y, diff.z}));
            sum_abs_error += static_cast<double>(diff.x + diff.y + diff.z);
            n_compared += 3;
        }
    }

    const float mean_abs_error = n_compared > 0
                                     ? static_cast<float>(sum_abs_error / static_cast<double>(n_compared))
                                     : 0.0f;

    EXPECT_LT(mean_abs_error, 1e-3f);
    EXPECT_LT(max_abs_error, 5e-3f);
}

TEST_F(RotatedShCorrectnessTest, ViewportParityWithExportUnderRotationAndNonUniformScale) {
    if (!fs::exists(bike_path)) {
        GTEST_SKIP() << "Missing test asset: " << bike_path;
    }

    auto loaded = lfs::io::load_ply(bike_path);
    ASSERT_TRUE(loaded.has_value()) << "Failed to load bike PLY: " << loaded.error();

    lfs::core::SplatData original = std::move(loaded.value());
    ASSERT_GT(original.size(), 0UL);
    ASSERT_TRUE(original.shN().is_valid());
    ASSERT_GE(original.get_max_sh_degree(), 1);

    const glm::mat4 non_uniform_scale = glm::scale(glm::mat4(1.0f), glm::vec3(1.7f, 0.6f, 1.35f));
    const glm::mat4 rotation = glm::rotate(
        glm::mat4(1.0f), glm::radians(31.0f), glm::normalize(glm::vec3(-0.53f, 0.41f, 0.74f)));
    const glm::mat4 translation = glm::translate(glm::mat4(1.0f), glm::vec3(-0.35f, 0.2f, 0.85f));
    const glm::mat4 world_transform = translation * rotation * non_uniform_scale;

    lfs::core::SplatData transformed(
        original.get_max_sh_degree(),
        original.means_raw().clone(),
        original.sh0_raw().clone(),
        original.shN_raw().is_valid() ? original.shN_raw().clone() : lfs::core::Tensor(),
        original.scaling_raw().clone(),
        original.rotation_raw().clone(),
        original.opacity_raw().clone(),
        original.get_scene_scale());
    transformed.set_active_sh_degree(original.get_active_sh_degree());
    ASSERT_NO_THROW(lfs::core::transform(transformed, world_transform));

    const CpuShData original_sh = to_cpu_sh(original);
    const CpuShData transformed_sh = to_cpu_sh(transformed);
    ASSERT_GE(transformed_sh.degree, 1);
    ASSERT_GE(transformed_sh.rest_coeffs, 3);

    const auto transformed_means_cpu = transformed.means().contiguous().to(lfs::core::Device::CPU);
    const float* const transformed_means_ptr = transformed_means_cpu.ptr<float>();

    const glm::mat3 rot = extract_rotation(world_transform);
    const glm::mat3 rot_inv = glm::inverse(rot);

    const std::array<glm::vec3, 5> camera_positions = {
        glm::vec3(2.2f, -1.1f, 0.7f),
        glm::vec3(-1.8f, 0.9f, 2.5f),
        glm::vec3(0.3f, 2.1f, -1.4f),
        glm::vec3(-2.6f, -1.9f, 1.2f),
        glm::vec3(1.1f, 0.5f, 3.0f)};

    const size_t sample_count = std::min<size_t>(64, original.size());
    const size_t stride = std::max<size_t>(1, original.size() / sample_count);

    float max_abs_error = 0.0f;
    double sum_abs_error = 0.0;
    size_t n_compared = 0;

    for (size_t i = 0, used = 0; i < original.size() && used < sample_count; i += stride, ++used) {
        const glm::vec3 mean_world(
            transformed_means_ptr[i * 3 + 0],
            transformed_means_ptr[i * 3 + 1],
            transformed_means_ptr[i * 3 + 2]);

        for (const auto& cam_world : camera_positions) {
            const glm::vec3 dir_world = mean_world - cam_world;
            const glm::vec3 dir_local_viewport = rot_inv * dir_world;
            const glm::vec3 viewport_color = eval_sh_color(original_sh, i, dir_local_viewport);
            const glm::vec3 export_color = eval_sh_color(transformed_sh, i, dir_world);
            const glm::vec3 diff = glm::abs(viewport_color - export_color);

            max_abs_error = std::max(max_abs_error, std::max({diff.x, diff.y, diff.z}));
            sum_abs_error += static_cast<double>(diff.x + diff.y + diff.z);
            n_compared += 3;
        }
    }

    const float mean_abs_error = n_compared > 0
                                     ? static_cast<float>(sum_abs_error / static_cast<double>(n_compared))
                                     : 0.0f;

    EXPECT_LT(mean_abs_error, 1e-3f);
    EXPECT_LT(max_abs_error, 5e-3f);
}

TEST_F(RotatedShCorrectnessTest, IdentityTransformPreservesShCoefficients) {
    if (!fs::exists(bike_path)) {
        GTEST_SKIP() << "Missing test asset: " << bike_path;
    }

    auto loaded = lfs::io::load_ply(bike_path);
    ASSERT_TRUE(loaded.has_value()) << "Failed to load bike PLY: " << loaded.error();

    lfs::core::SplatData original = std::move(loaded.value());
    ASSERT_TRUE(original.shN().is_valid());

    auto transformed = clone_splat_data(original);
    lfs::core::transform(transformed, glm::mat4(1.0f));

    const CpuShData orig_sh = to_cpu_sh(original);
    const CpuShData xform_sh = to_cpu_sh(transformed);
    ASSERT_EQ(orig_sh.rest_coeffs, xform_sh.rest_coeffs);

    const size_t sample_count = std::min<size_t>(64, original.size());
    const size_t stride = std::max<size_t>(1, original.size() / sample_count);
    const auto err = compare_sh_coefficients(orig_sh, xform_sh, sample_count, stride);

    EXPECT_EQ(err.max, 0.0f) << "Identity transform modified SH coefficients";
}

TEST_F(RotatedShCorrectnessTest, PureTranslationPreservesShCoefficients) {
    if (!fs::exists(bike_path)) {
        GTEST_SKIP() << "Missing test asset: " << bike_path;
    }

    auto loaded = lfs::io::load_ply(bike_path);
    ASSERT_TRUE(loaded.has_value()) << "Failed to load bike PLY: " << loaded.error();

    lfs::core::SplatData original = std::move(loaded.value());
    ASSERT_TRUE(original.shN().is_valid());

    const glm::mat4 translation = glm::translate(glm::mat4(1.0f), glm::vec3(5.0f, -3.0f, 7.0f));
    auto transformed = clone_splat_data(original);
    lfs::core::transform(transformed, translation);

    const CpuShData orig_sh = to_cpu_sh(original);
    const CpuShData xform_sh = to_cpu_sh(transformed);
    ASSERT_EQ(orig_sh.rest_coeffs, xform_sh.rest_coeffs);

    const size_t sample_count = std::min<size_t>(64, original.size());
    const size_t stride = std::max<size_t>(1, original.size() / sample_count);
    const auto err = compare_sh_coefficients(orig_sh, xform_sh, sample_count, stride);

    EXPECT_EQ(err.max, 0.0f) << "Pure translation modified SH coefficients";
}

TEST_F(RotatedShCorrectnessTest, DcComponentInvariantUnderRotation) {
    if (!fs::exists(bike_path)) {
        GTEST_SKIP() << "Missing test asset: " << bike_path;
    }

    auto loaded = lfs::io::load_ply(bike_path);
    ASSERT_TRUE(loaded.has_value()) << "Failed to load bike PLY: " << loaded.error();

    lfs::core::SplatData original = std::move(loaded.value());
    ASSERT_GT(original.size(), 0UL);

    const glm::mat4 rotation = glm::rotate(
        glm::mat4(1.0f), glm::radians(73.0f), glm::normalize(glm::vec3(0.3f, -0.5f, 0.8f)));
    auto transformed = clone_splat_data(original);
    lfs::core::transform(transformed, rotation);

    const auto orig_sh0 = original.sh0().contiguous().to(lfs::core::Device::CPU);
    const auto xform_sh0 = transformed.sh0().contiguous().to(lfs::core::Device::CPU);
    const float* orig_ptr = orig_sh0.ptr<float>();
    const float* xform_ptr = xform_sh0.ptr<float>();

    const size_t sample_count = std::min<size_t>(64, original.size());
    const size_t stride = std::max<size_t>(1, original.size() / sample_count);

    for (size_t i = 0, used = 0; used < sample_count; i += stride, ++used) {
        for (int ch = 0; ch < 3; ++ch) {
            EXPECT_EQ(orig_ptr[i * 3 + ch], xform_ptr[i * 3 + ch])
                << "DC coefficient changed at gaussian " << i << " channel " << ch;
        }
    }
}

TEST_F(RotatedShCorrectnessTest, RoundtripRotationRecoversSh) {
    if (!fs::exists(bike_path)) {
        GTEST_SKIP() << "Missing test asset: " << bike_path;
    }

    auto loaded = lfs::io::load_ply(bike_path);
    ASSERT_TRUE(loaded.has_value()) << "Failed to load bike PLY: " << loaded.error();

    lfs::core::SplatData original = std::move(loaded.value());
    ASSERT_TRUE(original.shN().is_valid());

    const glm::mat4 rotation = glm::rotate(
        glm::mat4(1.0f), glm::radians(73.0f), glm::normalize(glm::vec3(0.3f, -0.5f, 0.8f)));
    const glm::mat4 rotation_inv = glm::inverse(rotation);

    auto roundtripped = clone_splat_data(original);
    lfs::core::transform(roundtripped, rotation);
    lfs::core::transform(roundtripped, rotation_inv);

    const CpuShData orig_sh = to_cpu_sh(original);
    const CpuShData rt_sh = to_cpu_sh(roundtripped);
    ASSERT_EQ(orig_sh.rest_coeffs, rt_sh.rest_coeffs);

    const size_t sample_count = std::min<size_t>(64, original.size());
    const size_t stride = std::max<size_t>(1, original.size() / sample_count);
    const auto err = compare_sh_coefficients(orig_sh, rt_sh, sample_count, stride);

    EXPECT_LT(err.mean, 1e-3f) << "Roundtrip mean error too large";
    EXPECT_LT(err.max, 5e-3f) << "Roundtrip max error too large";
}

TEST_F(RotatedShCorrectnessTest, NinetyDegreeAxisAlignedRotation) {
    if (!fs::exists(bike_path)) {
        GTEST_SKIP() << "Missing test asset: " << bike_path;
    }

    auto loaded = lfs::io::load_ply(bike_path);
    ASSERT_TRUE(loaded.has_value()) << "Failed to load bike PLY: " << loaded.error();

    lfs::core::SplatData original = std::move(loaded.value());
    ASSERT_TRUE(original.shN().is_valid());
    ASSERT_GE(original.get_max_sh_degree(), 1);

    const std::array<glm::vec3, 3> axes = {
        glm::vec3(1, 0, 0), glm::vec3(0, 1, 0), glm::vec3(0, 0, 1)};

    const std::array<glm::vec3, 6> test_dirs = {
        glm::normalize(glm::vec3(1, 0, 0)),
        glm::normalize(glm::vec3(0, 1, 0)),
        glm::normalize(glm::vec3(0, 0, 1)),
        glm::normalize(glm::vec3(1, 1, 0)),
        glm::normalize(glm::vec3(0, 1, 1)),
        glm::normalize(glm::vec3(1, 0, 1))};

    const size_t sample_count = std::min<size_t>(64, original.size());
    const size_t stride = std::max<size_t>(1, original.size() / sample_count);

    for (const auto& axis : axes) {
        const glm::mat4 rot90 = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), axis);
        const glm::mat3 rot3 = extract_rotation(rot90);
        const glm::mat3 rot3_inv = glm::inverse(rot3);

        auto transformed = clone_splat_data(original);
        lfs::core::transform(transformed, rot90);

        const CpuShData orig_sh = to_cpu_sh(original);
        const CpuShData xform_sh = to_cpu_sh(transformed);

        float max_abs_error = 0.0f;
        double sum_abs_error = 0.0;
        size_t n_compared = 0;

        for (size_t i = 0, used = 0; used < sample_count; i += stride, ++used) {
            for (const auto& dir_world : test_dirs) {
                const glm::vec3 dir_local = glm::normalize(rot3_inv * dir_world);
                const glm::vec3 ref = eval_sh_color(orig_sh, i, dir_local);
                const glm::vec3 rot = eval_sh_color(xform_sh, i, dir_world);
                const glm::vec3 diff = glm::abs(ref - rot);

                max_abs_error = std::max(max_abs_error, std::max({diff.x, diff.y, diff.z}));
                sum_abs_error += static_cast<double>(diff.x + diff.y + diff.z);
                n_compared += 3;
            }
        }

        const float mean_abs_error = n_compared > 0
                                         ? static_cast<float>(sum_abs_error / static_cast<double>(n_compared))
                                         : 0.0f;

        EXPECT_LT(mean_abs_error, 1e-3f) << "90-degree rotation around axis ("
                                         << axis.x << "," << axis.y << "," << axis.z << ")";
        EXPECT_LT(max_abs_error, 5e-3f) << "90-degree rotation around axis ("
                                        << axis.x << "," << axis.y << "," << axis.z << ")";
    }
}

TEST_F(RotatedShCorrectnessTest, SequentialRotationsMatchComposed) {
    if (!fs::exists(bike_path)) {
        GTEST_SKIP() << "Missing test asset: " << bike_path;
    }

    auto loaded = lfs::io::load_ply(bike_path);
    ASSERT_TRUE(loaded.has_value()) << "Failed to load bike PLY: " << loaded.error();

    lfs::core::SplatData original = std::move(loaded.value());
    ASSERT_TRUE(original.shN().is_valid());

    const glm::mat4 r1 = glm::rotate(
        glm::mat4(1.0f), glm::radians(37.0f), glm::normalize(glm::vec3(1.0f, 0.0f, 0.0f)));
    const glm::mat4 r2 = glm::rotate(
        glm::mat4(1.0f), glm::radians(53.0f), glm::normalize(glm::vec3(0.0f, 0.7f, 0.7f)));

    auto sequential = clone_splat_data(original);
    lfs::core::transform(sequential, r1);
    lfs::core::transform(sequential, r2);

    auto composed = clone_splat_data(original);
    lfs::core::transform(composed, r2 * r1);

    const CpuShData seq_sh = to_cpu_sh(sequential);
    const CpuShData comp_sh = to_cpu_sh(composed);
    ASSERT_EQ(seq_sh.rest_coeffs, comp_sh.rest_coeffs);

    const size_t sample_count = std::min<size_t>(64, original.size());
    const size_t stride = std::max<size_t>(1, original.size() / sample_count);
    const auto err = compare_sh_coefficients(seq_sh, comp_sh, sample_count, stride);

    EXPECT_LT(err.mean, 2e-3f) << "Sequential vs composed mean error too large";
    EXPECT_LT(err.max, 1e-2f) << "Sequential vs composed max error too large";
}
