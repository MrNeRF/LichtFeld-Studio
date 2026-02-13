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

