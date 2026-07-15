/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

#include "core/splat_data.hpp"
#include "io/exporter.hpp"
#include "io/formats/spz.hpp"
#include "io/loader.hpp"
#include "load-spz.h"

namespace fs = std::filesystem;
using namespace lfs::core;
using namespace lfs::io;

class SpzFormatTest : public ::testing::Test {
protected:
    static constexpr float EPSILON = 1e-4f;
    static constexpr float SPZ_TOLERANCE = 0.15f; // SPZ uses lossy 8-bit quantization

    const fs::path test_ply = fs::path(PROJECT_ROOT_PATH) / "windmill.ply";
    const fs::path temp_dir = fs::temp_directory_path() / "lfs_spz_test";

    void SetUp() override {
        fs::create_directories(temp_dir);
    }

    void TearDown() override {
        fs::remove_all(temp_dir);
    }

    // Create SplatData with known values for testing
    static SplatData create_test_splat(size_t num_points, int sh_degree) {
        constexpr int SH_COEFFS[] = {0, 3, 8, 15};
        const size_t sh_coeffs = sh_degree > 0 ? SH_COEFFS[sh_degree] : 0;

        auto means = Tensor::empty({num_points, 3}, Device::CPU, DataType::Float32);
        auto sh0 = Tensor::empty({num_points, 1, 3}, Device::CPU, DataType::Float32);
        auto scaling = Tensor::empty({num_points, 3}, Device::CPU, DataType::Float32);
        auto rotation = Tensor::empty({num_points, 4}, Device::CPU, DataType::Float32);
        auto opacity = Tensor::empty({num_points, 1}, Device::CPU, DataType::Float32);

        Tensor shN;
        if (sh_coeffs > 0) {
            shN = Tensor::empty({num_points, sh_coeffs, 3}, Device::CPU, DataType::Float32);
        }

        auto* means_ptr = static_cast<float*>(means.data_ptr());
        auto* sh0_ptr = static_cast<float*>(sh0.data_ptr());
        auto* scaling_ptr = static_cast<float*>(scaling.data_ptr());
        auto* rotation_ptr = static_cast<float*>(rotation.data_ptr());
        auto* opacity_ptr = static_cast<float*>(opacity.data_ptr());

        for (size_t i = 0; i < num_points; ++i) {
            // Positions: spread out in space
            means_ptr[i * 3 + 0] = static_cast<float>(i % 10);
            means_ptr[i * 3 + 1] = static_cast<float>((i / 10) % 10);
            means_ptr[i * 3 + 2] = static_cast<float>(i / 100);

            // SH0 colors: values in typical SH range [-2, 2]
            sh0_ptr[i * 3 + 0] = 0.5f + 0.1f * static_cast<float>(i % 5);
            sh0_ptr[i * 3 + 1] = 0.3f + 0.1f * static_cast<float>((i + 1) % 5);
            sh0_ptr[i * 3 + 2] = 0.4f + 0.1f * static_cast<float>((i + 2) % 5);

            // Scales: log scale in valid SPZ range [-10, 6]
            scaling_ptr[i * 3 + 0] = -3.0f + 0.01f * static_cast<float>(i % 100);
            scaling_ptr[i * 3 + 1] = -3.0f + 0.01f * static_cast<float>((i + 1) % 100);
            scaling_ptr[i * 3 + 2] = -3.0f + 0.01f * static_cast<float>((i + 2) % 100);

            // Rotation: nontrivial normalized quaternions (wxyz format)
            constexpr float inv_sqrt_two = 0.70710678118f;
            const std::array<std::array<float, 4>, 4> rotations = {{
                {1.0f, 0.0f, 0.0f, 0.0f},
                {inv_sqrt_two, inv_sqrt_two, 0.0f, 0.0f},
                {inv_sqrt_two, 0.0f, inv_sqrt_two, 0.0f},
                {inv_sqrt_two, 0.0f, 0.0f, inv_sqrt_two},
            }};
            std::copy(rotations[i % rotations.size()].begin(),
                      rotations[i % rotations.size()].end(),
                      rotation_ptr + i * 4);

            // Opacity: logit values in SPZ-safe range (avoids inf from sigmoid)
            opacity_ptr[i] = -2.0f + 0.04f * static_cast<float>(i % 100);
        }

        // Fill higher order SH if present
        if (sh_coeffs > 0) {
            auto* shN_ptr = static_cast<float*>(shN.data_ptr());
            for (size_t i = 0; i < num_points * sh_coeffs * 3; ++i) {
                shN_ptr[i] = 0.1f * static_cast<float>(static_cast<int>(i % 10) - 5);
            }
        }

        return SplatData(
            sh_degree,
            std::move(means),
            std::move(sh0),
            std::move(shN),
            std::move(scaling),
            std::move(rotation),
            std::move(opacity),
            0.5f);
    }

    static std::vector<uint8_t> make_spz_header(
        const uint32_t point_count,
        const uint8_t sh_degree,
        const uint8_t fractional_bits) {
        std::vector<uint8_t> bytes;
        const auto append_u32 = [&](const uint32_t value) {
            bytes.push_back(static_cast<uint8_t>(value));
            bytes.push_back(static_cast<uint8_t>(value >> 8));
            bytes.push_back(static_cast<uint8_t>(value >> 16));
            bytes.push_back(static_cast<uint8_t>(value >> 24));
        };
        append_u32(0x5053474e);
        append_u32(3);
        append_u32(point_count);
        bytes.push_back(sh_degree);
        bytes.push_back(fractional_bits);
        bytes.push_back(0);
        bytes.push_back(0);
        return bytes;
    }

    static bool write_gzipped_spz(
        const fs::path& path,
        const std::vector<uint8_t>& unpacked) {
        std::vector<uint8_t> compressed;
        if (!spz::compressGzipped(
                unpacked.data(), unpacked.size(), &compressed)) {
            return false;
        }
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        stream.write(reinterpret_cast<const char*>(compressed.data()),
                     static_cast<std::streamsize>(compressed.size()));
        return stream.good();
    }
};

TEST_F(SpzFormatTest, RejectsOversizedHeaderBeforePayloadAllocation) {
    const fs::path path = temp_dir / "oversized_header.spz";
    ASSERT_TRUE(write_gzipped_spz(
        path,
        make_spz_header(spz::kMaxSpzPoints + 1, 3, 12)));

    const auto result = load_spz(path);

    EXPECT_FALSE(result.has_value());
}

TEST_F(SpzFormatTest, RejectsInvalidFractionalBitsBeforePayloadAllocation) {
    const fs::path path = temp_dir / "invalid_fractional_bits.spz";
    ASSERT_TRUE(write_gzipped_spz(
        path,
        make_spz_header(1, 0, spz::kMaxSpzFractionalBits + 1)));

    const auto result = load_spz(path);

    EXPECT_FALSE(result.has_value());
}

TEST_F(SpzFormatTest, RejectsTruncatedPayloadBeforeAttributeAllocation) {
    const fs::path path = temp_dir / "truncated_payload.spz";
    ASSERT_TRUE(write_gzipped_spz(path, make_spz_header(1, 0, 12)));

    const auto result = load_spz(path);

    EXPECT_FALSE(result.has_value());
}

TEST_F(SpzFormatTest, ExtremeEncodedOpacityLoadsAsFinite) {
    auto original = create_test_splat(1, 0);
    original.opacity_raw().ptr<float>()[0] = 1000.0f;
    const fs::path path = temp_dir / "finite_opacity.spz";
    ASSERT_TRUE(save_spz(original, {.output_path = path}).has_value());

    const auto result = load_spz(path);

    ASSERT_TRUE(result.has_value()) << result.error();
    const auto opacity = result->opacity_raw().cpu();
    EXPECT_TRUE(std::isfinite(opacity.ptr<float>()[0]));
}

// CRITICAL: Verify sh0 tensor shape is [N, 1, 3] - this caught our color bug
TEST_F(SpzFormatTest, Sh0TensorShapeIsCorrect) {
    auto original = create_test_splat(100, 1);

    const fs::path spz_path = temp_dir / "shape_test.spz";
    ASSERT_TRUE(save_spz(original, {.output_path = spz_path}).has_value());

    auto loader = Loader::create();
    const auto result = loader->load(spz_path);
    ASSERT_TRUE(result.has_value());

    const auto* splat_ptr = std::get_if<std::shared_ptr<SplatData>>(&result->data);
    ASSERT_NE(splat_ptr, nullptr);
    const auto& loaded = **splat_ptr;

    // sh0 MUST be [N, 1, 3] to match PLY loader - wrong shape causes color corruption
    EXPECT_EQ(loaded.sh0().ndim(), 3);
    EXPECT_EQ(loaded.sh0().size(0), 100);
    EXPECT_EQ(loaded.sh0().size(1), 1);
    EXPECT_EQ(loaded.sh0().size(2), 3);
}

// Verify all tensor shapes match between original and loaded
TEST_F(SpzFormatTest, AllTensorShapesPreserved) {
    auto original = create_test_splat(100, 1);

    const fs::path spz_path = temp_dir / "shapes.spz";
    ASSERT_TRUE(save_spz(original, {.output_path = spz_path}).has_value());

    auto loader = Loader::create();
    const auto result = loader->load(spz_path);
    ASSERT_TRUE(result.has_value());

    const auto* splat_ptr = std::get_if<std::shared_ptr<SplatData>>(&result->data);
    ASSERT_NE(splat_ptr, nullptr);
    const auto& loaded = **splat_ptr;

    EXPECT_EQ(loaded.size(), original.size());
    EXPECT_EQ(loaded.get_max_sh_degree(), original.get_max_sh_degree());

    // Check all tensor dimensions match
    EXPECT_EQ(loaded.means().ndim(), original.means().ndim());
    EXPECT_EQ(loaded.sh0().ndim(), original.sh0().ndim());
    EXPECT_EQ(loaded.scaling_raw().ndim(), original.scaling_raw().ndim());
    EXPECT_EQ(loaded.rotation_raw().ndim(), original.rotation_raw().ndim());
    EXPECT_EQ(loaded.opacity_raw().ndim(), original.opacity_raw().ndim());

    if (original.shN().is_valid()) {
        EXPECT_TRUE(loaded.shN().is_valid());
        EXPECT_EQ(loaded.shN().ndim(), original.shN().ndim());
    }
}

// Roundtrip test: values should be preserved within SPZ compression tolerance
TEST_F(SpzFormatTest, RoundtripPreservesValues) {
    auto original = create_test_splat(100, 1);

    const fs::path spz_path = temp_dir / "roundtrip.spz";
    ASSERT_TRUE(save_spz(original, {.output_path = spz_path}).has_value());

    auto loader = Loader::create();
    const auto result = loader->load(spz_path);
    ASSERT_TRUE(result.has_value());

    const auto* splat_ptr = std::get_if<std::shared_ptr<SplatData>>(&result->data);
    ASSERT_NE(splat_ptr, nullptr);
    const auto& loaded = **splat_ptr;

    const auto orig_means = original.means().contiguous().to(Device::CPU);
    const auto orig_sh0 = original.sh0().contiguous().to(Device::CPU);
    const auto orig_shN = original.shN_raw().contiguous().to(Device::CPU);
    const auto orig_scaling = original.scaling_raw().contiguous().to(Device::CPU);
    const auto orig_opacity = original.opacity_raw().contiguous().to(Device::CPU);

    const auto load_means = loaded.means().contiguous().to(Device::CPU);
    const auto load_sh0 = loaded.sh0().contiguous().to(Device::CPU);
    const auto load_shN = loaded.shN_raw().contiguous().to(Device::CPU);
    const auto load_scaling = loaded.scaling_raw().contiguous().to(Device::CPU);
    const auto load_opacity = loaded.opacity_raw().contiguous().to(Device::CPU);

    const auto* orig_means_ptr = static_cast<const float*>(orig_means.data_ptr());
    const auto* orig_sh0_ptr = static_cast<const float*>(orig_sh0.data_ptr());
    const auto* orig_shN_ptr = static_cast<const float*>(orig_shN.data_ptr());
    const auto* orig_scaling_ptr = static_cast<const float*>(orig_scaling.data_ptr());
    const auto* orig_opacity_ptr = static_cast<const float*>(orig_opacity.data_ptr());

    const auto* load_means_ptr = static_cast<const float*>(load_means.data_ptr());
    const auto* load_sh0_ptr = static_cast<const float*>(load_sh0.data_ptr());
    const auto* load_shN_ptr = static_cast<const float*>(load_shN.data_ptr());
    const auto* load_scaling_ptr = static_cast<const float*>(load_scaling.data_ptr());
    const auto* load_opacity_ptr = static_cast<const float*>(load_opacity.data_ptr());

    // Check positions (24-bit fixed point, high precision)
    for (size_t i = 0; i < 100 * 3; ++i) {
        EXPECT_NEAR(load_means_ptr[i], orig_means_ptr[i], 0.01f) << "Position mismatch at " << i;
    }

    // Check SH0 colors (8-bit quantization)
    for (size_t i = 0; i < 100 * 3; ++i) {
        EXPECT_NEAR(load_sh0_ptr[i], orig_sh0_ptr[i], SPZ_TOLERANCE) << "SH0 mismatch at " << i;
    }

    ASSERT_EQ(load_shN.numel(), orig_shN.numel());
    for (size_t i = 0; i < orig_shN.numel(); ++i) {
        EXPECT_NEAR(load_shN_ptr[i], orig_shN_ptr[i], SPZ_TOLERANCE) << "SHN mismatch at " << i;
    }

    // Check scales (8-bit quantization, range [-10, 6])
    for (size_t i = 0; i < 100 * 3; ++i) {
        EXPECT_NEAR(load_scaling_ptr[i], orig_scaling_ptr[i], SPZ_TOLERANCE) << "Scale mismatch at " << i;
    }

    // Check opacity
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_NEAR(load_opacity_ptr[i], orig_opacity_ptr[i], SPZ_TOLERANCE) << "Opacity mismatch at " << i;
    }
}

// Verify rotation quaternion conversion (SPZ xyzw <-> SplatData wxyz)
TEST_F(SpzFormatTest, RotationQuaternionConversion) {
    auto original = create_test_splat(10, 0);

    const fs::path spz_path = temp_dir / "rotation.spz";
    ASSERT_TRUE(save_spz(original, {.output_path = spz_path}).has_value());

    auto loader = Loader::create();
    const auto result = loader->load(spz_path);
    ASSERT_TRUE(result.has_value());

    const auto* splat_ptr = std::get_if<std::shared_ptr<SplatData>>(&result->data);
    ASSERT_NE(splat_ptr, nullptr);
    const auto& loaded = **splat_ptr;

    const auto orig_rotation = original.rotation_raw().contiguous().to(Device::CPU);
    const auto load_rotation = loaded.rotation_raw().contiguous().to(Device::CPU);
    const auto* orig_rot_ptr = static_cast<const float*>(orig_rotation.data_ptr());
    const auto* load_rot_ptr = static_cast<const float*>(load_rotation.data_ptr());

    // Verify quaternions represent same rotation (q and -q are equivalent)
    for (size_t i = 0; i < 10; ++i) {
        const float dot = orig_rot_ptr[i * 4 + 0] * load_rot_ptr[i * 4 + 0] +
                          orig_rot_ptr[i * 4 + 1] * load_rot_ptr[i * 4 + 1] +
                          orig_rot_ptr[i * 4 + 2] * load_rot_ptr[i * 4 + 2] +
                          orig_rot_ptr[i * 4 + 3] * load_rot_ptr[i * 4 + 3];
        EXPECT_NEAR(std::abs(dot), 1.0f, SPZ_TOLERANCE) << "Rotation mismatch at point " << i;
    }
}

// Test SH degree 0 (no higher-order coefficients)
TEST_F(SpzFormatTest, ShDegree0) {
    auto original = create_test_splat(50, 0);

    const fs::path spz_path = temp_dir / "sh0.spz";
    ASSERT_TRUE(save_spz(original, {.output_path = spz_path}).has_value());

    auto loader = Loader::create();
    const auto result = loader->load(spz_path);
    ASSERT_TRUE(result.has_value());

    const auto* splat_ptr = std::get_if<std::shared_ptr<SplatData>>(&result->data);
    ASSERT_NE(splat_ptr, nullptr);
    const auto& loaded = **splat_ptr;

    EXPECT_EQ(loaded.get_max_sh_degree(), 0);
    // SplatData keeps a valid zero-length swizzled buffer at degree 0 so every
    // renderer can use one storage contract regardless of SH degree.
    EXPECT_TRUE(loaded.shN().is_valid());
    EXPECT_EQ(loaded.shN().numel(), 0);
}

// Test with real PLY file if available
TEST_F(SpzFormatTest, RealPlyRoundtrip) {
    if (!fs::exists(test_ply)) {
        GTEST_SKIP() << "windmill.ply not found";
    }

    auto loader = Loader::create();

    // Load PLY
    const auto ply_result = loader->load(test_ply);
    ASSERT_TRUE(ply_result.has_value());
    const auto* ply_ptr = std::get_if<std::shared_ptr<SplatData>>(&ply_result->data);
    ASSERT_NE(ply_ptr, nullptr);
    const auto& ply_splat = **ply_ptr;

    // Export to SPZ
    const fs::path spz_path = temp_dir / "real.spz";
    ASSERT_TRUE(save_spz(ply_splat, {.output_path = spz_path}).has_value());

    // Load SPZ
    const auto spz_result = loader->load(spz_path);
    ASSERT_TRUE(spz_result.has_value());
    const auto* spz_ptr = std::get_if<std::shared_ptr<SplatData>>(&spz_result->data);
    ASSERT_NE(spz_ptr, nullptr);
    const auto& spz_splat = **spz_ptr;

    // Verify structure matches
    EXPECT_EQ(spz_splat.size(), ply_splat.size());
    EXPECT_EQ(spz_splat.get_max_sh_degree(), ply_splat.get_max_sh_degree());

    // CRITICAL: sh0 shape must match PLY loader
    EXPECT_EQ(spz_splat.sh0().ndim(), ply_splat.sh0().ndim());
    EXPECT_EQ(spz_splat.sh0().size(1), ply_splat.sh0().size(1)); // Must be 1
}
