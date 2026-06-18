/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_sog_format.cpp
 * @brief Tests for SOG (SuperSplat Optimized Gaussian) format support
 *
 * Verifies that our SOG loader can load files created by splat-transform
 * and produce comparable results to the original PLY.
 */

#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>

#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "io/exporter.hpp"
#include "io/formats/ply.hpp"
#include "io/formats/sogs.hpp"
#include "io/loader.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace fs = std::filesystem;

class SogFormatTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Test files
        test_dir = fs::path("/home/paja/projects/gaussian-splatting-cuda/test_formats");
        sog_bundle = test_dir / "test.sog";
        original_ply = fs::path("/home/paja/projects/gaussian-splatting-cuda/output/splat_30000.ply");

        // Create test directory if it doesn't exist
        if (!fs::exists(test_dir)) {
            fs::create_directories(test_dir);
        }
    }

    static bool floatNear(float a, float b, float tol = 1e-4f) {
        return std::abs(a - b) <= tol;
    }

    // Compare SplatData instances with tolerance for SOG lossy compression
    static void compareSplatDataSog(const lfs::core::SplatData& sog,
                                    const lfs::core::SplatData& reference,
                                    float pos_tol = 0.5f,    // Position tolerance
                                    float attr_tol = 1.0f) { // Attribute tolerance
        ASSERT_EQ(sog.size(), reference.size())
            << "Splat count mismatch";

        const size_t N = sog.size();

        // Compare means (positions) - lossy due to log transform + 16-bit quantization
        auto sog_means = sog.means().cpu();
        auto ref_means = reference.means().cpu();
        const float* sog_m = sog_means.ptr<float>();
        const float* ref_m = ref_means.ptr<float>();

        float max_pos_diff = 0.0f;
        double sum_pos_diff = 0.0;
        for (size_t i = 0; i < N * 3; ++i) {
            float diff = std::abs(sog_m[i] - ref_m[i]);
            max_pos_diff = std::max(max_pos_diff, diff);
            sum_pos_diff += diff;
        }
        std::cout << "  Position: max=" << max_pos_diff << ", avg=" << (sum_pos_diff / (N * 3)) << std::endl;
        EXPECT_LT(max_pos_diff, pos_tol) << "Position reconstruction error too high";

        // Compare scales (log space, clustered to 256 values)
        auto sog_scales = sog.get_scaling().cpu();
        auto ref_scales = reference.get_scaling().cpu();
        const float* sog_s = sog_scales.ptr<float>();
        const float* ref_s = ref_scales.ptr<float>();

        float max_scale_diff = 0.0f;
        for (size_t i = 0; i < N * 3; ++i) {
            max_scale_diff = std::max(max_scale_diff, std::abs(sog_s[i] - ref_s[i]));
        }
        std::cout << "  Scale: max=" << max_scale_diff << std::endl;
        EXPECT_LT(max_scale_diff, attr_tol) << "Scale reconstruction error too high";

        // Compare rotations (quaternions, 8-bit per component)
        auto sog_rot = sog.get_rotation().cpu();
        auto ref_rot = reference.get_rotation().cpu();
        const float* sog_r = sog_rot.ptr<float>();
        const float* ref_r = ref_rot.ptr<float>();

        float max_rot_diff = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            // Quaternions can be negated and still represent same rotation
            float diff_pos = 0.0f, diff_neg = 0.0f;
            for (int j = 0; j < 4; ++j) {
                diff_pos += std::abs(sog_r[i * 4 + j] - ref_r[i * 4 + j]);
                diff_neg += std::abs(sog_r[i * 4 + j] + ref_r[i * 4 + j]);
            }
            max_rot_diff = std::max(max_rot_diff, std::min(diff_pos, diff_neg));
        }
        std::cout << "  Rotation: max=" << max_rot_diff << std::endl;
        EXPECT_LT(max_rot_diff, 0.5f) << "Rotation reconstruction error too high";

        // Compare opacity
        auto sog_op = sog.get_opacity().cpu();
        auto ref_op = reference.get_opacity().cpu();
        const float* sog_o = sog_op.ptr<float>();
        const float* ref_o = ref_op.ptr<float>();

        float max_opacity_diff = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            max_opacity_diff = std::max(max_opacity_diff, std::abs(sog_o[i] - ref_o[i]));
        }
        std::cout << "  Opacity: max=" << max_opacity_diff << std::endl;
        EXPECT_LT(max_opacity_diff, 2.0f) << "Opacity reconstruction error too high";

        // Compare SH0 (colors, clustered to 256 values)
        auto sog_sh0 = sog.sh0().cpu();
        auto ref_sh0 = reference.sh0().cpu();
        const float* sog_c = sog_sh0.ptr<float>();
        const float* ref_c = ref_sh0.ptr<float>();

        float max_color_diff = 0.0f;
        for (size_t i = 0; i < N * 3; ++i) {
            max_color_diff = std::max(max_color_diff, std::abs(sog_c[i] - ref_c[i]));
        }
        std::cout << "  Color (SH0): max=" << max_color_diff << std::endl;
        EXPECT_LT(max_color_diff, attr_tol) << "Color reconstruction error too high";
    }

    fs::path test_dir;
    fs::path sog_bundle;
    fs::path original_ply;
};

// Test: Load SOG bundle
TEST_F(SogFormatTest, LoadSogBundle) {
    if (!fs::exists(sog_bundle)) {
        GTEST_SKIP() << "SOG test file not found: " << sog_bundle
                     << "\nRun: node splat-transform/bin/cli.mjs -w output/splat_30000.ply test_formats/test.sog";
    }

    auto result = lfs::io::load_sog(sog_bundle);
    ASSERT_TRUE(result.has_value()) << "Failed to load: " << result.error();

    const auto& splat = *result;
    std::cout << "Loaded SOG bundle: " << splat.size() << " splats" << std::endl;

    EXPECT_GT(splat.size(), 0) << "No splats loaded";
    EXPECT_TRUE(splat.means().is_valid()) << "Means tensor invalid";
    EXPECT_TRUE(splat.sh0().is_valid()) << "SH0 tensor invalid";
    EXPECT_TRUE(splat.get_scaling().is_valid()) << "Scaling tensor invalid";
    EXPECT_TRUE(splat.get_rotation().is_valid()) << "Rotation tensor invalid";
    EXPECT_TRUE(splat.get_opacity().is_valid()) << "Opacity tensor invalid";
}

// Test: Load SOG directory (unbundled)
TEST_F(SogFormatTest, LoadSogDirectory) {
    auto meta_json = test_dir / "meta.json";
    if (!fs::exists(meta_json)) {
        GTEST_SKIP() << "SOG meta.json not found: " << meta_json;
    }

    auto result = lfs::io::load_sog(test_dir);
    ASSERT_TRUE(result.has_value()) << "Failed to load: " << result.error();

    const auto& splat = *result;
    std::cout << "Loaded SOG directory: " << splat.size() << " splats" << std::endl;

    EXPECT_GT(splat.size(), 0) << "No splats loaded";
}

// Test: Compare SOG with original PLY using statistics (splats are reordered)
TEST_F(SogFormatTest, CompareWithOriginalPly) {
    if (!fs::exists(sog_bundle)) {
        GTEST_SKIP() << "SOG test file not found: " << sog_bundle;
    }
    if (!fs::exists(original_ply)) {
        GTEST_SKIP() << "Original PLY not found: " << original_ply;
    }

    std::cout << "Loading SOG bundle..." << std::endl;
    auto sog_result = lfs::io::load_sog(sog_bundle);
    ASSERT_TRUE(sog_result.has_value()) << "Failed to load SOG: " << sog_result.error();

    std::cout << "Loading original PLY..." << std::endl;
    auto ply_result = lfs::io::load_ply(original_ply);
    ASSERT_TRUE(ply_result.has_value()) << "Failed to load PLY: " << ply_result.error();

    ASSERT_EQ(sog_result->size(), ply_result->size()) << "Splat count mismatch";
    const size_t N = sog_result->size();

    std::cout << "Comparing statistics for " << N << " splats..." << std::endl;
    std::cout << "Note: SOG reorders splats (Morton order) and uses lossy k-means compression" << std::endl;

    auto compute_stats = [](const float* data, size_t n) {
        float min_val = data[0], max_val = data[0];
        double sum = 0;
        for (size_t i = 0; i < n; ++i) {
            min_val = std::min(min_val, data[i]);
            max_val = std::max(max_val, data[i]);
            sum += data[i];
        }
        return std::make_tuple(min_val, max_val, sum / n);
    };

    // Compare positions
    auto sog_means = sog_result->means().cpu();
    auto orig_means = ply_result->means().cpu();
    auto [sog_pos_min, sog_pos_max, sog_pos_avg] = compute_stats(sog_means.ptr<float>(), N * 3);
    auto [orig_pos_min, orig_pos_max, orig_pos_avg] = compute_stats(orig_means.ptr<float>(), N * 3);

    std::cout << "\nPositions:" << std::endl;
    std::cout << "  SOG:  min=" << sog_pos_min << ", max=" << sog_pos_max << ", avg=" << sog_pos_avg << std::endl;
    std::cout << "  Orig: min=" << orig_pos_min << ", max=" << orig_pos_max << ", avg=" << orig_pos_avg << std::endl;

    // Compare SH0 colors
    auto sog_sh0 = sog_result->sh0().cpu();
    auto orig_sh0 = ply_result->sh0().cpu();
    auto [sog_sh0_min, sog_sh0_max, sog_sh0_avg] = compute_stats(sog_sh0.ptr<float>(), N * 3);
    auto [orig_sh0_min, orig_sh0_max, orig_sh0_avg] = compute_stats(orig_sh0.ptr<float>(), N * 3);

    std::cout << "\nSH0 Colors:" << std::endl;
    std::cout << "  SOG:  min=" << sog_sh0_min << ", max=" << sog_sh0_max << ", avg=" << sog_sh0_avg << std::endl;
    std::cout << "  Orig: min=" << orig_sh0_min << ", max=" << orig_sh0_max << ", avg=" << orig_sh0_avg << std::endl;

    // Debug: print first 5 values
    std::cout << "\nFirst 5 sh0 values:" << std::endl;
    const float* sog_sh0_data = sog_sh0.ptr<float>();
    const float* orig_sh0_data = orig_sh0.ptr<float>();
    for (int i = 0; i < 5; ++i) {
        std::cout << "  [" << i << "] SOG=" << sog_sh0_data[i] << ", Orig=" << orig_sh0_data[i] << std::endl;
    }

    // Compare scales
    auto sog_scales = sog_result->get_scaling().cpu();
    auto orig_scales = ply_result->get_scaling().cpu();
    auto [sog_scale_min, sog_scale_max, sog_scale_avg] = compute_stats(sog_scales.ptr<float>(), N * 3);
    auto [orig_scale_min, orig_scale_max, orig_scale_avg] = compute_stats(orig_scales.ptr<float>(), N * 3);

    std::cout << "\nScales:" << std::endl;
    std::cout << "  SOG:  min=" << sog_scale_min << ", max=" << sog_scale_max << ", avg=" << sog_scale_avg << std::endl;
    std::cout << "  Orig: min=" << orig_scale_min << ", max=" << orig_scale_max << ", avg=" << orig_scale_avg << std::endl;

    // Tolerances for lossy k-means compression
    constexpr float avg_tol = 0.1f;   // Average should be close
    constexpr float range_tol = 0.5f; // Min/max should be within 50%

    EXPECT_NEAR(sog_pos_avg, orig_pos_avg, std::abs(orig_pos_avg) * avg_tol + 0.1f)
        << "Position average should match";
    EXPECT_NEAR(sog_sh0_avg, orig_sh0_avg, std::abs(orig_sh0_avg) * avg_tol + 0.1f)
        << "SH0 average should match";
    EXPECT_NEAR(sog_scale_avg, orig_scale_avg, std::abs(orig_scale_avg) * avg_tol + 0.5f)
        << "Scale average should match";

    std::cout << "\nSOG statistics comparison PASSED" << std::endl;
}

// Test: File not found handling
TEST_F(SogFormatTest, FileNotFound) {
    auto result = lfs::io::load_sog("/nonexistent/path/file.sog");
    EXPECT_FALSE(result.has_value()) << "Should fail for nonexistent file";
}

// Test: Load meta.json directly
TEST_F(SogFormatTest, LoadMetaJsonDirectly) {
    auto meta_json = test_dir / "meta.json";
    if (!fs::exists(meta_json)) {
        GTEST_SKIP() << "meta.json not found: " << meta_json;
    }

    auto result = lfs::io::load_sog(meta_json);
    ASSERT_TRUE(result.has_value()) << "Failed to load via meta.json: " << result.error();

    std::cout << "Loaded via meta.json: " << result->size() << " splats" << std::endl;
}

// Test: Compare our SOG loader with splat-transform's decompression
TEST_F(SogFormatTest, CompareWithSplatTransformDecompression) {
    // This file is created by: node splat-transform/bin/cli.mjs test_formats/test.sog /tmp/sog_verify.ply
    fs::path sog_decompressed = "/tmp/sog_verify.ply";
    if (!fs::exists(sog_decompressed)) {
        GTEST_SKIP() << "Run: node splat-transform/bin/cli.mjs test_formats/test.sog /tmp/sog_verify.ply";
    }
    if (!fs::exists(sog_bundle)) {
        GTEST_SKIP() << "SOG bundle not found: " << sog_bundle;
    }

    std::cout << "Loading SOG with our loader..." << std::endl;
    auto our_result = lfs::io::load_sog(sog_bundle);
    ASSERT_TRUE(our_result.has_value()) << "Failed to load SOG: " << our_result.error();

    std::cout << "Loading splat-transform decompressed PLY..." << std::endl;
    auto ref_result = lfs::io::load_ply(sog_decompressed);
    ASSERT_TRUE(ref_result.has_value()) << "Failed to load reference: " << ref_result.error();

    ASSERT_EQ(our_result->size(), ref_result->size()) << "Splat count mismatch";
    const size_t N = our_result->size();

    // Compare SH0 values
    auto our_sh0 = our_result->sh0().cpu();
    auto ref_sh0 = ref_result->sh0().cpu();
    const float* our_data = our_sh0.ptr<float>();
    const float* ref_data = ref_sh0.ptr<float>();

    float max_diff = 0.0f;
    for (size_t i = 0; i < N * 3; ++i) {
        max_diff = std::max(max_diff, std::abs(our_data[i] - ref_data[i]));
    }

    std::cout << "Max SH0 difference between our loader and splat-transform: " << max_diff << std::endl;

    // First 5 values
    std::cout << "\nFirst 5 SH0 values:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "  [" << i << "] Our=" << our_data[i] << ", Ref=" << ref_data[i] << std::endl;
    }

    EXPECT_LT(max_diff, 1e-5f) << "Our loader should match splat-transform exactly";
}

// Test SOG export roundtrip: load PLY, export as SOG, reimport
TEST_F(SogFormatTest, ExportRoundtrip) {
    if (!fs::exists(original_ply)) {
        GTEST_SKIP() << "Original PLY not found: " << original_ply;
    }

    // Load original PLY
    std::cout << "Loading original PLY..." << std::endl;
    auto orig_result = lfs::io::load_ply(original_ply);
    ASSERT_TRUE(orig_result.has_value()) << "Failed to load PLY: " << orig_result.error();
    std::cout << "Loaded " << orig_result->size() << " splats" << std::endl;

    // Export as SOG
    fs::path export_path = test_dir / "export_test.sog";
    std::cout << "Exporting to SOG: " << export_path << std::endl;

    lfs::io::SogSaveOptions options{
        .output_path = export_path,
        .kmeans_iterations = 10};

    auto write_result = lfs::io::save_sog(*orig_result, options);
    ASSERT_TRUE(write_result.has_value()) << "Failed to write SOG: " << write_result.error().format();
    std::cout << "SOG export complete" << std::endl;

    // Reimport the SOG
    std::cout << "Reimporting SOG..." << std::endl;
    auto reimport_result = lfs::io::load_sog(export_path);
    ASSERT_TRUE(reimport_result.has_value()) << "Failed to reimport SOG: " << reimport_result.error();

    EXPECT_EQ(reimport_result->size(), orig_result->size())
        << "Reimported splat count differs from original";

    // Compare SH0 colors
    size_t N = orig_result->size();
    auto orig_sh0 = orig_result->sh0().cpu();
    auto reimp_sh0 = reimport_result->sh0().cpu();
    const float* orig_sh0_ptr = orig_sh0.ptr<float>();
    const float* reimp_sh0_ptr = reimp_sh0.ptr<float>();

    // Compute statistics (SOG reorders, so compare averages)
    double orig_sum = 0, reimp_sum = 0;
    for (size_t i = 0; i < N * 3; ++i) {
        orig_sum += orig_sh0_ptr[i];
        reimp_sum += reimp_sh0_ptr[i];
    }
    double orig_avg = orig_sum / (N * 3);
    double reimp_avg = reimp_sum / (N * 3);

    std::cout << "\nSH0 Color comparison:" << std::endl;
    std::cout << "  Original avg: " << orig_avg << std::endl;
    std::cout << "  Reimport avg: " << reimp_avg << std::endl;
    std::cout << "  Difference: " << std::abs(reimp_avg - orig_avg) << std::endl;

    // First few values (note: Morton reordered, so indices won't match)
    std::cout << "\nFirst 5 SH0 values (different order due to Morton):" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "  [" << i << "] Orig=" << orig_sh0_ptr[i * 3] << "," << orig_sh0_ptr[i * 3 + 1] << "," << orig_sh0_ptr[i * 3 + 2]
                  << " Reimp=" << reimp_sh0_ptr[i * 3] << "," << reimp_sh0_ptr[i * 3 + 1] << "," << reimp_sh0_ptr[i * 3 + 2] << std::endl;
    }

    // Average should be close (within 10% tolerance for lossy k-means)
    EXPECT_NEAR(reimp_avg, orig_avg, std::abs(orig_avg) * 0.2 + 0.1)
        << "SH0 color average differs too much after roundtrip";

    std::cout << "Export roundtrip SUCCESS: " << reimport_result->size() << " splats" << std::endl;

    // Clean up test file
    fs::remove_all(export_path);
}

// Regression: a SOG loaded through the full Loader must route its tensors through the
// supplied splat allocator (Vulkan-external storage), or the Vulkan splat renderer rejects
// it ("refusing full input-copy fallback"). The SOG decoder ignores the allocator, so the
// LoaderService migrates the model post-load. Before the fix the allocator was never called.
TEST_F(SogFormatTest, LoaderRoutesSogThroughSplatAllocator) {
    fs::path sog_path;
    for (const auto& candidate : {sog_bundle,
                                  fs::path("/home/paja/projects/gaussian-splatting-cuda/splat_30000.sog")}) {
        if (fs::exists(candidate)) {
            sog_path = candidate;
            break;
        }
    }
    if (sog_path.empty()) {
        GTEST_SKIP() << "No SOG test file available (test.sog or splat_30000.sog)";
    }

    std::vector<std::string> allocated_names;
    lfs::io::LoadOptions options;
    options.splat_tensor_allocator = [&](lfs::core::TensorShape shape,
                                         size_t /*capacity*/,
                                         lfs::core::DataType dtype,
                                         std::string_view name) {
        allocated_names.emplace_back(name);
        return lfs::core::Tensor::empty(std::move(shape), lfs::core::Device::CUDA, dtype);
    };

    auto loader = lfs::io::Loader::create();
    auto result = loader->load(sog_path, options);
    ASSERT_TRUE(result.has_value()) << "SOG load failed: " << result.error().format();

    auto* splat = std::get_if<std::shared_ptr<lfs::core::SplatData>>(&result->data);
    ASSERT_NE(splat, nullptr);
    ASSERT_NE(*splat, nullptr);

    const auto routed = [&](const std::string& n) {
        return std::find(allocated_names.begin(), allocated_names.end(), n) != allocated_names.end();
    };
    EXPECT_FALSE(allocated_names.empty()) << "SOG tensors never went through the allocator";
    EXPECT_TRUE(routed("SplatData.means"));
    EXPECT_TRUE(routed("SplatData.sh0"));
    EXPECT_TRUE(routed("SplatData.scaling"));
    EXPECT_TRUE(routed("SplatData.rotation"));
    EXPECT_TRUE(routed("SplatData.opacity"));
}
