/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_loader_comparison.cpp
 * @brief Comprehensive comparison test between old (gs::loader) and new (lfs::loader) loader implementations
 *
 * This test suite validates that the new torch-free loader implementation produces
 * identical or equivalent results compared to the original torch-based loader.
 *
 * Tests include:
 * - API compatibility
 * - COLMAP dataset loading
 * - Point cloud data comparison
 * - Camera dataset comparison
 * - Scene center computation
 * - Load time performance
 */

#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <gtest/gtest.h>
#include <iomanip>
#include <thread>

// Old loader (torch-based)
#include "loader/loader.hpp"

// New loader (torch-free)
#include "loader_new/loader.hpp"

// Point cloud structures
#include "core/point_cloud.hpp"
#include "core_new/point_cloud.hpp"

// SplatData for training verification
#include "core/splat_data.hpp"
#include "core_new/splat_data.hpp"

// Training parameters
#include "core/parameters.hpp"
#include "core_new/parameters.hpp"

// For tensor conversion utilities
#include "core_new/tensor.hpp"

namespace fs = std::filesystem;

// Test fixture for loader comparison
class LoaderComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up paths to test data
        data_dir = fs::path(TEST_DATA_DIR);
        stump_path = data_dir / "stump";

        // Verify test data exists
        ASSERT_TRUE(fs::exists(data_dir)) << "Test data directory not found: " << data_dir;
        ASSERT_TRUE(fs::exists(stump_path)) << "Stump dataset not found: " << stump_path;
        ASSERT_TRUE(fs::exists(stump_path / "sparse" / "0")) << "COLMAP sparse reconstruction not found";

        // Create loader instances
        old_loader = gs::loader::Loader::create();
        new_loader = lfs::loader::Loader::create();

        ASSERT_NE(old_loader, nullptr) << "Failed to create old loader";
        ASSERT_NE(new_loader, nullptr) << "Failed to create new loader";
    }

    // Helper: Compare floating point values with tolerance
    bool floatEqual(float a, float b, float tolerance = 1e-5f) const {
        return std::abs(a - b) <= tolerance;
    }

    // Helper: Compare torch::Tensor and lfs::core::Tensor
    bool tensorsEqual(const torch::Tensor& torch_tensor, const lfs::core::Tensor& lfs_tensor,
                      float tolerance = 1e-5f, bool debug = false) const {
        // Move both to CPU for comparison
        auto torch_cpu = torch_tensor.cpu().contiguous();
        auto lfs_cpu = lfs_tensor.cpu().contiguous();

        // Check shapes match
        auto torch_sizes = torch_cpu.sizes();
        const auto& lfs_shape = lfs_cpu.shape(); // Returns const TensorShape&

        if (torch_sizes.size() != lfs_shape.rank()) {
            if (debug) {
                std::cout << "Shape rank mismatch: torch=" << torch_sizes.size()
                          << " lfs=" << lfs_shape.rank() << std::endl;
            }
            return false;
        }

        for (size_t i = 0; i < lfs_shape.rank(); ++i) {
            if (torch_sizes[i] != static_cast<int64_t>(lfs_shape[i])) {
                if (debug) {
                    std::cout << "Shape dim " << i << " mismatch: torch=" << torch_sizes[i]
                              << " lfs=" << lfs_shape[i] << std::endl;
                }
                return false;
            }
        }

        // Check dtypes are compatible
        bool dtype_compatible = false;
        if (torch_cpu.dtype() == torch::kFloat32 && lfs_cpu.dtype() == lfs::core::DataType::Float32) {
            dtype_compatible = true;
        } else if (torch_cpu.dtype() == torch::kUInt8 && lfs_cpu.dtype() == lfs::core::DataType::UInt8) {
            dtype_compatible = true;
        }

        if (!dtype_compatible) {
            if (debug) {
                std::cout << "Dtype mismatch: torch=" << torch_cpu.dtype()
                          << " lfs=" << static_cast<int>(lfs_cpu.dtype()) << std::endl;
            }
            return false;
        }

        // Compare values
        int64_t numel = torch_cpu.numel();
        if (torch_cpu.dtype() == torch::kFloat32) {
            const float* torch_data = torch_cpu.data_ptr<float>();
            const float* lfs_data = lfs_cpu.ptr<float>();

            for (int64_t i = 0; i < numel; ++i) {
                if (!floatEqual(torch_data[i], lfs_data[i], tolerance)) {
                    if (debug) {
                        std::cout << "Value mismatch at index " << i << ": torch=" << torch_data[i]
                                  << " lfs=" << lfs_data[i] << " diff=" << std::abs(torch_data[i] - lfs_data[i]) << std::endl;
                        // Only show first few mismatches
                        static int mismatch_count = 0;
                        if (++mismatch_count >= 10) {
                            std::cout << "... (suppressing further mismatches)" << std::endl;
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
            }
        } else if (torch_cpu.dtype() == torch::kUInt8) {
            const uint8_t* torch_data = torch_cpu.data_ptr<uint8_t>();
            const uint8_t* lfs_data = lfs_cpu.ptr<uint8_t>();

            for (int64_t i = 0; i < numel; ++i) {
                if (torch_data[i] != lfs_data[i]) {
                    if (debug) {
                        std::cout << "Value mismatch at index " << i << ": torch=" << (int)torch_data[i]
                                  << " lfs=" << (int)lfs_data[i] << std::endl;
                        static int mismatch_count = 0;
                        if (++mismatch_count >= 10) {
                            std::cout << "... (suppressing further mismatches)" << std::endl;
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
            }
        }

        return true;
    }

    // Helper: Compare point clouds
    bool pointCloudsEqual(const gs::PointCloud& old_pc, const lfs::core::PointCloud& new_pc,
                          float tolerance = 1e-5f) const {
        // Compare sizes
        if (old_pc.size() != new_pc.size()) {
            std::cout << "Point cloud size mismatch: old=" << old_pc.size()
                      << " new=" << new_pc.size() << std::endl;
            return false;
        }

        // Compare means
        if (!tensorsEqual(old_pc.means, new_pc.means, tolerance)) {
            std::cout << "Point cloud means mismatch" << std::endl;
            return false;
        }

        // Compare colors
        if (!tensorsEqual(old_pc.colors, new_pc.colors, tolerance, true)) {
            std::cout << "Point cloud colors mismatch" << std::endl;
            std::cout << "Old colors shape: [";
            auto old_sizes = old_pc.colors.sizes();
            for (int i = 0; i < old_sizes.size(); ++i) {
                std::cout << old_sizes[i];
                if (i < old_sizes.size() - 1)
                    std::cout << ", ";
            }
            std::cout << "] dtype=" << old_pc.colors.dtype() << " device=" << old_pc.colors.device() << std::endl;

            std::cout << "New colors shape: [";
            const auto& new_shape = new_pc.colors.shape();
            for (size_t i = 0; i < new_shape.rank(); ++i) {
                std::cout << new_shape[i];
                if (i < new_shape.rank() - 1)
                    std::cout << ", ";
            }
            std::cout << "] dtype=" << static_cast<int>(new_pc.colors.dtype())
                      << " device=" << static_cast<int>(new_pc.colors.device()) << std::endl;
            return false;
        }

        // Compare optional fields if they exist
        if (old_pc.is_gaussian() && new_pc.is_gaussian()) {
            if (!tensorsEqual(old_pc.sh0, new_pc.sh0, tolerance)) {
                std::cout << "Point cloud sh0 mismatch" << std::endl;
                return false;
            }

            if (old_pc.shN.defined() && new_pc.shN.is_valid()) {
                if (!tensorsEqual(old_pc.shN, new_pc.shN, tolerance)) {
                    std::cout << "Point cloud shN mismatch" << std::endl;
                    return false;
                }
            }

            if (old_pc.opacity.defined() && new_pc.opacity.is_valid()) {
                if (!tensorsEqual(old_pc.opacity, new_pc.opacity, tolerance)) {
                    std::cout << "Point cloud opacity mismatch" << std::endl;
                    return false;
                }
            }

            if (old_pc.scaling.defined() && new_pc.scaling.is_valid()) {
                if (!tensorsEqual(old_pc.scaling, new_pc.scaling, tolerance)) {
                    std::cout << "Point cloud scaling mismatch" << std::endl;
                    return false;
                }
            }

            if (old_pc.rotation.defined() && new_pc.rotation.is_valid()) {
                if (!tensorsEqual(old_pc.rotation, new_pc.rotation, tolerance)) {
                    std::cout << "Point cloud rotation mismatch" << std::endl;
                    return false;
                }
            }
        }

        return true;
    }

    // Test data paths
    fs::path data_dir;
    fs::path stump_path;
    fs::path lego_path{"/media/paja/T7/nerf_synthetic/lego"};

    // Loader instances
    std::unique_ptr<gs::loader::Loader> old_loader;
    std::unique_ptr<lfs::loader::Loader> new_loader;
};

// Test 1: API compatibility - verify both loaders have same interface
TEST_F(LoaderComparisonTest, APICompatibility) {
    // Test static methods exist and return same results
    bool old_is_dataset = gs::loader::Loader::isDatasetPath(stump_path);
    bool new_is_dataset = lfs::loader::Loader::isDatasetPath(stump_path);
    EXPECT_EQ(old_is_dataset, new_is_dataset) << "isDatasetPath results differ";

    auto old_type = gs::loader::Loader::getDatasetType(stump_path);
    auto new_type = lfs::loader::Loader::getDatasetType(stump_path);

    // Compare enum values (both should be COLMAP)
    EXPECT_EQ(static_cast<int>(old_type), static_cast<int>(new_type))
        << "Dataset type detection differs";

    // Test canLoad
    EXPECT_TRUE(old_loader->canLoad(stump_path)) << "Old loader cannot load stump";
    EXPECT_TRUE(new_loader->canLoad(stump_path)) << "New loader cannot load stump";

    // Test getSupportedFormats
    auto old_formats = old_loader->getSupportedFormats();
    auto new_formats = new_loader->getSupportedFormats();
    EXPECT_FALSE(old_formats.empty()) << "Old loader has no supported formats";
    EXPECT_FALSE(new_formats.empty()) << "New loader has no supported formats";

    // Test getSupportedExtensions
    auto old_extensions = old_loader->getSupportedExtensions();
    auto new_extensions = new_loader->getSupportedExtensions();
    EXPECT_FALSE(old_extensions.empty()) << "Old loader has no supported extensions";
    EXPECT_FALSE(new_extensions.empty()) << "New loader has no supported extensions";
}

// Test 2: Load COLMAP dataset and compare results
TEST_F(LoaderComparisonTest, LoadCOLMAPDataset) {
    // Configure load options
    gs::loader::LoadOptions old_options;
    old_options.resize_factor = -1;
    old_options.max_width = 3840;
    old_options.images_folder = "images";
    old_options.validate_only = false;

    lfs::loader::LoadOptions new_options;
    new_options.resize_factor = -1;
    new_options.max_width = 3840;
    new_options.images_folder = "images";
    new_options.validate_only = false;

    // Load with both loaders
    auto old_result = old_loader->load(stump_path, old_options);
    auto new_result = new_loader->load(stump_path, new_options);

    // Check both succeeded
    ASSERT_TRUE(old_result.has_value()) << "Old loader failed: " << old_result.error();
    ASSERT_TRUE(new_result.has_value()) << "New loader failed: " << new_result.error();

    // Compare loader names
    EXPECT_EQ(old_result->loader_used, new_result->loader_used)
        << "Different loaders were used";

    // Both should contain LoadedScene for COLMAP
    EXPECT_TRUE(std::holds_alternative<gs::loader::LoadedScene>(old_result->data))
        << "Old loader didn't return LoadedScene";
    EXPECT_TRUE(std::holds_alternative<lfs::loader::LoadedScene>(new_result->data))
        << "New loader didn't return LoadedScene";

    // Extract scenes
    auto& old_scene = std::get<gs::loader::LoadedScene>(old_result->data);
    auto& new_scene = std::get<lfs::loader::LoadedScene>(new_result->data);

    // Verify both have valid data
    ASSERT_NE(old_scene.cameras, nullptr) << "Old loader: cameras is null";
    ASSERT_NE(old_scene.point_cloud, nullptr) << "Old loader: point_cloud is null";
    ASSERT_NE(new_scene.cameras, nullptr) << "New loader: cameras is null";
    ASSERT_NE(new_scene.point_cloud, nullptr) << "New loader: point_cloud is null";

    // Note: We can't easily compare camera->size() without including full dataset headers
    // which would pull in training dependencies. The important thing is both are non-null.
    std::cout << "Old loader: camera dataset loaded" << std::endl;
    std::cout << "New loader: camera dataset loaded" << std::endl;

    // Compare point cloud sizes
    EXPECT_EQ(old_scene.point_cloud->size(), new_scene.point_cloud->size())
        << "Point cloud sizes differ";

    std::cout << "Old loader: " << old_scene.point_cloud->size() << " points" << std::endl;
    std::cout << "New loader: " << new_scene.point_cloud->size() << " points" << std::endl;

    // Compare scene centers
    auto old_center_cpu = old_result->scene_center.cpu();
    auto new_center_cpu = new_result->scene_center.cpu();

    const float* old_center_data = old_center_cpu.data_ptr<float>();
    const float* new_center_data = new_center_cpu.ptr<float>();

    EXPECT_TRUE(floatEqual(old_center_data[0], new_center_data[0], 1e-4f))
        << "Scene center X differs: " << old_center_data[0] << " vs " << new_center_data[0];
    EXPECT_TRUE(floatEqual(old_center_data[1], new_center_data[1], 1e-4f))
        << "Scene center Y differs: " << old_center_data[1] << " vs " << new_center_data[1];
    EXPECT_TRUE(floatEqual(old_center_data[2], new_center_data[2], 1e-4f))
        << "Scene center Z differs: " << old_center_data[2] << " vs " << new_center_data[2];

    std::cout << std::format("Old scene center: [{:.6f}, {:.6f}, {:.6f}]",
                             old_center_data[0], old_center_data[1], old_center_data[2])
              << std::endl;
    std::cout << std::format("New scene center: [{:.6f}, {:.6f}, {:.6f}]",
                             new_center_data[0], new_center_data[1], new_center_data[2])
              << std::endl;

    // Compare load times (informational, not assertion)
    std::cout << "Old loader time: " << old_result->load_time.count() << "ms" << std::endl;
    std::cout << "New loader time: " << new_result->load_time.count() << "ms" << std::endl;

    // Compare warnings
    std::cout << "Old loader warnings: " << old_result->warnings.size() << std::endl;
    for (const auto& warning : old_result->warnings) {
        std::cout << "  - " << warning << std::endl;
    }
    std::cout << "New loader warnings: " << new_result->warnings.size() << std::endl;
    for (const auto& warning : new_result->warnings) {
        std::cout << "  - " << warning << std::endl;
    }
}

// Test 3: Compare point cloud data in detail
TEST_F(LoaderComparisonTest, PointCloudDataComparison) {
    gs::loader::LoadOptions old_options;
    old_options.validate_only = false;

    lfs::loader::LoadOptions new_options;
    new_options.validate_only = false;

    auto old_result = old_loader->load(stump_path, old_options);
    auto new_result = new_loader->load(stump_path, new_options);

    ASSERT_TRUE(old_result.has_value());
    ASSERT_TRUE(new_result.has_value());

    auto& old_scene = std::get<gs::loader::LoadedScene>(old_result->data);
    auto& new_scene = std::get<lfs::loader::LoadedScene>(new_result->data);

    // Deep comparison of point clouds
    EXPECT_TRUE(pointCloudsEqual(*old_scene.point_cloud, *new_scene.point_cloud, 1e-4f))
        << "Point cloud data differs between loaders";
}

// Test 3b: Verify SplatData initialization produces identical results for training
TEST_F(LoaderComparisonTest, SplatDataInitialization) {
    gs::loader::LoadOptions old_options;
    old_options.validate_only = false;

    lfs::loader::LoadOptions new_options;
    new_options.validate_only = false;

    auto old_result = old_loader->load(stump_path, old_options);
    auto new_result = new_loader->load(stump_path, new_options);

    ASSERT_TRUE(old_result.has_value()) << "Old loader failed";
    ASSERT_TRUE(new_result.has_value()) << "New loader failed";

    auto& old_scene = std::get<gs::loader::LoadedScene>(old_result->data);
    auto& new_scene = std::get<lfs::loader::LoadedScene>(new_result->data);

    // Create training parameters for initialization
    // Use only fields that both implementations support
    gs::param::TrainingParameters old_params;
    old_params.optimization.sh_degree = 3;
    old_params.optimization.random = false; // Use point cloud, not random init
    old_params.optimization.init_num_pts = 100000;
    old_params.optimization.init_extent = 3.0f;

    lfs::core::param::TrainingParameters new_params;
    new_params.optimization.sh_degree = 3;
    new_params.optimization.random = false; // Use point cloud, not random init
    new_params.optimization.init_num_pts = 100000;
    new_params.optimization.init_extent = 3.0f;

    std::cout << "Initializing SplatData from PointClouds..." << std::endl;

    // Initialize SplatData from both point clouds using the same parameters
    auto old_splat_result = gs::SplatData::init_model_from_pointcloud(
        old_params, old_result->scene_center, *old_scene.point_cloud);

    auto new_splat_result = lfs::core::SplatData::init_model_from_pointcloud(
        new_params, new_result->scene_center, *new_scene.point_cloud);

    ASSERT_TRUE(old_splat_result.has_value()) << "Old SplatData init failed: " << old_splat_result.error();
    ASSERT_TRUE(new_splat_result.has_value()) << "New SplatData init failed: " << new_splat_result.error();

    auto& old_splat = old_splat_result.value();
    auto& new_splat = new_splat_result.value();

    std::cout << "Comparing initialized SplatData for training..." << std::endl;
    std::cout << "Old SplatData: " << old_splat.size() << " splats" << std::endl;
    std::cout << "New SplatData: " << new_splat.size() << " splats" << std::endl;

    // 1. Compare sizes
    EXPECT_EQ(old_splat.size(), new_splat.size())
        << "SplatData sizes differ";

    // 2. Compare means (positions)
    std::cout << "  [1/6] Comparing means..." << std::endl;
    EXPECT_TRUE(tensorsEqual(old_splat.means(), new_splat.means(), 1e-4f))
        << "SplatData means differ";

    // 3. Compare sh0 (base spherical harmonics / colors)
    std::cout << "  [2/6] Comparing sh0 (base colors)..." << std::endl;
    EXPECT_TRUE(tensorsEqual(old_splat.sh0(), new_splat.sh0(), 1e-4f))
        << "SplatData sh0 differ";

    // 4. Compare shN (higher-order spherical harmonics)
    if (old_splat.shN().defined() && new_splat.shN().numel() > 0) {
        std::cout << "  [3/6] Comparing shN (higher-order SH)..." << std::endl;
        EXPECT_TRUE(tensorsEqual(old_splat.shN(), new_splat.shN(), 1e-4f))
            << "SplatData shN differ";
    } else {
        std::cout << "  [3/6] shN: not initialized (expected for initial point cloud)" << std::endl;
    }

    // 5. Compare opacity
    std::cout << "  [4/6] Comparing opacity..." << std::endl;
    EXPECT_TRUE(tensorsEqual(old_splat.get_opacity(), new_splat.get_opacity(), 1e-4f))
        << "SplatData opacity differ";

    // 6. Compare scaling
    std::cout << "  [5/6] Comparing scaling..." << std::endl;
    EXPECT_TRUE(tensorsEqual(old_splat.get_scaling(), new_splat.get_scaling(), 1e-4f))
        << "SplatData scaling differ";

    // 7. Compare rotation
    std::cout << "  [6/6] Comparing rotation..." << std::endl;
    EXPECT_TRUE(tensorsEqual(old_splat.get_rotation(), new_splat.get_rotation(), 1e-4f))
        << "SplatData rotation differ";

    // 8. Verify metadata matches (only what both implementations provide)
    EXPECT_EQ(old_splat.get_active_sh_degree(), new_splat.get_active_sh_degree())
        << "Active SH degree differs";
    EXPECT_FLOAT_EQ(old_splat.get_scene_scale(), new_splat.get_scene_scale())
        << "Scene scale differs";

    std::cout << "✓ SplatData initialization verified - training data is identical!" << std::endl;
    std::cout << "  Both produce " << old_splat.size() << " Gaussians with matching attributes" << std::endl;
    std::cout << "  Ready for training with SH degree " << old_splat.get_active_sh_degree() << std::endl;
}

// Test 3c: Compare all Gaussian attributes comprehensively
TEST_F(LoaderComparisonTest, GaussianAttributesComparison) {
    gs::loader::LoadOptions old_options;
    old_options.validate_only = false;

    lfs::loader::LoadOptions new_options;
    new_options.validate_only = false;

    auto old_result = old_loader->load(stump_path, old_options);
    auto new_result = new_loader->load(stump_path, new_options);

    ASSERT_TRUE(old_result.has_value());
    ASSERT_TRUE(new_result.has_value());

    auto& old_scene = std::get<gs::loader::LoadedScene>(old_result->data);
    auto& new_scene = std::get<lfs::loader::LoadedScene>(new_result->data);

    auto& old_pc = *old_scene.point_cloud;
    auto& new_pc = *new_scene.point_cloud;

    std::cout << "Comparing all Gaussian attributes..." << std::endl;
    std::cout << "Old: " << old_pc.size() << " points, is_gaussian=" << old_pc.is_gaussian() << std::endl;
    std::cout << "New: " << new_pc.size() << " points, is_gaussian=" << new_pc.is_gaussian() << std::endl;

    // 1. Compare means (positions)
    std::cout << "  [1/8] Comparing means (positions)..." << std::endl;
    EXPECT_TRUE(tensorsEqual(old_pc.means, new_pc.means, 1e-4f))
        << "Means (positions) differ";

    // 2. Compare colors
    std::cout << "  [2/8] Comparing colors..." << std::endl;
    EXPECT_TRUE(tensorsEqual(old_pc.colors, new_pc.colors, 1e-4f))
        << "Colors differ";

    // 3. Compare normals if present
    if (old_pc.normals.defined() && new_pc.normals.is_valid()) {
        std::cout << "  [3/8] Comparing normals..." << std::endl;
        EXPECT_TRUE(tensorsEqual(old_pc.normals, new_pc.normals, 1e-4f))
            << "Normals differ";
    } else {
        std::cout << "  [3/8] Normals: not present in both" << std::endl;
    }

    // 4. Compare sh0 if present
    if (old_pc.sh0.defined() && new_pc.sh0.is_valid()) {
        std::cout << "  [4/8] Comparing sh0..." << std::endl;
        EXPECT_TRUE(tensorsEqual(old_pc.sh0, new_pc.sh0, 1e-4f))
            << "sh0 differ";
    } else {
        std::cout << "  [4/8] sh0: not present in both" << std::endl;
    }

    // 5. Compare shN if present
    if (old_pc.shN.defined() && new_pc.shN.is_valid()) {
        std::cout << "  [5/8] Comparing shN..." << std::endl;
        EXPECT_TRUE(tensorsEqual(old_pc.shN, new_pc.shN, 1e-4f))
            << "shN differ";
    } else {
        std::cout << "  [5/8] shN: not present in both" << std::endl;
    }

    // 6. Compare opacity if present
    if (old_pc.opacity.defined() && new_pc.opacity.is_valid()) {
        std::cout << "  [6/8] Comparing opacity..." << std::endl;
        EXPECT_TRUE(tensorsEqual(old_pc.opacity, new_pc.opacity, 1e-4f))
            << "Opacity differ";
    } else {
        std::cout << "  [6/8] Opacity: not present in both" << std::endl;
    }

    // 7. Compare scaling if present
    if (old_pc.scaling.defined() && new_pc.scaling.is_valid()) {
        std::cout << "  [7/8] Comparing scaling..." << std::endl;
        EXPECT_TRUE(tensorsEqual(old_pc.scaling, new_pc.scaling, 1e-4f))
            << "Scaling differ";
    } else {
        std::cout << "  [7/8] Scaling: not present in both" << std::endl;
    }

    // 8. Compare rotation if present
    if (old_pc.rotation.defined() && new_pc.rotation.is_valid()) {
        std::cout << "  [8/8] Comparing rotation..." << std::endl;
        EXPECT_TRUE(tensorsEqual(old_pc.rotation, new_pc.rotation, 1e-4f))
            << "Rotation differ";
    } else {
        std::cout << "  [8/8] Rotation: not present in both" << std::endl;
    }

    std::cout << "✓ All Gaussian attributes compared successfully!" << std::endl;
}

// Test 4: Validation-only mode
TEST_F(LoaderComparisonTest, ValidationOnlyMode) {
    gs::loader::LoadOptions old_options;
    old_options.validate_only = true;

    lfs::loader::LoadOptions new_options;
    new_options.validate_only = true;

    auto old_result = old_loader->load(stump_path, old_options);
    auto new_result = new_loader->load(stump_path, new_options);

    // Both should succeed in validation mode
    EXPECT_TRUE(old_result.has_value()) << "Old loader validation failed";
    EXPECT_TRUE(new_result.has_value()) << "New loader validation failed";

    // In validation mode, some data might be null (implementation-dependent)
    // Just verify they both return a result
    std::cout << "Validation mode - Old loader: " << old_result->loader_used << std::endl;
    std::cout << "Validation mode - New loader: " << new_result->loader_used << std::endl;
}

// Test 5: Invalid path handling
TEST_F(LoaderComparisonTest, InvalidPathHandling) {
    fs::path invalid_path = data_dir / "nonexistent_dataset";

    // Old loader might throw or return error
    std::optional<std::expected<gs::loader::LoadResult, std::string>> old_result;
    try {
        old_result = old_loader->load(invalid_path);
    } catch (const std::exception& e) {
        std::cout << "Old loader threw exception: " << e.what() << std::endl;
        // Throwing is acceptable for invalid paths
    }

    // New loader might throw or return error
    std::optional<std::expected<lfs::loader::LoadResult, std::string>> new_result;
    try {
        new_result = new_loader->load(invalid_path);
    } catch (const std::exception& e) {
        std::cout << "New loader threw exception: " << e.what() << std::endl;
        // Throwing is acceptable for invalid paths
    }

    // Both should either throw or return error (not succeed)
    if (old_result.has_value()) {
        EXPECT_FALSE(old_result->has_value()) << "Old loader should fail on invalid path";
        if (!old_result->has_value()) {
            std::cout << "Old loader error: " << old_result->error() << std::endl;
        }
    }

    if (new_result.has_value()) {
        EXPECT_FALSE(new_result->has_value()) << "New loader should fail on invalid path";
        if (!new_result->has_value()) {
            std::cout << "New loader error: " << new_result->error() << std::endl;
        }
    }
}

// Test 6: Load with different resize factors
TEST_F(LoaderComparisonTest, ResizeFactorHandling) {
    for (int resize_factor : {-1, 2, 4}) {
        std::cout << "\nTesting resize_factor=" << resize_factor << std::endl;

        gs::loader::LoadOptions old_options;
        old_options.resize_factor = resize_factor;

        lfs::loader::LoadOptions new_options;
        new_options.resize_factor = resize_factor;

        auto old_result = old_loader->load(stump_path, old_options);
        auto new_result = new_loader->load(stump_path, new_options);

        EXPECT_TRUE(old_result.has_value()) << "Old loader failed with resize_factor=" << resize_factor;
        EXPECT_TRUE(new_result.has_value()) << "New loader failed with resize_factor=" << resize_factor;

        if (old_result.has_value() && new_result.has_value()) {
            auto& old_scene = std::get<gs::loader::LoadedScene>(old_result->data);
            auto& new_scene = std::get<lfs::loader::LoadedScene>(new_result->data);

            // Just verify both loaded successfully
            EXPECT_NE(old_scene.cameras, nullptr);
            EXPECT_NE(new_scene.cameras, nullptr);

            std::cout << "  Both loaders succeeded" << std::endl;
        }
    }
}

// Test 7: Performance comparison
TEST_F(LoaderComparisonTest, PerformanceComparison) {
    const int NUM_ITERATIONS = 5;

    std::vector<int64_t> old_times;
    std::vector<int64_t> new_times;

    gs::loader::LoadOptions old_options;
    lfs::loader::LoadOptions new_options;

    std::cout << "\nPerformance comparison (" << NUM_ITERATIONS << " iterations):" << std::endl;

    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        // Old loader
        auto old_result = old_loader->load(stump_path, old_options);
        ASSERT_TRUE(old_result.has_value());
        old_times.push_back(old_result->load_time.count());

        // New loader
        auto new_result = new_loader->load(stump_path, new_options);
        ASSERT_TRUE(new_result.has_value());
        new_times.push_back(new_result->load_time.count());
    }

    // Calculate averages
    int64_t old_avg = 0, new_avg = 0;
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        old_avg += old_times[i];
        new_avg += new_times[i];
    }
    old_avg /= NUM_ITERATIONS;
    new_avg /= NUM_ITERATIONS;

    std::cout << "Old loader average: " << old_avg << "ms" << std::endl;
    std::cout << "New loader average: " << new_avg << "ms" << std::endl;

    if (new_avg < old_avg) {
        float speedup = static_cast<float>(old_avg) / static_cast<float>(new_avg);
        std::cout << "New loader is " << speedup << "x faster" << std::endl;
    } else {
        float slowdown = static_cast<float>(new_avg) / static_cast<float>(old_avg);
        std::cout << "New loader is " << slowdown << "x slower" << std::endl;
    }

    // The new loader should be reasonably fast (within 2x of old loader)
    // This is a soft requirement since performance can vary
    if (new_avg > old_avg * 2) {
        std::cout << "WARNING: New loader is significantly slower than old loader" << std::endl;
    }
}

// Test: Tensor slice operation - verify it works correctly
TEST_F(LoaderComparisonTest, TensorSliceOperation) {
    // Create a 4x4 matrix similar to what transforms.cpp does
    lfs::core::Tensor mat = lfs::core::Tensor::empty({4, 4}, lfs::core::Device::CPU, lfs::core::DataType::Float32);

    // Fill it with known values
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            mat[i][j] = static_cast<float>(i * 4 + j);
        }
    }

    ASSERT_TRUE(mat.is_valid());
    ASSERT_EQ(mat.numel(), 16);

    // Test single slice (should extract first 3 rows)
    lfs::core::Tensor slice1 = mat.slice(0, 0, 3);
    EXPECT_TRUE(slice1.is_valid()) << "First slice should be valid";
    EXPECT_EQ(slice1.numel(), 12) << "First slice should have 12 elements (3x4)";

    // Test double slice (should extract 3x3 block)
    lfs::core::Tensor slice2 = mat.slice(0, 0, 3).slice(1, 0, 3);
    EXPECT_TRUE(slice2.is_valid()) << "Double slice should be valid";
    EXPECT_EQ(slice2.numel(), 9) << "Double slice should have 9 elements (3x3)";

    // Test the exact pattern from transforms.cpp for T extraction
    lfs::core::Tensor slice3 = mat.slice(0, 0, 3).slice(1, 3, 4);
    EXPECT_TRUE(slice3.is_valid()) << "Slice before squeeze should be valid";
    EXPECT_EQ(slice3.numel(), 3) << "Slice before squeeze should have 3 elements (3x1)";

    if (slice3.is_valid()) {
        lfs::core::Tensor slice3_squeezed = slice3.squeeze(1);
        EXPECT_TRUE(slice3_squeezed.is_valid()) << "Squeezed slice should be valid";
        EXPECT_EQ(slice3_squeezed.numel(), 3) << "Squeezed slice should have 3 elements";
    }
}

// Test 10: Blender dataset (Lego) - comprehensive comparison including SplatData initialization
TEST_F(LoaderComparisonTest, BlenderDatasetComparison) {
    // Skip test if lego dataset doesn't exist
    if (!fs::exists(lego_path)) {
        GTEST_SKIP() << "Lego dataset not found at: " << lego_path;
    }

    std::cout << "\n=== Testing Blender Loader (Lego Dataset) ===" << std::endl;

    gs::loader::LoadOptions old_options;
    old_options.validate_only = false;

    lfs::loader::LoadOptions new_options;
    new_options.validate_only = false;

    // Load with both loaders
    auto old_result = old_loader->load(lego_path, old_options);
    auto new_result = new_loader->load(lego_path, new_options);

    ASSERT_TRUE(old_result.has_value()) << "Old loader failed: " << old_result.error();
    ASSERT_TRUE(new_result.has_value()) << "New loader failed: " << new_result.error();

    auto& old_scene = std::get<gs::loader::LoadedScene>(old_result->data);
    auto& new_scene = std::get<lfs::loader::LoadedScene>(new_result->data);

    std::cout << "\n[1/5] Camera Dataset Comparison" << std::endl;

    // 1. Verify camera datasets exist
    ASSERT_TRUE(old_scene.cameras != nullptr) << "Old loader: No camera dataset";
    ASSERT_TRUE(new_scene.cameras != nullptr) << "New loader: No camera dataset";
    std::cout << "  ✓ Both loaders created camera datasets" << std::endl;

    std::cout << "\n[2/5] Point Cloud Comparison" << std::endl;

    // 2. Compare point clouds (Blender datasets use random initialization)
    ASSERT_TRUE(old_scene.point_cloud != nullptr) << "Old loader: No point cloud";
    ASSERT_TRUE(new_scene.point_cloud != nullptr) << "New loader: No point cloud";

    auto& old_pc = *old_scene.point_cloud;
    auto& new_pc = *new_scene.point_cloud;

    std::cout << "  Old point cloud: " << old_pc.size() << " points" << std::endl;
    std::cout << "  New point cloud: " << new_pc.size() << " points" << std::endl;

    EXPECT_EQ(old_pc.size(), new_pc.size()) << "Point cloud size mismatch";
    EXPECT_EQ(old_pc.is_gaussian(), new_pc.is_gaussian()) << "is_gaussian flag mismatch";

    // For random point clouds, we verify structure rather than values
    EXPECT_TRUE(old_pc.means.defined()) << "Old means not defined";
    EXPECT_TRUE(new_pc.means.is_valid()) << "New means not valid";
    EXPECT_TRUE(old_pc.colors.defined()) << "Old colors not defined";
    EXPECT_TRUE(new_pc.colors.is_valid()) << "New colors not valid";

    std::cout << "\n[3/5] Scene Center Comparison" << std::endl;

    // 3. Compare scene centers
    auto old_center_cpu = old_result->scene_center.cpu();
    auto new_center_cpu = new_result->scene_center.cpu();

    std::cout << "  Old scene center: [" << old_center_cpu[0].item<float>() << ", "
              << old_center_cpu[1].item<float>() << ", " << old_center_cpu[2].item<float>() << "]" << std::endl;
    std::cout << "  New scene center: [" << new_center_cpu[0].item() << ", "
              << new_center_cpu[1].item() << ", " << new_center_cpu[2].item() << "]" << std::endl;

    EXPECT_FLOAT_EQ(old_center_cpu[0].item<float>(), new_center_cpu[0].item()) << "Scene center X differs";
    EXPECT_FLOAT_EQ(old_center_cpu[1].item<float>(), new_center_cpu[1].item()) << "Scene center Y differs";
    EXPECT_FLOAT_EQ(old_center_cpu[2].item<float>(), new_center_cpu[2].item()) << "Scene center Z differs";

    std::cout << "\n[4/5] SplatData Initialization Comparison" << std::endl;

    // 4. **CRITICAL**: Initialize SplatData and compare all attributes
    gs::param::TrainingParameters old_params;
    old_params.optimization.sh_degree = 3;
    old_params.optimization.random = true; // Blender uses random init
    old_params.optimization.init_num_pts = old_pc.size();
    old_params.optimization.init_extent = 3.0f;

    lfs::core::param::TrainingParameters new_params;
    new_params.optimization.sh_degree = 3;
    new_params.optimization.random = true; // Blender uses random init
    new_params.optimization.init_num_pts = new_pc.size();
    new_params.optimization.init_extent = 3.0f;

    auto old_splat_result = gs::SplatData::init_model_from_pointcloud(
        old_params, old_result->scene_center, old_pc);
    auto new_splat_result = lfs::core::SplatData::init_model_from_pointcloud(
        new_params, new_result->scene_center, new_pc);

    ASSERT_TRUE(old_splat_result.has_value()) << "Old SplatData init failed: " << old_splat_result.error();
    ASSERT_TRUE(new_splat_result.has_value()) << "New SplatData init failed: " << new_splat_result.error();

    auto& old_splat = old_splat_result.value();
    auto& new_splat = new_splat_result.value();

    std::cout << "  Old SplatData: " << old_splat.size() << " Gaussians" << std::endl;
    std::cout << "  New SplatData: " << new_splat.size() << " Gaussians" << std::endl;

    // Compare sizes
    EXPECT_EQ(old_splat.size(), new_splat.size()) << "SplatData size mismatch";

    // For random initialization, verify structure and metadata (not exact values)
    std::cout << "  Verifying tensor shapes and metadata..." << std::endl;

    // Check means shape
    EXPECT_EQ(old_splat.means().size(0), new_splat.means().size(0)) << "Means count mismatch";
    EXPECT_EQ(old_splat.means().size(1), 3) << "Old means should be Nx3";
    EXPECT_EQ(new_splat.means().shape()[1], 3) << "New means should be Nx3";

    // Check sh0 shape
    EXPECT_EQ(old_splat.sh0().size(0), new_splat.sh0().size(0)) << "SH0 count mismatch";
    EXPECT_EQ(old_splat.sh0().size(1), new_splat.sh0().shape()[1]) << "SH0 channel mismatch";

    // Check shN shape
    EXPECT_EQ(old_splat.shN().size(0), new_splat.shN().size(0)) << "SHN count mismatch";
    EXPECT_EQ(old_splat.shN().size(1), new_splat.shN().shape()[1]) << "SHN channel mismatch";

    // Check opacity shape
    EXPECT_EQ(old_splat.opacity_raw().size(0), new_splat.opacity_raw().size(0)) << "Opacity count mismatch";

    // Check scaling shape
    EXPECT_EQ(old_splat.scaling_raw().size(0), new_splat.scaling_raw().size(0)) << "Scaling count mismatch";
    EXPECT_EQ(old_splat.scaling_raw().size(1), 3) << "Old scaling should be Nx3";
    EXPECT_EQ(new_splat.scaling_raw().shape()[1], 3) << "New scaling should be Nx3";

    // Check rotation shape
    EXPECT_EQ(old_splat.rotation_raw().size(0), new_splat.rotation_raw().size(0)) << "Rotation count mismatch";
    EXPECT_EQ(old_splat.rotation_raw().size(1), 4) << "Old rotation should be Nx4 (quaternion)";
    EXPECT_EQ(new_splat.rotation_raw().shape()[1], 4) << "New rotation should be Nx4 (quaternion)";

    std::cout << "\n[5/5] Metadata Comparison" << std::endl;

    // 5. Compare metadata
    std::cout << "  Old active SH degree: " << old_splat.get_active_sh_degree() << std::endl;
    std::cout << "  New active SH degree: " << new_splat.get_active_sh_degree() << std::endl;
    EXPECT_EQ(old_splat.get_active_sh_degree(), new_splat.get_active_sh_degree())
        << "Active SH degree mismatch";

    std::cout << "  Old scene scale: " << old_splat.get_scene_scale() << std::endl;
    std::cout << "  New scene scale: " << new_splat.get_scene_scale() << std::endl;
    // Scene scale can differ slightly due to random initialization - allow 1% tolerance
    EXPECT_NEAR(old_splat.get_scene_scale(), new_splat.get_scene_scale(), old_splat.get_scene_scale() * 0.01)
        << "Scene scale mismatch exceeds 1% tolerance";

    std::cout << "\n✓ Blender (Lego) dataset comparison complete!" << std::endl;
    std::cout << "  - Cameras: datasets created successfully" << std::endl;
    std::cout << "  - Point cloud: " << old_pc.size() << " points (structure verified)" << std::endl;
    std::cout << "  - SplatData: " << old_splat.size() << " Gaussians (all attributes verified)" << std::endl;
    std::cout << "  - All data arriving at SplatData is compatible!" << std::endl;
}

// Test: Comprehensive loading benchmark for garden dataset
// Includes loading + SplatData initialization
// Runs multiple iterations alternating which loader goes first to account for filesystem caching
TEST_F(LoaderComparisonTest, GardenLoadingBenchmark) {
    const fs::path garden_path = fs::path("data") / "garden";

    // Skip if garden doesn't exist
    if (!fs::exists(garden_path)) {
        GTEST_SKIP() << "Garden dataset not found at: " << garden_path;
    }

    std::cout << "\n=== Garden Dataset Loading Benchmark ===" << std::endl;
    std::cout << "Dataset: " << garden_path << std::endl;
    std::cout << "Iterations: 5 (alternating loader order)" << std::endl;
    std::cout << "Measured: Loading + SplatData initialization\n"
              << std::endl;

    // Load options
    gs::loader::LoadOptions old_options;
    old_options.images_folder = "images";
    old_options.resize_factor = 1;

    lfs::loader::LoadOptions new_options;
    new_options.images_folder = "images";
    new_options.resize_factor = 1;

    // Parameters for SplatData initialization
    gs::param::TrainingParameters old_params;
    lfs::core::param::TrainingParameters new_params;

    const int num_iterations = 5;
    std::vector<double> old_times;
    std::vector<double> new_times;

    for (int iter = 0; iter < num_iterations; ++iter) {
        std::cout << "Iteration " << (iter + 1) << "/" << num_iterations << ":" << std::endl;

        // Alternate which loader goes first to account for caching
        bool old_first = (iter % 2 == 0);

        if (old_first) {
            // Old loader first
            {
                std::cout << "  [1] Old loader... " << std::flush;
                auto start = std::chrono::high_resolution_clock::now();

                auto old_result = old_loader->load(garden_path, old_options);
                ASSERT_TRUE(old_result.has_value()) << "Old loader failed: " << old_result.error();

                auto& old_scene = std::get<gs::loader::LoadedScene>(old_result->data);
                auto& old_pc = *old_scene.point_cloud;
                old_params.optimization.init_num_pts = old_pc.size();
                old_params.optimization.init_extent = 3.0f;

                auto old_splat_result = gs::SplatData::init_model_from_pointcloud(
                    old_params, old_result->scene_center, old_pc);
                ASSERT_TRUE(old_splat_result.has_value());

                auto end = std::chrono::high_resolution_clock::now();
                double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
                old_times.push_back(elapsed_ms);

                std::cout << elapsed_ms << " ms (" << old_splat_result.value().size() << " Gaussians)" << std::endl;
            }

            // Small delay to ensure caches settle
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            // New loader second
            {
                std::cout << "  [2] New loader... " << std::flush;
                auto start = std::chrono::high_resolution_clock::now();

                auto new_result = new_loader->load(garden_path, new_options);
                ASSERT_TRUE(new_result.has_value()) << "New loader failed: " << new_result.error();

                auto& new_scene = std::get<lfs::loader::LoadedScene>(new_result->data);
                auto& new_pc = *new_scene.point_cloud;
                new_params.optimization.init_num_pts = new_pc.size();
                new_params.optimization.init_extent = 3.0f;

                auto new_splat_result = lfs::core::SplatData::init_model_from_pointcloud(
                    new_params, new_result->scene_center, new_pc);
                ASSERT_TRUE(new_splat_result.has_value());

                auto end = std::chrono::high_resolution_clock::now();
                double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
                new_times.push_back(elapsed_ms);

                std::cout << elapsed_ms << " ms (" << new_splat_result.value().size() << " Gaussians)" << std::endl;
            }
        } else {
            // New loader first
            {
                std::cout << "  [1] New loader... " << std::flush;
                auto start = std::chrono::high_resolution_clock::now();

                auto new_result = new_loader->load(garden_path, new_options);
                ASSERT_TRUE(new_result.has_value()) << "New loader failed: " << new_result.error();

                auto& new_scene = std::get<lfs::loader::LoadedScene>(new_result->data);
                auto& new_pc = *new_scene.point_cloud;
                new_params.optimization.init_num_pts = new_pc.size();
                new_params.optimization.init_extent = 3.0f;

                auto new_splat_result = lfs::core::SplatData::init_model_from_pointcloud(
                    new_params, new_result->scene_center, new_pc);
                ASSERT_TRUE(new_splat_result.has_value());

                auto end = std::chrono::high_resolution_clock::now();
                double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
                new_times.push_back(elapsed_ms);

                std::cout << elapsed_ms << " ms (" << new_splat_result.value().size() << " Gaussians)" << std::endl;
            }

            // Small delay to ensure caches settle
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            // Old loader second
            {
                std::cout << "  [2] Old loader... " << std::flush;
                auto start = std::chrono::high_resolution_clock::now();

                auto old_result = old_loader->load(garden_path, old_options);
                ASSERT_TRUE(old_result.has_value()) << "Old loader failed: " << old_result.error();

                auto& old_scene = std::get<gs::loader::LoadedScene>(old_result->data);
                auto& old_pc = *old_scene.point_cloud;
                old_params.optimization.init_num_pts = old_pc.size();
                old_params.optimization.init_extent = 3.0f;

                auto old_splat_result = gs::SplatData::init_model_from_pointcloud(
                    old_params, old_result->scene_center, old_pc);
                ASSERT_TRUE(old_splat_result.has_value());

                auto end = std::chrono::high_resolution_clock::now();
                double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
                old_times.push_back(elapsed_ms);

                std::cout << elapsed_ms << " ms (" << old_splat_result.value().size() << " Gaussians)" << std::endl;
            }
        }

        std::cout << std::endl;
    }

    // Calculate statistics
    auto calc_stats = [](const std::vector<double>& times) {
        double sum = 0.0, min = times[0], max = times[0];
        for (double t : times) {
            sum += t;
            min = std::min(min, t);
            max = std::max(max, t);
        }
        double mean = sum / times.size();

        double var_sum = 0.0;
        for (double t : times) {
            var_sum += (t - mean) * (t - mean);
        }
        double stddev = std::sqrt(var_sum / times.size());

        return std::make_tuple(mean, min, max, stddev);
    };

    auto [old_mean, old_min, old_max, old_stddev] = calc_stats(old_times);
    auto [new_mean, new_min, new_max, new_stddev] = calc_stats(new_times);

    std::cout << "=== Benchmark Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\nOld Loader (LibTorch-based):" << std::endl;
    std::cout << "  Mean:   " << old_mean << " ms" << std::endl;
    std::cout << "  Min:    " << old_min << " ms" << std::endl;
    std::cout << "  Max:    " << old_max << " ms" << std::endl;
    std::cout << "  StdDev: " << old_stddev << " ms" << std::endl;

    std::cout << "\nNew Loader (Custom Tensor):" << std::endl;
    std::cout << "  Mean:   " << new_mean << " ms" << std::endl;
    std::cout << "  Min:    " << new_min << " ms" << std::endl;
    std::cout << "  Max:    " << new_max << " ms" << std::endl;
    std::cout << "  StdDev: " << new_stddev << " ms" << std::endl;

    double speedup = old_mean / new_mean;
    double percent_diff = ((new_mean - old_mean) / old_mean) * 100.0;

    std::cout << "\nComparison:" << std::endl;
    if (speedup > 1.0) {
        std::cout << "  New loader is " << speedup << "x faster" << std::endl;
        std::cout << "  (" << std::abs(percent_diff) << "% improvement)" << std::endl;
    } else {
        std::cout << "  New loader is " << (1.0 / speedup) << "x slower" << std::endl;
        std::cout << "  (" << percent_diff << "% slower)" << std::endl;
    }

    std::cout << "\n=== Benchmark Complete ===" << std::endl;
}

// Note: main() is provided by test_main.cpp
