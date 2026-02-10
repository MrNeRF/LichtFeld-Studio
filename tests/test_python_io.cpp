/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

#include "core/point_cloud.hpp"
#include "core/splat_data.hpp"
#include "io/exporter.hpp"
#include "io/loader.hpp"

namespace fs = std::filesystem;
using namespace lfs::core;
using namespace lfs::io;

class PythonIOTest : public ::testing::Test {
protected:
    static constexpr float EPSILON = 1e-5f;
    static constexpr float PLY_TOLERANCE = 1e-4f;

    const fs::path data_dir = fs::path(PROJECT_ROOT_PATH) / "data";
    const fs::path bicycle_dir = data_dir / "bicycle";
    const fs::path temp_dir = fs::temp_directory_path() / "lfs_py_io_test";

    void SetUp() override {
        fs::create_directories(temp_dir);
    }

    void TearDown() override {
        fs::remove_all(temp_dir);
    }

    static SplatData create_test_splat(size_t num_points, int sh_degree = 0) {
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

        auto* means_ptr = means.ptr<float>();
        auto* sh0_ptr = sh0.ptr<float>();
        auto* scaling_ptr = scaling.ptr<float>();
        auto* rotation_ptr = rotation.ptr<float>();
        auto* opacity_ptr = opacity.ptr<float>();

        for (size_t i = 0; i < num_points; ++i) {
            means_ptr[i * 3 + 0] = static_cast<float>(i % 10);
            means_ptr[i * 3 + 1] = static_cast<float>((i / 10) % 10);
            means_ptr[i * 3 + 2] = static_cast<float>(i / 100);

            sh0_ptr[i * 3 + 0] = 0.5f + 0.1f * static_cast<float>(i % 5);
            sh0_ptr[i * 3 + 1] = 0.3f + 0.1f * static_cast<float>((i + 1) % 5);
            sh0_ptr[i * 3 + 2] = 0.4f + 0.1f * static_cast<float>((i + 2) % 5);

            scaling_ptr[i * 3 + 0] = -3.0f + 0.01f * static_cast<float>(i % 100);
            scaling_ptr[i * 3 + 1] = -3.0f + 0.01f * static_cast<float>((i + 1) % 100);
            scaling_ptr[i * 3 + 2] = -3.0f + 0.01f * static_cast<float>((i + 2) % 100);

            rotation_ptr[i * 4 + 0] = 1.0f;
            rotation_ptr[i * 4 + 1] = 0.0f;
            rotation_ptr[i * 4 + 2] = 0.0f;
            rotation_ptr[i * 4 + 3] = 0.0f;

            opacity_ptr[i] = -2.0f + 0.04f * static_cast<float>(i % 100);
        }

        if (sh_coeffs > 0) {
            auto* shN_ptr = shN.ptr<float>();
            for (size_t i = 0; i < num_points * sh_coeffs * 3; ++i) {
                shN_ptr[i] = 0.1f * static_cast<float>((i % 10) - 5);
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
            1.0f);
    }
};

// Test Loader creation
TEST_F(PythonIOTest, LoaderCreation) {
    auto loader = Loader::create();
    ASSERT_NE(loader, nullptr);
}

// Test supported formats/extensions
TEST_F(PythonIOTest, SupportedFormats) {
    auto loader = Loader::create();

    auto formats = loader->getSupportedFormats();
    EXPECT_FALSE(formats.empty());

    auto extensions = loader->getSupportedExtensions();
    EXPECT_FALSE(extensions.empty());

    bool has_ply = false;
    for (const auto& ext : extensions) {
        if (ext == ".ply")
            has_ply = true;
    }
    EXPECT_TRUE(has_ply) << "Should support PLY format";
}

// Test dataset detection
TEST_F(PythonIOTest, IsDatasetPath) {
    EXPECT_TRUE(Loader::isDatasetPath(bicycle_dir)) << "bicycle should be detected as dataset";
    EXPECT_FALSE(Loader::isDatasetPath(temp_dir / "nonexistent.ply")) << "PLY file is not a dataset";
}

// Test dataset type detection
TEST_F(PythonIOTest, DatasetTypeDetection) {
    auto type = Loader::getDatasetType(bicycle_dir);
    EXPECT_EQ(type, DatasetType::COLMAP) << "bicycle should be COLMAP dataset";

    auto unknown_type = Loader::getDatasetType(temp_dir);
    EXPECT_EQ(unknown_type, DatasetType::Unknown);
}

// Test loading COLMAP dataset
TEST_F(PythonIOTest, LoadCOLMAPDataset) {
    if (!fs::exists(bicycle_dir / "sparse")) {
        GTEST_SKIP() << "bicycle sparse data not available";
    }

    auto loader = Loader::create();
    LoadOptions options;
    options.resize_factor = 8; // Downscale for faster test
    options.images_folder = "images_8";

    auto result = loader->load(bicycle_dir, options);
    ASSERT_TRUE(result.has_value()) << "Failed to load: " << result.error().format();

    EXPECT_FALSE(result->loader_used.empty());
    EXPECT_GT(result->load_time.count(), 0);

    // Should be a LoadedScene, not SplatData
    EXPECT_TRUE(std::holds_alternative<LoadedScene>(result->data));
}

// Test PLY save/load roundtrip
TEST_F(PythonIOTest, PlySaveLoadRoundtrip) {
    const size_t num_points = 1000;
    auto original = create_test_splat(num_points, 1);

    fs::path output_path = temp_dir / "test_output.ply";

    // Save
    PlySaveOptions save_options;
    save_options.output_path = output_path;
    save_options.binary = true;

    auto save_result = save_ply(original, save_options);
    ASSERT_TRUE(save_result.has_value()) << "Failed to save: " << save_result.error().format();
    EXPECT_TRUE(fs::exists(output_path));

    // Load back
    auto loader = Loader::create();
    auto load_result = loader->load(output_path);
    ASSERT_TRUE(load_result.has_value()) << "Failed to load: " << load_result.error().format();

    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<SplatData>>(load_result->data));
    auto& loaded = *std::get<std::shared_ptr<SplatData>>(load_result->data);

    // Verify point count
    EXPECT_EQ(loaded.size(), num_points);
    EXPECT_EQ(loaded.get_max_sh_degree(), original.get_max_sh_degree());

    // Verify means are close
    auto orig_means = original.get_means();
    auto load_means = loaded.get_means();
    ASSERT_EQ(orig_means.shape(), load_means.shape());

    auto orig_cpu = orig_means.cpu().contiguous();
    auto load_cpu = load_means.cpu().contiguous();
    float* orig_ptr = orig_cpu.ptr<float>();
    float* load_ptr = load_cpu.ptr<float>();

    for (size_t i = 0; i < std::min(size_t(100), num_points * 3); ++i) {
        EXPECT_NEAR(orig_ptr[i], load_ptr[i], PLY_TOLERANCE) << "Mismatch at index " << i;
    }
}

// Test ASCII PLY save
TEST_F(PythonIOTest, PlyAsciiSave) {
    const size_t num_points = 100;
    auto splat = create_test_splat(num_points);

    fs::path output_path = temp_dir / "test_ascii.ply";

    PlySaveOptions options;
    options.output_path = output_path;
    options.binary = false;

    auto result = save_ply(splat, options);
    ASSERT_TRUE(result.has_value()) << "Failed to save ASCII PLY: " << result.error().format();
    EXPECT_TRUE(fs::exists(output_path));

    // Verify file contains ASCII header
    std::ifstream file(output_path);
    std::string line;
    std::getline(file, line);
    EXPECT_EQ(line, "ply");
    std::getline(file, line);
    EXPECT_TRUE(line.find("ascii") != std::string::npos) << "Expected ASCII format";
}

// Test SPZ save (if available)
TEST_F(PythonIOTest, SpzSave) {
    const size_t num_points = 500;
    auto splat = create_test_splat(num_points, 1);

    fs::path output_path = temp_dir / "test_output.spz";

    SpzSaveOptions options;
    options.output_path = output_path;

    auto result = save_spz(splat, options);
    ASSERT_TRUE(result.has_value()) << "Failed to save SPZ: " << result.error().format();
    EXPECT_TRUE(fs::exists(output_path));
    EXPECT_GT(fs::file_size(output_path), 0);
}

// Test SOG save (SuperSplat format)
TEST_F(PythonIOTest, SogSave) {
    const size_t num_points = 500;
    auto splat = create_test_splat(num_points, 1);

    fs::path output_path = temp_dir / "test_output.sog";

    SogSaveOptions options;
    options.output_path = output_path;
    options.kmeans_iterations = 5;
    options.use_gpu = true;

    auto result = save_sog(splat, options);
    ASSERT_TRUE(result.has_value()) << "Failed to save SOG: " << result.error().format();
    EXPECT_TRUE(fs::exists(output_path));
    EXPECT_GT(fs::file_size(output_path), 0);
}

// Test HTML export
TEST_F(PythonIOTest, HtmlExport) {
    const size_t num_points = 500;
    auto splat = create_test_splat(num_points, 1);

    fs::path output_path = temp_dir / "test_viewer.html";

    HtmlExportOptions options;
    options.output_path = output_path;
    options.kmeans_iterations = 5;

    auto result = export_html(splat, options);
    ASSERT_TRUE(result.has_value()) << "Failed to export HTML: " << result.error().format();
    EXPECT_TRUE(fs::exists(output_path));

    // Verify HTML content
    std::ifstream file(output_path);
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    EXPECT_TRUE(content.find("<!DOCTYPE html>") != std::string::npos ||
                content.find("<html") != std::string::npos)
        << "Should be valid HTML";
}

// Test progress callback
TEST_F(PythonIOTest, ProgressCallback) {
    const size_t num_points = 1000;
    auto splat = create_test_splat(num_points);

    fs::path output_path = temp_dir / "test_progress.ply";

    int callback_count = 0;
    float last_progress = -1.0f;

    PlySaveOptions options;
    options.output_path = output_path;
    options.binary = true;
    options.progress_callback = [&](float progress, const std::string& stage) -> bool {
        EXPECT_GE(progress, 0.0f);
        EXPECT_LE(progress, 1.0f);
        EXPECT_GE(progress, last_progress) << "Progress should not decrease";
        EXPECT_FALSE(stage.empty());
        last_progress = progress;
        callback_count++;
        return true; // Continue
    };

    auto result = save_ply(splat, options);
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(callback_count, 0) << "Progress callback should have been called";
}

// Test error handling for invalid path
TEST_F(PythonIOTest, LoadInvalidPath) {
    auto loader = Loader::create();
    auto result = loader->load("/nonexistent/path/to/file.ply");
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::PATH_NOT_FOUND);
}

// Test error handling for invalid format
TEST_F(PythonIOTest, LoadInvalidFormat) {
    // Create a file with wrong content
    fs::path bad_file = temp_dir / "bad_file.ply";
    std::ofstream ofs(bad_file);
    ofs << "This is not a valid PLY file\n";
    ofs.close();

    auto loader = Loader::create();
    auto result = loader->load(bad_file);
    EXPECT_FALSE(result.has_value());
}

// Test canLoad
TEST_F(PythonIOTest, CanLoad) {
    auto loader = Loader::create();

    EXPECT_TRUE(loader->canLoad(bicycle_dir)) << "Should be able to load bicycle dataset";
    EXPECT_FALSE(loader->canLoad("/nonexistent/path")) << "Should not be able to load nonexistent path";
}

// Test loading real PLY file if available
TEST_F(PythonIOTest, LoadRealPlyFile) {
    fs::path benchmark_ply = fs::path(PROJECT_ROOT_PATH) / "results" / "benchmark" / "bicycle" / "splat_30000.ply";

    if (!fs::exists(benchmark_ply)) {
        GTEST_SKIP() << "benchmark PLY file not available at " << benchmark_ply;
    }

    auto loader = Loader::create();
    auto result = loader->load(benchmark_ply);
    ASSERT_TRUE(result.has_value()) << "Failed to load: " << result.error().format();

    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<SplatData>>(result->data));
    auto& splat = *std::get<std::shared_ptr<SplatData>>(result->data);

    EXPECT_GT(splat.size(), 0) << "Should have loaded some gaussians";
}

// Test PLY save with uint8 colors round-trip
TEST_F(PythonIOTest, PlySaveWithColorsUint8) {
    constexpr size_t N = 200;
    PointCloud pc;
    pc.means = Tensor::empty({N, 3}, Device::CPU, DataType::Float32);
    pc.colors = Tensor::empty({N, 3}, Device::CPU, DataType::UInt8);

    auto* means_ptr = pc.means.ptr<float>();
    auto* colors_ptr = pc.colors.ptr<uint8_t>();
    for (size_t i = 0; i < N; ++i) {
        means_ptr[i * 3 + 0] = static_cast<float>(i);
        means_ptr[i * 3 + 1] = static_cast<float>(i + 1);
        means_ptr[i * 3 + 2] = static_cast<float>(i + 2);
        colors_ptr[i * 3 + 0] = static_cast<uint8_t>(i % 256);
        colors_ptr[i * 3 + 1] = static_cast<uint8_t>((i * 3) % 256);
        colors_ptr[i * 3 + 2] = static_cast<uint8_t>((i * 7) % 256);
    }
    pc.attribute_names = {"x", "y", "z"};

    fs::path output_path = temp_dir / "colors_u8.ply";
    PlySaveOptions options;
    options.output_path = output_path;
    options.binary = true;

    auto result = save_ply(pc, options);
    ASSERT_TRUE(result.has_value()) << result.error().format();

    // Verify header contains red/green/blue properties
    std::ifstream file(output_path, std::ios::binary);
    std::string header_text;
    std::string line;
    while (std::getline(file, line) && line != "end_header") {
        header_text += line + "\n";
    }
    EXPECT_NE(header_text.find("property uchar red"), std::string::npos);
    EXPECT_NE(header_text.find("property uchar green"), std::string::npos);
    EXPECT_NE(header_text.find("property uchar blue"), std::string::npos);
}

// Test PLY save with float32 colors converts to uint8
TEST_F(PythonIOTest, PlySaveWithColorsFloat32) {
    constexpr size_t N = 100;
    PointCloud pc;
    pc.means = Tensor::empty({N, 3}, Device::CPU, DataType::Float32);
    pc.colors = Tensor::empty({N, 3}, Device::CPU, DataType::Float32);

    auto* means_ptr = pc.means.ptr<float>();
    auto* colors_ptr = pc.colors.ptr<float>();
    for (size_t i = 0; i < N; ++i) {
        means_ptr[i * 3 + 0] = static_cast<float>(i);
        means_ptr[i * 3 + 1] = 0.0f;
        means_ptr[i * 3 + 2] = 0.0f;
        colors_ptr[i * 3 + 0] = static_cast<float>(i) / static_cast<float>(N);
        colors_ptr[i * 3 + 1] = 0.5f;
        colors_ptr[i * 3 + 2] = 1.0f;
    }
    pc.attribute_names = {"x", "y", "z"};

    fs::path output_path = temp_dir / "colors_f32.ply";
    PlySaveOptions options;
    options.output_path = output_path;
    options.binary = true;

    auto result = save_ply(pc, options);
    ASSERT_TRUE(result.has_value()) << result.error().format();

    // Verify header has uchar color properties (float32 should be converted)
    std::ifstream file(output_path, std::ios::binary);
    std::string header_text;
    std::string line;
    while (std::getline(file, line) && line != "end_header") {
        header_text += line + "\n";
    }
    EXPECT_NE(header_text.find("property uchar red"), std::string::npos);
}

// Test PLY save uses fallback attribute names when attribute_names is empty
TEST_F(PythonIOTest, PlySaveFallbackAttributeNames) {
    constexpr size_t N = 50;
    PointCloud pc;
    pc.means = Tensor::empty({N, 3}, Device::CPU, DataType::Float32);
    pc.opacity = Tensor::empty({N, 1}, Device::CPU, DataType::Float32);

    auto* means_ptr = pc.means.ptr<float>();
    auto* opacity_ptr = pc.opacity.ptr<float>();
    for (size_t i = 0; i < N; ++i) {
        means_ptr[i * 3 + 0] = static_cast<float>(i);
        means_ptr[i * 3 + 1] = 0.0f;
        means_ptr[i * 3 + 2] = 0.0f;
        opacity_ptr[i] = 0.5f;
    }
    // Leave attribute_names empty to trigger fallback
    assert(pc.attribute_names.empty());

    fs::path output_path = temp_dir / "fallback_names.ply";
    PlySaveOptions options;
    options.output_path = output_path;
    options.binary = true;

    auto result = save_ply(pc, options);
    ASSERT_TRUE(result.has_value()) << result.error().format();

    // Parse header and verify fallback names are present
    std::ifstream file(output_path, std::ios::binary);
    std::string header_text;
    std::string line;
    while (std::getline(file, line) && line != "end_header") {
        header_text += line + "\n";
    }
    EXPECT_NE(header_text.find("property float x"), std::string::npos);
    EXPECT_NE(header_text.find("property float y"), std::string::npos);
    EXPECT_NE(header_text.find("property float z"), std::string::npos);
    EXPECT_NE(header_text.find("property float opacity"), std::string::npos);
}
