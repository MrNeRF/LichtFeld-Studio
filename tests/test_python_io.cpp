/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/point_cloud.hpp"
#include "core/splat_data.hpp"
#include "io/exporter.hpp"
#include "io/formats/ply.hpp"
#include "io/loader.hpp"
#include "tinyply.hpp"

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

    void write_text_file(const fs::path& path, const std::string& contents) const {
        fs::create_directories(path.parent_path());
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out.is_open()) << "Failed to open " << path;
        out << contents;
        out.close();
        ASSERT_TRUE(out.good()) << "Failed to write " << path;
    }

    std::string read_text_file(const fs::path& path) const {
        std::ifstream in(path, std::ios::binary);
        EXPECT_TRUE(in.is_open()) << "Failed to open " << path;
        return {
            std::istreambuf_iterator<char>(in),
            std::istreambuf_iterator<char>()};
    }

    void expect_existing_target_and_no_temp_files(const fs::path& output_path, const std::string& existing_contents) const {
        EXPECT_EQ(read_text_file(output_path), existing_contents);

        const auto temp_prefix = output_path.filename().string() + ".";
        for (const auto& entry : fs::directory_iterator(output_path.parent_path())) {
            EXPECT_FALSE(entry.path().filename().string().starts_with(temp_prefix))
                << "Temporary export file was not removed: " << entry.path();
        }
    }

    static void expect_progress_completed(const std::vector<float>& updates) {
        ASSERT_FALSE(updates.empty());
        EXPECT_FLOAT_EQ(updates.front(), 0.0f);
        EXPECT_FLOAT_EQ(updates.back(), 1.0f);
        for (size_t i = 1; i < updates.size(); ++i) {
            EXPECT_GE(updates[i], updates[i - 1]) << "Progress regressed at update " << i;
        }
    }

    void write_png(const fs::path& path) const {
        static const std::vector<unsigned char> png_1x1 = {
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
            0x89, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x44, 0x41,
            0x54, 0x78, 0x9C, 0x63, 0xF8, 0xCF, 0xC0, 0xF0,
            0x1F, 0x00, 0x05, 0x00, 0x01, 0xFF, 0x89, 0x99,
            0x3D, 0x1D, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,
            0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82};

        fs::create_directories(path.parent_path());
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out.is_open()) << "Failed to open " << path;
        out.write(reinterpret_cast<const char*>(png_1x1.data()),
                  static_cast<std::streamsize>(png_1x1.size()));
        out.close();
        ASSERT_TRUE(out.good()) << "Failed to write " << path;
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

    static SplatData create_layout_test_splat() {
        auto means = Tensor::from_vector(std::vector<float>{1.0f, 2.0f, 3.0f}, {size_t{1}, size_t{3}}, Device::CPU);
        auto sh0 = Tensor::from_vector(std::vector<float>{10.0f, 20.0f, 30.0f}, {size_t{1}, size_t{1}, size_t{3}}, Device::CPU);
        auto shN = Tensor::from_vector(
            std::vector<float>{
                100.0f, 200.0f, 300.0f,
                101.0f, 201.0f, 301.0f,
                102.0f, 202.0f, 302.0f},
            {size_t{1}, size_t{3}, size_t{3}}, Device::CPU);
        auto scaling = Tensor::zeros({size_t{1}, size_t{3}}, Device::CPU, DataType::Float32);
        auto rotation = Tensor::from_vector(std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f}, {size_t{1}, size_t{4}}, Device::CPU);
        auto opacity = Tensor::from_vector(std::vector<float>{0.25f}, {size_t{1}, size_t{1}}, Device::CPU);

        return SplatData(
            1,
            std::move(means),
            std::move(sh0),
            std::move(shN),
            std::move(scaling),
            std::move(rotation),
            std::move(opacity),
            1.0f);
    }

    static std::string read_ply_header(const fs::path& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open PLY header");
        }

        std::string header_text;
        std::string line;
        while (std::getline(file, line) && line != "end_header") {
            header_text += line + "\n";
        }
        return header_text;
    }

    static std::vector<float> read_ply_float_property(const fs::path& path, const std::string& property_name) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open PLY file");
        }

        tinyply::PlyFile ply;
        ply.parse_header(file);
        auto property = ply.request_properties_from_element("vertex", {property_name});
        ply.read(file);

        if (property->t != tinyply::Type::FLOAT32) {
            throw std::runtime_error("Expected float32 PLY property");
        }

        std::vector<float> values(property->count);
        std::memcpy(values.data(), property->buffer.get(), values.size() * sizeof(float));
        return values;
    }

    static void write_external_sh_layout_test_ply(const fs::path& path) {
        std::ofstream out(path, std::ios::binary);
        if (!out.is_open()) {
            throw std::runtime_error("Failed to open test PLY for writing");
        }

        std::string header =
            "ply\n"
            "format binary_little_endian 1.0\n"
            "element vertex 1\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float f_dc_0\n"
            "property float f_dc_1\n"
            "property float f_dc_2\n"
            "property float f_rest_0\n"
            "property float f_rest_1\n"
            "property float f_rest_2\n"
            "property float f_rest_3\n"
            "property float f_rest_4\n"
            "property float f_rest_5\n"
            "property float f_rest_6\n"
            "property float f_rest_7\n"
            "property float f_rest_8\n"
            "end_header\n";
        out.write(header.data(), static_cast<std::streamsize>(header.size()));

        const std::array<float, 15> vertex{
            1.0f, 2.0f, 3.0f,
            10.0f, 20.0f, 30.0f,
            100.0f, 101.0f, 102.0f,
            200.0f, 201.0f, 202.0f,
            300.0f, 301.0f, 302.0f};
        out.write(reinterpret_cast<const char*>(vertex.data()),
                  static_cast<std::streamsize>(vertex.size() * sizeof(float)));
        if (!out.good()) {
            throw std::runtime_error("Failed to write test PLY");
        }
    }

    static std::vector<float> read_binary_ply_body_as_floats(const fs::path& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) {
            throw std::runtime_error("Failed to open test PLY for reading");
        }

        const std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const size_t header_pos = bytes.find("end_header");
        if (header_pos == std::string::npos) {
            throw std::runtime_error("Missing end_header in test PLY");
        }

        const size_t newline_pos = bytes.find('\n', header_pos);
        if (newline_pos == std::string::npos) {
            throw std::runtime_error("Malformed PLY header");
        }

        const size_t body_offset = newline_pos + 1;
        const size_t body_bytes = bytes.size() - body_offset;
        if (body_bytes % sizeof(float) != 0) {
            throw std::runtime_error("PLY body is not float-aligned");
        }

        std::vector<float> values(body_bytes / sizeof(float));
        std::memcpy(values.data(), bytes.data() + body_offset, body_bytes);
        return values;
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

TEST_F(PythonIOTest, LoadTransformsDatasetConvertsPointCloudToColmapWorld) {
    const fs::path dataset_dir = temp_dir / "transforms_dataset";
    write_png(dataset_dir / "frame_0001.png");
    write_text_file(
        dataset_dir / "transforms_train.json",
        R"json({
  "w": 1,
  "h": 1,
  "camera_angle_x": 0.78539816339,
  "ply_file_path": "pointcloud.ply",
  "frames": [
    {
      "file_path": "frame_0001.png",
      "transform_matrix": [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
      ]
    }
  ]
})json");
    write_text_file(
        dataset_dir / "pointcloud.ply",
        R"ply(ply
format ascii 1.0
element vertex 2
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
0 1 -2 255 0 0
1 -3 4 0 255 0
)ply");

    auto loader = Loader::create();
    auto result = loader->load(dataset_dir);
    ASSERT_TRUE(result.has_value()) << "Failed to load: " << result.error().format();
    ASSERT_TRUE(std::holds_alternative<LoadedScene>(result->data));

    const auto& scene = std::get<LoadedScene>(result->data);
    ASSERT_EQ(scene.cameras.size(), 1u);
    ASSERT_TRUE(scene.point_cloud);
    ASSERT_EQ(scene.point_cloud->size(), 2u);

    auto means_cpu = scene.point_cloud->means.cpu().contiguous();
    auto acc = means_cpu.accessor<float, 2>();
    EXPECT_NEAR(acc(0, 0), 0.0f, EPSILON);
    EXPECT_NEAR(acc(0, 1), -1.0f, EPSILON);
    EXPECT_NEAR(acc(0, 2), 2.0f, EPSILON);
    EXPECT_NEAR(acc(1, 0), 1.0f, EPSILON);
    EXPECT_NEAR(acc(1, 1), 3.0f, EPSILON);
    EXPECT_NEAR(acc(1, 2), -4.0f, EPSILON);
}

TEST_F(PythonIOTest, LoadTransformsDatasetCanBeCancelled) {
    const fs::path dataset_dir = temp_dir / "cancelled_transforms_dataset";
    write_text_file(
        dataset_dir / "transforms_train.json",
        R"json({
  "w": 1,
  "h": 1,
  "camera_angle_x": 0.78539816339,
  "frames": [
    {
      "file_path": "frame_0001.png",
      "transform_matrix": [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
      ]
    }
  ]
})json");

    int cancel_checks = 0;
    auto loader = Loader::create();
    auto result = loader->load(dataset_dir, {
                                                .cancel_requested = [&cancel_checks]() {
                                                    return ++cancel_checks >= 3;
                                                },
                                            });

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::CANCELLED);
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

TEST_F(PythonIOTest, PlySaveReportsProgressDuringPreparation) {
    auto splat = create_test_splat(1000, 1);
    const fs::path output_path = temp_dir / "progress_output.ply";

    std::vector<float> updates;
    std::vector<std::string> stages;

    PlySaveOptions save_options;
    save_options.output_path = output_path;
    save_options.binary = true;
    save_options.progress_callback = [&](float progress, const std::string& stage) {
        updates.push_back(progress);
        stages.push_back(stage);
        return true;
    };

    auto save_result = save_ply(splat, save_options);
    ASSERT_TRUE(save_result.has_value()) << "Failed to save: " << save_result.error().format();

    ASSERT_GE(updates.size(), 6UL);
    EXPECT_FLOAT_EQ(updates.front(), 0.0f);
    EXPECT_FLOAT_EQ(updates.back(), 1.0f);

    for (size_t i = 1; i < updates.size(); ++i) {
        EXPECT_GE(updates[i], updates[i - 1]) << "Progress regressed at update " << i;
    }

    bool saw_mid_export_progress = false;
    bool saw_copy_stage = false;
    for (size_t i = 0; i < updates.size(); ++i) {
        saw_mid_export_progress |= updates[i] > 0.1f && updates[i] < 0.9f;
        saw_copy_stage |= stages[i].starts_with("Copying ");
    }

    EXPECT_TRUE(saw_mid_export_progress);
    EXPECT_TRUE(saw_copy_stage);
}

TEST_F(PythonIOTest, PlySaveCancellationAtFinalProgressDoesNotReportSuccess) {
    auto splat = create_test_splat(1000, 1);
    const fs::path output_path = temp_dir / "cancelled_progress_output.ply";

    PlySaveOptions save_options;
    save_options.output_path = output_path;
    save_options.binary = true;
    save_options.progress_callback = [](float progress, const std::string&) {
        return progress < 1.0f;
    };

    auto save_result = save_ply(splat, save_options);
    ASSERT_FALSE(save_result.has_value());
    EXPECT_EQ(save_result.error().code, ErrorCode::CANCELLED);
}

TEST_F(PythonIOTest, PlySaveCancellationKeepsExistingTarget) {
    auto splat = create_test_splat(1000, 1);
    const fs::path output_path = temp_dir / "keep_existing_on_cancel.ply";
    const std::string existing_contents = "existing ply data";
    write_text_file(output_path, existing_contents);

    PlySaveOptions save_options;
    save_options.output_path = output_path;
    save_options.binary = true;
    save_options.progress_callback = [](float progress, const std::string&) {
        return progress < 1.0f;
    };

    auto save_result = save_ply(splat, save_options);
    ASSERT_FALSE(save_result.has_value());
    EXPECT_EQ(save_result.error().code, ErrorCode::CANCELLED);

    expect_existing_target_and_no_temp_files(output_path, existing_contents);
}

TEST_F(PythonIOTest, PlyLoadMapsExternalChannelMajorShOrderToInternalLayout) {
    const fs::path input_path = temp_dir / "external_channel_major_sh.ply";
    write_external_sh_layout_test_ply(input_path);

    auto loaded = load_ply(input_path);
    ASSERT_TRUE(loaded.has_value()) << "Failed to load test PLY: " << loaded.error();

    EXPECT_EQ(loaded->size(), 1UL);
    EXPECT_EQ(loaded->get_max_sh_degree(), 1);

    const auto sh0 = loaded->sh0().cpu().contiguous();
    ASSERT_EQ(sh0.ndim(), 3);
    ASSERT_EQ(sh0.size(0), 1);
    ASSERT_EQ(sh0.size(1), 1);
    ASSERT_EQ(sh0.size(2), 3);

    const auto* sh0_ptr = sh0.ptr<float>();
    EXPECT_FLOAT_EQ(sh0_ptr[0], 10.0f);
    EXPECT_FLOAT_EQ(sh0_ptr[1], 20.0f);
    EXPECT_FLOAT_EQ(sh0_ptr[2], 30.0f);

    const auto shN = loaded->shN().cpu().contiguous();
    ASSERT_TRUE(shN.is_valid());
    ASSERT_EQ(shN.ndim(), 3);
    ASSERT_EQ(shN.size(0), 1);
    ASSERT_EQ(shN.size(1), 3);
    ASSERT_EQ(shN.size(2), 3);

    const std::array<float, 9> expected_rest{
        100.0f, 200.0f, 300.0f,
        101.0f, 201.0f, 301.0f,
        102.0f, 202.0f, 302.0f};
    const auto* shN_ptr = shN.ptr<float>();
    for (size_t i = 0; i < expected_rest.size(); ++i) {
        EXPECT_FLOAT_EQ(shN_ptr[i], expected_rest[i]) << "Mismatch at shN index " << i;
    }
}

TEST_F(PythonIOTest, PlySaveUsesExternalChannelMajorShOrder) {
    auto splat = create_layout_test_splat();
    const fs::path output_path = temp_dir / "channel_major_export.ply";

    PlySaveOptions save_options;
    save_options.output_path = output_path;
    save_options.binary = true;

    auto save_result = save_ply(splat, save_options);
    ASSERT_TRUE(save_result.has_value()) << "Failed to save test PLY: " << save_result.error().format();

    const auto values = read_binary_ply_body_as_floats(output_path);
    ASSERT_GE(values.size(), 18UL);

    EXPECT_FLOAT_EQ(values[0], 1.0f);
    EXPECT_FLOAT_EQ(values[1], 2.0f);
    EXPECT_FLOAT_EQ(values[2], 3.0f);

    const std::array<float, 3> expected_dc{10.0f, 20.0f, 30.0f};
    for (size_t i = 0; i < expected_dc.size(); ++i) {
        EXPECT_FLOAT_EQ(values[6 + i], expected_dc[i]) << "Mismatch at f_dc_" << i;
    }

    const std::array<float, 9> expected_rest{
        100.0f, 101.0f, 102.0f,
        200.0f, 201.0f, 202.0f,
        300.0f, 301.0f, 302.0f};
    for (size_t i = 0; i < expected_rest.size(); ++i) {
        EXPECT_FLOAT_EQ(values[9 + i], expected_rest[i]) << "Mismatch at f_rest_" << i;
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
    std::vector<float> updates;

    SpzSaveOptions options;
    options.output_path = output_path;
    options.progress_callback = [&](float progress, const std::string&) {
        updates.push_back(progress);
        return true;
    };

    auto result = save_spz(splat, options);
    ASSERT_TRUE(result.has_value()) << "Failed to save SPZ: " << result.error().format();
    EXPECT_TRUE(fs::exists(output_path));
    EXPECT_GT(fs::file_size(output_path), 0);
    expect_progress_completed(updates);
}

TEST_F(PythonIOTest, SpzSaveCancellationKeepsExistingTarget) {
    auto splat = create_test_splat(200, 1);
    const fs::path output_path = temp_dir / "keep_existing_on_cancel.spz";
    const std::string existing_contents = "existing spz data";
    write_text_file(output_path, existing_contents);

    SpzSaveOptions options;
    options.output_path = output_path;
    options.progress_callback = [](float progress, const std::string&) {
        return progress < 1.0f;
    };

    auto result = save_spz(splat, options);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::CANCELLED);
    expect_existing_target_and_no_temp_files(output_path, existing_contents);
}

// Test SOG save (SuperSplat format)
TEST_F(PythonIOTest, SogSave) {
    const size_t num_points = 500;
    auto splat = create_test_splat(num_points, 1);

    fs::path output_path = temp_dir / "test_output.sog";
    std::vector<float> updates;

    SogSaveOptions options;
    options.output_path = output_path;
    options.kmeans_iterations = 5;
    options.use_gpu = true;
    options.progress_callback = [&](float progress, const std::string&) {
        updates.push_back(progress);
        return true;
    };

    auto result = save_sog(splat, options);
    ASSERT_TRUE(result.has_value()) << "Failed to save SOG: " << result.error().format();
    EXPECT_TRUE(fs::exists(output_path));
    EXPECT_GT(fs::file_size(output_path), 0);
    expect_progress_completed(updates);
}

TEST_F(PythonIOTest, SogSaveCancellationKeepsExistingTarget) {
    auto splat = create_test_splat(200, 0);
    const fs::path output_path = temp_dir / "keep_existing_on_cancel.sog";
    const std::string existing_contents = "existing sog data";
    write_text_file(output_path, existing_contents);

    SogSaveOptions options;
    options.output_path = output_path;
    options.kmeans_iterations = 1;
    options.use_gpu = true;
    options.progress_callback = [](float progress, const std::string&) {
        return progress < 1.0f;
    };

    auto result = save_sog(splat, options);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::CANCELLED);
    expect_existing_target_and_no_temp_files(output_path, existing_contents);
}

// Test HTML export
TEST_F(PythonIOTest, HtmlExport) {
    const size_t num_points = 500;
    auto splat = create_test_splat(num_points, 1);

    fs::path output_path = temp_dir / "test_viewer.html";
    std::vector<float> updates;

    HtmlExportOptions options;
    options.output_path = output_path;
    options.kmeans_iterations = 5;
    options.progress_callback = [&](float progress, const std::string&) {
        updates.push_back(progress);
        return true;
    };

    auto result = export_html(splat, options);
    ASSERT_TRUE(result.has_value()) << "Failed to export HTML: " << result.error().format();
    EXPECT_TRUE(fs::exists(output_path));
    expect_progress_completed(updates);

    // Verify HTML content
    std::ifstream file(output_path);
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    EXPECT_TRUE(content.find("<!DOCTYPE html>") != std::string::npos ||
                content.find("<html") != std::string::npos)
        << "Should be valid HTML";
}

TEST_F(PythonIOTest, HtmlExportCancellationKeepsExistingTarget) {
    auto splat = create_test_splat(200, 0);
    const fs::path output_path = temp_dir / "keep_existing_on_cancel.html";
    const std::string existing_contents = "existing html data";
    write_text_file(output_path, existing_contents);

    HtmlExportOptions options;
    options.output_path = output_path;
    options.kmeans_iterations = 1;
    options.progress_callback = [](float progress, const std::string&) {
        return progress < 1.0f;
    };

    auto result = export_html(splat, options);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::CANCELLED);
    expect_existing_target_and_no_temp_files(output_path, existing_contents);
}

TEST_F(PythonIOTest, RadSaveCancellationKeepsExistingTarget) {
    auto splat = create_test_splat(200, 0);
    const fs::path output_path = temp_dir / "keep_existing_on_cancel.rad";
    const std::string existing_contents = "existing rad data";
    write_text_file(output_path, existing_contents);

    RadSaveOptions options;
    options.output_path = output_path;
    options.compression_level = 1;
    options.progress_callback = [](float progress, const std::string&) {
        return progress < 1.0f;
    };

    auto result = save_rad(splat, options);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::CANCELLED);
    expect_existing_target_and_no_temp_files(output_path, existing_contents);
}

TEST_F(PythonIOTest, RadSaveReportsProgressToCompletion) {
    auto splat = create_test_splat(200, 0);
    const fs::path output_path = temp_dir / "test_output.rad";
    std::vector<float> updates;

    RadSaveOptions options;
    options.output_path = output_path;
    options.compression_level = 1;
    options.progress_callback = [&](float progress, const std::string&) {
        updates.push_back(progress);
        return true;
    };

    auto result = save_rad(splat, options);
    ASSERT_TRUE(result.has_value()) << "Failed to save RAD: " << result.error().format();
    EXPECT_TRUE(fs::exists(output_path));
    EXPECT_GT(fs::file_size(output_path), 0);
    expect_progress_completed(updates);
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
    const std::string header_text = read_ply_header(output_path);
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
    const std::string header_text = read_ply_header(output_path);
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
    const std::string header_text = read_ply_header(output_path);
    EXPECT_NE(header_text.find("property float x"), std::string::npos);
    EXPECT_NE(header_text.find("property float y"), std::string::npos);
    EXPECT_NE(header_text.find("property float z"), std::string::npos);
    EXPECT_NE(header_text.find("property float opacity"), std::string::npos);
}

TEST_F(PythonIOTest, PlySaveWritesExtraAttributePayloads) {
    PointCloud pc;
    pc.means = Tensor::from_vector(
        std::vector<float>{0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f},
        {size_t{2}, size_t{3}}, Device::CPU);
    pc.attribute_names = {"x", "y", "z"};

    PlySaveOptions options;
    options.output_path = temp_dir / "extra_attributes_payload.ply";
    options.binary = true;
    options.extra_attributes = {
        PlyAttributeBlock{
            .values = Tensor::from_vector(std::vector<float>{0.25f, 0.75f}, {size_t{2}}, Device::CPU),
            .names = {"confidence"},
        },
        PlyAttributeBlock{
            .values = Tensor::from_vector(
                std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                {size_t{2}, size_t{3}}, Device::CPU),
            .names = {"velocity_0", "velocity_1", "velocity_2"},
        },
    };

    const auto result = save_ply(pc, options);
    ASSERT_TRUE(result.has_value()) << result.error().format();

    const std::string header_text = read_ply_header(options.output_path);
    EXPECT_NE(header_text.find("property float confidence"), std::string::npos);
    EXPECT_NE(header_text.find("property float velocity_0"), std::string::npos);
    EXPECT_NE(header_text.find("property float velocity_1"), std::string::npos);
    EXPECT_NE(header_text.find("property float velocity_2"), std::string::npos);

    EXPECT_EQ(read_ply_float_property(options.output_path, "confidence"),
              (std::vector<float>{0.25f, 0.75f}));
    EXPECT_EQ(read_ply_float_property(options.output_path, "velocity_0"),
              (std::vector<float>{1.0f, 4.0f}));
    EXPECT_EQ(read_ply_float_property(options.output_path, "velocity_1"),
              (std::vector<float>{2.0f, 5.0f}));
    EXPECT_EQ(read_ply_float_property(options.output_path, "velocity_2"),
              (std::vector<float>{3.0f, 6.0f}));
}

TEST_F(PythonIOTest, PlySaveFiltersExtraAttributesWhenDeletedMaskPresent) {
    auto splat = create_test_splat(3);
    splat.deleted() = Tensor::from_vector(std::vector<bool>{false, true, false}, {size_t{3}}, Device::CPU);

    PlySaveOptions options;
    options.output_path = temp_dir / "deleted_mask_extra_attributes.ply";
    options.binary = true;
    options.extra_attributes = {
        PlyAttributeBlock{
            .values = Tensor::from_vector(std::vector<float>{10.0f, 20.0f, 30.0f}, {size_t{3}}, Device::CPU),
            .names = {"confidence"},
        },
    };

    const auto result = save_ply(splat, options);
    ASSERT_TRUE(result.has_value()) << result.error().format();

    EXPECT_EQ(read_ply_float_property(options.output_path, "x"),
              (std::vector<float>{0.0f, 2.0f}));
    EXPECT_EQ(read_ply_float_property(options.output_path, "confidence"),
              (std::vector<float>{10.0f, 30.0f}));
}

TEST_F(PythonIOTest, PlySaveRejectsReservedExtraAttributeNames) {
    PointCloud pc;
    pc.means = Tensor::zeros({size_t{2}, size_t{3}}, Device::CPU, DataType::Float32);
    pc.attribute_names = {"x", "y", "z"};

    PlySaveOptions options;
    options.output_path = temp_dir / "reserved_extra_name.ply";
    options.binary = true;
    options.extra_attributes = {
        PlyAttributeBlock{
            .values = Tensor::ones({size_t{2}}, Device::CPU, DataType::Float32),
            .names = {"opacity"},
        },
    };

    const auto result = save_ply(pc, options);
    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().format().find("reserved"), std::string::npos);
}

TEST_F(PythonIOTest, PlySaveRejectsDuplicateExtraAttributeNames) {
    PointCloud pc;
    pc.means = Tensor::zeros({size_t{2}, size_t{3}}, Device::CPU, DataType::Float32);
    pc.attribute_names = {"x", "y", "z"};

    PlySaveOptions options;
    options.output_path = temp_dir / "duplicate_extra_name.ply";
    options.binary = true;
    options.extra_attributes = {
        PlyAttributeBlock{
            .values = Tensor::ones({size_t{2}}, Device::CPU, DataType::Float32),
            .names = {"confidence"},
        },
        PlyAttributeBlock{
            .values = Tensor::full({size_t{2}}, 2.0f, Device::CPU, DataType::Float32),
            .names = {"confidence"},
        },
    };

    const auto result = save_ply(pc, options);
    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().format().find("Duplicate PLY property name 'confidence'"), std::string::npos);
}

TEST_F(PythonIOTest, PlySaveRejectsWhitespaceInExtraAttributeNames) {
    PointCloud pc;
    pc.means = Tensor::zeros({size_t{2}, size_t{3}}, Device::CPU, DataType::Float32);
    pc.attribute_names = {"x", "y", "z"};

    PlySaveOptions options;
    options.output_path = temp_dir / "invalid_extra_name.ply";
    options.binary = true;
    options.extra_attributes = {
        PlyAttributeBlock{
            .values = Tensor::ones({size_t{2}}, Device::CPU, DataType::Float32),
            .names = {"bad name"},
        },
    };

    const auto result = save_ply(pc, options);
    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().format().find("without whitespace"), std::string::npos);
}

TEST_F(PythonIOTest, PlySaveRejectsExtraAttributeRowCountMismatch) {
    PointCloud pc;
    pc.means = Tensor::zeros({size_t{2}, size_t{3}}, Device::CPU, DataType::Float32);
    pc.attribute_names = {"x", "y", "z"};

    PlySaveOptions options;
    options.output_path = temp_dir / "mismatched_extra_rows.ply";
    options.binary = true;
    options.extra_attributes = {
        PlyAttributeBlock{
            .values = Tensor::ones({size_t{1}}, Device::CPU, DataType::Float32),
            .names = {"confidence"},
        },
    };

    const auto result = save_ply(pc, options);
    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().format().find("row count 1 does not match point count 2"), std::string::npos);
}

TEST_F(PythonIOTest, PlySaveRejectsEmptyExtraAttributesWhenDeletedMaskPresent) {
    auto splat = create_test_splat(3);
    splat.deleted() = Tensor::from_vector(std::vector<bool>{false, true, false}, {size_t{3}}, Device::CPU);

    PlySaveOptions options;
    options.output_path = temp_dir / "empty_extra_with_deleted_mask.ply";
    options.binary = true;
    options.extra_attributes = {
        PlyAttributeBlock{
            .values = Tensor(),
            .names = {"confidence"},
        },
    };

    const auto result = save_ply(splat, options);
    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().format().find("must not be empty"), std::string::npos);
}
