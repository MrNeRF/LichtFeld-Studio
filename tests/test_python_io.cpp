/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <iterator>
#include <limits>
#include <nlohmann/json.hpp>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/camera_types.h"
#include "core/error.hpp"
#include "core/image_io.hpp"
#include "core/point_cloud.hpp"
#include "core/splat_data.hpp"
#include "io/exporter.hpp"
#include "io/formats/ply.hpp"
#include "io/formats/rad.hpp"
#include "io/formats/transforms.hpp"
#include "io/loader.hpp"
#include "io/nvcodec_image_loader.hpp"
#include "io/pipelined_image_loader.hpp"
#include "io/project_container.hpp"
#include "io/project_document.hpp"
#include "licht_test_support.hpp"
#include "python/gil.hpp"
#include "python/runner.hpp"
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

    void write_png(const fs::path& path, int width = 1, int height = 1) const {
        ASSERT_GT(width, 0);
        ASSERT_GT(height, 0);

        // Minimal solid-color RGBA PNG for the given dimensions (no external deps).
        auto crc32 = [](const unsigned char* data, size_t len) -> uint32_t {
            uint32_t crc = 0xFFFFFFFFu;
            for (size_t i = 0; i < len; ++i) {
                crc ^= data[i];
                for (int b = 0; b < 8; ++b) {
                    const uint32_t mask = -(crc & 1u);
                    crc = (crc >> 1) ^ (0xEDB88320u & mask);
                }
            }
            return ~crc;
        };
        auto append_be32 = [](std::vector<unsigned char>& out, uint32_t value) {
            out.push_back(static_cast<unsigned char>((value >> 24) & 0xFF));
            out.push_back(static_cast<unsigned char>((value >> 16) & 0xFF));
            out.push_back(static_cast<unsigned char>((value >> 8) & 0xFF));
            out.push_back(static_cast<unsigned char>(value & 0xFF));
        };
        auto append_chunk = [&](std::vector<unsigned char>& out, const char type[4],
                                const std::vector<unsigned char>& data) {
            append_be32(out, static_cast<uint32_t>(data.size()));
            const size_t type_offset = out.size();
            out.insert(out.end(), type, type + 4);
            out.insert(out.end(), data.begin(), data.end());
            append_be32(out, crc32(out.data() + type_offset, 4 + data.size()));
        };

        std::vector<unsigned char> raw;
        raw.reserve(static_cast<size_t>(height) * (1u + static_cast<size_t>(width) * 4u));
        for (int y = 0; y < height; ++y) {
            raw.push_back(0); // filter: None
            for (int x = 0; x < width; ++x) {
                raw.push_back(0);
                raw.push_back(0);
                raw.push_back(0);
                raw.push_back(255);
            }
        }

        // Stored (uncompressed) deflate blocks — valid zlib stream for any size.
        std::vector<unsigned char> zlib_data;
        zlib_data.push_back(0x78);
        zlib_data.push_back(0x01);
        size_t offset = 0;
        const size_t raw_size = raw.size();
        while (offset < raw_size) {
            const size_t chunk = std::min<size_t>(65535, raw_size - offset);
            const bool final_block = (offset + chunk) >= raw_size;
            zlib_data.push_back(final_block ? 0x01 : 0x00);
            zlib_data.push_back(static_cast<unsigned char>(chunk & 0xFF));
            zlib_data.push_back(static_cast<unsigned char>((chunk >> 8) & 0xFF));
            const uint16_t nlen = static_cast<uint16_t>(~chunk);
            zlib_data.push_back(static_cast<unsigned char>(nlen & 0xFF));
            zlib_data.push_back(static_cast<unsigned char>((nlen >> 8) & 0xFF));
            zlib_data.insert(zlib_data.end(), raw.begin() + static_cast<std::ptrdiff_t>(offset),
                             raw.begin() + static_cast<std::ptrdiff_t>(offset + chunk));
            offset += chunk;
        }
        // Adler-32 of the raw scanline data
        uint32_t s1 = 1, s2 = 0;
        for (unsigned char byte : raw) {
            s1 = (s1 + byte) % 65521;
            s2 = (s2 + s1) % 65521;
        }
        append_be32(zlib_data, (s2 << 16) | s1);

        std::vector<unsigned char> png = {
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
        std::vector<unsigned char> ihdr;
        append_be32(ihdr, static_cast<uint32_t>(width));
        append_be32(ihdr, static_cast<uint32_t>(height));
        ihdr.push_back(8); // bit depth
        ihdr.push_back(6); // RGBA
        ihdr.push_back(0);
        ihdr.push_back(0);
        ihdr.push_back(0);
        append_chunk(png, "IHDR", ihdr);
        append_chunk(png, "IDAT", zlib_data);
        append_chunk(png, "IEND", {});

        fs::create_directories(path.parent_path());
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out.is_open()) << "Failed to open " << path;
        out.write(reinterpret_cast<const char*>(png.data()),
                  static_cast<std::streamsize>(png.size()));
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

    static void append_float(std::string& bytes, const float value) {
        bytes.append(reinterpret_cast<const char*>(&value), sizeof(value));
    }

    template <size_t N>
    static void append_floats(std::string& bytes, const std::array<float, N>& values) {
        for (const float value : values) {
            append_float(bytes, value);
        }
    }

    static void write_binary_test_ply(const fs::path& path, const std::string& header, const std::string& body) {
        std::ofstream out(path, std::ios::binary);
        if (!out.is_open()) {
            throw std::runtime_error("Failed to open test PLY for writing");
        }

        out.write(header.data(), static_cast<std::streamsize>(header.size()));
        out.write(body.data(), static_cast<std::streamsize>(body.size()));
        if (!out.good()) {
            throw std::runtime_error("Failed to write test PLY");
        }
    }

    static void write_basic_gaussian_ply(const fs::path& path,
                                         const std::vector<std::array<float, 11>>& vertices) {
        std::string body;
        body.reserve(vertices.size() * 11 * sizeof(float));
        for (const auto& vertex : vertices) {
            append_floats(body, vertex);
        }

        write_binary_test_ply(
            path,
            "ply\n"
            "format binary_little_endian 1.0\n"
            "element vertex " +
                std::to_string(vertices.size()) +
                "\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property float scale_0\n"
                "property float scale_1\n"
                "property float scale_2\n"
                "property float opacity\n"
                "property float rot_0\n"
                "property float rot_1\n"
                "property float rot_2\n"
                "property float rot_3\n"
                "end_header\n",
            body);
    }

    static void write_partial_rotation_ply(const fs::path& path) {
        std::string body;
        append_floats(body, std::array<float, 6>{1.0f, 2.0f, 3.0f, 1.0f, 0.0f, 0.0f});

        write_binary_test_ply(
            path,
            "ply\n"
            "format binary_little_endian 1.0\n"
            "element vertex 1\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float rot_0\n"
            "property float rot_1\n"
            "property float rot_2\n"
            "end_header\n",
            body);
    }

    static void write_gaussian_ply_with_extra_uchar_property(const fs::path& path) {
        std::string body;
        body.reserve(11 * sizeof(float) + 1);
        append_float(body, 1.0f);
        body.push_back(static_cast<char>(7));
        append_floats(body, std::array<float, 10>{2.0f, 3.0f, -2.0f, -2.0f, -2.0f, 0.25f, 1.0f, 0.0f, 0.0f, 0.0f});

        write_binary_test_ply(
            path,
            "ply\n"
            "format binary_little_endian 1.0\n"
            "element vertex 1\n"
            "property float x\n"
            "property uchar source_id\n"
            "property float y\n"
            "property float z\n"
            "property float scale_0\n"
            "property float scale_1\n"
            "property float scale_2\n"
            "property float opacity\n"
            "property float rot_0\n"
            "property float rot_1\n"
            "property float rot_2\n"
            "property float rot_3\n"
            "end_header\n",
            body);
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
    write_png(dataset_dir / "frame_0001.png");
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

TEST_F(PythonIOTest, RejectsMalformedTransformsCameraContractsBeforeCameraConstruction) {
    const fs::path transforms_path = temp_dir / "malformed_transforms.json";
    const auto expect_rejected = [&](nlohmann::json transforms) {
        write_text_file(transforms_path, transforms.dump());
        EXPECT_THROW(
            (void)read_transforms_cameras_and_images(transforms_path, {}),
            std::runtime_error);
    };

    const nlohmann::json identity_matrix = {
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 1.0},
    };
    nlohmann::json valid = {
        {"w", 64},
        {"h", 64},
        {"fl_x", 50.0},
        {"fl_y", 50.0},
        {"frames", nlohmann::json::array({
                       {{"file_path", "frame.png"}, {"transform_matrix", identity_matrix}},
                   })},
    };

    auto invalid = valid;
    invalid["w"] = 0;
    expect_rejected(std::move(invalid));

    invalid = valid;
    invalid["fl_x"] = 0.0;
    expect_rejected(std::move(invalid));

    invalid = valid;
    invalid["frames"][0]["transform_matrix"][2] = nlohmann::json::array({0.0, 0.0, 1.0});
    expect_rejected(std::move(invalid));

    invalid = valid;
    invalid["frames"][0]["transform_matrix"][2] = nlohmann::json::array({0.0, 0.0, 0.0, 0.0});
    expect_rejected(std::move(invalid));

    invalid = valid;
    invalid["frames"][0]["transform_matrix"][3] = nlohmann::json::array({0.0, 0.0, 1.0, 1.0});
    expect_rejected(std::move(invalid));
}

TEST_F(PythonIOTest, LoadTransformsPerFrameIntrinsics) {
    const fs::path dataset_dir = temp_dir / "per_frame_intrinsics";
    write_png(dataset_dir / "frame_a.png", 64, 48);
    write_png(dataset_dir / "frame_b.png", 128, 96);

    const nlohmann::json identity_matrix = {
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 1.0},
    };
    nlohmann::json transforms = {
        {"frames", nlohmann::json::array({
                       {
                           {"file_path", "frame_a.png"},
                           {"w", 64},
                           {"h", 48},
                           {"fl_x", 50.0},
                           {"fl_y", 45.0},
                           {"cx", 32.0},
                           {"cy", 24.0},
                           {"camera_model", "OPENCV"},
                           {"k1", 0.01},
                           {"k2", -0.02},
                           {"k3", 0.003},
                           {"p1", 0.0001},
                           {"p2", -0.0002},
                           {"k4", 0.001},
                           {"b1", 0.0003},
                           {"b2", -0.0004},
                           {"transform_matrix", identity_matrix},
                       },
                       {
                           {"file_path", "frame_b.png"},
                           {"w", 128},
                           {"h", 96},
                           {"fl_x", 90.0},
                           {"fl_y", 88.0},
                           {"cx", 64.0},
                           {"cy", 48.0},
                           {"camera_model", "OPENCV"},
                           {"k1", 0.05},
                           {"k2", -0.04},
                           {"k3", 0.006},
                           {"p1", 0.0005},
                           {"p2", -0.0006},
                           {"k4", 0.002},
                           {"b1", 0.0007},
                           {"b2", -0.0008},
                           {"transform_matrix", identity_matrix},
                       },
                   })},
    };
    write_text_file(dataset_dir / "transforms.json", transforms.dump());

    auto [cameras, center, splits] = read_transforms_cameras_and_images(dataset_dir / "transforms.json", {});
    (void)center;
    (void)splits;
    ASSERT_EQ(cameras.size(), 2u);

    EXPECT_EQ(cameras[0]._width, 64);
    EXPECT_EQ(cameras[0]._height, 48);
    EXPECT_FLOAT_EQ(cameras[0]._focal_x, 50.0f);
    EXPECT_FLOAT_EQ(cameras[0]._focal_y, 45.0f);
    EXPECT_FLOAT_EQ(cameras[0]._center_x, 32.0f);
    EXPECT_FLOAT_EQ(cameras[0]._center_y, 24.0f);
    EXPECT_EQ(cameras[0]._camera_model_type, CameraModelType::PINHOLE);
    ASSERT_EQ(cameras[0]._radial_distortion.numel(), 3u);
    ASSERT_EQ(cameras[0]._tangential_distortion.numel(), 2u);
    {
        const auto* radial = cameras[0]._radial_distortion.ptr<float>();
        const auto* tangential = cameras[0]._tangential_distortion.ptr<float>();
        EXPECT_FLOAT_EQ(radial[0], 0.01f);
        EXPECT_FLOAT_EQ(radial[1], -0.02f);
        EXPECT_FLOAT_EQ(radial[2], 0.003f);
        EXPECT_FLOAT_EQ(tangential[0], 0.0001f);
        EXPECT_FLOAT_EQ(tangential[1], -0.0002f);
    }

    EXPECT_EQ(cameras[1]._width, 128);
    EXPECT_EQ(cameras[1]._height, 96);
    EXPECT_FLOAT_EQ(cameras[1]._focal_x, 90.0f);
    EXPECT_FLOAT_EQ(cameras[1]._focal_y, 88.0f);
    EXPECT_FLOAT_EQ(cameras[1]._center_x, 64.0f);
    EXPECT_FLOAT_EQ(cameras[1]._center_y, 48.0f);
    EXPECT_EQ(cameras[1]._camera_model_type, CameraModelType::PINHOLE);
    ASSERT_EQ(cameras[1]._radial_distortion.numel(), 3u);
    ASSERT_EQ(cameras[1]._tangential_distortion.numel(), 2u);
    {
        const auto* radial = cameras[1]._radial_distortion.ptr<float>();
        const auto* tangential = cameras[1]._tangential_distortion.ptr<float>();
        EXPECT_FLOAT_EQ(radial[0], 0.05f);
        EXPECT_FLOAT_EQ(radial[1], -0.04f);
        EXPECT_FLOAT_EQ(radial[2], 0.006f);
        EXPECT_FLOAT_EQ(tangential[0], 0.0005f);
        EXPECT_FLOAT_EQ(tangential[1], -0.0006f);
    }

    // Distinct per-frame values prove resolution is not root-only.
    EXPECT_NE(cameras[0]._focal_x, cameras[1]._focal_x);
    EXPECT_NE(cameras[0]._width, cameras[1]._width);
}

TEST_F(PythonIOTest, LoadTransformsRootIntrinsicsFrameOverride) {
    const fs::path dataset_dir = temp_dir / "root_override_intrinsics";
    write_png(dataset_dir / "frame_0.png", 100, 80);
    write_png(dataset_dir / "frame_1.png", 100, 80);

    const nlohmann::json identity_matrix = {
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 1.0},
    };
    nlohmann::json transforms = {
        {"w", 100},
        {"h", 80},
        {"fl_x", 70.0},
        {"fl_y", 65.0},
        {"cx", 50.0},
        {"cy", 40.0},
        {"camera_model", "PINHOLE"},
        {"frames", nlohmann::json::array({
                       {
                           {"file_path", "frame_0.png"},
                           {"transform_matrix", identity_matrix},
                       },
                       {
                           {"file_path", "frame_1.png"},
                           {"fl_x", 110.0},
                           {"fl_y", 105.0},
                           {"cx", 49.0},
                           {"cy", 39.0},
                           {"transform_matrix", identity_matrix},
                       },
                   })},
    };
    write_text_file(dataset_dir / "transforms.json", transforms.dump());

    auto [cameras, center, splits] = read_transforms_cameras_and_images(dataset_dir / "transforms.json", {});
    (void)center;
    (void)splits;
    ASSERT_EQ(cameras.size(), 2u);

    EXPECT_FLOAT_EQ(cameras[0]._focal_x, 70.0f);
    EXPECT_FLOAT_EQ(cameras[0]._focal_y, 65.0f);
    EXPECT_FLOAT_EQ(cameras[0]._center_x, 50.0f);
    EXPECT_FLOAT_EQ(cameras[0]._center_y, 40.0f);

    EXPECT_FLOAT_EQ(cameras[1]._focal_x, 110.0f);
    EXPECT_FLOAT_EQ(cameras[1]._focal_y, 105.0f);
    EXPECT_FLOAT_EQ(cameras[1]._center_x, 49.0f);
    EXPECT_FLOAT_EQ(cameras[1]._center_y, 39.0f);
}

TEST_F(PythonIOTest, TransformsIdentityFrameYieldsIdentityExtrinsics) {
    const fs::path dataset_dir = temp_dir / "identity_extrinsics";
    write_png(dataset_dir / "frame.png", 32, 32);

    const nlohmann::json identity_matrix = {
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 1.0},
    };
    nlohmann::json transforms = {
        {"w", 32},
        {"h", 32},
        {"fl_x", 20.0},
        {"fl_y", 20.0},
        {"frames", nlohmann::json::array({
                       {
                           {"file_path", "frame.png"},
                           {"transform_matrix", identity_matrix},
                       },
                   })},
    };
    write_text_file(dataset_dir / "transforms.json", transforms.dump());

    auto [cameras, center, splits] = read_transforms_cameras_and_images(dataset_dir / "transforms.json", {});
    (void)center;
    (void)splits;
    ASSERT_EQ(cameras.size(), 1u);

    ASSERT_EQ(cameras[0]._R.ndim(), 2u);
    ASSERT_EQ(cameras[0]._R.size(0), 3u);
    ASSERT_EQ(cameras[0]._R.size(1), 3u);
    ASSERT_EQ(cameras[0]._T.numel(), 3u);

    auto R = cameras[0]._R.cpu().contiguous();
    auto T = cameras[0]._T.cpu().contiguous();
    auto R_acc = R.accessor<float, 2>();
    const auto* T_ptr = T.ptr<float>();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(R_acc(i, j), (i == j) ? 1.0f : 0.0f, 1e-5f)
                << "R(" << i << "," << j << ")";
        }
        EXPECT_NEAR(T_ptr[i], 0.0f, 1e-5f) << "T[" << i << "]";
    }
}

TEST_F(PythonIOTest, TransformsTranslationFrameMatchesFlippedWorld) {
    const fs::path dataset_dir = temp_dir / "translation_extrinsics";
    write_png(dataset_dir / "frame.png", 32, 32);

    const nlohmann::json transform_matrix = {
        {1.0, 0.0, 0.0, 1.0},
        {0.0, 1.0, 0.0, 2.0},
        {0.0, 0.0, 1.0, 3.0},
        {0.0, 0.0, 0.0, 1.0},
    };
    nlohmann::json transforms = {
        {"w", 32},
        {"h", 32},
        {"fl_x", 20.0},
        {"fl_y", 20.0},
        {"frames", nlohmann::json::array({
                       {
                           {"file_path", "frame.png"},
                           {"transform_matrix", transform_matrix},
                       },
                   })},
    };
    write_text_file(dataset_dir / "transforms.json", transforms.dump());

    auto [cameras, center, splits] = read_transforms_cameras_and_images(dataset_dir / "transforms.json", {});
    (void)center;
    (void)splits;
    ASSERT_EQ(cameras.size(), 1u);

    auto R = cameras[0]._R.cpu().contiguous();
    auto T = cameras[0]._T.cpu().contiguous();
    auto R_acc = R.accessor<float, 2>();
    const auto* T_ptr = T.ptr<float>();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(R_acc(i, j), (i == j) ? 1.0f : 0.0f, 1e-5f)
                << "R(" << i << "," << j << ")";
        }
    }
    EXPECT_NEAR(T_ptr[0], -1.0f, 1e-5f);
    EXPECT_NEAR(T_ptr[1], 2.0f, 1e-5f);
    EXPECT_NEAR(T_ptr[2], 3.0f, 1e-5f);
}

TEST_F(PythonIOTest, LoadTransformsCrlfFile) {
    const fs::path dataset_dir = temp_dir / "crlf_transforms";
    write_png(dataset_dir / "frame.png", 16, 16);

    std::string json = R"json({
  "w": 16,
  "h": 16,
  "fl_x": 10.0,
  "fl_y": 10.0,
  "frames": [
    {
      "file_path": "frame.png",
      "transform_matrix": [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
      ]
    }
  ]
})json";
    for (size_t pos = 0; (pos = json.find('\n', pos)) != std::string::npos;) {
        json.replace(pos, 1, "\r\n");
        pos += 2;
    }
    write_text_file(dataset_dir / "transforms.json", json);

    auto [cameras, center, splits] = read_transforms_cameras_and_images(dataset_dir / "transforms.json", {});
    (void)center;
    (void)splits;
    ASSERT_EQ(cameras.size(), 1u);
    EXPECT_EQ(cameras[0]._width, 16);
    EXPECT_EQ(cameras[0]._height, 16);
    EXPECT_FLOAT_EQ(cameras[0]._focal_x, 10.0f);
}

TEST_F(PythonIOTest, TransformsCameraAngleXPerFrame) {
    const fs::path dataset_dir = temp_dir / "camera_angle_x_frame";
    constexpr int resolution = 100;
    write_png(dataset_dir / "frame.png", resolution, resolution);

    const nlohmann::json identity_matrix = {
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 1.0},
    };
    nlohmann::json transforms = {
        {"w", resolution},
        {"h", resolution},
        {"frames", nlohmann::json::array({
                       {
                           {"file_path", "frame.png"},
                           {"camera_angle_x", std::numbers::pi_v<double> / 2.0},
                           {"transform_matrix", identity_matrix},
                       },
                   })},
    };
    write_text_file(dataset_dir / "transforms.json", transforms.dump());

    auto [cameras, center, splits] = read_transforms_cameras_and_images(dataset_dir / "transforms.json", {});
    (void)center;
    (void)splits;
    ASSERT_EQ(cameras.size(), 1u);

    const float expected = 0.5f * static_cast<float>(resolution) /
                           std::tan(static_cast<float>(std::numbers::pi_v<float> / 4.0f));
    EXPECT_NEAR(cameras[0]._focal_x, expected, 1e-4f);
    EXPECT_NEAR(cameras[0]._focal_y, expected, 1e-4f);
}

// Test loading COLMAP dataset
TEST_F(PythonIOTest, LoadCOLMAPDataset) {
    if (!fs::exists(bicycle_dir / "sparse")) {
        GTEST_SKIP() << "bicycle sparse data not available";
    }

    auto loader = Loader::create();
    LoadOptions options;
    // The committed masks are quarter-resolution and intentionally pair with images_4.
    options.resize_factor = 4;
    options.images_folder = "images_4";

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
    ASSERT_TRUE(loaded.has_value())
        << "Failed to load test PLY: " << lfs::format_for_developer(loaded.error());

    EXPECT_EQ(loaded->value.size(), 1UL);
    EXPECT_EQ(loaded->value.get_max_sh_degree(), 1);

    const auto sh0 = loaded->value.sh0().cpu().contiguous();
    ASSERT_EQ(sh0.ndim(), 3);
    ASSERT_EQ(sh0.size(0), 1);
    ASSERT_EQ(sh0.size(1), 1);
    ASSERT_EQ(sh0.size(2), 3);

    const auto* sh0_ptr = sh0.ptr<float>();
    EXPECT_FLOAT_EQ(sh0_ptr[0], 10.0f);
    EXPECT_FLOAT_EQ(sh0_ptr[1], 20.0f);
    EXPECT_FLOAT_EQ(sh0_ptr[2], 30.0f);

    const auto shN = loaded->value.shN_canonical_cpu().contiguous();
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

TEST_F(PythonIOTest, PlyLoadDiscardsInvalidGaussianRowsBeforeImport) {
    const fs::path input_path = temp_dir / "invalid_gaussian_rows.ply";
    const float nan = std::numeric_limits<float>::quiet_NaN();
    write_basic_gaussian_ply(
        input_path,
        {
            std::array<float, 11>{1.0f, 2.0f, 3.0f, -2.0f, -2.0f, -2.0f, 0.25f, 1.0f, 0.0f, 0.0f, 0.0f},
            std::array<float, 11>{0.0f, 0.0f, 0.0f, -2.0f, -2.0f, -2.0f, 0.25f, 0.0f, 0.0f, 0.0f, 0.0f},
            std::array<float, 11>{nan, 4.0f, 5.0f, -2.0f, -2.0f, -2.0f, 0.25f, 1.0f, 0.0f, 0.0f, 0.0f},
        });

    auto loaded = load_ply(input_path);
    ASSERT_TRUE(loaded.has_value())
        << "Failed to load test PLY: " << lfs::format_for_developer(loaded.error());
    EXPECT_EQ(loaded->value.size(), 1UL);

    const auto means = loaded->value.means().cpu().contiguous();
    ASSERT_EQ(means.numel(), 3);
    const auto* means_ptr = means.ptr<float>();
    EXPECT_FLOAT_EQ(means_ptr[0], 1.0f);
    EXPECT_FLOAT_EQ(means_ptr[1], 2.0f);
    EXPECT_FLOAT_EQ(means_ptr[2], 3.0f);

    const auto rotation = loaded->value.rotation_raw().cpu().contiguous();
    ASSERT_EQ(rotation.numel(), 4);
    const auto* rotation_ptr = rotation.ptr<float>();
    EXPECT_FLOAT_EQ(rotation_ptr[0], 1.0f);
    EXPECT_FLOAT_EQ(rotation_ptr[1], 0.0f);
    EXPECT_FLOAT_EQ(rotation_ptr[2], 0.0f);
    EXPECT_FLOAT_EQ(rotation_ptr[3], 0.0f);
}

TEST_F(PythonIOTest, PlyLoadRejectsPartialRotationSchema) {
    const fs::path input_path = temp_dir / "partial_rotation_schema.ply";
    write_partial_rotation_ply(input_path);

    const auto loaded = load_ply(input_path);
    ASSERT_FALSE(loaded.has_value());
    EXPECT_NE(loaded.error().detail().find("rotation properties must include rot_0, rot_1, rot_2, and rot_3"),
              std::string::npos);
}

TEST_F(PythonIOTest, PlyLoadRejectsNonemptyElementsBeforeVertices) {
    const fs::path input_path = temp_dir / "pre_vertex_element.ply";
    std::string body;
    append_float(body, 123.0f);
    append_floats(body, std::array<float, 11>{
                            1.0f, 2.0f, 3.0f,
                            -2.0f, -2.0f, -2.0f,
                            0.25f,
                            1.0f, 0.0f, 0.0f, 0.0f});
    write_binary_test_ply(
        input_path,
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element metadata 1\n"
        "property float value\n"
        "element vertex 1\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float scale_0\n"
        "property float scale_1\n"
        "property float scale_2\n"
        "property float opacity\n"
        "property float rot_0\n"
        "property float rot_1\n"
        "property float rot_2\n"
        "property float rot_3\n"
        "end_header\n",
        body);

    const auto result = load_ply(input_path);
    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().detail().find("before vertex"), std::string::npos);
}

TEST_F(PythonIOTest, PlyClassifiersRequireBoundedCompleteHeaders) {
    const fs::path input_path = temp_dir / "unterminated_header.ply";
    std::string contents =
        "ply\n"
        "format ascii 1.0\n"
        "element face 1\n"
        "property float opacity\n"
        "property float scale_0\n"
        "property float rot_0\n";
    contents.append(2 * 1024 * 1024, 'x');
    write_text_file(input_path, contents);

    EXPECT_FALSE(ply_has_faces(input_path));
    EXPECT_FALSE(is_gaussian_splat_ply(input_path));
}

TEST_F(PythonIOTest, PlyLoadAccountsForNonFloatVertexProperties) {
    const fs::path input_path = temp_dir / "extra_uchar_property.ply";
    write_gaussian_ply_with_extra_uchar_property(input_path);

    auto loaded = load_ply(input_path);
    ASSERT_TRUE(loaded.has_value())
        << "Failed to load test PLY: " << lfs::format_for_developer(loaded.error());
    EXPECT_EQ(loaded->value.size(), 1UL);

    const auto means = loaded->value.means().cpu().contiguous();
    ASSERT_EQ(means.numel(), 3);
    const auto* means_ptr = means.ptr<float>();
    EXPECT_FLOAT_EQ(means_ptr[0], 1.0f);
    EXPECT_FLOAT_EQ(means_ptr[1], 2.0f);
    EXPECT_FLOAT_EQ(means_ptr[2], 3.0f);
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
    EXPECT_TRUE(content.find("window.__lfsInitMeasureTool") != std::string::npos)
        << "Should embed the measure tool init hook";
    EXPECT_TRUE(content.find("export { Gizmo, TranslateGizmo };") == std::string::npos)
        << "Gizmo export statement should be stripped";
    EXPECT_TRUE(content.find("export { initMeasureTool };") == std::string::npos)
        << "Measure-tool export statement should be stripped";
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

TEST_F(PythonIOTest, RadSaveLoadRoundtripPreservesCompactSplat) {
    const auto original = create_test_splat(64, 1);
    const auto output_path = temp_dir / "compact_roundtrip.rad";

    RadSaveOptions options;
    options.output_path = output_path;
    options.compression_level = 1;
    const auto save_result = save_rad(original, options);
    ASSERT_TRUE(save_result.has_value()) << save_result.error().format();

    const auto loaded = load_rad(output_path);
    ASSERT_TRUE(loaded.has_value()) << loaded.error();
    ASSERT_TRUE(loaded->lod_tree);
    ASSERT_TRUE(loaded->lod_tree->has_tree());
    EXPECT_GE(loaded->size(), original.size());
    EXPECT_EQ(loaded->get_max_sh_degree(), original.get_max_sh_degree());

    const auto original_means = original.means_raw().cpu().contiguous();
    const auto loaded_means = loaded->means_raw().cpu().contiguous();
    const auto* original_ptr = original_means.ptr<float>();
    const auto* loaded_ptr = loaded_means.ptr<float>();

    std::vector<std::array<float, 3>> expected_positions;
    expected_positions.reserve(original.size());
    for (size_t i = 0; i < original.size(); ++i) {
        expected_positions.push_back(
            {original_ptr[i * 3], original_ptr[i * 3 + 1], original_ptr[i * 3 + 2]});
    }

    std::vector<std::array<float, 3>> leaf_positions;
    for (size_t i = 0; i < static_cast<size_t>(loaded->size()); ++i) {
        if (loaded->lod_tree->child_count_at(i) == 0) {
            leaf_positions.push_back(
                {loaded_ptr[i * 3], loaded_ptr[i * 3 + 1], loaded_ptr[i * 3 + 2]});
        }
    }

    std::sort(expected_positions.begin(), expected_positions.end());
    std::sort(leaf_positions.begin(), leaf_positions.end());
    EXPECT_EQ(leaf_positions, expected_positions);
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
    EXPECT_NE(result.error().format().find("must be valid"), std::string::npos);
}

TEST_F(PythonIOTest, PipelinedLoaderStatsRemainResponsiveDuringCompletions) {
    const auto image_path = temp_dir / "stats_source.png";
    write_png(image_path);

    PipelinedLoaderConfig config;
    config.jpeg_batch_size = 4;
    config.decoder_pool_size = 4;
    config.prefetch_count = 64;
    config.output_queue_size = 64;
    config.io_threads = 2;
    config.cold_process_threads = 2;

    PipelinedImageLoader loader(config);
    LoadParams params{.resize_factor = 1, .max_width = 0};
    constexpr size_t request_count = 256;
    for (size_t i = 0; i < request_count; ++i) {
        loader.prefetch(i, image_path, params);
    }

    std::atomic<bool> stop_polling{false};
    auto poller = std::async(std::launch::async, [&] {
        while (!stop_polling.load(std::memory_order_acquire)) {
            (void)loader.get_stats();
        }
    });

    for (size_t i = 0; i < request_count; ++i) {
        const auto ready = loader.get();
        EXPECT_TRUE(ready.tensor.is_valid());
    }
    stop_polling.store(true, std::memory_order_release);

    ASSERT_EQ(poller.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    poller.get();
    EXPECT_EQ(loader.get_stats().total_images_loaded, request_count);
}

TEST_F(PythonIOTest, PipelinedLoaderReportsPrimaryImageFailure) {
    PipelinedLoaderConfig config;
    config.jpeg_batch_size = 1;
    config.decoder_pool_size = 1;
    config.prefetch_count = 1;
    config.output_queue_size = 1;
    config.io_threads = 1;
    config.cold_process_threads = 1;

    PipelinedImageLoader loader(config);
    const auto missing_path = temp_dir / "missing_training_image.png";
    loader.prefetch(17, missing_path, LoadParams{});

    try {
        (void)loader.get();
        FAIL() << "Expected the primary image failure to reach the consumer";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("missing_training_image.png"), std::string::npos);
    }
    EXPECT_EQ(loader.in_flight_count(), 0u);
}

TEST_F(PythonIOTest, PipelinedLoaderShutdownReleasesQueuedGpuTensorsBeforeDecodeStream) {
    if (!NvCodecImageLoader::is_available()) {
        GTEST_SKIP() << "nvImageCodec is unavailable";
    }

    constexpr size_t width = 320;
    constexpr size_t height = 240;
    const auto image_path = temp_dir / "shutdown_stream_source.jpg";
    auto image = Tensor::empty(
        {height, width, size_t{3}}, Device::CPU, DataType::UInt8);
    std::fill_n(image.ptr<uint8_t>(), image.numel(), uint8_t{127});
    save_image_u8(image_path, std::move(image));

    PipelinedLoaderConfig config;
    config.jpeg_batch_size = 1;
    config.decoder_pool_size = 1;
    config.prefetch_count = 1;
    config.output_queue_size = 1;
    config.io_threads = 1;
    config.cold_process_threads = 1;

    PipelinedImageLoader loader(config);
    loader.prefetch(0, image_path, LoadParams{});

    const auto ready_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (loader.ready_count() == 0 &&
           std::chrono::steady_clock::now() < ready_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    ASSERT_EQ(loader.ready_count(), 1u);
    EXPECT_GT(loader.get_gpu_memory_stats().total_bytes(), 0u);

    // Do not consume the output. shutdown() must retire this stream-owned tensor
    // before it releases the allocator's stream references and destroys the stream.
    loader.shutdown();

    const auto stats = loader.get_stats();
    EXPECT_EQ(stats.prefetch_queue_size, 0u);
    EXPECT_EQ(stats.hot_queue_size, 0u);
    EXPECT_EQ(stats.cold_queue_size, 0u);
    EXPECT_EQ(stats.output_queue_size, 0u);
    EXPECT_EQ(stats.pending_pairs_count, 0u);
    EXPECT_EQ(loader.get_gpu_memory_stats().total_bytes(), 0u);
    EXPECT_EQ(loader.in_flight_count(), 0u);

    void* probe = nullptr;
    ASSERT_EQ(cudaMallocAsync(&probe, 4096, nullptr), cudaSuccess);
    ASSERT_EQ(cudaFreeAsync(probe, nullptr), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

namespace {
    bool containsLichtfeldModule(const fs::path& dir) {
        std::error_code ec;
        if (!fs::exists(dir, ec)) {
            return false;
        }
        for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
            std::error_code file_ec;
            if (!it->is_regular_file(file_ec) || file_ec) {
                continue;
            }
            const auto filename = it->path().filename().string();
            const auto ext = it->path().extension().string();
            if ((ext == ".so" || ext == ".pyd") && filename.rfind("lichtfeld", 0) == 0) {
                return true;
            }
        }
        return false;
    }

    fs::path findPythonModuleDir() {
        std::error_code ec;
        const auto cwd = fs::current_path(ec);
        const auto project_root = fs::path(PROJECT_ROOT_PATH);
        for (const auto& candidate : {
                 cwd / "src" / "python",
                 cwd.parent_path() / "src" / "python",
                 project_root / "build" / "src" / "python",
             }) {
            if (containsLichtfeldModule(candidate)) {
                return candidate;
            }
        }
        return {};
    }

    void prependPythonPath(const fs::path& path) {
        const auto value = path.string();
        const char* existing = std::getenv("PYTHONPATH");
#ifdef _WIN32
        const char separator = ';';
#else
        const char separator = ':';
#endif
        const std::string combined =
            existing && *existing ? value + separator + std::string(existing) : value;
#ifdef _WIN32
        _putenv_s("PYTHONPATH", combined.c_str());
#else
        setenv("PYTHONPATH", combined.c_str(), 1);
#endif
    }

    std::string consumePythonError() {
        if (!PyErr_Occurred()) {
            return "unknown Python error";
        }
        PyObject* type = nullptr;
        PyObject* value = nullptr;
        PyObject* traceback = nullptr;
        PyErr_Fetch(&type, &value, &traceback);
        PyErr_NormalizeException(&type, &value, &traceback);
        std::string message = "unknown Python error";
        if (value) {
            PyObject* as_str = PyObject_Str(value);
            if (as_str) {
                if (const char* utf8 = PyUnicode_AsUTF8(as_str)) {
                    message = utf8;
                }
                Py_DECREF(as_str);
            }
        }
        Py_XDECREF(type);
        Py_XDECREF(value);
        Py_XDECREF(traceback);
        return message;
    }
} // namespace

TEST_F(PythonIOTest, InspectProjectFromTwoThreadsReturnsSameUuid) {
    using namespace lfs::io::project;
    using namespace lfs::test::licht;

    const auto container_path = temp_dir / "concurrent_inspect.licht";
    const auto project_uuid = fixed_uuid(11);
    const auto payload = byte_vector("inspect-concurrency");
    {
        auto created = ProjectWriter::create(
            container_path,
            CreateOptions{
                .project_uuid = project_uuid,
                .file_uuid = fixed_uuid(12),
                .role = ContainerRole::Master,
                .creation_time_unix_ns = 1'735'689'600'000'000'000,
                .index_compression = IndexCompression::StoredForDeterministicTests,
                .disk_reserve_bytes = 0,
            });
        ASSERT_TRUE(created) << lfs::format_for_developer(created.error());
        auto writer = std::move(*created);
        auto planned = writer.plan_commit(CommitOptions{
            .kind = CommitKind::Explicit,
            .commit_uuid = fixed_uuid(13),
            .snapshot_uuid = fixed_uuid(14),
            .wallclock_unix_ns = 1'735'689'601'000'000'000,
        });
        ASSERT_TRUE(planned) << lfs::format_for_developer(planned.error());
        auto preflighted = writer.preflight(payload.size());
        ASSERT_TRUE(preflighted) << lfs::format_for_developer(preflighted.error());
        auto written = writer.write_chunk(fixed_key("TEST", 15), payload);
        ASSERT_TRUE(written) << lfs::format_for_developer(written.error());
        auto committed = writer.commit();
        ASSERT_TRUE(committed) << lfs::format_for_developer(committed.error());
    }

    const auto module_dir = findPythonModuleDir();
    ASSERT_FALSE(module_dir.empty()) << "Could not locate built lichtfeld module";
    prependPythonPath(module_dir);
    const auto init = lfs::python::ensure_initialized();
    ASSERT_TRUE(init) << lfs::format_for_developer(init.error());

    const auto script = std::format(R"PY(
import threading
import lichtfeld as lf
path = r"{}"
uuids = [None, None]
errors = [None, None]

def worker(index):
    try:
        uuids[index] = lf.io.inspect_project(path).project_uuid
    except Exception as exc:
        errors[index] = repr(exc)

threads = [
    threading.Thread(target=worker, args=(0,)),
    threading.Thread(target=worker, args=(1,)),
]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join(timeout=30)
result_alive = any(thread.is_alive() for thread in threads)
result_errors = [error for error in errors if error is not None]
result_uuid_a = uuids[0]
result_uuid_b = uuids[1]
)PY",
                                    container_path.generic_string());

    const lfs::python::GilAcquire gil;
    PyObject* namespace_dict = PyDict_New();
    ASSERT_TRUE(namespace_dict);
    PyDict_SetItemString(namespace_dict, "__builtins__", PyEval_GetBuiltins());
    PyObject* exec_result =
        PyRun_String(script.c_str(), Py_file_input, namespace_dict, namespace_dict);
    if (!exec_result) {
        const auto python_error = consumePythonError();
        Py_DECREF(namespace_dict);
        FAIL() << python_error;
    }
    Py_DECREF(exec_result);

    auto* const alive_obj = PyDict_GetItemString(namespace_dict, "result_alive");
    auto* const errors_obj = PyDict_GetItemString(namespace_dict, "result_errors");
    auto* const uuid_a_obj = PyDict_GetItemString(namespace_dict, "result_uuid_a");
    auto* const uuid_b_obj = PyDict_GetItemString(namespace_dict, "result_uuid_b");
    ASSERT_TRUE(alive_obj);
    ASSERT_TRUE(errors_obj);
    ASSERT_TRUE(uuid_a_obj);
    ASSERT_TRUE(uuid_b_obj);
    EXPECT_FALSE(PyObject_IsTrue(alive_obj)) << "inspect_project hung on a worker thread";
    EXPECT_EQ(PyList_Size(errors_obj), 0);
    ASSERT_TRUE(PyUnicode_Check(uuid_a_obj));
    ASSERT_TRUE(PyUnicode_Check(uuid_b_obj));
    const std::string uuid_a = PyUnicode_AsUTF8(uuid_a_obj);
    const std::string uuid_b = PyUnicode_AsUTF8(uuid_b_obj);
    EXPECT_EQ(uuid_a, uuid_b);
    EXPECT_EQ(uuid_a, project_uuid.to_string());

    Py_DECREF(namespace_dict);
}

TEST_F(PythonIOTest, InspectProjectFallbackPreviewPath) {
    using namespace lfs::io::project;
    using namespace lfs::test::licht;

    const auto dataset_root = temp_dir / "fallback_dataset";
    const auto images_dir = dataset_root / "images";
    const auto container_path = temp_dir / "fallback_inspect.licht";
    fs::create_directories(dataset_root);

    auto document = make_empty_document(fixed_uuid(21), 100);
    auto snapshot = require_result(document->parameters().snapshot());
    snapshot.dataset.images = "images";
    require_status(document->edit_parameters().set_snapshot(snapshot));
    const auto dataset_uuid = require_result(upsert_path_reference(
        document->edit_references(), {}, dataset_root, "dataset.root", "dataset"));
    require_status(document->edit_project().set_dataset_reference(dataset_uuid));
    auto options = deterministic_document_save_options(0x70000000, 22, 300);
    auto saved = document->save(container_path, options);
    ASSERT_TRUE(saved) << lfs::format_for_developer(saved.error());

    fs::create_directories(images_dir);
    write_png(images_dir / "zulu.png", 8, 8);
    write_png(images_dir / "alpha.png", 8, 8);

    const auto module_dir = findPythonModuleDir();
    ASSERT_FALSE(module_dir.empty()) << "Could not locate built lichtfeld module";
    prependPythonPath(module_dir);
    const auto init = lfs::python::ensure_initialized();
    ASSERT_TRUE(init) << lfs::format_for_developer(init.error());

    const auto script = std::format(R"PY(
import lichtfeld as lf
i = lf.io.inspect_project(r"{}")
result_has_preview = i.has_preview
result_fallback = i.fallback_preview_path
)PY",
                                    container_path.generic_string());

    const lfs::python::GilAcquire gil;
    PyObject* namespace_dict = PyDict_New();
    ASSERT_TRUE(namespace_dict);
    PyDict_SetItemString(namespace_dict, "__builtins__", PyEval_GetBuiltins());
    PyObject* exec_result =
        PyRun_String(script.c_str(), Py_file_input, namespace_dict, namespace_dict);
    if (!exec_result) {
        const auto python_error = consumePythonError();
        Py_DECREF(namespace_dict);
        FAIL() << python_error;
    }
    Py_DECREF(exec_result);

    auto* const has_preview_obj = PyDict_GetItemString(namespace_dict, "result_has_preview");
    auto* const fallback_obj = PyDict_GetItemString(namespace_dict, "result_fallback");
    ASSERT_TRUE(has_preview_obj);
    ASSERT_TRUE(fallback_obj);
    EXPECT_FALSE(PyObject_IsTrue(has_preview_obj));
    ASSERT_TRUE(PyUnicode_Check(fallback_obj));
    const fs::path fallback = PyUnicode_AsUTF8(fallback_obj);
    EXPECT_EQ(fallback.filename(), "alpha.png");
    EXPECT_TRUE(fs::equivalent(fallback, images_dir / "alpha.png"));
    Py_DECREF(namespace_dict);
}

TEST_F(PythonIOTest, InspectProjectFallbackEmptyWhenPreviewEmbedded) {
    using namespace lfs::io::project;
    using namespace lfs::test::licht;

    const auto dataset_root = temp_dir / "embedded_dataset";
    const auto images_dir = dataset_root / "images";
    fs::create_directories(images_dir);
    write_png(images_dir / "scene.png", 16, 16);

    const auto container_path = temp_dir / "embedded_inspect.licht";
    auto document = make_empty_document(fixed_uuid(31), 100);
    auto snapshot = require_result(document->parameters().snapshot());
    snapshot.dataset.images = "images";
    require_status(document->edit_parameters().set_snapshot(snapshot));
    const auto dataset_uuid = require_result(upsert_path_reference(
        document->edit_references(), {}, dataset_root, "dataset.root", "dataset"));
    require_status(document->edit_project().set_dataset_reference(dataset_uuid));
    auto options = deterministic_document_save_options(0x70000000, 32, 300);
    auto saved = document->save(container_path, options);
    ASSERT_TRUE(saved) << lfs::format_for_developer(saved.error());

    const auto module_dir = findPythonModuleDir();
    ASSERT_FALSE(module_dir.empty()) << "Could not locate built lichtfeld module";
    prependPythonPath(module_dir);
    const auto init = lfs::python::ensure_initialized();
    ASSERT_TRUE(init) << lfs::format_for_developer(init.error());

    const auto script = std::format(R"PY(
import lichtfeld as lf
i = lf.io.inspect_project(r"{}")
result_has_preview = i.has_preview
result_fallback = i.fallback_preview_path
)PY",
                                    container_path.generic_string());

    const lfs::python::GilAcquire gil;
    PyObject* namespace_dict = PyDict_New();
    ASSERT_TRUE(namespace_dict);
    PyDict_SetItemString(namespace_dict, "__builtins__", PyEval_GetBuiltins());
    PyObject* exec_result =
        PyRun_String(script.c_str(), Py_file_input, namespace_dict, namespace_dict);
    if (!exec_result) {
        const auto python_error = consumePythonError();
        Py_DECREF(namespace_dict);
        FAIL() << python_error;
    }
    Py_DECREF(exec_result);

    auto* const has_preview_obj = PyDict_GetItemString(namespace_dict, "result_has_preview");
    auto* const fallback_obj = PyDict_GetItemString(namespace_dict, "result_fallback");
    ASSERT_TRUE(has_preview_obj);
    ASSERT_TRUE(fallback_obj);
    EXPECT_TRUE(PyObject_IsTrue(has_preview_obj));
    ASSERT_TRUE(PyUnicode_Check(fallback_obj));
    EXPECT_STREQ(PyUnicode_AsUTF8(fallback_obj), "");
    Py_DECREF(namespace_dict);
}
