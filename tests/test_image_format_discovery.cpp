/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/image_exr.hpp"
#include "core/image_io.hpp"
#include "io/filesystem_utils.hpp"
#include "io/loaders/colmap_loader.hpp"
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>

namespace fs = std::filesystem;
namespace {
    class ImageFormatDiscoveryTest : public ::testing::Test {
    protected:
        fs::path temp_dir_;
        void SetUp() override {
            temp_dir_ = fs::temp_directory_path() / ("lfs-image-discovery-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            ASSERT_TRUE(fs::create_directory(temp_dir_));
        }
        void TearDown() override {
            std::error_code error;
            fs::remove_all(temp_dir_, error);
        }
        void write_text_file(const fs::path& path, const std::string& contents) {
            fs::create_directories(path.parent_path());
            std::ofstream out(path, std::ios::binary);
            out << contents;
            ASSERT_TRUE(out.good());
        }
        void write_minimal_colmap_text_dataset(const fs::path& path, const std::string& name) {
            write_text_file(path / "cameras.txt", "1 PINHOLE 2 2 1 1 1 1\n");
            write_text_file(path / "images.txt", "1 1 0 0 0 0 0 0 1 " + name + "\n\n");
        }
    };
    TEST_F(ImageFormatDiscoveryTest, FlatExrAndWebpDatasetsUseExistingCpuDecoders) {
        for (const auto extension : {".exr", ".webp"}) {
            const auto dataset = temp_dir_ / std::string(extension + 1);
            const auto name = std::string("frame") + extension;
            write_minimal_colmap_text_dataset(dataset, name);
            write_text_file(dataset / "cameras.txt", "1 PINHOLE 2 2 1 1 1 1\n");
            const auto path = dataset / name;
            if (std::string_view(extension) == ".webp") {
                // Independent 2x2 lossless red fixture encoded with FFmpeg/libwebp.
                const std::array<unsigned char, 36> bytes{82, 73, 70, 70, 28, 0, 0, 0, 87, 69, 66, 80, 86, 80, 56, 76, 15, 0, 0, 0, 47, 1, 64, 0, 0, 7, 16, 245, 143, 254, 7, 34, 162, 255, 1, 0};
                std::ofstream out(path, std::ios::binary);
                out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
                ASSERT_TRUE(out.good());
            } else {
                namespace codec = lfs::core::image_codecs;
                const std::array<float, 12> pixels{1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0};
                std::ofstream out(path, std::ios::binary);
                const auto write = [](void* user, const void* data, uint64_t size, uint64_t offset) noexcept -> int64_t {
                    auto& stream = *static_cast<std::ofstream*>(user);
                    stream.seekp(static_cast<std::streamoff>(offset));
                    stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
                    return stream.good() ? static_cast<int64_t>(size) : -1;
                };
                const auto encoded = codec::encode_exr({2, 2, 3, 6 * sizeof(float), {reinterpret_cast<const std::uint8_t*>(pixels.data()), sizeof(pixels)}},
                                                       {name, &out, write}, {});
                ASSERT_TRUE(encoded.has_value());
            }
            auto [samples, width, height, channels] = lfs::core::load_image(path, 1, 0);
            const auto owned = std::unique_ptr<unsigned char, decltype(&std::free)>(samples, &std::free);
            ASSERT_NE(samples, nullptr);
            ASSERT_EQ(width, 2);
            ASSERT_EQ(height, 2);
            ASSERT_EQ(channels, 3);
            for (int pixel = 0; pixel < 4; ++pixel) {
                EXPECT_GE(samples[pixel * 3], 250);
                EXPECT_LE(samples[pixel * 3 + 1], 2);
                EXPECT_LE(samples[pixel * 3 + 2], 2);
            }
            lfs::io::ColmapLoader loader;
            const auto loaded = loader.load(dataset, {.validate_only = true});
            ASSERT_TRUE(loaded.has_value()) << loaded.error().format();
            EXPECT_EQ(lfs::io::count_image_files(dataset, false), 1);
            for (const auto folder : {"a", "b"}) {
                fs::create_directories(dataset / "images" / folder);
                fs::copy_file(path, dataset / "images" / folder / name);
            }
            const auto ambiguous = loader.load(dataset, {.validate_only = true});
            ASSERT_FALSE(ambiguous.has_value());
            EXPECT_EQ(ambiguous.error().code, lfs::io::ErrorCode::INVALID_DATASET);
            EXPECT_NE(ambiguous.error().message.find("basename only"), std::string::npos);
        }
    }

} // namespace
