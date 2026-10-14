/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/image_exr.hpp"
#include "core/image_heif.hpp"
#include "core/image_io.hpp"
#include "core/tensor_backend.hpp"
#include "io/filesystem_utils.hpp"
#include "io/loaders/blender_loader.hpp"
#include "io/loaders/colmap_loader.hpp"
#include "io/project_document.hpp"
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
    TEST_F(ImageFormatDiscoveryTest, ExistingDatasetKeepsPreviewAndCountsWithFormatCompanions) {
        const auto dataset = temp_dir_ / "companions";
        write_minimal_colmap_text_dataset(dataset, "frame.jpg");
        const std::array<uint8_t, 12> rgb{255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0};
        std::string error;
        ASSERT_TRUE(lfs::core::image_codecs::write_jpeg(dataset / "frame.jpg", rgb.data(), 2, 2, 3, 95, std::nullopt, error));
        write_text_file(dataset / "frame.exr", "unreferenced depth/HDR companion");
        write_text_file(dataset / "frame.webp", "unreferenced image companion");
        lfs::io::ColmapLoader loader;
        ASSERT_TRUE(loader.load(dataset, {.validate_only = true}).has_value());
        EXPECT_EQ(lfs::io::count_image_files(dataset, false), 1);
        EXPECT_EQ(lfs::io::detect_dataset_info(dataset).image_count, 1);
        lfs::core::param::DatasetConfig config;
        config.data_path = dataset;
        config.images = ".";
        ASSERT_TRUE(lfs::io::project::first_dataset_image(config).has_value());
        EXPECT_EQ(lfs::io::project::first_dataset_image(config)->lexically_normal(), dataset / "frame.jpg");
        write_text_file(dataset / "second.JPG", "legacy uppercase extension");
        write_text_file(dataset / "second.WEBP", "same-stem uppercase companion");
        write_text_file(dataset / "nested" / "frame.exr", "standalone in a different directory");
        EXPECT_EQ(lfs::io::count_image_files(dataset, false), 2);
        EXPECT_EQ(lfs::io::count_image_files(dataset, true), 3);
        EXPECT_EQ(lfs::io::detect_dataset_info(dataset).image_count, 2);
        fs::remove(dataset / "frame.jpg");
        fs::remove(dataset / "second.JPG");
        EXPECT_EQ(lfs::io::count_image_files(dataset, false), 3);
        ASSERT_TRUE(lfs::io::project::first_dataset_image(config).has_value());
        EXPECT_EQ(lfs::io::project::first_dataset_image(config)->lexically_normal(), dataset / "frame.exr");
    }
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

    TEST_F(ImageFormatDiscoveryTest, HeifDatasetLoadsAndCompanionsPreserveLegacyCounts) {
        const auto fixture = fs::path(LFS_HEIF_FIXTURES) / "rgb12.avif";
        size_t case_index = 0;
        for (const auto extension : {".avif", ".AVIF", ".HEIF"}) {
            const auto dataset = temp_dir_ / std::to_string(case_index++);
            const auto name = std::string("frame") + extension;
            write_minimal_colmap_text_dataset(dataset, name);
            write_text_file(dataset / "cameras.txt", "1 PINHOLE 32 24 30 30 16 12\n");
            fs::copy_file(fixture, dataset / name);
            EXPECT_TRUE(lfs::io::is_image_file(dataset / name));
            EXPECT_EQ(lfs::io::count_image_files(dataset, false), 1);
            auto [pixels, w, h, c] = lfs::core::load_image_u16(dataset / name);
            std::unique_ptr<uint16_t, decltype(&lfs::core::free_image)> owner(pixels, lfs::core::free_image);
            ASSERT_NE(pixels, nullptr);
            EXPECT_EQ(w, 32);
            EXPECT_EQ(h, 24);
            EXPECT_EQ(c, 3);
            auto [raw, rw, rh, rc] = lfs::core::load_image_float(dataset / name);
            std::unique_ptr<float, decltype(&lfs::core::free_image)> raw_owner(raw, lfs::core::free_image);
            ASSERT_NE(raw, nullptr);
            EXPECT_EQ(rw, w);
            EXPECT_EQ(rh, h);
            EXPECT_EQ(rc, c);
            lfs::io::ColmapLoader loader;
            const auto loaded = loader.load(dataset, {.validate_only = true});
            ASSERT_TRUE(loaded.has_value()) << loaded.error().format();
            write_text_file(dataset / "frame.jpg", "unreferenced legacy image");
            EXPECT_EQ(lfs::io::count_image_files(dataset, false), 1);
            lfs::core::param::DatasetConfig config;
            config.data_path = dataset;
            config.images = ".";
            ASSERT_TRUE(lfs::io::project::first_dataset_image(config).has_value());
            EXPECT_EQ(lfs::io::project::first_dataset_image(config)->filename(), "frame.jpg");
        }
    }

    TEST_F(ImageFormatDiscoveryTest, HeifImageLoadersUseDisplayColorAndPreserveAlpha) {
        namespace codec = lfs::core::image_codecs;
        const auto root = fs::path(LFS_HEIF_FIXTURES);
        codec::Image expected;
        std::string error;
        ASSERT_TRUE(codec::decode_heif(root / "p3.avif", expected, error, true)) << error;
        auto [rgb, w, h, c] = lfs::core::load_image(root / "p3.avif");
        std::unique_ptr<uint8_t, decltype(&lfs::core::free_image)> rgb_owner(rgb, lfs::core::free_image);
        ASSERT_NE(rgb, nullptr);
        ASSERT_EQ(c, 3);
        ASSERT_EQ(w, 32);
        ASSERT_EQ(h, 24);
        EXPECT_TRUE(std::equal(expected.data.begin(), expected.data.end(), rgb));
        ASSERT_TRUE(codec::decode(root / "rgba8.avif", expected, error)) << error;
        auto [rgba, aw, ah, ac] = lfs::core::load_image_with_alpha(root / "rgba8.avif");
        std::unique_ptr<uint8_t, decltype(&lfs::core::free_image)> rgba_owner(rgba, lfs::core::free_image);
        ASSERT_NE(rgba, nullptr);
        ASSERT_EQ(ac, 4);
        ASSERT_EQ(aw, 32);
        ASSERT_EQ(ah, 24);
        for (size_t i = 3; i < expected.data.size(); i += 4)
            EXPECT_EQ(rgba[i], expected.data[i]);
        auto [small, sw, sh, sc] = lfs::core::load_image_with_alpha(root / "rgba8.avif", 2);
        std::unique_ptr<uint8_t, decltype(&lfs::core::free_image)> small_owner(small, lfs::core::free_image);
        ASSERT_NE(small, nullptr);
        EXPECT_EQ(sw, 16);
        EXPECT_EQ(sh, 12);
        EXPECT_EQ(sc, 4);
        EXPECT_THROW(lfs::core::load_image(root / "pq.avif"), std::runtime_error);
        auto [raw, rw, rh, rc] = lfs::core::load_image_float(root / "pq.avif");
        std::unique_ptr<float, decltype(&lfs::core::free_image)> raw_owner(raw, lfs::core::free_image);
        ASSERT_NE(raw, nullptr);
        EXPECT_EQ(rw, 32);
        EXPECT_EQ(rh, 24);
        EXPECT_EQ(rc, 3);
    }

    TEST_F(ImageFormatDiscoveryTest, NativeHeicDiscoveryPreservesLegacyCompanions) {
#if defined(_WIN32) || defined(__APPLE__)
        EXPECT_TRUE(lfs::io::is_image_file("frame.HEIC"));
        EXPECT_FALSE(lfs::io::is_legacy_image_file("frame.HEIC"));
        const auto path = temp_dir_ / "frame.HEIC";
        fs::copy_file(fs::path(LFS_HEIF_FIXTURES) / "system-p3icc.heic", path);
        EXPECT_EQ(lfs::io::count_image_files(temp_dir_, false), 1);
        write_text_file(temp_dir_ / "frame.jpg", "legacy companion");
        EXPECT_EQ(lfs::io::count_image_files(temp_dir_, false), 1);
        lfs::core::param::DatasetConfig config;
        config.data_path = temp_dir_;
        config.images = ".";
        ASSERT_TRUE(lfs::io::project::first_dataset_image(config).has_value());
        EXPECT_EQ(lfs::io::project::first_dataset_image(config)->lexically_normal(), temp_dir_ / "frame.jpg");
        std::string error;
        lfs::core::image_codecs::Probe info;
        if (!lfs::core::image_codecs::probe(path, info, error)) {
#ifdef _WIN32
            ASSERT_TRUE(error.find("provider open failed") != std::string::npos || error.find("provider factory failed") != std::string::npos) << error;
            GTEST_SKIP() << "Microsoft HEIF/HEVC extension unavailable: " << error;
#else
            FAIL() << "Apple system provider failed its qualified RGB8 input: " << error;
#endif
        }
        EXPECT_TRUE(lfs::core::probe_image_header_bounded(path, error)) << error;
        auto [pixels, w, h, c] = lfs::core::load_image(path);
        std::unique_ptr<uint8_t, decltype(&lfs::core::free_image)> owner(pixels, lfs::core::free_image);
        ASSERT_NE(pixels, nullptr);
        EXPECT_GT(w, 0);
        EXPECT_GT(h, 0);
        EXPECT_EQ(c, 3);
#else
        EXPECT_FALSE(lfs::io::is_image_file("frame.HEIC"));
#endif
    }
    TEST_F(ImageFormatDiscoveryTest, HeifBoundedHeaderAndMemoryAreReadable) {
        const auto path = temp_dir_ / "frame.avif";
        fs::copy_file(fs::path(LFS_HEIF_FIXTURES) / "p3.avif", path);
        std::string error;
        EXPECT_TRUE(lfs::core::probe_image_header_bounded(path, error)) << error;
        std::ifstream input(path, std::ios::binary);
        std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)), {});
        auto [memory, w, h, c] = lfs::core::load_image_from_memory(bytes.data(), bytes.size(), 1, 0);
        std::unique_ptr<uint8_t, decltype(&lfs::core::free_image)> owner(memory, lfs::core::free_image);
        auto [disk, dw, dh, dc] = lfs::core::load_image(path);
        std::unique_ptr<uint8_t, decltype(&lfs::core::free_image)> disk_owner(disk, lfs::core::free_image);
        ASSERT_NE(memory, nullptr);
        ASSERT_EQ(w, dw);
        ASSERT_EQ(h, dh);
        ASSERT_EQ(c, dc);
        EXPECT_EQ(std::memcmp(memory, disk, size_t(w) * h * c), 0);
    }
    TEST_F(ImageFormatDiscoveryTest, HeifBlenderDatasetCreatesCamerasWithoutUnreadableWarning) {
        if (!lfs::core::gpu_backend_available(lfs::core::default_gpu_backend()))
            GTEST_SKIP() << "Blender camera construction requires a GPU backend; bounded-header and memory decode are tested independently";
        fs::copy_file(fs::path(LFS_HEIF_FIXTURES) / "p3.avif", temp_dir_ / "frame.avif");
        write_text_file(temp_dir_ / "transforms.json", R"({"camera_angle_x":0.8,"camera_angle_y":0.6,"frames":[{"file_path":"frame.avif","transform_matrix":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}]})");
        lfs::io::BlenderLoader loader;
        const auto loaded = loader.load(temp_dir_);
        ASSERT_TRUE(loaded.has_value()) << loaded.error().format();
        for (const auto& warning : loaded->warnings)
            EXPECT_EQ(warning.find("unreadable"), std::string::npos) << warning;
    }
} // namespace
