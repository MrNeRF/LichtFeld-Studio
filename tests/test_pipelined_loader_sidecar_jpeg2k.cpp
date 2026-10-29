/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "core/image_io.hpp"
#include "core/path_utils.hpp"
#include "cuda_backend_test.hpp"
#include "io/nvcodec_image_loader.hpp"
#include "io/pipelined_image_loader.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

    namespace fs = std::filesystem;

    std::string path_string(const fs::path& path) {
        return path.string();
    }

    fs::path bicycle_image_path() {
        return fs::path(PROJECT_ROOT_PATH) / "data/bicycle/images_4/_DSC8679.JPG";
    }

    struct TempFileGuard {
        fs::path path;
        ~TempFileGuard() {
            std::error_code ec;
            fs::remove(path, ec);
        }
    };

    fs::path write_depth_png_from_bicycle_content(const std::string& filename, const uint16_t variant) {
        const fs::path source_path = bicycle_image_path();
        auto [source_data, source_width, source_height, source_channels] = lfs::core::load_image(source_path);
        if (!source_data || source_channels != 3)
            throw std::runtime_error("Failed to read bicycle source: " + path_string(source_path));
        std::vector<uint8_t> rgb(source_data, source_data + static_cast<size_t>(source_width) * source_height * 3);
        lfs::core::free_image(source_data);

        std::vector<uint16_t> depth(static_cast<size_t>(source_width) * source_height);
        for (int y = 0; y < source_height; ++y) {
            for (int x = 0; x < source_width; ++x) {
                const size_t rgb_idx = (static_cast<size_t>(y) * source_width + x) * 3;
                const uint32_t luma =
                    static_cast<uint32_t>(rgb[rgb_idx + 0]) * 77u +
                    static_cast<uint32_t>(rgb[rgb_idx + 1]) * 150u +
                    static_cast<uint32_t>(rgb[rgb_idx + 2]) * 29u;
                const uint16_t base = static_cast<uint16_t>((luma >> 8u) * 257u);
                depth[static_cast<size_t>(y) * source_width + x] =
                    static_cast<uint16_t>(base ^ variant ^
                                          static_cast<uint16_t>((x * 17 + y * 31) & 0xffu));
            }
        }
        depth.front() = 0u;
        depth.back() = 65535u;

        const fs::path output_path =
            fs::temp_directory_path() / filename;
        if (!lfs::core::save_png(output_path, depth.data(), source_width, source_height, 1, 16, 1))
            throw std::runtime_error("Failed to write depth PNG output");
        return output_path;
    }

    fs::path write_depth_png_from_bicycle_content() {
        return write_depth_png_from_bicycle_content("lfs_pipelined_sidecar_jpeg2k_depth16.png", 0u);
    }

    lfs::io::ReadyImage wait_for_ready(lfs::io::PipelinedImageLoader& loader) {
        auto ready = loader.try_get_for(std::chrono::seconds(20));
        if (!ready) {
            throw std::runtime_error("Timed out waiting for pipelined loader output");
        }
        if (ready->depth_ready)
            ready->depth_ready->wait();
        return std::move(*ready);
    }

} // namespace

class PipelinedImageLoaderSidecarJpeg2k : public lfs::test::CudaBackendTest {};

TEST_F(PipelinedImageLoaderSidecarJpeg2k, DepthFirstTouchTranscodesThenHotDecodes) {
    ASSERT_TRUE(lfs::io::NvCodecImageLoader::is_available());
    const TempFileGuard depth_file{write_depth_png_from_bicycle_content()};
    const fs::path& depth_path = depth_file.path;

    lfs::io::PipelinedLoaderConfig config;
    config.io_threads = 1;
    config.cold_process_threads = 1;
    config.jpeg_batch_size = 2;
    config.decoder_pool_size = 2;
    config.output_queue_size = 2;
    config.prefetch_count = 2;
    config.max_cache_bytes = 512ULL * 1024ULL * 1024ULL;

    lfs::io::PipelinedImageLoader loader(config);

    lfs::io::ImageRequest first;
    first.sequence_id = 1;
    first.path = bicycle_image_path();
    first.depth_path = depth_path;
    first.params.output_uint8 = false;

    loader.prefetch(std::vector<lfs::io::ImageRequest>{first});
    auto first_ready = wait_for_ready(loader);
    ASSERT_TRUE(first_ready.depth.has_value());
    const auto stats_after_first = loader.get_stats();
    EXPECT_GE(stats_after_first.jpeg_cache_entries, size_t{2});
    EXPECT_GT(stats_after_first.jpeg_cache_bytes, size_t{0});
    std::error_code remove_ec;
    fs::remove(depth_path, remove_ec);
    ASSERT_FALSE(fs::exists(depth_path));

    lfs::io::ImageRequest second = first;
    second.sequence_id = 2;
    loader.prefetch(std::vector<lfs::io::ImageRequest>{second});
    auto second_ready = wait_for_ready(loader);
    ASSERT_TRUE(second_ready.depth.has_value());

    const auto stats_after_second = loader.get_stats();
    EXPECT_GT(stats_after_second.hot_path_hits, stats_after_first.hot_path_hits);

    const std::vector<float> first_values = first_ready.depth->cpu().to_vector();
    const std::vector<float> second_values = second_ready.depth->cpu().to_vector();
    ASSERT_EQ(second_values.size(), first_values.size());

    constexpr float kTolerance = 1.0f / 65535.0f;
    for (size_t i = 0; i < first_values.size(); ++i) {
        ASSERT_LE(std::abs(second_values[i] - first_values[i]), kTolerance)
            << "depth mismatch at element " << i;
    }
}

TEST_F(PipelinedImageLoaderSidecarJpeg2k, SmallRamBudgetSpillsAndCleansOnExit) {
    ASSERT_TRUE(lfs::io::NvCodecImageLoader::is_available());
    const TempFileGuard depth_file{
        write_depth_png_from_bicycle_content("lfs_pipelined_sidecar_jpeg2k_spill_depth16.png", 0u)};

    lfs::io::PipelinedLoaderConfig config;
    config.io_threads = 1;
    config.cold_process_threads = 1;
    config.jpeg_batch_size = 2;
    config.decoder_pool_size = 2;
    config.output_queue_size = 2;
    config.prefetch_count = 2;
    config.max_cache_bytes = 1;
    config.min_spill_free_bytes = 0; // spill regardless of free space in the temp dir

    fs::path spill_dir;
    std::vector<float> first_values;
    {
        lfs::io::PipelinedImageLoader loader(config);
        spill_dir = loader.run_spill_directory();
        ASSERT_FALSE(spill_dir.empty());

        lfs::io::ImageRequest first;
        first.sequence_id = 1;
        first.path = bicycle_image_path();
        first.depth_path = depth_file.path;
        first.params.output_uint8 = false;

        loader.prefetch(std::vector<lfs::io::ImageRequest>{first});
        auto first_ready = wait_for_ready(loader);
        ASSERT_TRUE(first_ready.depth.has_value());
        first_values = first_ready.depth->cpu().to_vector();
        const auto stats = loader.get_stats();
        EXPECT_GT(stats.spill_cache_entries, size_t{0});
        EXPECT_GT(stats.spill_cache_bytes, size_t{0});
        EXPECT_TRUE(fs::is_directory(spill_dir));

        std::error_code remove_ec;
        fs::remove(depth_file.path, remove_ec);
        ASSERT_FALSE(fs::exists(depth_file.path));

        first.sequence_id = 2;
        loader.prefetch(std::vector<lfs::io::ImageRequest>{first});
        auto streamed = wait_for_ready(loader);
        ASSERT_TRUE(streamed.depth.has_value());
        const auto streamed_values = streamed.depth->cpu().to_vector();
        ASSERT_EQ(streamed_values.size(), first_values.size());
        constexpr float kTolerance = 1.0f / 65535.0f;
        for (size_t i = 0; i < first_values.size(); ++i) {
            ASSERT_LE(std::abs(streamed_values[i] - first_values[i]), kTolerance)
                << "spilled depth mismatch at element " << i;
        }
    }
    EXPECT_FALSE(fs::exists(spill_dir));
}

TEST_F(PipelinedImageLoaderSidecarJpeg2k, ParameterChangesCanonicalizeFreshPerRun) {
    const auto path = bicycle_image_path();
    if (!fs::is_regular_file(path))
        GTEST_SKIP() << "bicycle dataset is absent: " << path;

    lfs::core::UndistortParams undistort{};
    undistort.src_fx = 1800.0f;
    undistort.src_fy = 1800.0f;
    undistort.src_cx = 1920.0f;
    undistort.src_cy = 1080.0f;
    undistort.dst_fx = 1750.0f;
    undistort.dst_fy = 1750.0f;
    undistort.dst_cx = 960.0f;
    undistort.dst_cy = 540.0f;
    undistort.src_width = 3840;
    undistort.src_height = 2160;
    undistort.dst_width = 1920;
    undistort.dst_height = 1080;
    undistort.model_type = lfs::core::CameraModelType::PINHOLE;
    undistort.distortion[0] = -0.08f;
    undistort.num_distortion = 1;

    const auto run = [&](const int resize_factor, const int max_width,
                         const lfs::core::UndistortParams* params) {
        lfs::io::PipelinedLoaderConfig config;
        config.jpeg_batch_size = 1;
        config.prefetch_count = 1;
        config.output_queue_size = 1;
        lfs::io::PipelinedImageLoader loader(config);
        lfs::io::ImageRequest request;
        request.sequence_id = 1;
        request.path = path;
        request.params.resize_factor = resize_factor;
        request.params.max_width = max_width;
        request.undistort = params;
        request.params.undistort = params;
        loader.prefetch({request});
        const auto ready = loader.get();
        EXPECT_TRUE(ready.error.empty()) << ready.error;
        EXPECT_TRUE(ready.tensor.is_valid());
        return ready.tensor.shape();
    };

    const auto run_a = run(2, 900, &undistort);
    const auto run_b = run(1, 0, nullptr);
    const auto run_c = run(4, 0, nullptr);

    EXPECT_NE(run_a, run_b) << "undistort/resize-off run reused run A output";
    EXPECT_NE(run_b, run_c) << "resize change reused run B output";
    EXPECT_NE(run_a, run_c) << "all parameter variants must remain distinct";
}

namespace {
    enum class DepthEncoding { Png8,
                               Png16,
                               Tiff8,
                               Tiff16,
                               TiffFloat };

    class PipelinedDepthDecode : public ::testing::TestWithParam<std::tuple<DepthEncoding, bool, bool>> {
    protected:
        void SetUp() override {
            directory_ = fs::temp_directory_path() /
                         ("lfs_depth_decode_" + std::to_string(
                                                    std::chrono::steady_clock::now().time_since_epoch().count()));
            fs::create_directories(directory_);
        }
        void TearDown() override {
            std::error_code ec;
            fs::remove_all(directory_, ec);
        }

        // Independent little-endian, single-strip TIFF fixture; includes values
        // that would be lost by an accidental conversion through eight bits.
        void write_tiff(const fs::path& path, const int bits, const std::vector<uint32_t>& samples, int width = 8, int height = 4) {
            std::ofstream file(path, std::ios::binary);
            const auto put = [&](uint32_t value, const int bytes) {
                for (int i = 0; i < bytes; ++i) {
                    file.put(static_cast<char>(value & 255));
                    value >>= 8;
                }
            };
            file.write("II", 2);
            put(42, 2);
            put(8, 4);
            put(10, 2);
            const auto tag = [&](const uint16_t id, const uint16_t type, const uint32_t value) {
                put(id, 2);
                put(type, 2);
                put(1, 4);
                put(value, 4);
            };
            tag(256, 4, width);  // width
            tag(257, 4, height); // height
            tag(258, 3, bits);
            tag(259, 3, 1);      // uncompressed
            tag(262, 3, 1);      // grayscale
            tag(273, 4, 134);    // strip offset
            tag(277, 3, 1);      // channels
            tag(278, 4, height); // rows per strip
            tag(279, 4, samples.size() * (bits / 8));
            tag(339, 3, bits == 32 ? 3 : 1); // IEEE float or unsigned integer
            put(0, 4);
            for (const auto value : samples)
                put(value, bits / 8);
            ASSERT_TRUE(file.good());
        }

        fs::path directory_;
    };
} // namespace

TEST_P(PipelinedDepthDecode, PreservesSamplesOnColdAndCachedReads) {
    const auto [encoding, resize, spill] = GetParam();
    const bool png = encoding == DepthEncoding::Png8 || encoding == DepthEncoding::Png16;
    const int bits = encoding == DepthEncoding::TiffFloat ? 32 : encoding == DepthEncoding::Tiff16 || encoding == DepthEncoding::Png16 ? 16
                                                                                                                                       : 8;
    const int width = bits == 32 ? 1024 : 8;
    const int height = bits == 32 ? 768 : 4;
    const auto rgb_path = directory_ / "image.png";
    const auto depth_path = directory_ / (png ? "depth.PNG" : bits == 16 ? "depth.tif"
                                                                         : "depth.TIFF");
    const std::vector<uint8_t> rgb(width * height * 3, 128);
    ASSERT_TRUE(lfs::core::save_png(rgb_path, rgb.data(), width, height, 3, 8, 1));
    const std::array<uint16_t, 8> pattern{0, 1, 257, 12345, 32768, 48001, 65534, 65535};
    const std::array<float, 8> float_pattern{0.0f, 0.000001f, 0.1234567f, 0.5f, 1.0001f, 2.0f, 3.75f, 1000.25f};
    std::vector<uint8_t> u8;
    std::vector<uint16_t> u16;
    std::vector<uint32_t> samples;
    std::vector<float> expected;
    for (size_t i = 0; i < static_cast<size_t>(width * height); ++i) {
        const uint16_t value = pattern[i % pattern.size()];
        u8.push_back(static_cast<uint8_t>(value >> 8));
        u16.push_back(value);
        const float decoded = bits == 32 ? float_pattern[i % float_pattern.size()] : bits == 16 ? static_cast<float>(value) / 65535.0f
                                                                                                : static_cast<float>(u8.back()) / 255.0f;
        expected.push_back(decoded);
        samples.push_back(bits == 32 ? std::bit_cast<uint32_t>(decoded) : bits == 16 ? value
                                                                                     : u8.back());
    }
    if (png) {
        ASSERT_TRUE(lfs::core::save_png(depth_path, bits == 16 ? static_cast<void*>(u16.data()) : static_cast<void*>(u8.data()),
                                        width, height, 1, bits, 1));
    } else {
        write_tiff(depth_path, bits, samples, width, height);
    }

    lfs::io::PipelinedLoaderConfig config;
    config.io_threads = 1;
    config.cold_process_threads = 1;
    config.decoder_pool_size = 1;
    config.output_queue_size = 2;
    config.prefetch_count = 2;
    if (spill) {
        config.max_cache_bytes = 1;
        config.min_spill_free_bytes = 0; // spill regardless of free space in the temp dir
    }
    lfs::io::PipelinedImageLoader loader(config);
    lfs::io::ImageRequest request;
    request.path = rgb_path;
    request.depth_path = depth_path;
    request.sequence_id = 1;
    request.aux_target_width = resize ? width / 2 : width;
    request.aux_target_height = resize ? height / 2 : height;
    const auto reference = lfs::core::resize_depth_prior(
                               lfs::core::Tensor::from_vector(expected, {size_t(height), size_t(width)}, lfs::core::Device::CUDA),
                               request.aux_target_height, request.aux_target_width)
                               .cpu()
                               .to_vector();
    for (int pass = 0; pass < 2; ++pass) {
        SCOPED_TRACE(pass);
        const auto start = std::chrono::steady_clock::now();
        loader.prefetch({request});
        const auto ready = wait_for_ready(loader);
        ASSERT_TRUE(ready.depth.has_value());
        const auto values = ready.depth->cpu().to_vector();
        RecordProperty(pass == 0 ? "cold_ms" : "cached_ms",
                       std::to_string(std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - start)
                                          .count()));
        const auto cache = loader.get_stats();
        RecordProperty("total_cache_bytes", std::to_string(cache.jpeg_cache_bytes + cache.spill_cache_bytes));
        RecordProperty("depth_bytes", std::to_string(values.size() * sizeof(float)));
        ASSERT_EQ(values.size(), reference.size());
        for (size_t i = 0; i < values.size(); ++i) {
            if (bits == 32)
                ASSERT_EQ(values[i], reference[i]) << "sample " << i;
            else
                ASSERT_NEAR(values[i], reference[i], pass == 0 ? 1e-6f : 1.0f / 65535.0f) << "sample " << i;
        }
        if (pass == 0) {
            ASSERT_TRUE(fs::remove(depth_path));
            ASSERT_FALSE(fs::exists(depth_path));
        }
        ++request.sequence_id;
    }
    if (bits == 32) {
        const auto normal_path = directory_ / "normal.png";
        ASSERT_TRUE(lfs::core::save_png(normal_path, rgb.data(), width, height, 3, 8, 1));
        request.normal_path = normal_path;
        for (int pass = 0; pass < 2; ++pass) {
            loader.prefetch({request});
            auto ready = wait_for_ready(loader);
            ASSERT_TRUE(ready.depth.has_value());
            ASSERT_TRUE(ready.normal.has_value());
            EXPECT_EQ(ready.depth->cpu().to_vector(), reference);
            if (ready.normal_ready)
                ready.normal_ready->wait();
            if (pass == 0)
                ASSERT_TRUE(fs::remove(normal_path));
            ++request.sequence_id;
        }
    }
    const auto stats = loader.get_stats();
    if (spill)
        EXPECT_GT(stats.spill_cache_entries, 0u);
}

INSTANTIATE_TEST_SUITE_P(Formats, PipelinedDepthDecode,
                         ::testing::Combine(::testing::Values(DepthEncoding::Png8, DepthEncoding::Png16,
                                                              DepthEncoding::Tiff8, DepthEncoding::Tiff16,
                                                              DepthEncoding::TiffFloat),
                                            ::testing::Bool(), ::testing::Bool()));
