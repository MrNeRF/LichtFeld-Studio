/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "core/cuda/undistort/undistort.hpp"
#include "core/path_utils.hpp"
#include "io/nvcodec_image_loader.hpp"
#include "io/pipelined_image_loader.hpp"

#include <OpenImageIO/imageio.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
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
        std::unique_ptr<OIIO::ImageInput> input(OIIO::ImageInput::open(path_string(source_path)));
        if (!input) {
            throw std::runtime_error("Failed to open bicycle source: " + path_string(source_path));
        }

        const OIIO::ImageSpec source_spec = input->spec();
        std::vector<uint8_t> rgb(static_cast<size_t>(source_spec.width) * source_spec.height * 3);
        if (!input->read_image(0, 0, 0, 3, OIIO::TypeDesc::UINT8, rgb.data())) {
            throw std::runtime_error("Failed to read bicycle source: " + input->geterror());
        }
        input->close();

        std::vector<uint16_t> depth(static_cast<size_t>(source_spec.width) * source_spec.height);
        for (int y = 0; y < source_spec.height; ++y) {
            for (int x = 0; x < source_spec.width; ++x) {
                const size_t rgb_idx = (static_cast<size_t>(y) * source_spec.width + x) * 3;
                const uint32_t luma =
                    static_cast<uint32_t>(rgb[rgb_idx + 0]) * 77u +
                    static_cast<uint32_t>(rgb[rgb_idx + 1]) * 150u +
                    static_cast<uint32_t>(rgb[rgb_idx + 2]) * 29u;
                const uint16_t base = static_cast<uint16_t>((luma >> 8u) * 257u);
                depth[static_cast<size_t>(y) * source_spec.width + x] =
                    static_cast<uint16_t>(base ^ variant ^
                                          static_cast<uint16_t>((x * 17 + y * 31) & 0xffu));
            }
        }
        depth.front() = 0u;
        depth.back() = 65535u;

        const fs::path output_path =
            fs::temp_directory_path() / filename;
        std::unique_ptr<OIIO::ImageOutput> output(OIIO::ImageOutput::create(path_string(output_path)));
        if (!output) {
            throw std::runtime_error("Failed to create depth PNG output: " + OIIO::geterror());
        }

        OIIO::ImageSpec spec(source_spec.width, source_spec.height, 1, OIIO::TypeDesc::UINT16);
        spec.attribute("png:compressionLevel", 1);
        if (!output->open(path_string(output_path), spec)) {
            throw std::runtime_error("Failed to open depth PNG output: " + output->geterror());
        }
        if (!output->write_image(OIIO::TypeDesc::UINT16, depth.data())) {
            throw std::runtime_error("Failed to write depth PNG output: " + output->geterror());
        }
        output->close();
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
        if (ready->depth_ready_event) {
            const cudaError_t wait_status = cudaEventSynchronize(ready->depth_ready_event);
            if (wait_status != cudaSuccess) {
                throw std::runtime_error(std::string("depth event wait failed: ") +
                                         cudaGetErrorString(wait_status));
            }
            cudaEventDestroy(ready->depth_ready_event);
            ready->depth_ready_event = nullptr;
        }
        return std::move(*ready);
    }

} // namespace

TEST(PipelinedImageLoaderSidecarJpeg2k, DepthFirstTouchTranscodesThenHotDecodes) {
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    ASSERT_GT(device_count, 0);
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

TEST(PipelinedImageLoaderSidecarJpeg2k, SmallRamBudgetSpillsAndCleansOnExit) {
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    ASSERT_GT(device_count, 0);
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

TEST(PipelinedImageLoaderSidecarJpeg2k, ParameterChangesCanonicalizeFreshPerRun) {
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
