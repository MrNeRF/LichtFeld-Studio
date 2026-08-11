/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/video/video_encoder.hpp"
#include "io/video_frame_extractor.hpp"

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

    using lfs::io::ExtractionMode;
    using lfs::io::VideoFrameExtractor;

    constexpr int kWidth = 16;
    constexpr int kHeight = 16;
    constexpr int kChannels = 3;
    constexpr int kFixtureFrameCount = 50;
    constexpr double kFixtureEndTime = 0.5;

    struct TempDir {
        explicit TempDir(const std::string_view label) {
            const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
            path = std::filesystem::temp_directory_path() /
                   ("lfs_video_extract_" + std::string(label) + "_" + std::to_string(now));
            std::filesystem::create_directories(path);
        }

        ~TempDir() {
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
        }

        std::filesystem::path path;
    };

    struct CudaFloatBuffer {
        explicit CudaFloatBuffer(const std::size_t count) {
            status = cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(float));
        }

        ~CudaFloatBuffer() {
            if (ptr)
                cudaFree(ptr);
        }

        float* ptr = nullptr;
        cudaError_t status = cudaSuccess;
    };

    bool cudaAvailable() {
        int device_count = 0;
        return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
    }

    bool writeEncodedVideo(const std::filesystem::path& video_path,
                           const int frame_count,
                           const int framerate,
                           std::string& error) {
        lfs::io::video::VideoExportOptions options;
        options.preset = lfs::io::video::VideoPreset::CUSTOM;
        options.width = kWidth;
        options.height = kHeight;
        options.framerate = framerate;
        options.crf = 23;

        lfs::io::video::VideoEncoder encoder;
        if (const auto opened = encoder.open(video_path, options); !opened) {
            error = opened.error();
            return false;
        }

        std::vector<float> frame(static_cast<std::size_t>(kWidth) * kHeight * kChannels);
        CudaFloatBuffer device_frame(frame.size());
        if (device_frame.status != cudaSuccess) {
            error = cudaGetErrorString(device_frame.status);
            return false;
        }

        for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
            const float red = static_cast<float>(frame_index + 1) /
                              static_cast<float>(frame_count);
            for (int y = 0; y < kHeight; ++y) {
                for (int x = 0; x < kWidth; ++x) {
                    const std::size_t offset =
                        (static_cast<std::size_t>(y) * kWidth + x) * kChannels;
                    frame[offset + 0] = red;
                    frame[offset + 1] = static_cast<float>(x) / static_cast<float>(kWidth - 1);
                    frame[offset + 2] = static_cast<float>(y) / static_cast<float>(kHeight - 1);
                }
            }

            const cudaError_t copy_status = cudaMemcpy(
                device_frame.ptr, frame.data(), frame.size() * sizeof(float), cudaMemcpyHostToDevice);
            if (copy_status != cudaSuccess) {
                error = cudaGetErrorString(copy_status);
                return false;
            }

            if (const auto written = encoder.writeFrameGpu(device_frame.ptr, kWidth, kHeight); !written) {
                error = written.error();
                return false;
            }
        }

        if (const auto closed = encoder.close(); !closed) {
            error = closed.error();
            return false;
        }
        return true;
    }

    VideoFrameExtractor::Params extractionParams(
        const std::filesystem::path& video_path,
        const std::filesystem::path& output_dir) {
        VideoFrameExtractor::Params params;
        params.video_path = video_path;
        params.output_dir = output_dir;
        params.mode = ExtractionMode::INTERVAL;
        params.frame_interval = 1;
        params.format = lfs::io::ImageFormat::PNG;
        params.generate_metadata = true;
        params.end_time = kFixtureEndTime;
        return params;
    }

    std::size_t countPngFiles(const std::filesystem::path& output_dir) {
        std::size_t count = 0;
        for (const auto& entry : std::filesystem::directory_iterator{output_dir}) {
            if (entry.is_regular_file() && entry.path().extension() == ".png")
                ++count;
        }
        return count;
    }

    nlohmann::json readMetadata(const std::filesystem::path& output_dir) {
        std::ifstream file(output_dir / "extraction_metadata.json");
        return nlohmann::json::parse(file);
    }

} // namespace

TEST(VideoFrameExtractorOutputNaming, IntervalUsesSourceFrameNumbers) {
    if (!cudaAvailable())
        GTEST_SKIP() << "CUDA device required for VideoEncoder-based fixture";

    TempDir temp("interval");
    const std::filesystem::path video_path = temp.path / "source.mp4";
    const std::filesystem::path output_dir = temp.path / "frames";
    std::filesystem::create_directories(output_dir);

    std::string error;
    ASSERT_TRUE(writeEncodedVideo(video_path, kFixtureFrameCount, 10, error)) << error;

    auto params = extractionParams(video_path, output_dir);
    params.frame_interval = 2;

    VideoFrameExtractor extractor;
    ASSERT_TRUE(extractor.extract(params, error)) << error;
    EXPECT_TRUE(std::filesystem::exists(output_dir / "frame_1.png"));
    EXPECT_TRUE(std::filesystem::exists(output_dir / "frame_3.png"));
    EXPECT_TRUE(std::filesystem::exists(output_dir / "frame_5.png"));
    EXPECT_FALSE(std::filesystem::exists(output_dir / "frame_2.png"));
    EXPECT_EQ(3u, countPngFiles(output_dir));
}

TEST(VideoFrameExtractorOutputNaming, TrimmedRangeKeepsOriginalSourceFrameNumbers) {
    if (!cudaAvailable())
        GTEST_SKIP() << "CUDA device required for VideoEncoder-based fixture";

    TempDir temp("trim");
    const std::filesystem::path video_path = temp.path / "source.mp4";
    const std::filesystem::path output_dir = temp.path / "frames";
    std::filesystem::create_directories(output_dir);

    std::string error;
    ASSERT_TRUE(writeEncodedVideo(video_path, kFixtureFrameCount, 10, error)) << error;

    auto params = extractionParams(video_path, output_dir);
    params.start_time = 2.3;
    params.end_time = 2.41;

    VideoFrameExtractor extractor;
    ASSERT_TRUE(extractor.extract(params, error)) << error;
    EXPECT_TRUE(std::filesystem::exists(output_dir / "frame_24.png"));
    EXPECT_TRUE(std::filesystem::exists(output_dir / "frame_25.png"));
    EXPECT_FALSE(std::filesystem::exists(output_dir / "frame_4.png"));
    EXPECT_EQ(2u, countPngFiles(output_dir));
}

TEST(VideoFrameExtractorOutputNaming, RepeatedSourceFramesAreWrittenOnce) {
    if (!cudaAvailable())
        GTEST_SKIP() << "CUDA device required for VideoEncoder-based fixture";

    TempDir temp("duplicates");
    const std::filesystem::path video_path = temp.path / "source.mp4";
    const std::filesystem::path output_dir = temp.path / "frames";
    std::filesystem::create_directories(output_dir);

    std::string error;
    ASSERT_TRUE(writeEncodedVideo(video_path, kFixtureFrameCount, 10, error)) << error;

    auto params = extractionParams(video_path, output_dir);
    params.mode = ExtractionMode::FPS;
    params.fps = 30.0;
    params.start_time = 0.0;
    params.end_time = 0.5;

    VideoFrameExtractor extractor;
    ASSERT_TRUE(extractor.extract(params, error)) << error;
    EXPECT_TRUE(std::filesystem::exists(output_dir / "frame_1.png"));
    EXPECT_TRUE(std::filesystem::exists(output_dir / "frame_2.png"));
    EXPECT_TRUE(std::filesystem::exists(output_dir / "frame_3.png"));
    EXPECT_TRUE(std::filesystem::exists(output_dir / "frame_4.png"));
    EXPECT_TRUE(std::filesystem::exists(output_dir / "frame_5.png"));
    EXPECT_EQ(5u, countPngFiles(output_dir));

    const nlohmann::json metadata = readMetadata(output_dir);
    ASSERT_TRUE(metadata.contains("frames"));
    EXPECT_EQ(5u, metadata["frames"].size());
    EXPECT_EQ(5, metadata["performance"]["written_frames"].get<int>());
}
