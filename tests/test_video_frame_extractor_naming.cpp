/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "cuda_backend_test.hpp"

#include "core/tensor_backend.hpp"
#include "io/video/video_encoder.hpp"
#include "io/video_frame_extractor.hpp"

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <stb_image.h>

extern "C" {
#include <libavformat/avformat.h>
}

#include <algorithm>
#include <array>
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

    constexpr int kWidth = 64;
    constexpr int kHeight = 64;
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

            const auto tensor = lfs::core::Tensor::from_blob(
                device_frame.ptr, {kHeight, kWidth, kChannels},
                lfs::core::Device::GPU, lfs::core::DataType::Float32);
            if (const auto written = encoder.writeFrame(tensor); !written) {
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

    std::array<int, 3> firstFrameCenter(const std::filesystem::path& output_dir) {
        for (const auto& entry : std::filesystem::directory_iterator{output_dir}) {
            if (entry.path().extension() != ".png")
                continue;
            int width = 0, height = 0, channels = 0;
            auto* pixels = stbi_load(entry.path().string().c_str(), &width, &height, &channels, 3);
            if (!pixels || width <= 0 || height <= 0) {
                stbi_image_free(pixels);
                return {-1, -1, -1};
            }
            const size_t center = (static_cast<size_t>(height / 2) * width + width / 2) * 3;
            const std::array<int, 3> color{pixels[center], pixels[center + 1], pixels[center + 2]};
            stbi_image_free(pixels);
            return color;
        }
        return {-1, -1, -1};
    }

} // namespace

enum class VideoInput { CpuRgba,
                        VulkanTensor };

class VideoEncoderInputTest : public ::testing::TestWithParam<VideoInput> {};

TEST_P(VideoEncoderInputTest, SolidColorEncodesAndExtracts) {
    const bool vulkan = GetParam() == VideoInput::VulkanTensor;
    if (vulkan && !lfs::core::gpu_backend_available(lfs::core::GpuBackend::Vulkan))
        GTEST_SKIP() << "Vulkan tensor backend unavailable";

    TempDir temp("solid_color");
    const auto video_path = temp.path / "solid.mp4";
    const auto output_dir = temp.path / "frames";
    std::filesystem::create_directories(output_dir);

    lfs::io::video::VideoExportOptions options;
    options.preset = lfs::io::video::VideoPreset::CUSTOM;
    options.width = kWidth;
    options.height = kHeight;
    options.framerate = 10;
    lfs::io::video::VideoEncoder encoder;
    ASSERT_TRUE(encoder.open(video_path, options));
    if (vulkan) {
        std::vector<float> rgb(static_cast<size_t>(kWidth) * kHeight * kChannels, 0.0f);
        for (size_t pixel = 0; pixel < rgb.size() / kChannels; ++pixel)
            rgb[pixel * kChannels + 1] = 1.0f;
        const lfs::core::GpuBackendScope scope(lfs::core::GpuBackend::Vulkan);
        const auto frame = lfs::core::Tensor::from_vector(
            rgb, {kHeight, kWidth, kChannels}, lfs::core::Device::GPU);
        ASSERT_EQ(lfs::core::gpu_backend_of(frame), lfs::core::GpuBackend::Vulkan);
        for (int i = 0; i < 6; ++i) {
            const auto written = encoder.writeFrame(frame);
            ASSERT_TRUE(written) << written.error();
        }
    } else {
        std::vector<uint8_t> rgba(static_cast<size_t>(kWidth) * kHeight * 4);
        for (size_t pixel = 0; pixel < rgba.size() / 4; ++pixel) {
            rgba[pixel * 4] = 255;
            rgba[pixel * 4 + 3] = 255;
        }
        for (int i = 0; i < 6; ++i) {
            const auto written = encoder.writeFrame(rgba, kWidth, kHeight);
            ASSERT_TRUE(written) << written.error();
        }
    }
    ASSERT_TRUE(encoder.close());

    AVFormatContext* container = nullptr;
    ASSERT_GE(avformat_open_input(&container, video_path.string().c_str(), nullptr, nullptr), 0);
    ASSERT_GE(avformat_find_stream_info(container, nullptr), 0);
    const int video_stream = av_find_best_stream(container, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    ASSERT_GE(video_stream, 0);
    AVPacket* packet = av_packet_alloc();
    ASSERT_NE(packet, nullptr);
    int packet_count = 0;
    int64_t final_end = 0;
    while (av_read_frame(container, packet) >= 0) {
        if (packet->stream_index == video_stream) {
            EXPECT_GT(packet->duration, 0);
            final_end = std::max(final_end, packet->pts + packet->duration);
            ++packet_count;
        }
        av_packet_unref(packet);
    }
    EXPECT_EQ(packet_count, 6);
    EXPECT_GE(container->streams[video_stream]->duration, final_end);
    av_packet_free(&packet);
    avformat_close_input(&container);

    auto params = extractionParams(video_path, output_dir);
    std::string error;
    VideoFrameExtractor extractor;
    ASSERT_TRUE(extractor.extract(params, error)) << error;
    EXPECT_EQ(countPngFiles(output_dir), 6u);
    const auto center = firstFrameCenter(output_dir);
    EXPECT_GT(center[vulkan ? 1 : 0], 200);
    EXPECT_LT(center[vulkan ? 0 : 1], 70);
    EXPECT_LT(center[2], 70);
}

INSTANTIATE_TEST_SUITE_P(Backends, VideoEncoderInputTest,
                         ::testing::Values(VideoInput::CpuRgba, VideoInput::VulkanTensor));

class VideoFrameExtractorOutputNaming : public lfs::test::CudaBackendTest {};
class VideoFrameExtractorCudaOutcome : public lfs::test::CudaBackendTest {};

TEST_F(VideoFrameExtractorOutputNaming, IntervalUsesSourceFrameNumbers) {
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
    EXPECT_EQ(extractor.lastOutcome(), lfs::io::ExtractionOutcome::Completed);
    EXPECT_TRUE(std::filesystem::exists(output_dir / "frame_1.png"));
    EXPECT_TRUE(std::filesystem::exists(output_dir / "frame_3.png"));
    EXPECT_TRUE(std::filesystem::exists(output_dir / "frame_5.png"));
    EXPECT_FALSE(std::filesystem::exists(output_dir / "frame_2.png"));
    EXPECT_EQ(3u, countPngFiles(output_dir));

    const nlohmann::json metadata = readMetadata(output_dir);
    ASSERT_TRUE(metadata.contains("processing"));
    EXPECT_EQ(metadata["processing"]["decoder"]["backend"], "nvdec")
        << "regression must exercise the hardware decode path";
}

TEST(VideoFrameExtractorOutcome, CancellationDoesNotRelabelEarlierFailure) {
    VideoFrameExtractor::Params params;
    params.video_path = "/path/that/does/not/exist.mp4";
    params.cancel_requested = [] { return true; };

    VideoFrameExtractor extractor;
    std::string error;
    EXPECT_FALSE(extractor.extract(params, error));
    EXPECT_EQ(extractor.lastOutcome(), lfs::io::ExtractionOutcome::Failed);
    EXPECT_FALSE(error.empty());
}

TEST_F(VideoFrameExtractorCudaOutcome, ReportsExplicitCancellation) {
    TempDir temp("cancelled");
    const std::filesystem::path video_path = temp.path / "source.mp4";
    const std::filesystem::path output_dir = temp.path / "frames";
    std::filesystem::create_directories(output_dir);

    std::string error;
    ASSERT_TRUE(writeEncodedVideo(video_path, kFixtureFrameCount, 10, error)) << error;

    auto params = extractionParams(video_path, output_dir);
    params.cancel_requested = [] { return true; };

    VideoFrameExtractor extractor;
    EXPECT_FALSE(extractor.extract(params, error));
    EXPECT_EQ(extractor.lastOutcome(), lfs::io::ExtractionOutcome::Cancelled);

    params.cancel_requested = [] { return false; };
    ASSERT_TRUE(extractor.extract(params, error)) << error;
    EXPECT_EQ(extractor.lastOutcome(), lfs::io::ExtractionOutcome::Completed);
    EXPECT_TRUE(error.empty());
}

TEST_F(VideoFrameExtractorOutputNaming, TrimmedRangeKeepsOriginalSourceFrameNumbers) {
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

TEST_F(VideoFrameExtractorOutputNaming, RepeatedSourceFramesAreWrittenOnce) {
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
