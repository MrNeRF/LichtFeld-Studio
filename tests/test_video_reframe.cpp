/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/video/color_convert.cuh"
#include "io/video/video_encoder.hpp"
#include "io/video/video_frame_rate.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>
#include <vector>

extern "C" {
#include <libavformat/avformat.h>
}

namespace {
    TEST(VideoReframeTest, EncodedMp4PreservesFractionalRatesFrameCountsAndLegacyDimensions) {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
            GTEST_SKIP() << "No CUDA device";
        using namespace lfs::io::video;
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        for (const int output_width : {64, 320}) {
            const int output_height = output_width * 3 / 4;
            for (const VideoFrameRate rate : {VideoFrameRate{24, 1}, {24000, 1001}, {30000, 1001}, {60000, 1001}}) {
                for (const bool legacy : {false, true}) {
                    SCOPED_TRACE(rate.numerator);
                    SCOPED_TRACE(legacy);
                    VideoExportOptions options;
                    options.width = output_width;
                    options.height = output_height;
                    options.framerate = rate.numerator;
                    options.framerate_denominator = rate.denominator;
                    options.camera_width = legacy ? 0 : 16;
                    options.camera_height = legacy ? 0 : 9;
                    options.framing = VideoFrameFit::CROP;
                    const auto size = videoCameraRenderSize(output_width, output_height, options.camera_width, options.camera_height, options.framing);
                    std::vector<float> pixels(size.width * size.height * 3, 0.5f);
                    float* input = nullptr;
                    ASSERT_EQ(cudaMalloc(&input, pixels.size() * sizeof(float)), cudaSuccess);
                    ASSERT_EQ(cudaMemcpy(input, pixels.data(), pixels.size() * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
                    for (const int frames : {1, 60}) {
                        const auto path = std::filesystem::temp_directory_path() /
                                          ("lichtfeld-rate-" + std::to_string(stamp) + ".mp4");
                        VideoEncoder encoder;
                        const auto opened = encoder.open(path, options);
                        ASSERT_TRUE(opened) << opened.error();
                        for (int i = 0; i < frames; ++i) {
                            const auto written = encoder.writeFrameGpu(input, size.width, size.height);
                            ASSERT_TRUE(written) << written.error();
                        }
                        const auto closed = encoder.close();
                        ASSERT_TRUE(closed) << closed.error();
                        AVFormatContext* format = nullptr;
                        ASSERT_GE(avformat_open_input(&format, path.string().c_str(), nullptr, nullptr), 0);
                        ASSERT_GE(avformat_find_stream_info(format, nullptr), 0);
                        ASSERT_EQ(format->nb_streams, 1);
                        const auto* stream = format->streams[0];
                        EXPECT_EQ(stream->codecpar->width, output_width);
                        EXPECT_EQ(stream->codecpar->height, output_height);
                        EXPECT_EQ(stream->nb_frames, frames);
                        EXPECT_EQ(stream->codecpar->sample_aspect_ratio.num, 1);
                        EXPECT_EQ(stream->codecpar->sample_aspect_ratio.den, 1);
                        EXPECT_EQ(static_cast<int64_t>(stream->avg_frame_rate.num) * rate.denominator,
                                  static_cast<int64_t>(stream->avg_frame_rate.den) * rate.numerator);
                        EXPECT_NEAR(stream->duration * av_q2d(stream->time_base),
                                    static_cast<double>(frames) * rate.denominator / rate.numerator,
                                    av_q2d(stream->time_base));
                        avformat_close_input(&format);
                        std::filesystem::remove(path);
                    }
                    cudaFree(input);
                }
            }
        }
    }

    // Exercise the actual CUDA kernel against analytic ramps, including pixels
    // outside FIT, anisotropic STRETCH, automatic CROP, and partial CUDA blocks.
    TEST(VideoReframeTest, PixelCentersMatchAnalyticCameraImageWithoutWritingPastOutput) {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
            GTEST_SKIP() << "No CUDA device";
        using namespace lfs::io::video;
        constexpr int sw = 53, sh = 29, width = 34, height = 46;
        std::vector<float> source(sw * sh * 3);
        for (int y = 0; y < sh; ++y)
            for (int x = 0; x < sw; ++x) {
                source[(y * sw + x) * 3] = static_cast<float>(x) / (sw - 1);
                source[(y * sw + x) * 3 + 1] = static_cast<float>(y) / (sh - 1);
                source[(y * sw + x) * 3 + 2] = 0.75f;
            }
        float *input = nullptr, *output = nullptr;
        ASSERT_EQ(cudaMalloc(&input, source.size() * sizeof(float)), cudaSuccess);
        std::vector<float> result(width * height * 3 + 32, -17.0f);
        ASSERT_EQ(cudaMalloc(&output, result.size() * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(input, source.data(), source.size() * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
        for (const auto fit : {VideoFrameFit::FIT, VideoFrameFit::CROP, VideoFrameFit::STRETCH}) {
            SCOPED_TRACE(static_cast<int>(fit));
            std::fill(result.begin(), result.end(), -17.0f);
            ASSERT_EQ(cudaMemcpy(output, result.data(), result.size() * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
            const auto mapping = videoFrameMapping(sw, sh, width, height, fit);
            reframeRgbCuda(input, output, sw, sh, width, height, mapping);
            ASSERT_EQ(cudaMemcpy(result.data(), output, result.size() * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x) {
                    const float sx = (x + 0.5f) * mapping.scale_x + mapping.offset_x;
                    const float sy = (y + 0.5f) * mapping.scale_y + mapping.offset_y;
                    const bool black = sx < 0 || sx >= sw || sy < 0 || sy >= sh;
                    const int i = (y * width + x) * 3;
                    EXPECT_NEAR(result[i], black ? 0 : std::clamp(sx - 0.5f, 0.0f, sw - 1.0f) / (sw - 1), 0.000001f);
                    EXPECT_NEAR(result[i + 1], black ? 0 : std::clamp(sy - 0.5f, 0.0f, sh - 1.0f) / (sh - 1), 0.000001f);
                    EXPECT_NEAR(result[i + 2], black ? 0 : 0.75f, 0.000001f);
                }
            for (size_t i = width * height * 3; i < result.size(); ++i)
                EXPECT_EQ(result[i], -17.0f);
        }
        cudaFree(output);
        cudaFree(input);
    }
} // namespace
