/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// The tensor HDR tonemapper against libplacebo on generated HDR10, HLG and
// Dolby Vision frames. Builds without Vulkan check that the tensor path is the
// one HdrLibplaceboRenderer selects.

#include "core/tensor_backend.hpp"
#include "core/tensor_color.hpp"
#include "hdr_tonemap_tensor.hpp"
#include "media/decoded_video_frame_ffmpeg.hpp"
#include "media/hdr_renderer.hpp"
#include "media_studio_backends.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/dovi_meta.h>
#include <libavutil/hdr_dynamic_metadata.h>
}

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace {
    using namespace lfs::io;

    struct Clip {
        AVFormatContext* format = nullptr;
        AVFrame* frame = nullptr;

        ~Clip() {
            av_frame_free(&frame);
            avformat_close_input(&format);
        }
        AVStream* stream() const { return format->streams[0]; }
    };

    // A bars-over-gradient frame encoded as 10-bit 4:2:0 HEVC.
    std::unique_ptr<Clip> generate(const std::filesystem::path& path, const bool pq,
                                   const int width = 320, const int half_height = 90) {
        const std::string command = std::format(
            "'{}' -hide_banner -loglevel error -y "
            "-f lavfi -i smptehdbars=s={}x{} "
            "-f lavfi -i gradients=s={}x{}:c0=black:c1=white:x0=0:y0=0:x1={}:y1=0:nb_colors=2 "
            "-filter_complex '[0]format=yuv444p[a];[1]format=yuv444p[b];[a][b]vstack,format=yuv420p10le' "
            "-frames:v 1 -c:v libx265 -x265-params "
            "'log-level=error:lossless=1:colorprim=bt2020:colormatrix=bt2020nc:transfer={}{}' "
            "-color_primaries bt2020 -colorspace bt2020nc -color_trc {} '{}'",
            LFS_HDR_TEST_FFMPEG, width, half_height, width, half_height, width - 1,
            pq ? "smpte2084" : "arib-std-b67",
            pq ? ":hdr10=1:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(40000000,50):max-cll=4000,1000"
               : "",
            pq ? "smpte2084" : "arib-std-b67", path.string());
        if (std::system(command.c_str()) != 0)
            return nullptr;

        auto clip = std::make_unique<Clip>();
        if (avformat_open_input(&clip->format, path.c_str(), nullptr, nullptr) < 0 ||
            avformat_find_stream_info(clip->format, nullptr) < 0)
            return nullptr;
        const AVCodec* decoder = avcodec_find_decoder(clip->stream()->codecpar->codec_id);
        AVCodecContext* codec = avcodec_alloc_context3(decoder);
        AVPacket* packet = av_packet_alloc();
        clip->frame = av_frame_alloc();
        bool decoded = avcodec_parameters_to_context(codec, clip->stream()->codecpar) >= 0 &&
                       avcodec_open2(codec, decoder, nullptr) >= 0;
        while (decoded && av_read_frame(clip->format, packet) >= 0) {
            avcodec_send_packet(codec, packet);
            av_packet_unref(packet);
        }
        decoded = decoded && avcodec_send_packet(codec, nullptr) >= 0 &&
                  avcodec_receive_frame(codec, clip->frame) >= 0;
        av_packet_free(&packet);
        avcodec_free_context(&codec);
        return decoded ? std::move(clip) : nullptr;
    }

    // Profile 5 style RPU with an L1 block. The reshaped variant has a
    // polynomial luma and MMR chroma curve, each continuous at its pivot.
    void attachDolbyVision(AVFrame* frame, const bool identity) {
        size_t size = 0;
        AVDOVIMetadata* dovi = av_dovi_metadata_alloc(&size);
        AVDOVIRpuDataHeader* header = av_dovi_get_header(dovi);
        header->disable_residual_flag = 1;
        header->bl_bit_depth = 10;
        header->coef_log2_denom = 23;
        const auto fixed = [](const double value) { return std::llround(value * (1 << 23)); };
        constexpr double pivot = 400.0 / 1023.0;
        AVDOVIDataMapping* mapping = av_dovi_get_mapping(dovi);
        for (int c = 0; c < 3; ++c) {
            AVDOVIReshapingCurve& curve = mapping->curves[c];
            curve.num_pivots = identity ? 2 : 3;
            curve.pivots[0] = 0;
            curve.pivots[1] = identity ? 1023 : 400;
            curve.pivots[2] = 1023;
            for (int piece = 0; piece + 1 < curve.num_pivots; ++piece) {
                if (identity) {
                    curve.mapping_idc[piece] = AV_DOVI_MAPPING_POLYNOMIAL;
                    curve.poly_order[piece] = 1;
                    curve.poly_coef[piece][1] = fixed(1.0);
                } else if (c == 0) {
                    // s' = 0.02 + 0.9 s + 0.1 s^2, plus 0.3 (s^2 - pivot s) past the pivot
                    curve.mapping_idc[piece] = AV_DOVI_MAPPING_POLYNOMIAL;
                    curve.poly_order[piece] = 2;
                    curve.poly_coef[piece][0] = fixed(0.02);
                    curve.poly_coef[piece][1] = fixed(0.9 - 0.3 * pivot * piece);
                    curve.poly_coef[piece][2] = fixed(0.1 + 0.3 * piece);
                } else {
                    // Identity with cross terms, plus third order terms past the pivot
                    curve.mapping_idc[piece] = AV_DOVI_MAPPING_MMR;
                    curve.mmr_order[piece] = piece + 2;
                    curve.mmr_constant[piece] = fixed(-0.01);
                    for (int k = 0; k < 7; ++k) {
                        curve.mmr_coef[piece][0][k] = fixed(k == c ? 1.0 - 0.2 * pivot * piece : (k % 3 == 1 ? 0.05 : -0.05));
                        curve.mmr_coef[piece][1][k] = fixed((k == c ? (0.2 - 0.1 * pivot) * piece : 0.0) +
                                                            (k % 3 == 1 ? 0.025 : -0.025));
                        curve.mmr_coef[piece][2][k] = fixed(k == c ? 0.1 * piece : 0.0);
                    }
                }
            }
        }
        // BT.2020 YCbCr to R'G'B' and the RGB to LMS matrix Dolby Vision undoes.
        AVDOVIColorMetadata* color = av_dovi_get_color(dovi);
        constexpr std::array<double, 9> ycc{1.1678, 0.0, 1.6836, 1.1678, -0.1879, -0.6523, 1.1678, 2.1481, 0.0};
        constexpr std::array<double, 9> lms{0.3592, 0.6976, -0.0358, -0.1922, 1.1004, 0.0755, 0.0070, 0.0749, 0.8434};
        for (int i = 0; i < 9; ++i) {
            color->ycc_to_rgb_matrix[i] = av_d2q(ycc[i], 1 << 13);
            color->rgb_to_lms_matrix[i] = av_d2q(lms[i], 1 << 14);
        }
        color->ycc_to_rgb_offset[0] = av_make_q(64, 1023);
        color->ycc_to_rgb_offset[1] = color->ycc_to_rgb_offset[2] = av_make_q(512, 1023);
        color->source_min_pq = 62;
        color->source_max_pq = 3079;
        dovi->num_ext_blocks = 1;
        AVDOVIDmData* level1 = av_dovi_get_ext(dovi, 0);
        level1->level = 1;
        level1->l1 = {.min_pq = 0, .max_pq = 3200, .avg_pq = 1400};
        AVBufferRef* buffer = av_buffer_create(reinterpret_cast<uint8_t*>(dovi), size, nullptr, nullptr, 0);
        ASSERT_NE(av_frame_new_side_data_from_buf(frame, AV_FRAME_DATA_DOVI_METADATA, buffer), nullptr);
    }

    // HDR10+ scene statistics: MaxSCL 1500/1200/600 nits, average 120 nits.
    void attachHdr10Plus(AVFrame* frame) {
        AVDynamicHDRPlus* plus = av_dynamic_hdr_plus_create_side_data(frame);
        ASSERT_NE(plus, nullptr);
        plus->application_version = 1;
        plus->num_windows = 1;
        AVHDRPlusColorTransformParams& params = plus->params[0];
        params.maxscl[0] = av_make_q(15000, 100000);
        params.maxscl[1] = av_make_q(12000, 100000);
        params.maxscl[2] = av_make_q(6000, 100000);
        params.average_maxrgb = av_make_q(1200, 100000);
    }

    enum class Metadata { Static,
                          Hdr10Plus,
                          DolbyVisionIdentity,
                          DolbyVisionReshaped };

    struct Case {
        const char* name;
        bool pq;
        Metadata metadata;
        int width;
        int height;
        int rotation;
        bool rgba;
    };

    class HdrTensorTonemap : public testing::TestWithParam<Case> {
    public:
        static void SetUpTestSuite() { lfs::io::registerStudioMediaBackends(); }
    };

    TEST(HdrTonemapPerformance, DISABLED_FourK) {
        lfs::io::registerStudioMediaBackends();
        if (!std::filesystem::exists(LFS_HDR_TEST_FFMPEG))
            GTEST_SKIP() << "ffmpeg is unavailable";
        const auto directory = std::filesystem::temp_directory_path() / "lfs-hdr-benchmark";
        std::filesystem::create_directories(directory);
        const auto clip = generate(directory / "clip.mkv", true, 3840, 1080);
        std::filesystem::remove_all(directory);
        ASSERT_NE(clip, nullptr) << "ffmpeg could not encode the HEVC test clip";

        const auto benchmark = [&](auto& renderer, const char* name, const bool preview) {
            std::vector<unsigned char> pixels;
            std::string error;
            std::vector<double> elapsed;
            elapsed.reserve(31);
            for (int call = 0; call < 31; ++call) {
                const auto started = std::chrono::steady_clock::now();
                auto described = lfs::media::detail::describeDecodedVideoFrame(clip->frame, clip->stream());
                ASSERT_TRUE(described.has_value());
                const bool rendered = preview
                                          ? renderer.tonemapToSdrRgba(&*described, HdrFormat::HDR10,
                                                                      1920, 1080, 0, pixels, error)
                                          : renderer.tonemapToSdr(&*described, HdrFormat::HDR10,
                                                                  3840, 2160, pixels, error);
                ASSERT_TRUE(rendered) << error;
                elapsed.push_back(1000.0 * std::chrono::duration<double>(
                                               std::chrono::steady_clock::now() - started)
                                               .count());
            }
            std::sort(elapsed.begin() + 1, elapsed.end());
            const double mean = std::accumulate(elapsed.begin() + 1, elapsed.end(), 0.0) / 30.0;
            std::printf("%s %s: first %.3f ms, steady mean %.3f ms, median %.3f ms, min %.3f ms\n", name,
                        preview ? "4K->1080 RGBA" : "4K RGB", elapsed[0], mean, elapsed[16], elapsed[1]);
        };

        HdrTensorRenderer tensor;
        benchmark(tensor, "tensor", false);
        HdrTensorRenderer tensor_preview;
        benchmark(tensor_preview, "tensor", true);
#ifdef LFS_USE_VULKAN
        HdrLibplaceboRenderer libplacebo;
        benchmark(libplacebo, "libplacebo", false);
        HdrLibplaceboRenderer libplacebo_preview;
        benchmark(libplacebo_preview, "libplacebo", true);
#endif
    }

    TEST_P(HdrTensorTonemap, MatchesLibplacebo) {
        const Case& c = GetParam();
        if (!std::filesystem::exists(LFS_HDR_TEST_FFMPEG))
            GTEST_SKIP() << "ffmpeg is unavailable";
        const auto directory = std::filesystem::temp_directory_path() / std::format("lfs-hdr-{}", c.name);
        std::filesystem::create_directories(directory);
        const auto clip = generate(directory / "clip.mkv", c.pq);
        std::filesystem::remove_all(directory);
        ASSERT_NE(clip, nullptr) << "ffmpeg could not encode the HEVC test clip";
        if (c.metadata == Metadata::Hdr10Plus)
            attachHdr10Plus(clip->frame);
        const bool dolby_vision = c.metadata == Metadata::DolbyVisionIdentity || c.metadata == Metadata::DolbyVisionReshaped;
        if (dolby_vision)
            attachDolbyVision(clip->frame, c.metadata == Metadata::DolbyVisionIdentity);
        const HdrFormat format = dolby_vision ? HdrFormat::DOLBY_VISION_NATIVE
                                              : (c.pq ? HdrFormat::HDR10 : HdrFormat::HLG);

        // Two calls exercise the temporal peak state of the RGBA path.
        const auto render = [&](auto& renderer, std::vector<unsigned char>& pixels) {
            std::string error;
            ASSERT_TRUE(renderer.isAvailable(error)) << error;
            auto described = lfs::media::detail::describeDecodedVideoFrame(clip->frame, clip->stream());
            ASSERT_TRUE(described.has_value());
            for (int call = 0; call < (c.rgba ? 2 : 1); ++call) {
                ASSERT_TRUE(c.rgba ? renderer.tonemapToSdrRgba(&*described, format, c.width,
                                                               c.height, c.rotation, pixels, error)
                                   : renderer.tonemapToSdr(&*described, format, c.width, c.height,
                                                           pixels, error))
                    << error;
            }
            ASSERT_EQ(pixels.size(), size_t(c.width) * c.height * (c.rgba ? 4 : 3));
        };
        HdrTensorRenderer tensor;
        std::vector<unsigned char> actual;
        ASSERT_NO_FATAL_FAILURE(render(tensor, actual));
        EXPECT_EQ(lfs::core::default_gpu_backend(), lfs::core::GpuBackend::Metal);

        HdrLibplaceboRenderer reference;
        std::vector<unsigned char> expected;
        ASSERT_NO_FATAL_FAILURE(render(reference, expected));
#ifndef LFS_USE_VULKAN
        // Without Vulkan, HdrLibplaceboRenderer is the tensor implementation.
        EXPECT_EQ(expected, actual);
#else
        const size_t channels = c.rgba ? 4 : 3;
        std::array<int, 3> maximum{};
        std::array<double, 3> mean{};
        for (size_t i = 0; i < actual.size(); ++i) {
            const size_t channel = i % channels;
            if (channel == 3) {
                EXPECT_EQ(actual[i], 255);
                continue;
            }
            const int delta = std::abs(int(actual[i]) - int(expected[i]));
            maximum[channel] = std::max(maximum[channel], delta);
            mean[channel] += delta;
        }
        // The residual is mostly libplacebo's blue noise, which rand() seeds.
        for (size_t channel = 0; channel < 3; ++channel) {
            mean[channel] /= double(actual.size() / channels);
            EXPECT_LT(mean[channel], 0.5) << "channel " << channel;
            EXPECT_LE(maximum[channel], 3) << "channel " << channel;
        }
        std::printf("%s tensor vs libplacebo: mean |diff| %.3f %.3f %.3f, max |diff| %d %d %d\n", c.name,
                    mean[0], mean[1], mean[2], maximum[0], maximum[1], maximum[2]);
#endif
    }

    INSTANTIATE_TEST_SUITE_P(
        GeneratedClips, HdrTensorTonemap,
        testing::Values(Case{"hdr10", true, Metadata::Static, 320, 180, 0, false},
                        Case{"hlg", false, Metadata::Static, 320, 180, 0, false},
                        Case{"hdr10_plus", true, Metadata::Hdr10Plus, 320, 180, 0, false},
                        Case{"hdr10_downscaled", true, Metadata::Static, 200, 120, 0, false},
                        Case{"hdr10_upscaled", true, Metadata::Static, 480, 270, 0, false},
                        Case{"hdr10_preview_rotated", true, Metadata::Static, 180, 320, 90, true},
                        Case{"hlg_preview", false, Metadata::Static, 320, 180, 0, true},
                        Case{"dolby_vision_identity", true, Metadata::DolbyVisionIdentity, 320, 180, 0, false},
                        Case{"dolby_vision_reshaped", true, Metadata::DolbyVisionReshaped, 320, 180, 0, true}),
        [](const testing::TestParamInfo<Case>& info) { return std::string(info.param.name); });
} // namespace

TEST(HdrTensorColor, SharedProgramPreservesQuantization) {
    using namespace lfs::core;
    std::vector<float> pixels(34 * 18 * 3);
    for (size_t i = 0; i < pixels.size(); ++i)
        pixels[i] = float((i * 719) % 1024) / 511.f - .5f;
    for (int i = 0; i < 255; ++i) {
        const auto edge = (i + .5f) / 255.f;
        pixels[3 * i] = std::nextafter(edge, 0.f);
        pixels[3 * i + 1] = edge;
        pixels[3 * i + 2] = std::nextafter(edge, 1.f);
    }
    pixels[1000] = std::numeric_limits<float>::quiet_NaN();
    pixels[1001] = std::numeric_limits<float>::infinity();
    auto host = Tensor::from_blob(pixels.data(), {18, 34, 3}, Device::CPU, DataType::Float32);
    auto expected = rgb_to_yuv420p(host);
    ASSERT_TRUE(expected.has_value());
    for (const auto backend : {GpuBackend::Metal
#if LFS_TENSOR_VULKAN
                               ,
                               GpuBackend::Vulkan
#endif
         }) {
        GpuBackendScope scope(backend);
        auto input = host.gpu();
        input = Tensor::cat({input, Tensor::zeros_like(input)}, 1).slice(1, 0, 34);
        ASSERT_FALSE(input.is_contiguous());
        auto actual = rgb_to_yuv420p(input);
        ASSERT_TRUE(actual.has_value());
        EXPECT_EQ(actual->y.to_vector_uint8(), expected->y.to_vector_uint8());
        EXPECT_EQ(actual->u.to_vector_uint8(), expected->u.to_vector_uint8());
        EXPECT_EQ(actual->v.to_vector_uint8(), expected->v.to_vector_uint8());
        Yuv420Planes invalid{actual->y, actual->u, actual->u};
        EXPECT_FALSE(rgb_to_yuv420p_into(input, invalid));
    }
}
