/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/video/sws_packed_rgb.hpp"
#include "io/video_frame_extractor.hpp"
#include "io/video_player.hpp"

#include <gtest/gtest.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace {

    void fillGradient(AVFrame* frame, const int seed) {
        const auto* desc = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(frame->format));
        for (int plane = 0; plane < 3; ++plane) {
            const int w = plane == 0 ? frame->width : AV_CEIL_RSHIFT(frame->width, desc->log2_chroma_w);
            const int h = plane == 0 ? frame->height : AV_CEIL_RSHIFT(frame->height, desc->log2_chroma_h);
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    frame->data[plane][y * frame->linesize[plane] + x] =
                        static_cast<uint8_t>((x * 7 + y * 3 + plane * 50 + seed * 11) & 0xff);
        }
    }

    // A video that the player and extractor decode on the CPU, at a width that is not a multiple of 16.
    bool writeMjpegVideo(const std::filesystem::path& path, const int width, const int height, const int frames) {
        AVFormatContext* format = nullptr;
        if (avformat_alloc_output_context2(&format, nullptr, "avi", path.string().c_str()) < 0 || !format)
            return false;
        const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
        AVCodecContext* encoder = codec ? avcodec_alloc_context3(codec) : nullptr;
        AVStream* stream = encoder ? avformat_new_stream(format, nullptr) : nullptr;
        bool ok = stream != nullptr;
        AVFrame* frame = av_frame_alloc();
        AVPacket* packet = av_packet_alloc();
        if (ok) {
            encoder->width = width;
            encoder->height = height;
            encoder->pix_fmt = AV_PIX_FMT_YUVJ420P;
            encoder->time_base = {1, 10};
            stream->time_base = encoder->time_base;
            ok = avcodec_open2(encoder, codec, nullptr) >= 0 &&
                 avcodec_parameters_from_context(stream->codecpar, encoder) >= 0 &&
                 avio_open(&format->pb, path.string().c_str(), AVIO_FLAG_WRITE) >= 0 &&
                 avformat_write_header(format, nullptr) >= 0;
        }
        const auto drain = [&] {
            while (avcodec_receive_packet(encoder, packet) == 0) {
                av_packet_rescale_ts(packet, encoder->time_base, stream->time_base);
                packet->stream_index = stream->index;
                ok = ok && av_interleaved_write_frame(format, packet) >= 0;
            }
        };
        for (int i = 0; ok && i < frames; ++i) {
            frame->format = encoder->pix_fmt;
            frame->width = width;
            frame->height = height;
            ok = av_frame_get_buffer(frame, 0) >= 0;
            if (!ok)
                break;
            fillGradient(frame, i);
            frame->pts = i;
            ok = avcodec_send_frame(encoder, frame) >= 0;
            av_frame_unref(frame);
            drain();
        }
        if (ok) {
            avcodec_send_frame(encoder, nullptr);
            drain();
            ok = av_write_trailer(format) >= 0;
        }
        av_packet_free(&packet);
        av_frame_free(&frame);
        avcodec_free_context(&encoder);
        if (format && format->pb)
            avio_closep(&format->pb);
        avformat_free_context(format);
        return ok;
    }

    std::filesystem::path tempDir(const std::string& name) {
        const auto dir = std::filesystem::temp_directory_path() / ("lfs_cpu_rgb_" + name);
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        return dir;
    }

} // namespace

TEST(VideoCpuRgbConversion, PackedOutputStaysInsideTheFrame) {
    for (const auto format : {AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUVJ420P, AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUVJ422P}) {
        for (const int width : {2, 6, 14, 18, 34, 100, 648}) {
            SCOPED_TRACE(std::string(av_get_pix_fmt_name(format)) + " width " + std::to_string(width));
            constexpr int height = 10;
            AVFrame* source = av_frame_alloc();
            source->format = format;
            source->width = width;
            source->height = height;
            ASSERT_GE(av_frame_get_buffer(source, 0), 0);
            fillGradient(source, width);
            SwsContext* context = sws_getContext(width, height, format, width, height, AV_PIX_FMT_RGB24,
                                                 SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
            ASSERT_NE(context, nullptr);

            constexpr std::size_t guard = 256;
            const std::size_t packed = static_cast<std::size_t>(width) * height * 3;
            std::vector<uint8_t> destination(packed + guard, 0xa5);
            std::vector<uint8_t> scratch;
            EXPECT_GT(lfs::io::scaleToPackedRgb24(context, source->data, source->linesize, height,
                                                  destination.data(), width, height, scratch),
                      0);
            for (std::size_t i = packed; i < destination.size(); ++i)
                ASSERT_EQ(destination[i], 0xa5) << "byte " << i - packed << " after the frame";

            // Same conversion into a generously padded buffer: the packed rows must match it.
            // Aligned like the helper's rows so swscale takes the same code path.
            const int stride = (width * 3 + 63) / 64 * 64;
            std::vector<uint8_t> reference_storage(static_cast<std::size_t>(stride) * height + 128);
            uint8_t* const reference =
                reference_storage.data() + (64 - reinterpret_cast<std::uintptr_t>(reference_storage.data()) % 64) % 64;
            uint8_t* dst_data[4] = {reference, nullptr, nullptr, nullptr};
            int dst_linesize[4] = {stride, 0, 0, 0};
            ASSERT_GT(sws_scale(context, source->data, source->linesize, 0, height, dst_data, dst_linesize), 0);
            for (int y = 0; y < height; ++y)
                ASSERT_EQ(0, std::memcmp(destination.data() + static_cast<std::size_t>(y) * width * 3,
                                         reference + static_cast<std::size_t>(y) * stride,
                                         static_cast<std::size_t>(width) * 3))
                    << "row " << y;
            sws_freeContext(context);
            av_frame_free(&source);
        }
    }
}

// The preview and the extractor used to convert straight into tightly packed frames; for CPU-decoded
// videos whose width is not a multiple of 16 that overran the heap buffer and aborted the process.
TEST(VideoCpuRgbConversion, CpuDecodedVideoWithUnalignedWidthPlaysAndExtracts) {
    const auto dir = tempDir("mjpeg");
    const auto video = dir / "unaligned.avi";
    ASSERT_TRUE(writeMjpegVideo(video, 648, 420, 12));
    EXPECT_EXIT(
        {
            {
                lfs::io::VideoPlayer player;
                if (!player.open(video) || player.width() != 648 || player.height() != 420)
                    std::_Exit(2);
                int frames = 0;
                player.togglePlayPause();
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
                while (frames < 6 && std::chrono::steady_clock::now() < deadline) {
                    if (player.update(0.1) && player.currentFrameData())
                        ++frames;
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }
                if (frames < 6)
                    std::_Exit(3);
            }
            lfs::io::VideoFrameExtractor extractor;
            lfs::io::VideoFrameExtractor::Params params;
            params.video_path = video;
            params.output_dir = dir / "frames";
            params.mode = lfs::io::ExtractionMode::INTERVAL;
            params.frame_interval = 1;
            std::string error;
            if (!extractor.extract(params, error))
                std::_Exit(4);
            std::size_t written = 0;
            for (const auto& entry : std::filesystem::directory_iterator(params.output_dir))
                written += entry.path().extension() == ".png";
            std::_Exit(written == 12 ? 0 : 5);
        },
        ::testing::ExitedWithCode(0), "");
    std::filesystem::remove_all(dir);
}
