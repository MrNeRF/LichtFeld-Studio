/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/video/video_encoder.hpp"
#include "io/video_frame_extractor.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/pixfmt.h>
}

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <stb_image.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

    constexpr int WIDTH = 64;
    constexpr int HEIGHT = 64;
    constexpr int FPS = 10;

    struct TempDir {
        TempDir() {
            const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
            path = std::filesystem::temp_directory_path() /
                   ("lfs_videotoolbox_test_" + std::to_string(stamp));
            std::filesystem::create_directories(path);
        }

        ~TempDir() {
            std::error_code error;
            std::filesystem::remove_all(path, error);
        }

        std::filesystem::path path;
    };

    bool writeH264(const std::filesystem::path& path) {
        lfs::io::video::VideoExportOptions options;
        options.preset = lfs::io::video::VideoPreset::CUSTOM;
        options.width = WIDTH;
        options.height = HEIGHT;
        options.framerate = FPS;
        lfs::io::video::VideoEncoder encoder;
        if (const auto opened = encoder.open(path, options); !opened) {
            std::cerr << "H.264 fixture open: " << opened.error() << '\n';
            return false;
        }

        std::vector<std::uint8_t> rgba(WIDTH * HEIGHT * 4, 0);
        for (int pixel = 0; pixel < WIDTH * HEIGHT; ++pixel) {
            rgba[pixel * 4] = 255;
            rgba[pixel * 4 + 3] = 255;
        }
        for (int frame = 0; frame < 6; ++frame) {
            if (const auto written = encoder.writeFrame(rgba, WIDTH, HEIGHT); !written) {
                std::cerr << "H.264 fixture frame: " << written.error() << '\n';
                return false;
            }
        }
        if (const auto closed = encoder.close(); !closed) {
            std::cerr << "H.264 fixture close: " << closed.error() << '\n';
            return false;
        }
        return true;
    }

    bool writeRawVideo(const std::filesystem::path& path) {
        AVFormatContext* format = nullptr;
        if (avformat_alloc_output_context2(&format, nullptr, "nut",
                                           path.string().c_str()) < 0 ||
            !format)
            return false;

        const auto close = [&]() {
            if (format->pb)
                avio_closep(&format->pb);
            avformat_free_context(format);
        };
        AVStream* const stream = avformat_new_stream(format, nullptr);
        if (!stream) {
            close();
            return false;
        }
        stream->time_base = {1, FPS};
        stream->avg_frame_rate = {FPS, 1};
        stream->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
        stream->codecpar->codec_id = AV_CODEC_ID_RAWVIDEO;
        stream->codecpar->format = AV_PIX_FMT_RGB24;
        stream->codecpar->width = WIDTH;
        stream->codecpar->height = HEIGHT;
        if (avio_open(&format->pb, path.string().c_str(), AVIO_FLAG_WRITE) < 0 ||
            avformat_write_header(format, nullptr) < 0) {
            close();
            return false;
        }

        bool wrote_frames = true;
        for (int frame = 0; frame < 2; ++frame) {
            AVPacket* packet = av_packet_alloc();
            if (!packet || av_new_packet(packet, WIDTH * HEIGHT * 3) < 0) {
                av_packet_free(&packet);
                wrote_frames = false;
                break;
            }
            for (int pixel = 0; pixel < WIDTH * HEIGHT; ++pixel) {
                packet->data[pixel * 3] = 255;
                packet->data[pixel * 3 + 1] = 0;
                packet->data[pixel * 3 + 2] = 0;
            }
            packet->stream_index = stream->index;
            packet->pts = av_rescale_q(frame, AVRational{1, FPS}, stream->time_base);
            packet->dts = packet->pts;
            packet->duration = av_rescale_q(1, AVRational{1, FPS}, stream->time_base);
            if (av_interleaved_write_frame(format, packet) < 0)
                wrote_frames = false;
            av_packet_free(&packet);
            if (!wrote_frames)
                break;
        }
        const bool trailer_ok = av_write_trailer(format) >= 0;
        close();
        return wrote_frames && trailer_ok;
    }

    bool extractAndCheck(const std::filesystem::path& video,
                         const std::filesystem::path& output,
                         const char* expected_backend,
                         const std::size_t expected_frames,
                         const double end_time) {
        std::filesystem::create_directories(output);
        lfs::io::VideoFrameExtractor::Params params;
        params.video_path = video;
        params.output_dir = output;
        params.mode = lfs::io::ExtractionMode::INTERVAL;
        params.frame_interval = 1;
        params.format = lfs::io::ImageFormat::PNG;
        params.generate_metadata = true;
        params.end_time = end_time;

        lfs::io::VideoFrameExtractor extractor;
        std::string error;
        if (!extractor.extract(params, error)) {
            std::cerr << "Extraction failed: " << error << '\n';
            return false;
        }
        if (extractor.lastOutcome() != lfs::io::ExtractionOutcome::Completed) {
            std::cerr << "Extraction did not complete\n";
            return false;
        }
        std::size_t frames = 0;
        for (const auto& entry : std::filesystem::directory_iterator(output)) {
            if (entry.path().extension() == ".png") {
                if (entry.file_size() == 0)
                    return false;
                ++frames;
            }
        }
        if (frames != expected_frames) {
            std::cerr << "Expected " << expected_frames << " frames, got " << frames << '\n';
            return false;
        }
        std::ifstream metadata_file(output / "extraction_metadata.json");
        if (!metadata_file)
            return false;
        const nlohmann::json metadata = nlohmann::json::parse(metadata_file);
        const std::string backend =
            metadata["processing"]["decoder"]["backend"].get<std::string>();
        if (backend != expected_backend) {
            std::cerr << "Expected decoder " << expected_backend << ", got "
                      << backend << '\n';
            return false;
        }
        return true;
    }

} // namespace

TEST(VideoToolboxExtractor, HardwareH264) {
    TempDir temp;
    const auto h264 = temp.path / "hardware.mp4";
    ASSERT_TRUE(writeH264(h264));
    EXPECT_TRUE(extractAndCheck(h264, temp.path / "hardware_frames",
                                "videotoolbox", 6, 0.5));
}

// The export encoder (VideoToolbox on Macs) keeps the fixture's solid red.
TEST(VideoToolboxEncoder, EncodedFramesDecodeToTheirColor) {
    TempDir temp;
    const auto h264 = temp.path / "red.mp4";
    ASSERT_TRUE(writeH264(h264));
    ASSERT_TRUE(extractAndCheck(h264, temp.path / "red_frames", "videotoolbox", 6, 0.5));
    for (const auto& entry : std::filesystem::directory_iterator(temp.path / "red_frames")) {
        if (entry.path().extension() != ".png")
            continue;
        int width = 0, height = 0, channels = 0;
        stbi_uc* const pixels = stbi_load(entry.path().string().c_str(), &width, &height, &channels, 3);
        ASSERT_NE(pixels, nullptr) << entry.path();
        EXPECT_EQ(width, WIDTH);
        EXPECT_EQ(height, HEIGHT);
        const stbi_uc* const center = pixels + ((HEIGHT / 2) * WIDTH + WIDTH / 2) * 3;
        EXPECT_GT(center[0], 230) << entry.path();
        EXPECT_LT(center[1], 25) << entry.path();
        EXPECT_LT(center[2], 25) << entry.path();
        stbi_image_free(pixels);
    }
}

TEST(VideoToolboxExtractor, SoftwareFallback) {
    TempDir temp;
    const auto raw = temp.path / "software.nut";
    ASSERT_TRUE(writeRawVideo(raw));
    EXPECT_TRUE(extractAndCheck(raw, temp.path / "software_frames",
                                "ffmpeg_software", 1, -1.0));
}
