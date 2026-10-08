/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/windows/video_extractor_dialog.hpp"
#include "io/video/video_encoder.hpp"
#include "io/video_frame_extractor.hpp"
#include "io/video_player.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Elements/ElementFormControlSelect.h>
#include <gtest/gtest.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

    using lfs::io::ExtractionMode;
    using lfs::io::VideoFrameExtractor;

    bool writeProbedVideo(const std::filesystem::path& path, const char* container,
                          AVCodecID codec_id, int frame_count, int leading_frames);

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

    bool cudaAvailable() {
        int device_count = 0;
        return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
    }

    bool writeEncodedVideo(const std::filesystem::path& video_path,
                           const int frame_count,
                           const int framerate,
                           std::string& error,
                           const int width = kWidth,
                           const int height = kHeight) {
        lfs::io::video::VideoExportOptions options;
        options.preset = lfs::io::video::VideoPreset::CUSTOM;
        options.width = width;
        options.height = height;
        options.framerate = framerate;
        options.crf = 23;

        lfs::io::video::VideoEncoder encoder;
        if (const auto opened = encoder.open(video_path, options); !opened) {
            error = opened.error();
            return false;
        }

        std::vector<float> frame(static_cast<std::size_t>(width) * height * kChannels);
        CudaFloatBuffer device_frame(frame.size());
        if (device_frame.status != cudaSuccess) {
            error = cudaGetErrorString(device_frame.status);
            return false;
        }

        for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
            const float red = static_cast<float>(frame_index + 1) /
                              static_cast<float>(frame_count);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const std::size_t offset =
                        (static_cast<std::size_t>(y) * width + x) * kChannels;
                    frame[offset + 0] = red;
                    frame[offset + 1] = static_cast<float>(x) / static_cast<float>(width - 1);
                    frame[offset + 2] = static_cast<float>(y) / static_cast<float>(height - 1);
                }
            }

            const cudaError_t copy_status = cudaMemcpy(
                device_frame.ptr, frame.data(), frame.size() * sizeof(float), cudaMemcpyHostToDevice);
            if (copy_status != cudaSuccess) {
                error = cudaGetErrorString(copy_status);
                return false;
            }

            if (const auto written = encoder.writeFrameGpu(device_frame.ptr, width, height); !written) {
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

namespace lfs::gui {
    class VideoExtractorDialogTestAccess {
    public:
        static double end(const VideoExtractorDialog& dialog) { return dialog.trim_end_; }

        static void reset(VideoExtractorDialog& dialog) { dialog.handleClick("btn-trim-reset"); }

        static void syncEndInput(VideoExtractorDialog& dialog, Rml::Element* input) {
            dialog.trim_end_input_el_ = input;
            input->SetId("trim-end-input");
            input->AddEventListener(Rml::EventId::Change, &dialog.listener_);
            input->AddEventListener(Rml::EventId::Blur, &dialog.listener_);
            dialog.syncTimeline();
            input->DispatchEvent(Rml::EventId::Blur, {});
            input->RemoveEventListener(Rml::EventId::Change, &dialog.listener_);
            input->RemoveEventListener(Rml::EventId::Blur, &dialog.listener_);
        }

        static void sharpnessWindow(VideoExtractorDialog& dialog, Rml::Element* toggle,
                                    Rml::ElementFormControlSelect* mode) {
            dialog.sharpness_toggle_el_ = toggle;
            dialog.sharpness_mode_select_el_ = mode;
        }

        static VideoExtractionParams request(VideoExtractorDialog& dialog,
                                             const std::filesystem::path& output_dir,
                                             const bool interval, const bool edit_end) {
            dialog.output_dir_ = output_dir;
            dialog.mode_selection_ = interval ? 1 : 0;
            if (edit_end)
                dialog.handleChange("trim-end-input");
            dialog.beginExtractionFromUi();
            EXPECT_TRUE(dialog.pending_params_set_);
            return dialog.pending_params_;
        }
    };
} // namespace lfs::gui

TEST(VideoExtractorDialogTrim, FpsFullRangeKeepsThePreviewEnd) {
    TempDir temp("dialog_fps_end");
    const auto video_path = temp.path / "source.mp4";
    const auto output_dir = temp.path / "frames";
    std::filesystem::create_directories(output_dir);
    std::ofstream(output_dir / "frame_1.png").put('x'); // Capture the pending overwrite request.
    ASSERT_TRUE(writeProbedVideo(video_path, "mp4", AV_CODEC_ID_MPEG4, 50, 0));
    lfs::gui::VideoExtractorDialog dialog;
    ASSERT_TRUE(dialog.openVideoPath(video_path));
    const double expected_end = lfs::gui::VideoExtractorDialogTestAccess::end(dialog);
    const auto request = lfs::gui::VideoExtractorDialogTestAccess::request(dialog, output_dir, false, false);
    EXPECT_DOUBLE_EQ(request.end_time, expected_end);
}

TEST(VideoExtractorDialogTrim, ExplicitEndAtPreviewDurationStaysBounded) {
    TempDir temp("dialog_explicit_end");
    const auto video_path = temp.path / "source.mp4";
    const auto output_dir = temp.path / "frames";
    std::filesystem::create_directories(output_dir);
    std::ofstream(output_dir / "frame_1.png").put('x');
    ASSERT_TRUE(writeProbedVideo(video_path, "mp4", AV_CODEC_ID_MPEG4, 50, 0));
    lfs::gui::VideoExtractorDialog dialog;
    ASSERT_TRUE(dialog.openVideoPath(video_path));
    const double expected_end = lfs::gui::VideoExtractorDialogTestAccess::end(dialog);
    const auto request = lfs::gui::VideoExtractorDialogTestAccess::request(dialog, output_dir, true, true);
    EXPECT_DOUBLE_EQ(request.end_time, expected_end);
}

TEST(VideoExtractorDialogTrim, AutomaticIntervalAndResetStillReadToEnd) {
    TempDir temp("dialog_auto_end");
    const auto video_path = temp.path / "source.mp4";
    const auto output_dir = temp.path / "frames";
    std::filesystem::create_directories(output_dir);
    std::ofstream(output_dir / "frame_1.png").put('x');
    ASSERT_TRUE(writeProbedVideo(video_path, "mp4", AV_CODEC_ID_MPEG4, 50, 0));
    lfs::gui::VideoExtractorDialog dialog;
    ASSERT_TRUE(dialog.openVideoPath(video_path));
    using Access = lfs::gui::VideoExtractorDialogTestAccess;
    EXPECT_DOUBLE_EQ(Access::request(dialog, output_dir, true, false).end_time, -1.0);
    EXPECT_DOUBLE_EQ(Access::request(dialog, output_dir, true, true).end_time, Access::end(dialog));
    Access::reset(dialog);
    EXPECT_DOUBLE_EQ(Access::request(dialog, output_dir, true, false).end_time, -1.0);
    EXPECT_DOUBLE_EQ(Access::request(dialog, output_dir, false, false).end_time, Access::end(dialog));
}

TEST(VideoExtractorDialogTrim, SharpnessWindowKeepsThePreviewEnd) {
    ASSERT_TRUE(Rml::Initialise());
    struct RmlLifetime {
        ~RmlLifetime() { Rml::Shutdown(); }
    } rml_lifetime;
    auto toggle = Rml::Factory::InstanceElement(nullptr, "input", "input", {});
    auto select = Rml::Factory::InstanceElement(nullptr, "select", "select", {});
    auto end_input = Rml::Factory::InstanceElement(nullptr, "input", "input", {});
    ASSERT_TRUE(toggle);
    ASSERT_TRUE(end_input);
    end_input->SetAttribute("type", "text");
    auto* mode = dynamic_cast<Rml::ElementFormControlSelect*>(select.get());
    ASSERT_NE(mode, nullptr);
    toggle->SetAttribute("checked", "");
    mode->Add("Threshold", "threshold");
    mode->Add("Window", "window");
    mode->SetSelection(1);

    TempDir temp("dialog_window_end");
    const auto video_path = temp.path / "source.mp4";
    const auto output_dir = temp.path / "frames";
    std::filesystem::create_directories(output_dir);
    std::ofstream(output_dir / "frame_1.png").put('x');
    ASSERT_TRUE(writeProbedVideo(video_path, "mp4", AV_CODEC_ID_MPEG4, 50, 0));
    lfs::gui::VideoExtractorDialog dialog;
    ASSERT_TRUE(dialog.openVideoPath(video_path));
    lfs::gui::VideoExtractorDialogTestAccess::syncEndInput(dialog, end_input.get());
    EXPECT_DOUBLE_EQ(lfs::gui::VideoExtractorDialogTestAccess::request(dialog, output_dir, true, false).end_time, -1.0);
    lfs::gui::VideoExtractorDialogTestAccess::sharpnessWindow(dialog, toggle.get(), mode);
    const double expected_end = lfs::gui::VideoExtractorDialogTestAccess::end(dialog);
    for (const bool interval : {false, true}) {
        const auto request = lfs::gui::VideoExtractorDialogTestAccess::request(dialog, output_dir, interval, false);
        ASSERT_TRUE(request.sharpness_enabled && request.sharpness_window_mode);
        EXPECT_DOUBLE_EQ(request.end_time, expected_end);
    }
}

TEST(VideoEncoderTimingTest, EncodedFramesHaveRequestedDurationAndRate) {
    if (!cudaAvailable())
        GTEST_SKIP() << "CUDA device required for VideoEncoder-based fixture";

    TempDir temp("encoder_timing");
    for (const int fps : {2, 24}) {
        for (const int frame_count : {1, 2, 7}) {
            SCOPED_TRACE(::testing::Message() << "fps=" << fps << " frames=" << frame_count);
            const auto path = temp.path / (std::to_string(fps) + "_" + std::to_string(frame_count) + ".mp4");
            std::string error;
            ASSERT_TRUE(writeEncodedVideo(path, frame_count, fps, error, 160, 90)) << error;
            AVFormatContext* context = nullptr;
            ASSERT_EQ(avformat_open_input(&context, path.string().c_str(), nullptr, nullptr), 0);
            const auto close_context = [](AVFormatContext* ctx) { avformat_close_input(&ctx); };
            std::unique_ptr<AVFormatContext, decltype(close_context)> owner(context, close_context);
            ASSERT_GE(avformat_find_stream_info(context, nullptr), 0);
            const int index = av_find_best_stream(context, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
            ASSERT_GE(index, 0);
            const auto* stream = context->streams[index];
            EXPECT_EQ(stream->nb_frames, frame_count);
            EXPECT_DOUBLE_EQ(av_q2d(stream->avg_frame_rate), fps);
            EXPECT_NEAR(stream->duration * av_q2d(stream->time_base),
                        static_cast<double>(frame_count) / fps, 1e-6);

            AVPacket* packet = av_packet_alloc();
            ASSERT_NE(packet, nullptr);
            const auto free_packet = [](AVPacket* p) { av_packet_free(&p); };
            std::unique_ptr<AVPacket, decltype(free_packet)> packet_owner(packet, free_packet);
            std::vector<double> timestamps;
            int received = 0;
            while (av_read_frame(context, packet) >= 0) {
                if (packet->stream_index == index) {
                    timestamps.push_back(packet->pts * av_q2d(stream->time_base));
                    EXPECT_NEAR(packet->duration * av_q2d(stream->time_base), 1.0 / fps, 1e-6);
                    ++received;
                }
                av_packet_unref(packet);
            }
            EXPECT_EQ(received, frame_count);
            std::sort(timestamps.begin(), timestamps.end());
            for (size_t i = 0; i < timestamps.size(); ++i)
                EXPECT_NEAR(timestamps[i], static_cast<double>(i) / fps, 1e-6);
        }
    }
}

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

TEST(VideoFrameExtractorOutcome, ReportsExplicitCancellation) {
    if (!cudaAvailable())
        GTEST_SKIP() << "CUDA device required for VideoEncoder-based fixture";

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

namespace {
    // Writes a small CPU-encoded video whose container header may need packet probing to expose
    // the video stream (MPEG-TS, or MPEG-4 Part 2 in MP4 with leading frames before an edit list).
    bool writeProbedVideo(const std::filesystem::path& path, const char* const container,
                          const AVCodecID codec_id, const int frame_count, const int leading_frames) {
        AVFormatContext* format = nullptr;
        const std::string path_utf8 = path.string();
        if (avformat_alloc_output_context2(&format, nullptr, container, path_utf8.c_str()) < 0 || !format)
            return false;
        const AVCodec* const codec = avcodec_find_encoder(codec_id);
        AVStream* const stream = codec ? avformat_new_stream(format, nullptr) : nullptr;
        AVCodecContext* encoder = stream ? avcodec_alloc_context3(codec) : nullptr;
        bool ok = encoder != nullptr;
        if (ok) {
            encoder->width = 64;
            encoder->height = 48;
            encoder->pix_fmt = AV_PIX_FMT_YUV420P;
            encoder->time_base = AVRational{1, 25};
            encoder->framerate = AVRational{25, 1};
            encoder->gop_size = 10;
            encoder->max_b_frames = 0;
            if (format->oformat->flags & AVFMT_GLOBALHEADER)
                encoder->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
            ok = avcodec_open2(encoder, codec, nullptr) >= 0 &&
                 avcodec_parameters_from_context(stream->codecpar, encoder) >= 0;
        }
        if (ok) {
            stream->time_base = encoder->time_base;
            ok = avio_open(&format->pb, path_utf8.c_str(), AVIO_FLAG_WRITE) >= 0 &&
                 avformat_write_header(format, nullptr) >= 0;
        }
        AVFrame* frame = av_frame_alloc();
        AVPacket* packet = av_packet_alloc();
        const auto drain = [&] {
            while (avcodec_receive_packet(encoder, packet) == 0) {
                av_packet_rescale_ts(packet, encoder->time_base, stream->time_base);
                packet->stream_index = stream->index;
                ok = av_interleaved_write_frame(format, packet) >= 0 && ok;
            }
        };
        if (ok) {
            frame->format = encoder->pix_fmt;
            frame->width = encoder->width;
            frame->height = encoder->height;
            ok = av_frame_get_buffer(frame, 0) >= 0;
        }
        for (int index = 0; ok && index < frame_count; ++index) {
            ok = av_frame_make_writable(frame) >= 0;
            for (int y = 0; y < frame->height; ++y)
                for (int x = 0; x < frame->width; ++x)
                    frame->data[0][y * frame->linesize[0] + x] = static_cast<uint8_t>((x + y + index * 8) & 255);
            for (int plane = 1; plane < 3; ++plane)
                for (int y = 0; y < frame->height / 2; ++y)
                    for (int x = 0; x < frame->width / 2; ++x)
                        frame->data[plane][y * frame->linesize[plane] + x] = 128;
            frame->pts = index - leading_frames;
            ok = ok && avcodec_send_frame(encoder, frame) >= 0;
            drain();
        }
        if (ok) {
            avcodec_send_frame(encoder, nullptr);
            drain();
            ok = av_write_trailer(format) >= 0 && ok;
        }
        av_packet_free(&packet);
        av_frame_free(&frame);
        avcodec_free_context(&encoder);
        if (format->pb)
            avio_closep(&format->pb);
        avformat_free_context(format);
        return ok;
    }
} // namespace

TEST(VideoStreamProbe, VideosThatNeedPacketProbingOpenAndExtract) {
    struct Case {
        const char* name;
        const char* container;
        AVCodecID codec;
        int leading_frames;
    };
    for (const Case video : {Case{"source.ts", "mpegts", AV_CODEC_ID_MPEG2VIDEO, 0},
                             Case{"source.mp4", "mp4", AV_CODEC_ID_MPEG4, 6}}) {
        SCOPED_TRACE(video.name);
        TempDir temp("probe");
        const auto video_path = temp.path / video.name;
        constexpr int frame_count = 40;
        ASSERT_TRUE(writeProbedVideo(video_path, video.container, video.codec, frame_count, video.leading_frames));

        lfs::io::VideoPlayer player;
        ASSERT_TRUE(player.open(video_path));
        EXPECT_GT(player.duration(), 0.0);

        const auto output_dir = temp.path / "frames";
        std::filesystem::create_directories(output_dir);
        auto params = extractionParams(video_path, output_dir);
        params.end_time = -1.0;
        params.generate_metadata = false;
        VideoFrameExtractor extractor;
        std::string error;
        ASSERT_TRUE(extractor.extract(params, error)) << error;
        EXPECT_GT(countPngFiles(output_dir), 0u);
    }
}

TEST(VideoFrameExtractorTrim, SharpnessWindowsKeepTheDurationBoundary) {
    TempDir temp("window_duration");
    const auto video_path = temp.path / "source.mkv";
    ASSERT_TRUE(writeProbedVideo(video_path, "matroska", AV_CODEC_ID_MPEG4, 25, 0));

    // Model a recording whose duration estimate is shorter than its packet timeline.
    std::ifstream input(video_path, std::ios::binary);
    std::string bytes{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    input.close();
    const auto duration_offset = bytes.find(std::string("\x44\x89\x88", 3));
    ASSERT_NE(duration_offset, std::string::npos);
    ASSERT_LE(duration_offset + 11, bytes.size());
    const uint64_t duration_bits = std::bit_cast<uint64_t>(520.0);
    for (int i = 0; i < 8; ++i)
        bytes[duration_offset + 3 + i] = static_cast<char>(duration_bits >> (56 - 8 * i));
    std::ofstream(video_path, std::ios::binary).write(bytes.data(), bytes.size());

    auto bounded = extractionParams(video_path, temp.path / "bounded");
    bounded.end_time = 0.52;
    bounded.frame_interval = 10;
    bounded.sharpness.enabled = true;
    bounded.sharpness.window_mode = true;
    bounded.sharpness.window_candidates_target = 0;
    auto automatic = bounded;
    automatic.output_dir = temp.path / "automatic";
    automatic.end_time = -1.0;
    std::string error;
    VideoFrameExtractor bounded_extractor;
    ASSERT_TRUE(bounded_extractor.extract(bounded, error)) << error;
    VideoFrameExtractor automatic_extractor;
    ASSERT_TRUE(automatic_extractor.extract(automatic, error)) << error;
    const auto reference = readMetadata(bounded.output_dir);
    const auto actual = readMetadata(automatic.output_dir);
    ASSERT_EQ(reference["frames"].size(), 2u);
    EXPECT_EQ(actual["frames"], reference["frames"]);
    EXPECT_EQ(actual["processing"]["frame_selection"]["estimated_targets"], 2);
}

TEST(VideoFrameExtractorTrim, EndPastTheStreamExtractsToTheLastFrame) {
    if (!cudaAvailable())
        GTEST_SKIP() << "CUDA device required for VideoEncoder-based fixture";

    TempDir temp("trim_past_end");
    const std::filesystem::path video_path = temp.path / "source.mp4";
    std::string error;
    ASSERT_TRUE(writeEncodedVideo(video_path, kFixtureFrameCount, 10, error)) << error;

    const auto full_dir = temp.path / "full";
    const auto past_dir = temp.path / "past";
    std::filesystem::create_directories(full_dir);
    std::filesystem::create_directories(past_dir);

    auto full = extractionParams(video_path, full_dir);
    full.end_time = -1.0;
    VideoFrameExtractor full_extractor;
    ASSERT_TRUE(full_extractor.extract(full, error)) << error;

    // A frame-count estimate a few frames longer than the stream, as the preview reports for some files.
    auto past = extractionParams(video_path, past_dir);
    past.end_time = kFixtureFrameCount / 10.0 + 0.3;
    VideoFrameExtractor past_extractor;
    ASSERT_TRUE(past_extractor.extract(past, error)) << error;
    EXPECT_EQ(countPngFiles(past_dir), countPngFiles(full_dir));
    EXPECT_EQ(countPngFiles(past_dir), static_cast<std::size_t>(kFixtureFrameCount));

    auto late_start = extractionParams(video_path, temp.path / "late");
    late_start.start_time = kFixtureFrameCount / 10.0 + 0.3;
    late_start.end_time = -1.0;
    VideoFrameExtractor late_extractor;
    EXPECT_FALSE(late_extractor.extract(late_start, error));
}

TEST(VideoFrameExtractorTrim, PlayerFullRangeKeepsTheLastFrame) {
    if (!cudaAvailable())
        GTEST_SKIP() << "CUDA device required for VideoEncoder-based fixture";

    TempDir temp("trim_full_range");
    const std::filesystem::path video_path = temp.path / "source.mp4";
    const auto output_dir = temp.path / "frames";
    std::filesystem::create_directories(output_dir);
    constexpr int frame_count = 7;
    std::string error;
    ASSERT_TRUE(writeEncodedVideo(video_path, frame_count, 25, error)) << error;

    lfs::io::VideoPlayer player;
    ASSERT_TRUE(player.open(video_path));
    // The dialog's default range ends at the player's duration, kept as a float.
    const float trim_end = static_cast<float>(player.duration());

    auto params = extractionParams(video_path, output_dir);
    params.end_time = lfs::io::extractionEndTime(trim_end, player.duration());
    VideoFrameExtractor extractor;
    ASSERT_TRUE(extractor.extract(params, error)) << error;
    EXPECT_EQ(countPngFiles(output_dir), static_cast<std::size_t>(frame_count));
}
