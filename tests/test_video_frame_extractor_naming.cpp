/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/windows/video_extractor_dialog.hpp"
#include "io/video/video_encoder.hpp"
#include "io/video_frame_extractor.hpp"
#include "io/video_player.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Elements/ElementFormControlInput.h>
#include <RmlUi/Core/Elements/ElementFormControlSelect.h>
#include <RmlUi/Core/RenderInterface.h>
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

        static bool attach(VideoExtractorDialog& dialog, Rml::ElementDocument* document) {
            dialog.document_ = document;
            dialog.cacheElements();
            dialog.syncLocale();
            dialog.syncControls();
            return dialog.elements_cached_;
        }

        static void sync(VideoExtractorDialog& dialog) { dialog.syncControls(); }
        static float fps(const VideoExtractorDialog& dialog) { return dialog.fps_; }

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

namespace {
    bool writeProbedVideoWithRate(const std::filesystem::path& path, const char* container,
                                  AVCodecID codec_id, int frame_count, int leading_frames,
                                  int rate_numerator, int rate_denominator);

    class FpsTestRenderInterface final : public Rml::RenderInterface {
    public:
        Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex>, Rml::Span<const int>) override { return 1; }
        void RenderGeometry(Rml::CompiledGeometryHandle, Rml::Vector2f, Rml::TextureHandle) override {}
        void ReleaseGeometry(Rml::CompiledGeometryHandle) override {}
        Rml::TextureHandle LoadTexture(Rml::Vector2i& dimensions, const Rml::String&) override {
            dimensions = {16, 16};
            return 1;
        }
        Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>, Rml::Vector2i) override { return 1; }
        void ReleaseTexture(Rml::TextureHandle) override {}
        void EnableScissorRegion(bool) override {}
        void SetScissorRegion(Rml::Rectanglei) override {}
    };

    class VideoExtractorFpsInputTest : public ::testing::Test {
    protected:
        using Access = lfs::gui::VideoExtractorDialogTestAccess;
        void SetUp() override {
            ASSERT_TRUE(Rml::Initialise());
            const auto root = std::filesystem::path(PROJECT_ROOT_PATH);
            ASSERT_TRUE(Rml::LoadFontFace((root / "src/visualizer/gui/assets/fonts/Inter-Regular.ttf").string()));
            context = Rml::CreateContext("video_fps_input", {640, 900}, &renderer);
            ASSERT_NE(context, nullptr);
            document = context->LoadDocument((root / "src/visualizer/gui/rmlui/resources/video_extractor.rml").string());
            ASSERT_NE(document, nullptr);
            dialog = std::make_unique<lfs::gui::VideoExtractorDialog>();
            ASSERT_TRUE(Access::attach(*dialog, document));
            input = dynamic_cast<Rml::ElementFormControlInput*>(document->GetElementById("fps-value"));
            slider = dynamic_cast<Rml::ElementFormControlInput*>(document->GetElementById("fps-slider"));
            ASSERT_NE(input, nullptr);
            ASSERT_NE(slider, nullptr);
            document->Show();
            context->Update();
        }
        void TearDown() override {
            // Keep listeners alive until the document has been destroyed.
            if (context)
                Rml::RemoveContext("video_fps_input");
            if (dialog)
                dialog->shutdown();
            dialog.reset();
            Rml::Shutdown();
        }
        void enter(const std::string& value) {
            input->SetValue(value);
            input->DispatchEvent(Rml::EventId::Change, {});
            Access::sync(*dialog);
            context->Update();
        }
        FpsTestRenderInterface renderer;
        Rml::Context* context = nullptr;
        Rml::ElementDocument* document = nullptr;
        Rml::ElementFormControlInput* input = nullptr;
        Rml::ElementFormControlInput* slider = nullptr;
        std::unique_ptr<lfs::gui::VideoExtractorDialog> dialog;
    };

    TEST_F(VideoExtractorFpsInputTest, TypedValueAndSliderStaySynchronized) {
        EXPECT_EQ(input->GetValue(), "1");
        enter("2");
        EXPECT_NEAR(Access::fps(*dialog), 2.0f, 1e-5f);
        EXPECT_EQ(input->GetValue(), "2");
        EXPECT_NEAR(std::stof(slider->GetValue()), 2.0f, 1e-5f);
        enter("2.3");
        EXPECT_NEAR(Access::fps(*dialog), 2.3f, 1e-5f);
        EXPECT_EQ(input->GetValue(), "2.3");
        slider->SetValue("4.0");
        slider->DispatchEvent(Rml::EventId::Change, {});
        Access::sync(*dialog);
        EXPECT_EQ(input->GetValue(), "4");
    }

    TEST_F(VideoExtractorFpsInputTest, ClampKeepsFallbackRangeAndTypedPrecision) {
        for (const auto& [text, expected] : std::vector<std::pair<std::string, float>>{
                 {"0", 0.1f},
                 {"-2", 0.1f},
                 {"100", 30.0f},
                 {"0.1", 0.1f},
                 {"30", 30.0f},
                 {"2.36", 2.36f}}) {
            SCOPED_TRACE(text);
            enter(text);
            EXPECT_NEAR(Access::fps(*dialog), expected, 1e-5f);
            EXPECT_EQ(input->GetValue(), std::format("{}", expected));
            EXPECT_NEAR(std::stof(slider->GetValue()), expected, 0.051f);
        }
    }

    TEST_F(VideoExtractorFpsInputTest, InvalidTextPreservesTheLastValidValue) {
        enter("2");
        for (const auto& text : {"", "-", ".", "abc", "2fps", "2,5", "nan", "inf", "1e999"}) {
            SCOPED_TRACE(text);
            enter(text);
            EXPECT_NEAR(Access::fps(*dialog), 2.0f, 1e-5f);
            EXPECT_EQ(input->GetValue(), "2");
        }
    }

    TEST_F(VideoExtractorFpsInputTest, KeyboardEditingSurvivesRefreshAndCommitsOnBlur) {
        TempDir temp("fps_keyboard_video");
        const auto source = temp.path / "source.mp4";
        ASSERT_TRUE(writeProbedVideoWithRate(source, "mp4", AV_CODEC_ID_MPEG4, 4, 0, 30, 1));
        ASSERT_TRUE(dialog->openVideoPath(source));
        Access::sync(*dialog);
        context->Update();
        ASSERT_TRUE(input->Focus());
        input->SetSelectionRange(0, static_cast<int>(input->GetValue().size()));
        context->ProcessTextInput("2.");
        Access::sync(*dialog);
        EXPECT_EQ(input->GetValue(), "2.");
        context->ProcessTextInput("3");
        Access::sync(*dialog);
        EXPECT_EQ(input->GetValue(), "2.3");
        EXPECT_NEAR(Access::fps(*dialog), 2.3f, 1e-5f);
        input->SetSelectionRange(0, static_cast<int>(input->GetValue().size()));
        context->ProcessTextInput("100");
        Access::sync(*dialog);
        EXPECT_EQ(input->GetValue(), "100");
        ASSERT_TRUE(slider->Focus());
        Access::sync(*dialog);
        EXPECT_EQ(input->GetValue(), "30");
        EXPECT_NEAR(Access::fps(*dialog), 30.0f, 1e-5f);
    }

    TEST_F(VideoExtractorFpsInputTest, ModeSwitchPreservesTypedFpsAndInterval) {
        enter("2");
        auto* mode = dynamic_cast<Rml::ElementFormControlSelect*>(document->GetElementById("mode-select"));
        ASSERT_NE(mode, nullptr);
        mode->SetSelection(1);
        Access::sync(*dialog);
        EXPECT_EQ(document->GetElementById("fps-row")->GetProperty<Rml::Style::Display>("display"), Rml::Style::Display::None);
        mode->SetSelection(0);
        Access::sync(*dialog);
        EXPECT_NEAR(Access::fps(*dialog), 2.0f, 1e-5f);
        EXPECT_EQ(input->GetValue(), "2");
        EXPECT_EQ(document->GetElementById("interval-input")->GetAttribute<Rml::String>("value", ""), "1");
    }

    TEST_F(VideoExtractorFpsInputTest, MaximumTracksTheLoadedVideoAndClampsOnSourceChange) {
        TempDir temp("source_fps_limit");
        for (const auto [numerator, denominator] : {std::pair{30000, 1001}, std::pair{60, 1}, std::pair{24, 1}}) {
            const float rate = static_cast<float>(numerator) / denominator;
            const auto source = temp.path / (std::to_string(numerator) + ".mp4");
            ASSERT_TRUE(writeProbedVideoWithRate(source, "mp4", AV_CODEC_ID_MPEG4, 4, 0, numerator, denominator));
            ASSERT_TRUE(dialog->openVideoPath(source));
            Access::sync(*dialog);
            EXPECT_LE(Access::fps(*dialog), rate + 1e-5f);
            EXPECT_NEAR(std::stof(slider->GetAttribute<Rml::String>("max", "")), rate, 0.001f);
            enter("120");
            EXPECT_NEAR(Access::fps(*dialog), rate, 0.001f);
            enter("29.97");
            EXPECT_NEAR(Access::fps(*dialog), std::min(29.97f, rate), 1e-5f);
        }
    }

    TEST_F(VideoExtractorFpsInputTest, FractionalSourceLimitUsesTheSliderDisplayWithoutChangingTheRate) {
        TempDir temp("fps_fractional_limit");
        const auto source = temp.path / "source.mp4";
        ASSERT_TRUE(writeProbedVideoWithRate(source, "mp4", AV_CODEC_ID_MPEG4, 4, 0, 1801, 60));
        ASSERT_TRUE(dialog->openVideoPath(source));
        Access::sync(*dialog);
        const float source_rate = std::stof(slider->GetAttribute<Rml::String>("max", ""));
        ASSERT_GT(source_rate, 30.0f);
        ASSERT_LT(source_rate, 30.05f);
        enter("120");
        EXPECT_FLOAT_EQ(std::stof(input->GetValue()), std::stof(slider->GetValue()));
        EXPECT_EQ(input->GetValue(), "30");
        EXPECT_FLOAT_EQ(Access::fps(*dialog), source_rate);
        Access::sync(*dialog);
        EXPECT_EQ(input->GetValue(), "30");
        enter("2.36");
        EXPECT_EQ(input->GetValue(), "2.36");
        EXPECT_FLOAT_EQ(Access::fps(*dialog), 2.36f);
    }

    TEST_F(VideoExtractorFpsInputTest, ControlsRequireALoadedVideo) {
        EXPECT_TRUE(input->HasAttribute("disabled"));
        EXPECT_TRUE(slider->HasAttribute("disabled"));
        TempDir temp("fps_controls_video");
        const auto source = temp.path / "source.mp4";
        ASSERT_TRUE(writeProbedVideoWithRate(source, "mp4", AV_CODEC_ID_MPEG4, 4, 0, 16, 1));
        ASSERT_TRUE(dialog->openVideoPath(source));
        Access::sync(*dialog);
        EXPECT_FALSE(input->HasAttribute("disabled"));
        EXPECT_FALSE(slider->HasAttribute("disabled"));
        EXPECT_FLOAT_EQ(std::stof(slider->GetAttribute<Rml::String>("max", "")), 16.0f);
    }

    TEST_F(VideoExtractorFpsInputTest, SliderEventsNormalizeDecimalsWithoutRoundingTypedValues) {
        TempDir temp("fps_slider_precision");
        const auto source = temp.path / "source.mp4";
        ASSERT_TRUE(writeProbedVideoWithRate(source, "mp4", AV_CODEC_ID_MPEG4, 4, 0, 30000, 1001));
        ASSERT_TRUE(dialog->openVideoPath(source));
        Access::sync(*dialog);
        for (const float value : {2.3000002f, 4.6999998f, 15.100001f}) {
            Rml::Dictionary parameters;
            parameters["value"] = value;
            slider->DispatchEvent(Rml::EventId::Change, parameters);
            Access::sync(*dialog);
            EXPECT_EQ(input->GetValue(), std::format("{:.1f}", value));
        }
        Rml::Dictionary parameters;
        parameters["value"] = 30.0f;
        slider->DispatchEvent(Rml::EventId::Change, parameters);
        Access::sync(*dialog);
        EXPECT_FLOAT_EQ(Access::fps(*dialog), static_cast<float>(30000.0 / 1001.0));
        EXPECT_FLOAT_EQ(std::stof(input->GetValue()), std::stof(slider->GetValue()));
        enter("2.36");
        EXPECT_EQ(input->GetValue(), "2.36");
        EXPECT_FLOAT_EQ(Access::fps(*dialog), 2.36f);
    }

    TEST_F(VideoExtractorFpsInputTest, ExtractionRequestUsesTypedFps) {
        TempDir temp("typed_fps_request");
        const auto video_path = temp.path / "source.mp4";
        const auto output_dir = temp.path / "frames";
        std::filesystem::create_directories(output_dir);
        std::ofstream(output_dir / "frame_1.png").put('x');
        ASSERT_TRUE(writeProbedVideo(video_path, "mp4", AV_CODEC_ID_MPEG4, 50, 0));
        ASSERT_TRUE(dialog->openVideoPath(video_path));
        enter("2");
        const auto request = Access::request(*dialog, output_dir, false, false);
        EXPECT_DOUBLE_EQ(request.fps, 2.0);
        EXPECT_EQ(request.mode, ExtractionMode::FPS);
        EXPECT_TRUE(std::filesystem::exists(output_dir / "frame_1.png"));
    }
} // namespace

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
    bool writeProbedVideoWithRate(const std::filesystem::path& path, const char* const container,
                                  const AVCodecID codec_id, const int frame_count, const int leading_frames,
                                  const int rate_numerator, const int rate_denominator) {
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
            encoder->time_base = AVRational{rate_denominator, rate_numerator > 0 ? rate_numerator : 25};
            encoder->framerate = AVRational{rate_numerator > 0 ? rate_numerator : 25, rate_denominator};
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
                if (rate_numerator > 0 && packet->duration == 0)
                    packet->duration = 1;
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

namespace {
    bool writeProbedVideo(const std::filesystem::path& path, const char* container,
                          AVCodecID codec_id, int frame_count, int leading_frames) {
        return writeProbedVideoWithRate(path, container, codec_id, frame_count, leading_frames, 0, 1);
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
