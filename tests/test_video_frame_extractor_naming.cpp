/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/image_io.hpp"
#include "gui/windows/video_extractor_dialog.hpp"
#include "io/video/frame_color_range.hpp"
#include "io/video/probed_stream_rewind.hpp"
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
#include <libavutil/dict.h>
#include <libavutil/display.h>
#include <libavutil/error.h>
#include <libswscale/swscale.h>
}

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/stat.h>
#endif

#include "core/event_bridge/localization_manager.hpp"
#include "core/image_io.hpp"

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

#if defined(__linux__)
    uint64_t processReadChars() {
        std::ifstream io("/proc/self/io");
        std::string key;
        uint64_t value = std::numeric_limits<uint64_t>::max();
        while (io >> key >> value) {
            if (key == "rchar:")
                return value;
        }
        return std::numeric_limits<uint64_t>::max();
    }
#endif

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

    uint64_t hashPixels(const unsigned char* pixels, const std::size_t size) {
        uint64_t hash = 14695981039346656037ull;
        for (std::size_t index = 0; index < size; ++index)
            hash = (hash ^ pixels[index]) * 1099511628211ull;
        return hash;
    }

    nlohmann::json readMetadata(const std::filesystem::path& output_dir) {
        std::ifstream file(output_dir / "extraction_metadata.json");
        return nlohmann::json::parse(file);
    }

} // namespace

class VideoColorRange : public ::testing::TestWithParam<const char*> {};

TEST_P(VideoColorRange, PreviewAndExtractionPreserveLuminance) {
    if (!cudaAvailable())
        GTEST_SKIP() << "Requires NVDEC";
    const std::string name = GetParam();
    const auto path = std::filesystem::path(PROJECT_ROOT_PATH) / "tests/data/video" / name;
    constexpr int width = 256;
    constexpr int height = 256;
    const auto check = [&](const uint8_t* pixels, int channels, int tolerance) {
        ASSERT_NE(pixels, nullptr);
        int max_error = 0;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const int expected = x;
                for (int c = 0; c < 3; ++c)
                    max_error = std::max(max_error, std::abs(pixels[(y * width + x) * channels + c] - expected));
            }
        }
        EXPECT_LE(max_error, tolerance);
    };
    lfs::io::VideoPlayer player;
    ASSERT_TRUE(player.open(path)) << player.takeError();
    ASSERT_EQ(player.width(), width);
    ASSERT_EQ(player.height(), height);
    check(player.currentFrameData(), player.currentFrameChannels(), 2);
    player.close();

    TempDir temp("color_range");
    for (const auto format : {lfs::io::ImageFormat::PNG, lfs::io::ImageFormat::JPG}) {
        SCOPED_TRACE(format == lfs::io::ImageFormat::PNG ? "PNG" : "JPEG");
        const auto output = temp.path / (format == lfs::io::ImageFormat::PNG ? "png" : "jpg");
        std::filesystem::create_directories(output);
        auto params = extractionParams(path, output);
        params.end_time = -1.0;
        params.format = format;
        params.jpg_quality = 100;
        VideoFrameExtractor extractor;
        std::string error;
        ASSERT_TRUE(extractor.extract(params, error)) << error;
        const auto first = output / (format == lfs::io::ImageFormat::PNG ? "frame_1.png" : "frame_1.jpg");
        auto [pixels, image_width, image_height, channels] = lfs::core::load_image(first);
        ASSERT_NE(pixels, nullptr);
        EXPECT_EQ(image_width, width);
        EXPECT_EQ(image_height, height);
        if (image_width == width && image_height == height)
            check(pixels, channels, 2);
        lfs::core::free_image(pixels);
    }
}

INSTANTIATE_TEST_SUITE_P(Codecs, VideoColorRange,
                         ::testing::Values("vp9_full.webm", "av1_full.mkv", "vp9_limited.webm",
                                           "av1_limited.mkv", "h264_full.mp4", "hevc_full.mp4"));

TEST(VideoColorRangeMetadata, OnlyRepairsFullRangeCuvidVp9AndAv1) {
    for (const char* name : {"vp9_cuvid", "av1_cuvid", "h264_cuvid", "hevc_cuvid", "vp9", "libdav1d"}) {
        SCOPED_TRACE(name);
        AVCodec codec{};
        codec.name = name;
        AVCodecContext decoder{};
        decoder.codec = &codec;
        for (const auto stream_range : {AVCOL_RANGE_UNSPECIFIED, AVCOL_RANGE_MPEG, AVCOL_RANGE_JPEG}) {
            for (const auto frame_range : {AVCOL_RANGE_UNSPECIFIED, AVCOL_RANGE_MPEG, AVCOL_RANGE_JPEG}) {
                AVCodecParameters stream{};
                stream.color_range = stream_range;
                AVFrame frame{};
                frame.color_range = frame_range;
                frame.colorspace = AVCOL_SPC_BT709;
                frame.color_trc = AVCOL_TRC_BT709;
                frame.pts = 123;
                lfs::io::video::restoreCuvidFullRange(&frame, &decoder, &stream);
                const bool repair = (std::string_view(name) == "vp9_cuvid" ||
                                     std::string_view(name) == "av1_cuvid") &&
                                    stream_range == AVCOL_RANGE_JPEG && frame_range == AVCOL_RANGE_MPEG;
                EXPECT_EQ(frame.color_range, repair ? AVCOL_RANGE_JPEG : frame_range);
                EXPECT_EQ(frame.colorspace, AVCOL_SPC_BT709);
                EXPECT_EQ(frame.color_trc, AVCOL_TRC_BT709);
                EXPECT_EQ(frame.pts, 123);
            }
        }
    }
}

namespace lfs::gui {
    class VideoExtractorDialogTestAccess {
    public:
        static const std::filesystem::path& videoPath(const VideoExtractorDialog& dialog) { return dialog.video_path_; }
        static const std::filesystem::path& outputPath(const VideoExtractorDialog& dialog) { return dialog.output_dir_; }
        static void outputPath(VideoExtractorDialog& dialog, const std::filesystem::path& path) { dialog.output_dir_ = path; }
        static bool naming(VideoExtractorDialog& dialog, int selection, const std::string& custom,
                           const std::filesystem::path& video, std::string& pattern, std::string& error) {
            dialog.naming_selection_ = selection;
            dialog.filename_pattern_ = custom;
            dialog.video_path_ = video;
            return dialog.resolveFilenamePattern(pattern, error);
        }
        static void firstSourceFrameNumber(VideoExtractorDialog& dialog, const int value) {
            dialog.first_source_frame_number_ = value;
        }
        static void configureBatch(VideoExtractorDialog& dialog,
                                   const std::vector<std::filesystem::path>& videos,
                                   const std::filesystem::path& output_root) {
            dialog.batch_items_.clear();
            for (const auto& video : videos)
                dialog.batch_items_.push_back({.video_path = video});
            dialog.output_dir_ = output_root;
            dialog.batch_output_root_explicit_ = !output_root.empty();
            dialog.selected_batch_item_.reset();
            dialog.refreshBatchOutputDirectories();
        }
        static void addBatch(VideoExtractorDialog& dialog,
                             const std::vector<std::filesystem::path>& videos) {
            (void)queueBatch(dialog, videos);
            ASSERT_TRUE(waitForBatchProbes(dialog));
        }
        static bool queueBatch(VideoExtractorDialog& dialog,
                               const std::vector<std::filesystem::path>& videos) {
            return dialog.addBatchVideos(videos);
        }
        static void addFromBrowse(VideoExtractorDialog& dialog,
                                  std::vector<std::filesystem::path> videos) {
            dialog.addVideosFromBrowse(std::move(videos));
            ASSERT_TRUE(waitForBatchProbes(dialog));
        }
        static bool waitForBatchProbes(VideoExtractorDialog& dialog) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            for (;;) {
                dialog.applyCompletedBatchProbes();
                if (!dialog.batch_probe_active_.load() && !dialog.batch_probe_results_ready_.load())
                    return true;
                if (std::chrono::steady_clock::now() >= deadline)
                    return false;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        static std::size_t batchItemCount(const VideoExtractorDialog& dialog) {
            return dialog.batch_items_.size();
        }
        static void batchProbeState(VideoExtractorDialog& dialog, const bool active, const bool results_ready) {
            dialog.batch_probe_active_.store(active);
            dialog.batch_probe_results_ready_.store(results_ready);
        }
        static bool hasDynamicState(const VideoExtractorDialog& dialog) { return dialog.hasDynamicState(); }
        static void outputRoot(VideoExtractorDialog& dialog, const std::filesystem::path& output_root) {
            dialog.output_dir_ = output_root;
            dialog.batch_output_root_explicit_ = true;
            dialog.refreshBatchOutputDirectories();
        }
        static void fps(VideoExtractorDialog& dialog, const float value) { dialog.fps_ = value; }
        static void sharedOutputSettings(VideoExtractorDialog& dialog, const int width, const int height,
                                         const bool hdr_to_sdr) {
            dialog.custom_width_ = width;
            dialog.custom_height_ = height;
            dialog.hdr_to_sdr_ = hdr_to_sdr;
            // Batch parameters re-read both sizes from their fields, as typing leaves them.
            for (auto [element, value] : {std::pair{dialog.custom_width_input_el_, width},
                                          std::pair{dialog.custom_height_input_el_, height}}) {
                if (auto* const input = dynamic_cast<Rml::ElementFormControlInput*>(element))
                    input->SetValue(std::to_string(value));
            }
        }
        static void selectBatch(VideoExtractorDialog& dialog, const std::size_t index) {
            dialog.selectBatchItem(index);
        }
        static int customWidth(const VideoExtractorDialog& dialog) { return dialog.custom_width_; }
        static int customHeight(const VideoExtractorDialog& dialog) { return dialog.custom_height_; }
        static bool hdrToSdr(const VideoExtractorDialog& dialog) { return dialog.hdr_to_sdr_; }
        static void batchItemSettings(VideoExtractorDialog& dialog, const std::size_t index,
                                      const double start_time, const double end_time,
                                      const int rotation, const double duration,
                                      const double source_fps) {
            auto& item = dialog.batch_items_.at(index);
            item.start_time = start_time;
            item.end_time = end_time;
            item.end_time_is_auto = false;
            item.rotation = rotation;
            item.duration = duration;
            item.source_fps = source_fps;
        }
        static bool buildBatch(VideoExtractorDialog& dialog,
                               std::vector<VideoExtractionParams>& params,
                               std::string& error) {
            return dialog.buildBatchParams(params, error);
        }
        static void batchProgress(VideoExtractorDialog& dialog, const int active_index,
                                  const int current, const int total) {
            dialog.extracting_.store(true);
            dialog.active_batch_index_.store(active_index);
            dialog.updateProgress(current, total);
        }
        static void syncBatchList(VideoExtractorDialog& dialog) { dialog.syncBatchList(); }
        static void syncExtractionStatus(VideoExtractorDialog& dialog) { dialog.syncExtractionStatus(); }
        static bool attach(VideoExtractorDialog& dialog, Rml::ElementDocument* document) {
            dialog.document_ = document;
            dialog.cacheElements();
            dialog.syncLocale();
            sync(dialog);
            return dialog.elements_cached_;
        }
        static void sync(VideoExtractorDialog& dialog) {
            dialog.syncControls();
            dialog.syncOutputPreview();
        }
        static void extracting(VideoExtractorDialog& dialog, bool active) { dialog.extracting_.store(active); }
        static bool begin(VideoExtractorDialog& dialog, const std::filesystem::path& output) {
            dialog.output_dir_ = output;
            dialog.pending_params_set_ = false;
            dialog.beginExtractionFromUi();
            return dialog.pending_params_set_ || dialog.extracting_.load();
        }
        static nlohmann::json sourceState(const VideoExtractorDialog& dialog) {
            const auto status = dialog.getExtractionStatusSnapshot();
            return {
                {"video", dialog.video_path_.string()},
                {"output", dialog.output_dir_.string()},
                {"start", dialog.trim_start_},
                {"end", dialog.trim_end_},
                {"auto_end", dialog.trim_end_is_auto_},
                {"duration", dialog.player_->duration()},
                {"player_open", dialog.player_->isOpen()},
                {"rotation", dialog.rotation_deg_},
                {"current", dialog.current_frame_.load()},
                {"total", dialog.total_frames_.load()},
                {"discarded", dialog.discarded_frames_.load()},
                {"error", status.error_message},
                {"status", static_cast<int>(status.status_message)},
            };
        }
        static void progress(VideoExtractorDialog& dialog) { dialog.updateProgress(17, 50, 2); }
        static void stop(VideoExtractorDialog& dialog) { dialog.requestStopExtraction(); }
        static bool stopping(const VideoExtractorDialog& dialog) { return dialog.stop_extraction_requested_.load(); }
        static bool active(const VideoExtractorDialog& dialog) { return dialog.extracting_.load(); }
        static void finish(VideoExtractorDialog& dialog, int outcome) {
            if (outcome == 0)
                dialog.setExtractionComplete();
            else if (outcome == 1)
                dialog.setExtractionStopped();
            else
                dialog.setExtractionError("Extraction failed");
        }
        static void cancelOverwrite(VideoExtractorDialog& dialog) { dialog.handleClick("overwrite-cancel"); }
        static bool pending(const VideoExtractorDialog& dialog) { return dialog.pending_params_set_; }
        static void deleteExisting(VideoExtractorDialog& dialog) { dialog.handleClick("overwrite-delete"); }
        static void proceedOverwrite(VideoExtractorDialog& dialog) { dialog.handleClick("overwrite-proceed"); }
        static bool waitIdle(const VideoExtractorDialog& dialog) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
            while (dialog.extracting_.load()) {
                if (std::chrono::steady_clock::now() > deadline)
                    return false;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            return true;
        }
        static double end(const VideoExtractorDialog& dialog) { return dialog.trim_end_; }

        static void timeline(VideoExtractorDialog& dialog) { dialog.syncTimeline(); }
        static void click(VideoExtractorDialog& dialog, const std::string& id) { dialog.handleClick(id); }
        static void dragTrim(VideoExtractorDialog& dialog, bool end, float fraction) {
            auto* const timeline = dialog.timeline_el_;
            const float x = timeline->GetAbsoluteLeft() + fraction * timeline->GetBox().GetSize(Rml::BoxArea::Border).x;
            dialog.setTrimFromTimeline(end ? VideoExtractorDialog::TimelineDragTarget::TrimEnd
                                           : VideoExtractorDialog::TimelineDragTarget::TrimStart,
                                       x);
        }

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
            const auto params = dialog.pending_params_;
            dialog.pending_params_set_ = false;
            return params;
        }
    };
} // namespace lfs::gui

namespace {
    bool writeProbedVideoWithRate(const std::filesystem::path& path, const char* container,
                                  AVCodecID codec_id, int frame_count, int leading_frames,
                                  int rate_numerator, int rate_denominator,
                                  bool variable_rate = false, int frame_pattern_step = 8,
                                  int rotation_degrees = 0);

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

    TEST_F(VideoExtractorFpsInputTest, UneditedTrimPreservesEverySourceFrame) {
        for (const int frames : {1, 2, 3, 182, 184, 185}) {
            SCOPED_TRACE(frames);
            TempDir temp("trim_precision");
            const auto source = temp.path / "source.mp4";
            const auto output = temp.path / "frames";
            std::filesystem::create_directories(output);
            std::ofstream(output / "frame_1.png").put('x');
            ASSERT_TRUE(writeProbedVideoWithRate(source, "mp4", AV_CODEC_ID_MPEG4, frames, 0, 30, 1));
            ASSERT_TRUE(dialog->openVideoPath(source));
            const double expected_end = Access::end(*dialog);
            for (const bool interval : {false, true}) {
                SCOPED_TRACE(interval);
                Access::reset(*dialog);
                Access::timeline(*dialog);
                // Focusing and leaving either field must not quantize its value.
                for (const auto* id : {"trim-start-input", "trim-end-input"})
                    document->GetElementById(id)->DispatchEvent(Rml::EventId::Blur, {});
                Access::timeline(*dialog);
                const auto request = Access::request(*dialog, output, interval, false);
                EXPECT_DOUBLE_EQ(request.start_time, 0.0);
                EXPECT_DOUBLE_EQ(Access::end(*dialog), expected_end);
                EXPECT_DOUBLE_EQ(request.end_time, interval ? -1.0 : expected_end);
            }
        }
    }

    TEST_F(VideoExtractorFpsInputTest, ExplicitFractionalTrimSurvivesBlurAndStart) {
        TempDir temp("explicit_trim_precision");
        const auto source = temp.path / "source.mp4";
        const auto output = temp.path / "frames";
        std::filesystem::create_directories(output);
        std::ofstream(output / "frame_1.png").put('x');
        ASSERT_TRUE(writeProbedVideoWithRate(source, "mp4", AV_CODEC_ID_MPEG4, 185, 0, 30, 1));
        ASSERT_TRUE(dialog->openVideoPath(source));
        for (const auto& [start, end] : std::vector<std::pair<float, float>>{{2.0f, 3.0f}, {2.0333333f, 3.0666667f}}) {
            Access::reset(*dialog);
            Access::timeline(*dialog);
            for (const auto& [id, value] : std::vector<std::pair<const char*, float>>{
                     {"trim-start-input", start},
                     {"trim-end-input", end}}) {
                auto* field = dynamic_cast<Rml::ElementFormControlInput*>(document->GetElementById(id));
                ASSERT_NE(field, nullptr);
                field->SetValue(std::format("{}", value));
                field->DispatchEvent(Rml::EventId::Change, {});
                Access::timeline(*dialog);
                field->DispatchEvent(Rml::EventId::Blur, {});
                Access::timeline(*dialog);
            }
            for (const bool interval : {false, true}) {
                const auto request = Access::request(*dialog, output, interval, false);
                EXPECT_DOUBLE_EQ(request.start_time, static_cast<double>(start));
                EXPECT_DOUBLE_EQ(request.end_time, static_cast<double>(end));
            }
        }
    }

    TEST_F(VideoExtractorFpsInputTest, ShortVideoTrimButtonsKeepAValidRange) {
        for (const int frames : {1, 2, 3}) {
            SCOPED_TRACE(frames);
            TempDir temp("short_trim_buttons");
            const auto source = temp.path / "source.mp4";
            const auto output = temp.path / "frames";
            std::filesystem::create_directories(output);
            std::ofstream(output / "frame_1.png").put('x');
            ASSERT_TRUE(writeProbedVideoWithRate(source, "mp4", AV_CODEC_ID_MPEG4, frames, 0, 30, 1));
            ASSERT_TRUE(dialog->openVideoPath(source));
            const double duration = Access::end(*dialog);
            for (const auto* id : {"btn-trim-start-set", "btn-trim-end-set", "btn-trim-reset"}) {
                Access::reset(*dialog);
                Access::click(*dialog, id);
                Access::timeline(*dialog);
                const auto request = Access::request(*dialog, output, false, false);
                EXPECT_DOUBLE_EQ(request.start_time, 0.0);
                EXPECT_DOUBLE_EQ(request.end_time, duration);
            }
        }
    }

    TEST_F(VideoExtractorFpsInputTest, ShortVideoTrimFieldsAndTimelineKeepAValidRange) {
        TempDir temp("short_trim_edits");
        const auto source = temp.path / "source.mp4";
        const auto output = temp.path / "frames";
        std::filesystem::create_directories(output);
        std::ofstream(output / "frame_1.png").put('x');
        ASSERT_TRUE(writeProbedVideoWithRate(source, "mp4", AV_CODEC_ID_MPEG4, 1, 0, 30, 1));
        ASSERT_TRUE(dialog->openVideoPath(source));
        const double duration = Access::end(*dialog);
        for (const bool end : {false, true}) {
            for (const bool timeline : {false, true}) {
                Access::reset(*dialog);
                Access::timeline(*dialog);
                context->Update();
                if (timeline) {
                    Access::dragTrim(*dialog, end, end ? 0.0f : 1.0f);
                } else {
                    auto* field = dynamic_cast<Rml::ElementFormControlInput*>(
                        document->GetElementById(end ? "trim-end-input" : "trim-start-input"));
                    ASSERT_NE(field, nullptr);
                    field->SetValue(end ? "-1" : "99");
                    field->DispatchEvent(Rml::EventId::Change, {});
                }
                Access::timeline(*dialog);
                const auto request = Access::request(*dialog, output, false, false);
                EXPECT_DOUBLE_EQ(request.start_time, 0.0);
                EXPECT_DOUBLE_EQ(request.end_time, duration);
            }
        }
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

// Fails when samples one keyframe apart seek again (6 seeks at 2 fps), when samples two keyframes apart stop seeking
// (1 seek at 0.5 fps), or when decoding on selects another frame than a seek would.
TEST(VideoFrameExtractorOutputNaming, SparseFpsSeeksOnlyAcrossWholeKeyframeIntervals) {
    if (!cudaAvailable())
        GTEST_SKIP() << "Requires NVDEC";

    TempDir temp("sparse_seek");
    const std::filesystem::path video_path = temp.path / "source.mp4";
    std::string error;
    // The encoder writes one keyframe per second.
    ASSERT_TRUE(writeEncodedVideo(video_path, 90, 30, error, 160, 90)) << error;

    for (const auto [fps, seeks, frames] : {std::tuple{2.0, 1, 6u}, std::tuple{0.5, 2, 2u}}) {
        SCOPED_TRACE(::testing::Message() << "fps=" << fps);
        const std::filesystem::path output_dir = temp.path / ("frames_" + std::to_string(seeks));
        std::filesystem::create_directories(output_dir);
        auto params = extractionParams(video_path, output_dir);
        params.mode = ExtractionMode::FPS;
        params.fps = fps;
        params.end_time = -1.0;

        VideoFrameExtractor extractor;
        ASSERT_TRUE(extractor.extract(params, error)) << error;
        const auto metadata = readMetadata(output_dir);
        const auto& selection = metadata.at("processing").at("frame_selection");
        ASSERT_EQ(selection.at("strategy"), "keyframe_seek");
        EXPECT_EQ(selection.at("keyframe_seek_count"), seeks);
        const auto& written = metadata.at("frames");
        ASSERT_EQ(written.size(), frames);
        for (std::size_t i = 0; i < written.size(); ++i)
            EXPECT_NEAR(written[i].at("timestamp").get<double>(), static_cast<double>(i) / fps, 1e-6);
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

namespace {
    class VideoFrameExtractorWriteTest : public ::testing::TestWithParam<std::tuple<lfs::io::ImageFormat, bool>> {
    protected:
        void SetUp() override {
            ASSERT_TRUE(writeProbedVideoWithRate(temp.path / "source.mp4", "mp4", AV_CODEC_ID_MPEG4, 3, 0, 25, 1));
            params = extractionParams(temp.path / "source.mp4", temp.path / "frames");
            params.end_time = -1.0;
            params.format = std::get<0>(GetParam());
            params.sharpness.enabled = std::get<1>(GetParam());
            params.sharpness.window_mode = params.sharpness.enabled;
            params.sharpness.window_candidates_target = 0;
            std::filesystem::create_directories(params.output_dir);
        }

        std::filesystem::path framePath(int frame) const {
            return params.output_dir / ("frame_" + std::to_string(frame) +
                                        (params.format == lfs::io::ImageFormat::PNG ? ".png" : ".jpg"));
        }

        void expectFailure(const std::filesystem::path& failed_path) {
            std::string error;
            EXPECT_FALSE(extractor.extract(params, error));
            EXPECT_EQ(extractor.lastOutcome(), lfs::io::ExtractionOutcome::Failed);
            EXPECT_NE(error.find(failed_path.filename().string()), std::string::npos) << error;
        }

        TempDir temp{"write"};
        VideoFrameExtractor::Params params;
        VideoFrameExtractor extractor;
    };

    TEST_P(VideoFrameExtractorWriteTest, UnwritableFrameFailsAndCanRetry) {
        const auto blocked = framePath(1);
        std::filesystem::create_directory(blocked);
        expectFailure(blocked);
        EXPECT_FALSE(std::filesystem::exists(params.output_dir / "extraction_metadata.json"));

        std::filesystem::remove(blocked);
        std::string error;
        ASSERT_TRUE(extractor.extract(params, error)) << error;
        EXPECT_EQ(extractor.lastOutcome(), lfs::io::ExtractionOutcome::Completed);
        EXPECT_TRUE(error.empty());
        const auto metadata = readMetadata(params.output_dir);
        ASSERT_EQ(metadata["frames"].size(), 3u);
        for (int frame = 1; frame <= 3; ++frame) {
            EXPECT_GT(std::filesystem::file_size(framePath(frame)), 0u);
            EXPECT_EQ(metadata["frames"][frame - 1]["source_frame"], frame);
        }
    }

    TEST_P(VideoFrameExtractorWriteTest, LaterWriteFailurePreservesSavedFrames) {
        std::filesystem::create_directory(framePath(2));
        expectFailure(framePath(2));
        EXPECT_GT(std::filesystem::file_size(framePath(1)), 0u);
        EXPECT_FALSE(std::filesystem::exists(framePath(3)));
    }

    TEST_P(VideoFrameExtractorWriteTest, WritableOutputKeepsFramesAndMetadataConsistent) {
        std::string error;
        ASSERT_TRUE(extractor.extract(params, error)) << error;
        EXPECT_EQ(extractor.lastOutcome(), lfs::io::ExtractionOutcome::Completed);
        const auto metadata = readMetadata(params.output_dir);
        ASSERT_EQ(metadata["frames"].size(), 3u);
        std::vector<std::string> reference;
        for (int frame = 1; frame <= 3; ++frame) {
            EXPECT_EQ(metadata["frames"][frame - 1]["file"], framePath(frame).filename().string());
            EXPECT_EQ(metadata["frames"][frame - 1]["source_frame"], frame);
            std::ifstream file(framePath(frame), std::ios::binary);
            reference.emplace_back(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
            EXPECT_FALSE(reference.back().empty());
        }
        params.generate_metadata = false;
        ASSERT_TRUE(extractor.extract(params, error)) << error;
        for (int frame = 1; frame <= 3; ++frame) {
            std::ifstream file(framePath(frame), std::ios::binary);
            const std::string bytes{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
            EXPECT_EQ(bytes, reference[frame - 1]);
        }
    }

    TEST_P(VideoFrameExtractorWriteTest, RequestedMetadataWriteFailureIsReported) {
        const auto metadata = params.output_dir / "extraction_metadata.json";
        std::filesystem::create_directory(metadata);
        expectFailure(metadata);
        for (int frame = 1; frame <= 3; ++frame)
            EXPECT_GT(std::filesystem::file_size(framePath(frame)), 0u);

        params.generate_metadata = false;
        std::string error;
        EXPECT_TRUE(extractor.extract(params, error)) << error;
        EXPECT_EQ(extractor.lastOutcome(), lfs::io::ExtractionOutcome::Completed);
    }

#ifndef _WIN32
    TEST_P(VideoFrameExtractorWriteTest, FullDeviceWriteFails) {
        if (!std::filesystem::exists("/dev/full"))
            GTEST_SKIP() << "A full device is required";
        std::filesystem::create_symlink("/dev/full", framePath(1));
        expectFailure(framePath(1));
    }

    TEST_P(VideoFrameExtractorWriteTest, FullDeviceMetadataWriteFails) {
        if (!std::filesystem::exists("/dev/full"))
            GTEST_SKIP() << "A full device is required";
        const auto metadata = params.output_dir / "extraction_metadata.json";
        std::filesystem::create_symlink("/dev/full", metadata);
        expectFailure(metadata);
    }

    TEST_P(VideoFrameExtractorWriteTest, ReadOnlyOutputDirectoryFails) {
        std::filesystem::permissions(params.output_dir, std::filesystem::perms::owner_read |
                                                            std::filesystem::perms::owner_exec);
        struct RestorePermissions {
            std::filesystem::path path;
            ~RestorePermissions() {
                std::error_code ec;
                std::filesystem::permissions(path, std::filesystem::perms::owner_all, ec);
            }
        } restore{params.output_dir};
        std::ofstream probe(params.output_dir / "probe");
        if (probe)
            GTEST_SKIP() << "Process can bypass directory permissions";
        expectFailure(framePath(1));
        EXPECT_TRUE(std::filesystem::is_empty(params.output_dir));
    }
#endif

    INSTANTIATE_TEST_SUITE_P(FormatsAndSelection, VideoFrameExtractorWriteTest,
                             ::testing::Combine(::testing::Values(lfs::io::ImageFormat::PNG, lfs::io::ImageFormat::JPG),
                                                ::testing::Bool()));
} // namespace

namespace {
    using NamingAccess = lfs::gui::VideoExtractorDialogTestAccess;
    TEST(VideoExtractorNaming, PresetsAndCustomPatternsPreserveLegacyNumbering) {
        lfs::gui::VideoExtractorDialog dialog;
        const std::array expected{"clip.test_frame_00001", "clip.test_00001", "frame_00001", "00001"};
        for (int preset = 0; preset < 4; ++preset) {
            std::string pattern, error;
            ASSERT_TRUE(NamingAccess::naming(dialog, preset, "unused", "clip.test.mp4", pattern, error));
            EXPECT_EQ(lfs::io::formatFrameFilenameStem(pattern, 1), expected[preset]);
            EXPECT_NE(lfs::io::formatFrameFilenameStem(pattern, 13).find("00013"), std::string::npos);
        }
        for (const auto& [custom, result] : std::vector<std::pair<std::string, std::string>>{
                 {"vacanza", "vacanza7"},
                 {"%%", "%7"},
                 {"100%%_%05d", "100%_00007"},
                 {"frame_%0000d", "frame_00007"},
                 {"", "frame_7"},
                 {"%d_%06d", "7_000007"},
                 {"{{video}}_%d", "{video}_7"},
                 {"{{{video}}}_%d", "{video}_7"}}) {
            std::string pattern, error;
            ASSERT_TRUE(NamingAccess::naming(dialog, 4, custom, "video.mp4", pattern, error));
            EXPECT_EQ(lfs::io::formatFrameFilenameStem(pattern, 7), result);
        }
    }
    TEST(VideoExtractorNaming, SourceNameIsLiteralAndUnicodeWorks) {
        lfs::gui::VideoExtractorDialog dialog;
        std::string pattern, error;
        ASSERT_TRUE(NamingAccess::naming(dialog, 4, "{video}_{video}_%06d", "100%05d{unknown}.mp4", pattern, error));
        EXPECT_EQ(lfs::io::formatFrameFilenameStem(pattern, 7), "100%05d{unknown}_100%05d{unknown}_000007");
        ASSERT_TRUE(NamingAccess::naming(dialog, 0, "", std::filesystem::u8path("caf\xc3\xa9 \xe6\x97\xa5\xe6\x9c\xac.test.mp4"), pattern, error));
        EXPECT_EQ(lfs::io::formatFrameFilenameStem(pattern, 7), "caf\xc3\xa9 \xe6\x97\xa5\xe6\x9c\xac.test_frame_00007");
    }
    TEST(VideoExtractorNaming, RejectsInvalidPatternsWithoutTruncatingValidLongInput) {
        lfs::gui::VideoExtractorDialog dialog;
        for (const auto& custom : {std::string("{unknown}_%d"), std::string("{video_%d"),
                                   std::string("../frame_%d"), std::string("bad\0name_%d", 11)}) {
            std::string pattern, error;
            EXPECT_FALSE(NamingAccess::naming(dialog, 4, custom, "video.mp4", pattern, error)) << custom;
            EXPECT_FALSE(error.empty());
        }
        std::string pattern, error;
        for (const auto& custom : {"C:\\frame_%d", "CON.%d", "CONIN$.%d", "COM\xc2\xb9.%d"}) {
#ifdef _WIN32
            EXPECT_FALSE(NamingAccess::naming(dialog, 4, custom, "video.mp4", pattern, error));
#else
            EXPECT_TRUE(NamingAccess::naming(dialog, 4, custom, "video.mp4", pattern, error));
#endif
        }
        const std::string custom = std::string(250, 'a') + "_%05d";
        ASSERT_TRUE(NamingAccess::naming(dialog, 4, custom, "video.mp4", pattern, error));
        EXPECT_EQ(pattern, custom);
    }
    TEST(VideoExtractorNaming, StemEndingInDotOrSpaceProducesValidFilenames) {
        TempDir temp("naming_stem_suffix");
        const auto source = temp.path / "clip.mp4";
        ASSERT_TRUE(writeProbedVideo(source, "mp4", AV_CODEC_ID_MPEG4, 5, 0));
        lfs::gui::VideoExtractorDialog dialog;
        for (const auto* custom : {"x_%d.", "x_%d "}) {
            VideoFrameExtractor::Params params;
            params.video_path = source;
            params.output_dir = temp.path / (custom[4] == '.' ? "dot" : "space");
            params.mode = ExtractionMode::INTERVAL;
            params.frame_interval = 1;
            std::string error;
            ASSERT_TRUE(NamingAccess::naming(dialog, 4, custom, source, params.filename_pattern, error));
            VideoFrameExtractor extractor;
            ASSERT_TRUE(extractor.extract(params, error)) << error;
            EXPECT_TRUE(std::filesystem::exists(params.output_dir / (lfs::io::formatFrameFilenameStem(params.filename_pattern, 1) + ".png")));
        }
    }

    TEST(VideoExtractorNaming, GeneratedFrameNamesFollowThePattern) {
        using lfs::io::isGeneratedFrameFilename;
        for (const auto* name : {"frame_00001.png", "frame_00012.jpg", "frame_1.png", "frame_1234567.png"})
            EXPECT_TRUE(isGeneratedFrameFilename("frame_%05d", name)) << name;
        for (const auto* name : {"frame_00000.png", "frame_00001.PNG", "frame_00001.jpeg", "frame_final_edit.png",
                                 "frame_.png", "xframe_00001.png", "frame_00001x.png", "frame_00001.png.bak",
                                 "IMG_1234.JPG", "holiday.png", "_DSC8679.JPG", "my_garden.jpeg",
                                 "extraction_metadata.json"})
            EXPECT_FALSE(isGeneratedFrameFilename("frame_%05d", name)) << name;
        EXPECT_TRUE(isGeneratedFrameFilename("frame_%05d", "frame_00000.png", true));
        EXPECT_TRUE(isGeneratedFrameFilename("clip 50%%_{x}_frame_%05d", "clip 50%_{x}_frame_00003.png"));
        EXPECT_FALSE(isGeneratedFrameFilename("clip 50%%_{x}_frame_%05d", "clip 50%%_{x}_frame_00003.png"));
        EXPECT_TRUE(isGeneratedFrameFilename("shot_", "shot_7.png"));
        EXPECT_FALSE(isGeneratedFrameFilename("shot_", "shot_.png"));
        EXPECT_TRUE(isGeneratedFrameFilename("img2%03d", "img2001.png"));
        EXPECT_FALSE(isGeneratedFrameFilename("img2%03d", "img2.png"));
        for (const std::string pattern : {"frame_%05d", "%05d", "a%db%d", "x_%%_%0003", "{y}_%d_end", "plain"}) {
            for (const int number : {1, 7, 99999, 123456, std::numeric_limits<int>::max()}) {
                EXPECT_TRUE(isGeneratedFrameFilename(pattern, lfs::io::formatFrameFilenameStem(pattern, number) + ".png"))
                    << pattern << ' ' << number;
            }
        }
    }

    TEST(VideoFrameExtractorOutputNaming, FirstSourceFrameNumberOffsetsOutputNames) {
        TempDir temp("first_source_frame_number");
        const auto source = temp.path / "clip.mp4";
        ASSERT_TRUE(writeProbedVideo(source, "mp4", AV_CODEC_ID_MPEG4, 65, 0));
        auto params = extractionParams(source, temp.path / "frames");
        params.frame_interval = 30;
        params.end_time = -1.0;
        params.first_source_frame_number = 0;

        VideoFrameExtractor extractor;
        std::string error;
        ASSERT_TRUE(extractor.extract(params, error)) << error;
        EXPECT_TRUE(std::filesystem::exists(params.output_dir / "frame_0.png"));
        EXPECT_TRUE(std::filesystem::exists(params.output_dir / "frame_30.png"));
        EXPECT_TRUE(std::filesystem::exists(params.output_dir / "frame_60.png"));
        EXPECT_FALSE(std::filesystem::exists(params.output_dir / "frame_1.png"));

        std::ifstream metadata_file(params.output_dir / "extraction_metadata.json");
        ASSERT_TRUE(metadata_file.good());
        EXPECT_EQ(nlohmann::json::parse(metadata_file).at("first_source_frame_number"), 0);
    }

    TEST(VideoExtractorNaming, ExistingZeroBasedOutputIsDetectedAcrossNumberBases) {
        TempDir temp("zero_based_existing_output");
        const auto source = temp.path / "clip.mp4";
        const auto output = temp.path / "frames";
        ASSERT_TRUE(writeProbedVideo(source, "mp4", AV_CODEC_ID_MPEG4, 5, 0));
        std::filesystem::create_directories(output);
        std::ofstream(output / "frame_00000.png") << "earlier zero-based frame";

        lfs::gui::VideoExtractorDialog dialog;
        ASSERT_TRUE(dialog.openVideoPath(source));
        // The default base is one, but zero-based output from an earlier run still conflicts.
        ASSERT_TRUE(NamingAccess::begin(dialog, output));
        ASSERT_TRUE(NamingAccess::pending(dialog));
        NamingAccess::deleteExisting(dialog);
        ASSERT_TRUE(NamingAccess::waitIdle(dialog));
        EXPECT_FALSE(std::filesystem::exists(output / "frame_00000.png"));
        EXPECT_TRUE(std::filesystem::exists(output / "frame_00001.png"));
    }

    TEST(VideoExtractorNaming, ExistingImagesInTheOutputFolderAreKept) {
        TempDir temp("naming_keep_existing_images");
        const auto source = temp.path / "clip.mp4";
        ASSERT_TRUE(writeProbedVideo(source, "mp4", AV_CODEC_ID_MPEG4, 5, 0));
        const auto output = temp.path / "frames";
        std::filesystem::create_directories(output);
        const std::vector<std::string> existing{"IMG_1234.JPG", "holiday.png", "frame_final_edit.png", "notes.txt"};
        for (const auto& name : existing)
            std::ofstream(output / name) << "existing";
        lfs::gui::VideoExtractorDialog dialog;
        ASSERT_TRUE(dialog.openVideoPath(source));

        // Without earlier frames there is nothing to replace: the run starts without asking.
        ASSERT_TRUE(NamingAccess::begin(dialog, output));
        EXPECT_FALSE(NamingAccess::pending(dialog));
        ASSERT_TRUE(NamingAccess::waitIdle(dialog));
        EXPECT_TRUE(std::filesystem::exists(output / "frame_00001.png"));
        for (const auto& name : existing)
            EXPECT_TRUE(std::filesystem::exists(output / name)) << name;

        // Existing generated output asks first. Proceed keeps unrelated generated output.
        std::ofstream(output / "frame_00001.png") << "old frame";
        std::ofstream(output / "frame_00099.jpg") << "earlier";
        std::ofstream(output / "extraction_metadata.json") << "{}";
        ASSERT_TRUE(NamingAccess::begin(dialog, output));
        ASSERT_TRUE(NamingAccess::pending(dialog));
        NamingAccess::proceedOverwrite(dialog);
        ASSERT_TRUE(NamingAccess::waitIdle(dialog));
        EXPECT_GT(std::filesystem::file_size(output / "frame_00001.png"), 100u);
        EXPECT_TRUE(std::filesystem::exists(output / "frame_00099.jpg"));
        EXPECT_TRUE(std::filesystem::exists(output / "extraction_metadata.json"));
        for (const auto& name : existing)
            EXPECT_TRUE(std::filesystem::exists(output / name)) << name;

        // Deleting existing output leaves metadata alone when this run does not write metadata.
        ASSERT_TRUE(NamingAccess::begin(dialog, output));
        ASSERT_TRUE(NamingAccess::pending(dialog));
        NamingAccess::deleteExisting(dialog);
        ASSERT_TRUE(NamingAccess::waitIdle(dialog));
        EXPECT_FALSE(std::filesystem::exists(output / "frame_00099.jpg"));
        EXPECT_TRUE(std::filesystem::exists(output / "extraction_metadata.json"));
        EXPECT_TRUE(std::filesystem::exists(output / "frame_00001.png"));
        for (const auto& name : existing)
            EXPECT_TRUE(std::filesystem::exists(output / name)) << name;
    }

    TEST(VideoExtractorNaming, ZeroBasedExistingOutputAsksBeforeExtraction) {
        TempDir temp("zero_based_existing_output");
        const auto source = temp.path / "clip.mp4";
        ASSERT_TRUE(writeProbedVideo(source, "mp4", AV_CODEC_ID_MPEG4, 5, 0));
        const auto output = temp.path / "frames";
        std::filesystem::create_directories(output);
        std::ofstream(output / "frame_00000.png") << "earlier";

        lfs::gui::VideoExtractorDialog dialog;
        ASSERT_TRUE(dialog.openVideoPath(source));
        NamingAccess::firstSourceFrameNumber(dialog, 0);
        ASSERT_TRUE(NamingAccess::begin(dialog, output));
        EXPECT_TRUE(NamingAccess::pending(dialog));
        NamingAccess::cancelOverwrite(dialog);
        EXPECT_TRUE(std::filesystem::exists(output / "frame_00000.png"));
    }

    TEST(VideoExtractorNaming, ResolvedPresetProducesRealFilesAndMetadata) {
        TempDir temp("naming_real_files");
        const auto source = temp.path / "100%05d.mp4";
        ASSERT_TRUE(writeProbedVideo(source, "mp4", AV_CODEC_ID_MPEG4, 50, 0));
        lfs::gui::VideoExtractorDialog dialog;
        VideoFrameExtractor::Params params;
        params.video_path = source;
        params.output_dir = temp.path / "frames";
        params.mode = ExtractionMode::INTERVAL;
        params.frame_interval = 5;
        params.end_time = 0.5;
        params.generate_metadata = true;
        std::string error;
        ASSERT_TRUE(NamingAccess::naming(dialog, 0, "", source, params.filename_pattern, error));
        VideoFrameExtractor extractor;
        ASSERT_TRUE(extractor.extract(params, error)) << error;
        EXPECT_TRUE(std::filesystem::exists(params.output_dir / "100%05d_frame_00001.png"));
        EXPECT_TRUE(std::filesystem::exists(params.output_dir / "100%05d_frame_00006.png"));
        EXPECT_FALSE(std::filesystem::exists(params.output_dir / "100%05d_frame_00002.png"));
        std::ifstream metadata_file(params.output_dir / "extraction_metadata.json");
        ASSERT_TRUE(metadata_file.good());
        const auto metadata = nlohmann::json::parse(metadata_file);
        EXPECT_EQ(metadata.at("filename_pattern").get<std::string>(), params.filename_pattern);
    }

    class NamingTestRenderInterface final : public Rml::RenderInterface {
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

    class VideoExtractorNamingUiTest : public ::testing::Test {
    protected:
        void SetUp() override {
            ASSERT_TRUE(Rml::Initialise());
            const auto root = std::filesystem::path(PROJECT_ROOT_PATH);
            ASSERT_TRUE(lfs::event::LocalizationManager::getInstance().initialize((root / "src/visualizer/gui/resources/locales").string()));
            ASSERT_TRUE(Rml::LoadFontFace((root / "src/visualizer/gui/assets/fonts/Inter-Regular.ttf").string()));
            context = Rml::CreateContext("video_naming", {640, 900}, &renderer);
            ASSERT_NE(context, nullptr);
            document = context->LoadDocument((root / "src/visualizer/gui/rmlui/resources/video_extractor.rml").string());
            ASSERT_NE(document, nullptr);
            dialog = std::make_unique<lfs::gui::VideoExtractorDialog>();
            ASSERT_TRUE(NamingAccess::attach(*dialog, document));
            select = dynamic_cast<Rml::ElementFormControlSelect*>(document->GetElementById("naming-select"));
            input = dynamic_cast<Rml::ElementFormControlInput*>(document->GetElementById("pattern-input"));
            ASSERT_NE(select, nullptr);
            ASSERT_NE(input, nullptr);
            document->Show();
            context->Update();
        }
        void TearDown() override {
            if (context)
                Rml::RemoveContext("video_naming");
            if (dialog)
                dialog->shutdown();
            dialog.reset();
            Rml::Shutdown();
        }
        void sync() {
            NamingAccess::sync(*dialog);
            context->Update();
        }
        void choose(int preset) {
            select->SetSelection(preset);
            sync();
        }
        void enter(const std::string& value) {
            input->SetValue(value);
            input->DispatchEvent(Rml::EventId::Change, {});
            sync();
        }
        NamingTestRenderInterface renderer;
        Rml::Context* context = nullptr;
        Rml::ElementDocument* document = nullptr;
        Rml::ElementFormControlSelect* select = nullptr;
        Rml::ElementFormControlInput* input = nullptr;
        std::unique_ptr<lfs::gui::VideoExtractorDialog> dialog;
    };
    TEST_F(VideoExtractorNamingUiTest, VisibilityCustomMemoryAndValidation) {
        EXPECT_EQ(select->GetSelection(), 2);
        auto* row = document->GetElementById("custom-pattern-row");
        EXPECT_EQ(row->GetProperty<Rml::Style::Display>("display"), Rml::Style::Display::None);
        choose(4);
        EXPECT_NE(row->GetProperty<Rml::Style::Display>("display"), Rml::Style::Display::None);
        enter("camera_%06d");
        choose(0);
        choose(4);
        EXPECT_EQ(input->GetValue(), "camera_%06d");
        enter("{unknown}");
        EXPECT_NE(document->GetElementById("naming-error")->GetProperty<Rml::Style::Display>("display"), Rml::Style::Display::None);
        EXPECT_TRUE(document->GetElementById("btn-start")->HasAttribute("disabled"));
        choose(2);
        EXPECT_EQ(document->GetElementById("naming-error")->GetProperty<Rml::Style::Display>("display"), Rml::Style::Display::None);
        NamingAccess::extracting(*dialog, true);
        sync();
        EXPECT_TRUE(select->HasAttribute("disabled"));
        EXPECT_TRUE(input->HasAttribute("disabled"));
        NamingAccess::extracting(*dialog, false);
    }
    TEST_F(VideoExtractorNamingUiTest, KeyboardEditingAndBlurPreserveCustomPattern) {
        choose(4);
        ASSERT_TRUE(input->Focus());
        input->SetSelectionRange(0, static_cast<int>(input->GetValue().size()));
        context->ProcessTextInput("{{label}}_{video}_%06d");
        sync();
        EXPECT_EQ(input->GetValue(), "{{label}}_{video}_%06d");
        ASSERT_TRUE(select->Focus());
        sync();
        choose(2);
        choose(4);
        EXPECT_EQ(input->GetValue(), "{{label}}_{video}_%06d");
    }

    TEST_F(VideoExtractorNamingUiTest, MetadataOnlyPromptsOnlyWhenItWillBeReplaced) {
        TempDir temp("metadata_only_conflict");
        const auto source = temp.path / "source.mp4";
        ASSERT_TRUE(writeProbedVideo(source, "mp4", AV_CODEC_ID_MPEG4, 5, 0));
        ASSERT_TRUE(dialog->openVideoPath(source));
        auto* const metadata_toggle = document->GetElementById("generate-metadata");
        ASSERT_NE(metadata_toggle, nullptr);

        const auto retained_metadata = temp.path / "retained_metadata";
        std::filesystem::create_directories(retained_metadata);
        std::ofstream(retained_metadata / "extraction_metadata.json") << "{}";
        ASSERT_TRUE(NamingAccess::begin(*dialog, retained_metadata));
        EXPECT_FALSE(NamingAccess::pending(*dialog));
        ASSERT_TRUE(NamingAccess::waitIdle(*dialog));
        EXPECT_TRUE(std::filesystem::exists(retained_metadata / "extraction_metadata.json"));

        metadata_toggle->SetAttribute("checked", "checked");
        const auto replaced_metadata = temp.path / "replaced_metadata";
        std::filesystem::create_directories(replaced_metadata);
        std::ofstream(replaced_metadata / "extraction_metadata.json") << "{}";
        ASSERT_TRUE(NamingAccess::begin(*dialog, replaced_metadata));
        EXPECT_TRUE(NamingAccess::pending(*dialog));
        EXPECT_EQ(document->GetElementById("overwrite-message")->GetInnerRML(),
                  LOC("video_extractor.overwrite_prompt_metadata"));
        NamingAccess::cancelOverwrite(*dialog);
    }

    TEST_F(VideoExtractorNamingUiTest, SourceChangesRefreshPreviewAndInvalidPatternBlocksRealStart) {
        TempDir temp("naming_source_change");
        const auto first = temp.path / "first.mp4";
        const auto second = temp.path / "second.mp4";
        ASSERT_TRUE(writeProbedVideo(first, "mp4", AV_CODEC_ID_MPEG4, 50, 0));
        ASSERT_TRUE(writeProbedVideo(second, "mp4", AV_CODEC_ID_MPEG4, 50, 0));
        choose(4);
        enter("{video}_custom_%05d");
        ASSERT_TRUE(dialog->openVideoPath(first));
        sync();
        EXPECT_NE(document->GetElementById("pattern-example")->GetInnerRML().find("first_custom_00001.png"), std::string::npos);
        ASSERT_TRUE(dialog->openVideoPath(second));
        sync();
        EXPECT_EQ(input->GetValue(), "{video}_custom_%05d");
        EXPECT_NE(document->GetElementById("pattern-example")->GetInnerRML().find("second_custom_00001.png"), std::string::npos);
        const auto output = temp.path / "frames";
        std::filesystem::create_directories(output);
        enter("{invalid}_%d");
        EXPECT_FALSE(NamingAccess::begin(*dialog, output));
        sync();
        EXPECT_TRUE(document->GetElementById("btn-start")->HasAttribute("disabled"));
        EXPECT_TRUE(std::filesystem::is_empty(output));
        enter("{video}_%05d");
        EXPECT_FALSE(document->GetElementById("btn-start")->HasAttribute("disabled"));
    }

    TEST_F(VideoExtractorNamingUiTest, SourceReplacementDuringExtractionPreservesJobIdentity) {
        TempDir temp("busy_source");
        const auto first = temp.path / "first.mp4";
        const auto second = temp.path / "second.mp4";
        ASSERT_TRUE(writeProbedVideo(first, "mp4", AV_CODEC_ID_MPEG4, 50, 0));
        ASSERT_TRUE(writeProbedVideo(second, "mp4", AV_CODEC_ID_MPEG4, 20, 0));
        ASSERT_TRUE(dialog->openVideoPath(first));
        NamingAccess::extracting(*dialog, true);
        NamingAccess::progress(*dialog);
        sync();
        const auto expected = NamingAccess::sourceState(*dialog);
        for (const auto& path : {second, first, temp.path / "missing.mp4"}) {
            EXPECT_FALSE(dialog->openVideoPath(path));
            EXPECT_EQ(NamingAccess::sourceState(*dialog), expected);
            EXPECT_TRUE(NamingAccess::active(*dialog));
        }
        NamingAccess::finish(*dialog, 0);
        EXPECT_EQ(NamingAccess::sourceState(*dialog)["video"], first.string());
        EXPECT_EQ(NamingAccess::sourceState(*dialog)["output"], (temp.path / "first_frames").string());
    }

    TEST_F(VideoExtractorNamingUiTest, StopRequestKeepsSourceUntilWorkerFinishes) {
        TempDir temp("stopping_source");
        const auto source = temp.path / "source.mp4";
        ASSERT_TRUE(writeProbedVideo(source, "mp4", AV_CODEC_ID_MPEG4, 50, 0));
        ASSERT_TRUE(dialog->openVideoPath(source));
        NamingAccess::extracting(*dialog, true);
        NamingAccess::stop(*dialog);
        const auto expected = NamingAccess::sourceState(*dialog);
        EXPECT_FALSE(dialog->openVideoPath(temp.path / "missing.mp4"));
        EXPECT_TRUE(NamingAccess::active(*dialog));
        EXPECT_TRUE(NamingAccess::stopping(*dialog));
        EXPECT_EQ(NamingAccess::sourceState(*dialog), expected);
        NamingAccess::finish(*dialog, 1);
        EXPECT_TRUE(dialog->openVideoPath(source));
    }

    TEST_F(VideoExtractorNamingUiTest, PendingOverwriteKeepsSourceUntilCancelled) {
        TempDir temp("pending_source");
        const auto first = temp.path / "first.mp4";
        const auto second = temp.path / "second.mp4";
        ASSERT_TRUE(writeProbedVideo(first, "mp4", AV_CODEC_ID_MPEG4, 50, 0));
        ASSERT_TRUE(writeProbedVideo(second, "mp4", AV_CODEC_ID_MPEG4, 20, 0));
        ASSERT_TRUE(dialog->openVideoPath(first));
        const auto output = temp.path / "custom_output";
        std::filesystem::create_directories(output);
        std::ofstream(output / "frame_00001.png").put('x');
        ASSERT_TRUE(NamingAccess::begin(*dialog, output));
        ASSERT_TRUE(NamingAccess::pending(*dialog));
        const auto expected = NamingAccess::sourceState(*dialog);
        EXPECT_FALSE(dialog->openVideoPath(second));
        EXPECT_EQ(NamingAccess::sourceState(*dialog), expected);
        EXPECT_EQ(std::filesystem::file_size(output / "frame_00001.png"), 1);
        NamingAccess::cancelOverwrite(*dialog);
        EXPECT_TRUE(dialog->openVideoPath(second));
        EXPECT_EQ(NamingAccess::sourceState(*dialog)["output"], (temp.path / "second_frames").string());
    }

    TEST_F(VideoExtractorNamingUiTest, SourceCanChangeAfterEveryTerminalOutcome) {
        TempDir temp("finished_source");
        const auto first = temp.path / "first.mp4";
        const auto second = temp.path / "second.mp4";
        ASSERT_TRUE(writeProbedVideo(first, "mp4", AV_CODEC_ID_MPEG4, 50, 0));
        ASSERT_TRUE(writeProbedVideo(second, "mp4", AV_CODEC_ID_MPEG4, 20, 0));
        for (int outcome = 0; outcome < 3; ++outcome) {
            ASSERT_TRUE(dialog->openVideoPath(first));
            NamingAccess::extracting(*dialog, true);
            NamingAccess::finish(*dialog, outcome);
            ASSERT_TRUE(dialog->openVideoPath(second));
            const auto state = NamingAccess::sourceState(*dialog);
            EXPECT_EQ(state["video"], second.string());
            EXPECT_EQ(state["output"], (temp.path / "second_frames").string());
            EXPECT_EQ(state["error"], "");
            EXPECT_EQ(state["status"], 0);
            EXPECT_TRUE(state["player_open"].get<bool>());
        }
    }

    TEST_F(VideoExtractorNamingUiTest, RepeatedRefreshPreservesActiveSource) {
        TempDir temp("refresh_source");
        const auto source = temp.path / "source.mp4";
        ASSERT_TRUE(writeProbedVideo(source, "mp4", AV_CODEC_ID_MPEG4, 50, 0));
        ASSERT_TRUE(dialog->openVideoPath(source));
        NamingAccess::extracting(*dialog, true);
        NamingAccess::progress(*dialog);
        sync();
        const auto expected = NamingAccess::sourceState(*dialog);
        const auto start = std::chrono::steady_clock::now();
        constexpr int updates = 1000;
        for (int i = 0; i < updates; ++i)
            sync();
        const auto elapsed = std::chrono::steady_clock::now() - start;
        RecordProperty("refresh_ns", std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count() / updates));
        EXPECT_EQ(NamingAccess::sourceState(*dialog), expected);
        EXPECT_TRUE(document->GetElementById("btn-start")->HasAttribute("disabled"));
        EXPECT_TRUE(document->GetElementById("btn-browse-video")->HasAttribute("disabled"));
        EXPECT_TRUE(document->GetElementById("btn-browse-output")->HasAttribute("disabled"));
        NamingAccess::extracting(*dialog, false);
        sync();
        EXPECT_FALSE(document->GetElementById("btn-browse-video")->HasAttribute("disabled"));
        EXPECT_FALSE(document->GetElementById("btn-browse-output")->HasAttribute("disabled"));
    }

    TEST_F(VideoExtractorNamingUiTest, RealRequestMatchesPreviewAndFormat) {
        TempDir temp("naming_gui_request");
        const auto source = temp.path / "100%05d.mp4";
        ASSERT_TRUE(writeProbedVideo(source, "mp4", AV_CODEC_ID_MPEG4, 50, 0));
        ASSERT_TRUE(dialog->openVideoPath(source));
        const auto output = temp.path / "output";
        std::filesystem::create_directories(output);
        std::ofstream(output / "existing.png").put('x');
        // An earlier frame with this naming captures the pending request.
        std::ofstream(output / "100%05d_frame_00042.png").put('x');
        choose(0);
        const auto request = NamingAccess::request(*dialog, output, false, false);
        EXPECT_EQ(lfs::io::formatFrameFilenameStem(request.filename_pattern, 1), "100%05d_frame_00001");
        EXPECT_NE(document->GetElementById("pattern-example")->GetInnerRML().find("100%05d_frame_00001"), std::string::npos);
        auto* format = dynamic_cast<Rml::ElementFormControlSelect*>(document->GetElementById("format-select"));
        ASSERT_NE(format, nullptr);
        format->SetSelection(1);
        sync();
        EXPECT_NE(document->GetElementById("pattern-example")->GetInnerRML().find(".jpg"), std::string::npos);
        EXPECT_EQ(std::filesystem::file_size(output / "existing.png"), 1);
    }

    TEST_F(VideoExtractorNamingUiTest, BatchParamsUseSeparateDirectoriesAndPerItemSettings) {
        const auto root = std::filesystem::path("batch_output");
        NamingAccess::configureBatch(*dialog,
                                     {"capture_a/clip.mp4", "capture_b/clip.mp4"},
                                     root);
        NamingAccess::batchItemSettings(*dialog, 0, 1.25, 4.5, 90, 12.0, 24.0);
        NamingAccess::batchItemSettings(*dialog, 1, 0.5, 3.0, 270, 8.0, 30.0);

        std::vector<lfs::gui::VideoExtractionParams> params;
        std::string error;
        ASSERT_TRUE(NamingAccess::buildBatch(*dialog, params, error)) << error;
        ASSERT_EQ(params.size(), 2u);
        EXPECT_EQ(params[0].output_dir, root / "clip");
        EXPECT_EQ(params[1].output_dir, root / "clip_2");
        EXPECT_DOUBLE_EQ(params[0].start_time, 1.25);
        EXPECT_DOUBLE_EQ(params[0].end_time, 4.5);
        EXPECT_EQ(params[0].rotation, 90);
        EXPECT_DOUBLE_EQ(params[1].start_time, 0.5);
        EXPECT_DOUBLE_EQ(params[1].end_time, 3.0);
        EXPECT_EQ(params[1].rotation, 270);
    }

    TEST_F(VideoExtractorNamingUiTest, BatchProbePreservesUnselectedVideoRotationAndFps) {
        TempDir temp("batch_unselected_rotation");
        const auto first = temp.path / "first.mp4";
        const auto portrait = temp.path / "portrait.mp4";
        ASSERT_TRUE(writeProbedVideoWithRate(first, "mp4", AV_CODEC_ID_MPEG4, 6, 0, 60, 1));
        ASSERT_TRUE(writeProbedVideoWithRate(portrait, "mp4", AV_CODEC_ID_MPEG4, 6, 0, 24, 1,
                                             false, 8, 90));

        NamingAccess::addBatch(*dialog, {first, portrait});
        NamingAccess::outputRoot(*dialog, temp.path / "output");
        NamingAccess::fps(*dialog, 30.0f);

        std::vector<lfs::gui::VideoExtractionParams> params;
        std::string error;
        ASSERT_TRUE(NamingAccess::buildBatch(*dialog, params, error)) << error;
        ASSERT_EQ(params.size(), 2u);
        EXPECT_EQ(params[1].rotation, 90);
        EXPECT_DOUBLE_EQ(params[1].fps, 24.0);
        EXPECT_GT(params[1].end_time, 0.0);
    }

    TEST_F(VideoExtractorNamingUiTest, BatchProbesPublishOnlyAfterTheWorkerCompletes) {
        TempDir temp("batch_probe_async_publish");
        const auto first = temp.path / "first.mp4";
        const auto second = temp.path / "second.mp4";
        ASSERT_TRUE(writeProbedVideoWithRate(first, "mp4", AV_CODEC_ID_MPEG4, 4, 0, 30, 1));
        ASSERT_TRUE(writeProbedVideoWithRate(second, "mp4", AV_CODEC_ID_MPEG4, 4, 0, 30, 1));

        ASSERT_TRUE(NamingAccess::queueBatch(*dialog, {first, second}));
        EXPECT_EQ(NamingAccess::batchItemCount(*dialog), 0u);
        ASSERT_TRUE(NamingAccess::waitForBatchProbes(*dialog));
        EXPECT_EQ(NamingAccess::batchItemCount(*dialog), 2u);
    }

    TEST_F(VideoExtractorNamingUiTest, BatchProbeDoesNotRequireContinuousRedraw) {
        NamingAccess::batchProbeState(*dialog, true, true);
        EXPECT_FALSE(NamingAccess::hasDynamicState(*dialog));
    }

    TEST_F(VideoExtractorNamingUiTest, BatchAddReportsTheNumberOfUnreadableVideos) {
        TempDir temp("batch_unreadable_videos");

        NamingAccess::addBatch(*dialog, {temp.path / "missing_a.mp4", temp.path / "missing_b.mp4"});

        EXPECT_EQ(NamingAccess::sourceState(*dialog)["error"].get<std::string>(),
                  "2 videos could not be opened; see log for details.");
    }

    TEST_F(VideoExtractorNamingUiTest, SelectingBatchItemPreservesSharedOutputSettings) {
        TempDir temp("batch_shared_settings");
        const auto first = temp.path / "first.mp4";
        const auto second = temp.path / "second.mp4";
        ASSERT_TRUE(writeProbedVideoWithRate(first, "mp4", AV_CODEC_ID_MPEG4, 4, 0, 30, 1));
        ASSERT_TRUE(writeProbedVideoWithRate(second, "mp4", AV_CODEC_ID_MPEG4, 4, 0, 24, 1));

        NamingAccess::addBatch(*dialog, {first, second});
        NamingAccess::sharedOutputSettings(*dialog, 800, 600, true);
        NamingAccess::selectBatch(*dialog, 1);

        EXPECT_EQ(NamingAccess::customWidth(*dialog), 800);
        EXPECT_EQ(NamingAccess::customHeight(*dialog), 600);
        EXPECT_TRUE(NamingAccess::hdrToSdr(*dialog));
    }

    TEST_F(VideoExtractorNamingUiTest, DroppedVideoIsAddedWithoutChangingTheBatchOutputRoot) {
        TempDir temp("batch_drop");
        const auto first = temp.path / "first.mp4";
        const auto dropped = temp.path / "dropped.mp4";
        const auto output_root = temp.path / "output";
        ASSERT_TRUE(writeProbedVideoWithRate(first, "mp4", AV_CODEC_ID_MPEG4, 4, 0, 30, 1));
        ASSERT_TRUE(writeProbedVideoWithRate(dropped, "mp4", AV_CODEC_ID_MPEG4, 4, 0, 30, 1));

        NamingAccess::addBatch(*dialog, {first});
        NamingAccess::outputRoot(*dialog, output_root);
        ASSERT_TRUE(dialog->openVideoPath(dropped));
        ASSERT_TRUE(NamingAccess::waitForBatchProbes(*dialog));
        EXPECT_EQ(NamingAccess::outputPath(*dialog), output_root);

        std::vector<lfs::gui::VideoExtractionParams> params;
        std::string error;
        ASSERT_TRUE(NamingAccess::buildBatch(*dialog, params, error)) << error;
        ASSERT_EQ(params.size(), 2u);
        EXPECT_EQ(params[0].output_dir, output_root / "first");
        EXPECT_EQ(params[1].output_dir, output_root / "dropped");
    }

    TEST_F(VideoExtractorNamingUiTest, SecondBrowseSelectionPromotesTheExistingVideoToABatch) {
        TempDir temp("batch_second_browse");
        const auto first = temp.path / "clip.mp4";
        const auto second = temp.path / "other" / "clip.mp4";
        const auto output_root = temp.path / "output";
        std::filesystem::create_directories(second.parent_path());
        ASSERT_TRUE(writeProbedVideoWithRate(first, "mp4", AV_CODEC_ID_MPEG4, 4, 0, 30, 1));
        ASSERT_TRUE(writeProbedVideoWithRate(second, "mp4", AV_CODEC_ID_MPEG4, 4, 0, 30, 1));

        const auto first_with_dot_segment = first.parent_path() / "." / first.filename();
        ASSERT_TRUE(dialog->openVideoPath(first_with_dot_segment));
        NamingAccess::sharedOutputSettings(*dialog, 800, 600, false);
        NamingAccess::addFromBrowse(*dialog, {second});
        NamingAccess::outputRoot(*dialog, output_root);

        std::vector<lfs::gui::VideoExtractionParams> params;
        std::string error;
        ASSERT_TRUE(NamingAccess::buildBatch(*dialog, params, error)) << error;
        ASSERT_EQ(params.size(), 2u);
        EXPECT_EQ(params[0].output_dir, output_root / "clip");
        EXPECT_EQ(params[1].output_dir, output_root / "clip_2");
        EXPECT_EQ(params[0].custom_width, 800);
        EXPECT_EQ(params[1].custom_height, 600);
    }

    TEST_F(VideoExtractorNamingUiTest, BatchRequiresAnExplicitOutputRoot) {
        NamingAccess::configureBatch(*dialog, {"capture/clip.mp4"}, {});
        std::vector<lfs::gui::VideoExtractionParams> params;
        std::string error;
        EXPECT_FALSE(NamingAccess::buildBatch(*dialog, params, error));
        EXPECT_EQ(error, LOC("video_extractor.batch_no_output_root"));
    }

    TEST_F(VideoExtractorNamingUiTest, BatchRowsDisambiguateDuplicateNamesAndExposePaths) {
        NamingAccess::configureBatch(*dialog,
                                     {"capture_a/clip.mp4", "capture_b/clip.mp4"},
                                     "batch_output");
        NamingAccess::syncBatchList(*dialog);
        const auto rows = document->GetElementById("batch-video-list")->GetInnerRML();
        EXPECT_NE(rows.find("clip.mp4 (1)"), std::string::npos);
        EXPECT_NE(rows.find("clip.mp4 (2)"), std::string::npos);
        const auto* const first = document->GetElementById("batch-video-0");
        ASSERT_NE(first, nullptr);
        const auto tooltip = first->GetAttribute<Rml::String>("title", "");
        EXPECT_NE(tooltip.find("capture_a/clip.mp4"), std::string::npos);
        EXPECT_NE(tooltip.find("batch_output/clip"), std::string::npos);
    }

    TEST_F(VideoExtractorNamingUiTest, BatchProgressIsGlobalAcrossVideos) {
        NamingAccess::configureBatch(*dialog,
                                     {"capture/a.mp4", "capture/b.mp4", "capture/c.mp4"},
                                     "batch_output");
        NamingAccess::batchProgress(*dialog, 1, 25, 100);
        NamingAccess::syncExtractionStatus(*dialog);

        const auto* const progress = document->GetElementById("progress-bar");
        ASSERT_NE(progress, nullptr);
        EXPECT_NEAR(std::stof(progress->GetAttribute<Rml::String>("value", "")), 1.25f / 3.0f, 1.0e-4f);
        EXPECT_NE(document->GetElementById("progress-text")->GetInnerRML().find("2/3"), std::string::npos);
    }
} // namespace

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
    int completed = -1;
    params.progress_callback = [&](int current, int, int discarded) {
        completed = current - discarded;
    };

    VideoFrameExtractor extractor;
    ASSERT_TRUE(extractor.extract(params, error)) << error;
    EXPECT_TRUE(std::filesystem::exists(output_dir / "frame_1.png"));
    EXPECT_TRUE(std::filesystem::exists(output_dir / "frame_2.png"));
    EXPECT_TRUE(std::filesystem::exists(output_dir / "frame_3.png"));
    EXPECT_TRUE(std::filesystem::exists(output_dir / "frame_4.png"));
    EXPECT_TRUE(std::filesystem::exists(output_dir / "frame_5.png"));
    EXPECT_EQ(5u, countPngFiles(output_dir));
    EXPECT_EQ(5, completed);

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
                                  const int rate_numerator, const int rate_denominator,
                                  const bool variable_rate, const int frame_pattern_step,
                                  const int rotation_degrees) {
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
            if (ok && rotation_degrees != 0) {
                AVPacketSideData* const display_matrix = av_packet_side_data_new(
                    &stream->codecpar->coded_side_data, &stream->codecpar->nb_coded_side_data,
                    AV_PKT_DATA_DISPLAYMATRIX, 9 * sizeof(int32_t), 0);
                ok = display_matrix != nullptr;
                if (ok)
                    av_display_rotation_set(reinterpret_cast<int32_t*>(display_matrix->data), rotation_degrees);
            }
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
                    frame->data[0][y * frame->linesize[0] + x] =
                        static_cast<uint8_t>((x + y + index * frame_pattern_step) & 255);
            for (int plane = 1; plane < 3; ++plane)
                for (int y = 0; y < frame->height / 2; ++y)
                    for (int x = 0; x < frame->width / 2; ++x)
                        frame->data[plane][y * frame->linesize[plane] + x] = 128;
            frame->pts = (variable_rate && index >= 90 ? 90 + (index - 90) * 2 : index) - leading_frames;
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

TEST(VideoFrameExtractorOutputNaming, VariableRateUsesNominalTimestampNumbers) {
    for (const bool variable_rate : {false, true}) {
        for (const auto* container : {"mp4", "matroska"}) {
            SCOPED_TRACE(container);
            SCOPED_TRACE(variable_rate);
            TempDir temp("timestamp_numbers");
            const auto source = temp.path / "source.video";
            constexpr int frame_count = 185;
            ASSERT_TRUE(writeProbedVideoWithRate(source, container, AV_CODEC_ID_MPEG4,
                                                 frame_count, 0, 30, 1, variable_rate));
            for (const int interval : {1, 3}) {
                SCOPED_TRACE(interval);
                const auto output = temp.path / std::to_string(interval);
                auto params = extractionParams(source, output);
                params.end_time = -1.0;
                params.frame_interval = interval;
                int completed = -1;
                params.progress_callback = [&](int current, int, int discarded) {
                    completed = current - discarded;
                };
                VideoFrameExtractor extractor;
                std::string error;
                ASSERT_TRUE(extractor.extract(params, error)) << error;
                const int expected_count = (frame_count + interval - 1) / interval;
                EXPECT_EQ(countPngFiles(output), expected_count);
                EXPECT_EQ(completed, expected_count);
                const auto metadata = readMetadata(output);
                EXPECT_EQ(metadata["frames"].size(), expected_count);
                if (metadata["frames"].size() != expected_count)
                    continue;
                for (int index = 0; index < frame_count; index += interval) {
                    const int number = (variable_rate && index >= 90 ? 90 + (index - 90) * 2 : index) + 1;
                    const auto& frame = metadata["frames"][index / interval];
                    EXPECT_EQ(frame["source_frame"], number);
                    EXPECT_EQ(frame["file"], "frame_" + std::to_string(number) + ".png");
                    EXPECT_NEAR(frame["timestamp"].get<double>(), (number - 1) / 30.0, 0.001);
                    EXPECT_TRUE(std::filesystem::exists(output / frame["file"].get<std::string>()));
                }
            }
        }
    }
}

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

namespace {
    // Writes an MPEG-4 video of `video_frames` frames at 30 fps beside a silent PCM track of `audio_seconds`, so the
    // movie ends after its video, as camera recordings with a longer audio track do.
    bool writeVideoWithLongerAudio(const std::filesystem::path& path, const int video_frames,
                                   const double audio_seconds) {
        AVFormatContext* format = nullptr;
        const std::string path_utf8 = path.string();
        if (avformat_alloc_output_context2(&format, nullptr, "mov", path_utf8.c_str()) < 0 || !format)
            return false;
        const AVCodec* const video_codec = avcodec_find_encoder(AV_CODEC_ID_MPEG4);
        const AVCodec* const audio_codec = avcodec_find_encoder(AV_CODEC_ID_PCM_S16LE);
        AVStream* const video = video_codec ? avformat_new_stream(format, nullptr) : nullptr;
        AVStream* const audio = audio_codec ? avformat_new_stream(format, nullptr) : nullptr;
        AVCodecContext* video_encoder = video ? avcodec_alloc_context3(video_codec) : nullptr;
        AVCodecContext* audio_encoder = audio ? avcodec_alloc_context3(audio_codec) : nullptr;
        bool ok = video_encoder && audio_encoder;
        if (ok) {
            video_encoder->width = 64;
            video_encoder->height = 48;
            video_encoder->pix_fmt = AV_PIX_FMT_YUV420P;
            video_encoder->time_base = AVRational{1, 30};
            video_encoder->framerate = AVRational{30, 1};
            video_encoder->gop_size = 10;
            video_encoder->max_b_frames = 0;
            audio_encoder->sample_fmt = AV_SAMPLE_FMT_S16;
            audio_encoder->sample_rate = 8000;
            audio_encoder->time_base = AVRational{1, 8000};
            av_channel_layout_default(&audio_encoder->ch_layout, 1);
            if (format->oformat->flags & AVFMT_GLOBALHEADER) {
                video_encoder->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
                audio_encoder->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
            }
            ok = avcodec_open2(video_encoder, video_codec, nullptr) >= 0 &&
                 avcodec_open2(audio_encoder, audio_codec, nullptr) >= 0 &&
                 avcodec_parameters_from_context(video->codecpar, video_encoder) >= 0 &&
                 avcodec_parameters_from_context(audio->codecpar, audio_encoder) >= 0;
        }
        if (ok) {
            video->time_base = video_encoder->time_base;
            audio->time_base = audio_encoder->time_base;
            ok = avio_open(&format->pb, path_utf8.c_str(), AVIO_FLAG_WRITE) >= 0 &&
                 avformat_write_header(format, nullptr) >= 0;
        }
        AVFrame* frame = av_frame_alloc();
        AVPacket* packet = av_packet_alloc();
        const auto drain = [&](AVCodecContext* const encoder, AVStream* const stream) {
            while (avcodec_receive_packet(encoder, packet) == 0) {
                if (encoder == video_encoder && packet->duration == 0)
                    packet->duration = 1;
                av_packet_rescale_ts(packet, encoder->time_base, stream->time_base);
                packet->stream_index = stream->index;
                ok = av_interleaved_write_frame(format, packet) >= 0 && ok;
            }
        };
        if (ok) {
            frame->format = video_encoder->pix_fmt;
            frame->width = video_encoder->width;
            frame->height = video_encoder->height;
            ok = av_frame_get_buffer(frame, 0) >= 0;
        }
        for (int index = 0; ok && index < video_frames; ++index) {
            ok = av_frame_make_writable(frame) >= 0;
            for (int y = 0; y < frame->height; ++y)
                for (int x = 0; x < frame->width; ++x)
                    frame->data[0][y * frame->linesize[0] + x] = static_cast<uint8_t>((x + y + 3 * index) & 255);
            for (int plane = 1; plane < 3; ++plane)
                for (int y = 0; y < frame->height / 2; ++y)
                    std::fill_n(frame->data[plane] + y * frame->linesize[plane], frame->width / 2, uint8_t{128});
            frame->pts = index;
            ok = ok && avcodec_send_frame(video_encoder, frame) >= 0;
            drain(video_encoder, video);
        }
        if (ok) {
            avcodec_send_frame(video_encoder, nullptr);
            drain(video_encoder, video);
        }
        constexpr int kSamplesPerFrame = 800;
        av_frame_unref(frame);
        if (ok) {
            frame->format = audio_encoder->sample_fmt;
            frame->sample_rate = audio_encoder->sample_rate;
            frame->nb_samples = kSamplesPerFrame;
            ok = av_channel_layout_copy(&frame->ch_layout, &audio_encoder->ch_layout) >= 0 &&
                 av_frame_get_buffer(frame, 0) >= 0;
        }
        const int audio_samples = static_cast<int>(audio_seconds * audio_encoder->sample_rate);
        for (int first = 0; ok && first < audio_samples; first += kSamplesPerFrame) {
            ok = av_frame_make_writable(frame) >= 0;
            std::fill_n(frame->data[0], kSamplesPerFrame * sizeof(int16_t), uint8_t{0});
            frame->pts = first;
            ok = ok && avcodec_send_frame(audio_encoder, frame) >= 0;
            drain(audio_encoder, audio);
        }
        if (ok) {
            avcodec_send_frame(audio_encoder, nullptr);
            drain(audio_encoder, audio);
            ok = av_write_trailer(format) >= 0 && ok;
        }
        av_packet_free(&packet);
        av_frame_free(&frame);
        avcodec_free_context(&audio_encoder);
        avcodec_free_context(&video_encoder);
        if (format->pb)
            avio_closep(&format->pb);
        avformat_free_context(format);
        return ok;
    }
} // namespace

// Catches sampling to the movie's duration: a 4 s audio track beside 3 s of video made sparse FPS extraction ask the
// decoder for frames past the video's end, so the whole extraction failed.
TEST(VideoFrameExtractorTrim, SparseFpsStopsAtTheVideoWhenAudioOutlastsIt) {
    if (!cudaAvailable())
        GTEST_SKIP() << "CUDA device required for the hardware decoder that selects sparse seeking";

    TempDir temp("audio_outlasts_video");
    const auto source = temp.path / "source.mov";
    ASSERT_TRUE(writeVideoWithLongerAudio(source, 90, 4.0));

    auto params = extractionParams(source, temp.path / "frames");
    params.mode = ExtractionMode::FPS;
    params.fps = 3;
    params.end_time = -1;
    VideoFrameExtractor extractor;
    std::string error;
    ASSERT_TRUE(extractor.extract(params, error)) << error;
    EXPECT_EQ(countPngFiles(params.output_dir), 9u);
    const auto metadata = readMetadata(params.output_dir);
    EXPECT_NEAR(metadata["source_duration"].get<double>(), 3.0, 1e-6);
    EXPECT_EQ(metadata["frames"].size(), 9u);
}

TEST(VideoFrameExtractorTrim, SparseFpsStopsAtTheLastDecodableFrameInAStreamCopy) {
    if (!cudaAvailable())
        GTEST_SKIP() << "Requires NVDEC for sparse FPS extraction";

    TempDir temp("stream_copy_tail");
    const auto source = std::filesystem::path(PROJECT_ROOT_PATH) /
                        "tests/data/video/h264_stream_copy.mp4";
    for (const double end_time : {-1.0, 0.4}) {
        SCOPED_TRACE(end_time);
        auto params = extractionParams(source, temp.path / std::to_string(end_time));
        params.mode = ExtractionMode::FPS;
        params.fps = 3.0;
        params.end_time = end_time;

        VideoFrameExtractor extractor;
        std::string error;
        ASSERT_TRUE(extractor.extract(params, error)) << error;
        EXPECT_EQ(extractor.lastOutcome(), lfs::io::ExtractionOutcome::Completed);
        EXPECT_EQ(countPngFiles(params.output_dir), 1u);
        const auto metadata = readMetadata(params.output_dir);
        EXPECT_EQ(metadata["processing"]["frame_selection"]["strategy"], "keyframe_seek");
        EXPECT_EQ(metadata["processing"]["frame_selection"]["estimated_targets"], 1);
        ASSERT_EQ(metadata["frames"].size(), 1u);
        EXPECT_DOUBLE_EQ(metadata["frames"][0]["timestamp"].get<double>(), 0.0);
    }
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

namespace {
    // Independent software decode: compare image contents, not just the output count.
    std::vector<std::vector<uint8_t>> decodeVp9Reference(const std::filesystem::path& source) {
        std::vector<std::vector<uint8_t>> result;
        AVFormatContext* format = nullptr;
        if (avformat_open_input(&format, source.string().c_str(), nullptr, nullptr) < 0)
            return result;
        avformat_find_stream_info(format, nullptr);
        const int stream = av_find_best_stream(format, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_VP9);
        AVCodecContext* context = avcodec_alloc_context3(codec);
        avcodec_parameters_to_context(context, format->streams[stream]->codecpar);
        context->pkt_timebase = format->streams[stream]->time_base;
        if (avcodec_open2(context, codec, nullptr) >= 0) {
            AVPacket* packet = av_packet_alloc();
            AVFrame* frame = av_frame_alloc();
            SwsContext* scaler = nullptr;
            const auto drain = [&] {
                while (avcodec_receive_frame(context, frame) == 0) {
                    scaler = sws_getCachedContext(scaler, frame->width, frame->height,
                                                  static_cast<AVPixelFormat>(frame->format),
                                                  frame->width, frame->height, AV_PIX_FMT_RGB24,
                                                  SWS_BILINEAR, nullptr, nullptr, nullptr);
                    auto& rgb = result.emplace_back(frame->width * frame->height * 3);
                    uint8_t* planes[] = {rgb.data(), nullptr, nullptr, nullptr};
                    int strides[] = {frame->width * 3, 0, 0, 0};
                    sws_scale(scaler, frame->data, frame->linesize, 0, frame->height, planes, strides);
                    av_frame_unref(frame);
                }
            };
            while (av_read_frame(format, packet) >= 0) {
                if (packet->stream_index == stream) {
                    avcodec_send_packet(context, packet);
                    drain();
                }
                av_packet_unref(packet);
            }
            avcodec_send_packet(context, nullptr);
            drain();
            sws_freeContext(scaler);
            av_frame_free(&frame);
            av_packet_free(&packet);
        }
        avcodec_free_context(&context);
        avformat_close_input(&format);
        return result;
    }
} // namespace

TEST(VideoFrameExtractorOutputNaming, Vp9KeepsEverySourceTimestamp) {
    if (!cudaAvailable())
        GTEST_SKIP() << "CUDA device required for NVDEC timestamp regression";
    TempDir temp("vp9_timestamps");
    const auto source = std::filesystem::path(PROJECT_ROOT_PATH) /
                        "tests/data/video/vp9_timestamps.webm";
    const auto reference = decodeVp9Reference(source);
    ASSERT_EQ(reference.size(), 185u);
    auto params = extractionParams(source, temp.path);
    params.end_time = -1.0;
    VideoFrameExtractor extractor;
    std::string error;
    ASSERT_TRUE(extractor.extract(params, error)) << error;
    const auto metadata = readMetadata(temp.path);
    ASSERT_EQ(metadata["processing"]["decoder"]["backend"], "nvdec");
    EXPECT_EQ(countPngFiles(temp.path), reference.size());
    for (int index = 1; index <= 185; ++index) {
        SCOPED_TRACE(index);
        const auto path = temp.path /
                          (lfs::io::formatFrameFilenameStem(params.filename_pattern, index) + ".png");
        EXPECT_TRUE(std::filesystem::exists(path));
        if (!std::filesystem::exists(path))
            continue;
        const auto [pixels, width, height, channels] = lfs::core::load_image(path);
        ASSERT_NE(pixels, nullptr);
        const std::unique_ptr<unsigned char, decltype(&lfs::core::free_image)> image(pixels, lfs::core::free_image);
        ASSERT_EQ(width, 128);
        ASSERT_EQ(height, 128);
        ASSERT_EQ(channels, 3);
        const auto& expected = reference[index - 1];
        double difference = 0.0;
        for (std::size_t pixel = 0; pixel < expected.size(); ++pixel)
            difference += std::abs(static_cast<int>(pixels[pixel]) - expected[pixel]);
        EXPECT_LT(difference / expected.size(), 4.0);
        for (int candidate = std::max(1, index - 2); candidate <= std::min(185, index + 2); ++candidate) {
            double other_difference = 0.0;
            const auto& other = reference[candidate - 1];
            for (std::size_t pixel = 0; pixel < other.size(); ++pixel)
                other_difference += std::abs(static_cast<int>(pixels[pixel]) - other[pixel]);
            EXPECT_LE(difference, other_difference) << "better matching source frame " << candidate;
        }
    }
}

TEST(VideoFrameExtractorOutputNaming, ContainerTimeBasesPreserveIntervalFrames) {
    for (const char* container : {"mp4", "matroska"}) {
        for (const int denominator : {1000, 1001}) {
            SCOPED_TRACE(container);
            SCOPED_TRACE(denominator);
            TempDir temp("time_base_guard");
            const auto source = temp.path / "source.video";
            ASSERT_TRUE(writeProbedVideoWithRate(source, container, AV_CODEC_ID_MPEG4,
                                                 60, 0, 30000, denominator));
            auto params = extractionParams(source, temp.path / "all");
            params.end_time = -1.0;
            VideoFrameExtractor extractor;
            std::string error;
            ASSERT_TRUE(extractor.extract(params, error)) << error;
            ASSERT_EQ(countPngFiles(params.output_dir), 60u);
            const auto reference = params.output_dir;
            params.output_dir = temp.path / "interval";
            params.frame_interval = 3;
            ASSERT_TRUE(extractor.extract(params, error)) << error;
            EXPECT_EQ(countPngFiles(params.output_dir), 20u);
            for (int index = 1; index <= 60; index += 3) {
                const auto name = lfs::io::formatFrameFilenameStem(params.filename_pattern, index) + ".png";
                std::ifstream expected(reference / name, std::ios::binary);
                std::ifstream actual(params.output_dir / name, std::ios::binary);
                ASSERT_TRUE(expected.is_open());
                ASSERT_TRUE(actual.is_open());
                EXPECT_EQ(std::string(std::istreambuf_iterator<char>(expected), {}),
                          std::string(std::istreambuf_iterator<char>(actual), {}))
                    << index;
            }
        }
    }
}

TEST(VideoFrameExtractorOutputNaming, SmallVp9KeepsSoftwareFallback) {
    TempDir temp("vp9_software");
    const auto source = std::filesystem::path(PROJECT_ROOT_PATH) /
                        "tests/data/video/vp9_small.webm";
    auto params = extractionParams(source, temp.path);
    params.end_time = -1.0;
    VideoFrameExtractor extractor;
    std::string error;
    ASSERT_TRUE(extractor.extract(params, error)) << error;
    EXPECT_EQ(countPngFiles(temp.path), 10u);
    for (int index = 1; index <= 10; ++index)
        EXPECT_TRUE(std::filesystem::exists(temp.path /
                                            (lfs::io::formatFrameFilenameStem(params.filename_pattern, index) + ".png")))
            << index;
}

TEST(VideoFrameExtractorOutputNaming, Vp9SamplingKeepsSourceFrameNumbers) {
    if (!cudaAvailable())
        GTEST_SKIP() << "CUDA device required for NVDEC timestamp regression";
    TempDir temp("vp9_sampling");
    const auto source = std::filesystem::path(PROJECT_ROOT_PATH) /
                        "tests/data/video/vp9_timestamps.webm";
    for (const bool sparse : {false, true}) {
        auto params = extractionParams(source, temp.path / (sparse ? "fps" : "interval"));
        params.end_time = -1.0;
        params.mode = sparse ? ExtractionMode::FPS : ExtractionMode::INTERVAL;
        params.fps = 1.0;
        params.frame_interval = 3;
        VideoFrameExtractor extractor;
        std::string error;
        ASSERT_TRUE(extractor.extract(params, error)) << error;
        EXPECT_EQ(countPngFiles(params.output_dir), sparse ? 7u : 62u);
        const int step = sparse ? 30 : 3;
        for (int index = 1; index <= 185; index += step)
            EXPECT_TRUE(std::filesystem::exists(params.output_dir /
                                                (lfs::io::formatFrameFilenameStem(params.filename_pattern, index) + ".png")))
                << index;
    }
}

namespace {
    std::filesystem::path decoderFixture(const char* name) {
        return std::filesystem::path(PROJECT_ROOT_PATH) / "tests/data/video" / name;
    }

    void checkDecoderFixture(const char* name, const int width, const int height) {
        const auto source = decoderFixture(name);
        lfs::io::VideoPlayer player;
        ASSERT_TRUE(player.open(source)) << player.takeError();
        EXPECT_TRUE(player.takeError().empty());
        EXPECT_EQ(player.sourceWidth(), width);
        EXPECT_EQ(player.sourceHeight(), height);
        EXPECT_DOUBLE_EQ(player.fps(), 10.0);
        EXPECT_NEAR(player.duration(), 0.5, 0.001);
        ASSERT_NE(player.currentFrameData(), nullptr);
        const size_t size = static_cast<size_t>(player.width()) * player.height() * player.currentFrameChannels();
        const std::vector<uint8_t> first(player.currentFrameData(), player.currentFrameData() + size);
        EXPECT_GT(*std::max_element(first.begin(), first.end()), 200);
        player.seek(0.3);
        EXPECT_NEAR(player.currentTime(), 0.3, 0.001);
        player.seek(0.0);
        ASSERT_NE(player.currentFrameData(), nullptr);
        EXPECT_EQ(first, std::vector<uint8_t>(player.currentFrameData(), player.currentFrameData() + size));
        player.close();
        EXPECT_FALSE(player.isOpen());
        ASSERT_TRUE(player.open(source)) << player.takeError();
        EXPECT_EQ(first, std::vector<uint8_t>(player.currentFrameData(), player.currentFrameData() + size));

        TempDir temp("decoder_fallback");
        auto params = extractionParams(source, temp.path / "frames");
        params.end_time = -1.0;
        VideoFrameExtractor extractor;
        std::string error;
        ASSERT_TRUE(extractor.extract(params, error)) << error;
        EXPECT_EQ(countPngFiles(params.output_dir), 5u);
    }
} // namespace

TEST(VideoDecoderFallback, H264444OpensSeeksAndExtracts) {
    checkDecoderFixture("h264_444.mp4", 64, 48);
}

TEST(VideoDecoderFallback, OddSizedH264444OpensSeeksAndExtracts) {
    checkDecoderFixture("h264_444_odd.mp4", 65, 49);
}

TEST(VideoDecoderFallback, H264420StillOpensSeeksAndExtracts) {
    checkDecoderFixture("h264_420.mp4", 64, 48);
}

TEST(VideoDecoderFallback, LongRawH264PreviewOpenReadsOnlyTheProbeWindow) {
    TempDir temp("long_raw_h264_probe");
    const auto source = temp.path / "long_stream.h264";
    constexpr int frame_count = 300000;
    ASSERT_TRUE(writeProbedVideoWithRate(source, "h264", AV_CODEC_ID_H264,
                                         frame_count, 0, 30, 1, false, 3));
    const auto source_size = std::filesystem::file_size(source);
    ASSERT_GT(source_size, 8 * 1024 * 1024);

    lfs::io::VideoPlayer player;
#if defined(__linux__)
    const auto bytes_before_open = processReadChars();
#endif
    ASSERT_TRUE(player.open(source)) << player.takeError();
#if defined(__linux__)
    const auto bytes_after_open = processReadChars();
    ASSERT_NE(bytes_before_open, std::numeric_limits<uint64_t>::max());
    ASSERT_NE(bytes_after_open, std::numeric_limits<uint64_t>::max());
    const auto open_read_bytes = bytes_after_open - bytes_before_open;
    EXPECT_LT(open_read_bytes, source_size / 2)
        << "VideoPlayer::open read " << open_read_bytes
        << " of " << source_size << " source bytes";
#endif
    EXPECT_NE(player.currentFrameData(), nullptr);
    EXPECT_DOUBLE_EQ(player.duration(), 0.0);
    player.close();
}

TEST(VideoFrameExtractorOutputNaming, TimestampedGeneratedClipKeepsFrameHashes) {
    TempDir temp("timestamped_frame_hashes");
    const auto source = temp.path / "timestamped.mp4";
    constexpr int frame_count = 6;
    ASSERT_TRUE(writeProbedVideoWithRate(source, "mp4", AV_CODEC_ID_MPEG4,
                                         frame_count, 0, 25, 1, false, 3));

    auto params = extractionParams(source, temp.path / "frames");
    params.end_time = -1.0;
    VideoFrameExtractor extractor;
    std::string error;
    ASSERT_TRUE(extractor.extract(params, error)) << error;
    ASSERT_EQ(countPngFiles(params.output_dir), frame_count);

    // Keep the original NVDEC pixel reference independent of PNG compression.
    const std::vector<uint64_t> expected_hashes{
        4447340953764640790ull, 6700247232541181701ull,
        18018889043452171844ull, 15686283614649312191ull,
        3742041256909694354ull, 8649771673398531241ull};
    std::vector<uint64_t> hashes;
    for (int index = 1; index <= frame_count; ++index) {
        SCOPED_TRACE(index);
        const auto frame = params.output_dir / ("frame_" + std::to_string(index) + ".png");
        ASSERT_TRUE(std::filesystem::exists(frame));
        const auto [pixels, width, height, channels] = lfs::core::load_image(frame);
        ASSERT_NE(pixels, nullptr);
        const std::unique_ptr<unsigned char, decltype(&lfs::core::free_image)> image(pixels, lfs::core::free_image);
        ASSERT_EQ(width, 64);
        ASSERT_EQ(height, 48);
        ASSERT_EQ(channels, 3);
        hashes.push_back(hashPixels(pixels, static_cast<std::size_t>(width) * height * channels));
    }
    EXPECT_EQ(hashes, expected_hashes);
}

#if defined(__linux__)
TEST(VideoDecoderFallback, TimestampedInputWithoutDurationKeepsTheExistingError) {
    TempDir temp("timestamped_without_duration");
    const auto source = temp.path / "timestamped.ts";
    ASSERT_TRUE(writeProbedVideoWithRate(source, "mpegts", AV_CODEC_ID_MPEG2VIDEO,
                                         8, 0, 25, 1, false, 3));
    const auto fifo = temp.path / "timestamped.fifo";
    ASSERT_EQ(::mkfifo(fifo.c_str(), 0600), 0);

    std::thread writer([&] {
        std::ifstream input(source, std::ios::binary);
        std::ofstream output(fifo, std::ios::binary);
        output << input.rdbuf();
    });
    auto params = extractionParams(fifo, temp.path / "frames");
    params.end_time = -1.0;
    VideoFrameExtractor extractor;
    std::string error;
    const bool extracted = extractor.extract(params, error);
    writer.join();

    EXPECT_FALSE(extracted);
    EXPECT_EQ(error, "Could not determine video duration");
}
#endif

TEST(VideoDecoderFallback, RawH264WithoutPacketTimestampsOpensAndExtracts) {
    TempDir temp("raw_h264_no_timestamps");
    const auto source = temp.path / "raw_stream.mp4";
    constexpr int frame_count = 12;
    ASSERT_TRUE(writeProbedVideoWithRate(source, "h264", AV_CODEC_ID_H264,
                                         frame_count, 0, 30, 1, false));

    AVFormatContext* format = nullptr;
    ASSERT_EQ(avformat_open_input(&format, source.string().c_str(), nullptr, nullptr), 0);
    AVPacket* packet = av_packet_alloc();
    ASSERT_NE(packet, nullptr);
    ASSERT_EQ(av_read_frame(format, packet), 0);
    EXPECT_EQ(packet->pts, AV_NOPTS_VALUE);
    EXPECT_EQ(packet->dts, AV_NOPTS_VALUE);
    av_packet_free(&packet);
    avformat_close_input(&format);

    lfs::io::VideoPlayer player;
    ASSERT_TRUE(player.open(source)) << player.takeError();
    EXPECT_NE(player.currentFrameData(), nullptr);
    const double fps = player.fps();
    ASSERT_GT(fps, 1.0);
    ASSERT_LT(fps, 1000.0);
    EXPECT_DOUBLE_EQ(player.duration(), 0.0);
    const double seek_time = 5.0 / fps;
    player.seek(seek_time);
    EXPECT_NEAR(player.currentTime(), seek_time, 0.001);
    player.close();

    auto params = extractionParams(source, temp.path / "frames");
    params.end_time = -1.0;
    VideoFrameExtractor extractor;
    std::string error;
    ASSERT_TRUE(extractor.extract(params, error)) << error;
    EXPECT_EQ(countPngFiles(params.output_dir), frame_count);
    for (int number = 1; number <= frame_count; ++number)
        EXPECT_TRUE(std::filesystem::exists(params.output_dir /
                                            ("frame_" + std::to_string(number) + ".png")));
}

TEST(VideoDecoderFallback, FailedReplacementClearsDialogPathsAndCanRecover) {
    using Access = lfs::gui::VideoExtractorDialogTestAccess;
    lfs::gui::VideoExtractorDialog dialog;
    const auto source = decoderFixture("h264_420.mp4");
    ASSERT_TRUE(dialog.openVideoPath(source));
    EXPECT_EQ(Access::videoPath(dialog), source);
    Access::outputPath(dialog, "custom_output");
    ASSERT_TRUE(dialog.openVideoPath(source));
    EXPECT_EQ(Access::outputPath(dialog), "custom_output");
    EXPECT_FALSE(dialog.openVideoPath(decoderFixture("missing.mp4")));
    EXPECT_TRUE(Access::videoPath(dialog).empty());
    EXPECT_TRUE(Access::outputPath(dialog).empty());
    ASSERT_TRUE(dialog.openVideoPath(source));
    EXPECT_EQ(Access::videoPath(dialog), source);
    EXPECT_EQ(Access::outputPath(dialog), source.parent_path() / "h264_420_frames");
}
