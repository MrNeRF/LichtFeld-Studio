/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/argument_parser.hpp"
#include "core/image_io.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "io/filesystem_utils.hpp"
#include "io/video/video_encoder.hpp"
#include "preprocessing/video_mask_cells.hpp"
#include "preprocessing/video_mask_extraction.hpp"
#include "preprocessing/video_mask_recovery.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <gtest/gtest.h>

namespace {
    using namespace lfs;
    using namespace lfs::preprocessing;
    const auto photo = std::filesystem::path(PROJECT_ROOT_PATH) / "tests/data/nn/lpips_crop_a.png";
    struct OutputDirectory {
        std::filesystem::path path = std::filesystem::temp_directory_path() /
                                     ("lfs_video_mask_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        OutputDirectory() { std::filesystem::create_directory(path); }
        ~OutputDirectory() {
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
        }
    };
    struct Stream {
        cudaStream_t value = nullptr;
        Stream() { EXPECT_EQ(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), cudaSuccess); }
        ~Stream() {
            cudaStreamSynchronize(value);
            core::CudaMemoryPool::instance().release_stream(value);
            cudaStreamDestroy(value);
        }
    };
    VideoMaskFrame frame(const std::filesystem::path& output, int index) {
        const auto name = std::format("{:05d}.jpg", index);
        const auto [width, height, channels] = core::get_image_info(photo);
        return {.image_path = photo, .mask_path = video_mask_path(output, name), .width = width, .height = height, .time_seconds = index / 30.0};
    }
} // namespace

TEST(VideoMaskingValidation, InvalidInputsAndMemoryAdmission) {
    ::testing::Test::RecordProperty("catches", "Missing destination validation overwrites frames; sequence-sized admission or ignored budget breaks bounded allocation");
    OutputDirectory output;
    std::vector frames{frame(output.path, 0)};
    VideoMaskOptions options;
    const auto plan = plan_video_mask_resources(frames, options, size_t(8) << 30);
    ASSERT_TRUE(plan.has_value());
    frames.resize(100, frames.front());
    const auto longer = plan_video_mask_resources(frames, options, size_t(8) << 30);
    ASSERT_TRUE(longer.has_value());
    EXPECT_EQ(plan->worker_bytes, longer->worker_bytes);
    options.gpu_memory_budget_bytes = plan->worker_bytes;
    EXPECT_FALSE(plan_video_mask_resources(frames, options, size_t(8) << 30).has_value());
    frames.resize(1);
    std::filesystem::copy_file(frames[0].image_path, output.path / "source.jpg");
    frames[0].image_path = output.path / "source.jpg";
    frames[0].mask_path = frames[0].image_path;
    const auto invalid = mask_video_sequence(frames, {}, {}, nullptr);
    ASSERT_FALSE(invalid.has_value());
    EXPECT_EQ(invalid.error().code(), ErrorCode::InvalidArgument);
    frames[0] = frame(output.path, 0);
    frames.push_back(frames.front());
    EXPECT_FALSE(mask_video_sequence(frames, {}, {}, nullptr).has_value());
    frames.resize(1);
    options.cancel_requested = [] { return true; };
    const auto cancelled = mask_video_sequence(frames, {}, options, nullptr);
    ASSERT_FALSE(cancelled.has_value());
    EXPECT_EQ(cancelled.error().code(), ErrorCode::Cancelled);
}

TEST(VideoMaskingRecovery, PatienceAndRealMaskBounds) {
    ::testing::Test::RecordProperty("catches", "Resetting recovery on an unreliable frame or dropping the inclusive mask edge loses the recovery seed");
    preprocessing::detail::MaskRecovery forward, reverse;
    forward.accept();
    reverse.accept();
    const preprocessing::detail::RecoveryQuality weak{0.2f, 1, 0.1f, false};
    EXPECT_TRUE(forward.observe(weak, 0.5f, 0.4f, 3).empty());
    EXPECT_TRUE(forward.observe(weak, 0.5f, 0.4f, 3).empty());
    EXPECT_EQ(forward.observe(weak, 0.5f, 0.4f, 3), "no_reliable_frame");
    EXPECT_EQ(reverse.unreliable_frames, 0);
    EXPECT_EQ(reverse.observe({0.9f, 1, 0, true}, 0.5f, 0.4f, 3), "empty_prediction");
    EXPECT_TRUE(forward.observe({0.9f, 1, 0.1f, false}, 0.5f, 0.4f, 3).empty());
    EXPECT_EQ(forward.unreliable_frames, 0);
    auto [pixels, width, height, channels] = core::load_image(photo);
    const int side = std::min(width, height);
    std::vector<float> mask(size_t(side) * side);
    int left = side, top = side, right = -1, bottom = -1;
    for (int y = 0; y < side; ++y)
        for (int x = 0; x < side; ++x) {
            const bool keep = pixels[(size_t(y) * width + x) * channels] > 200;
            mask[size_t(y) * side + x] = keep;
            if (keep) {
                left = std::min(left, x);
                top = std::min(top, y);
                right = std::max(right, x);
                bottom = std::max(bottom, y);
            }
        }
    core::free_image(pixels);
    const auto box = preprocessing::detail::recovery_box(mask, side, width, height);
    ASSERT_TRUE(box.has_value());
    EXPECT_EQ((*box)[0], float(left) * width / side);
    EXPECT_EQ((*box)[1], float(top) * height / side);
    EXPECT_EQ((*box)[2], float(right + 1) * width / side);
    EXPECT_EQ((*box)[3], float(bottom + 1) * height / side);
}

TEST(VideoMaskExtraction, OptInLayoutAndWrittenFrameCallbacks) {
    ::testing::Test::RecordProperty("catches", "Unconditional nesting changes legacy output; omitted callbacks or circle-mask writes leave frames without training masks");
    OutputDirectory output;
    const auto video = output.path / "photo.mp4";
    auto [pixels, image_width, image_height, image_channels] = core::load_image(photo);
    std::vector<float> rgb(size_t(image_width) * image_height * 3);
    for (size_t i = 0; i < size_t(image_width) * image_height; ++i)
        for (size_t c = 0; c < 3; ++c)
            rgb[3 * i + c] = pixels[image_channels * i + c] / 255.0f;
    core::free_image(pixels);
    const auto gpu_frame = core::Tensor::from_vector(rgb, {size_t(image_height), size_t(image_width), 3}, core::Device::CPU)
                               .to(core::Device::CUDA);
    io::video::VideoExportOptions encoding;
    encoding.preset = io::video::VideoPreset::CUSTOM;
    encoding.width = image_width;
    encoding.height = image_height;
    encoding.framerate = 4;
    io::video::VideoEncoder encoder;
    ASSERT_TRUE(encoder.open(video, encoding).has_value());
    for (int i = 0; i < 8; ++i) {
        const auto written = encoder.writeFrameGpu(gpu_frame.ptr<float>(), image_width, image_height, gpu_frame.stream());
        ASSERT_TRUE(written.has_value()) << written.error();
    }
    ASSERT_TRUE(encoder.close().has_value());
    for (const bool enabled : {false, true})
        for (const auto format : {io::ImageFormat::PNG, io::ImageFormat::JPG}) {
            io::VideoFrameExtractor::Params params;
            params.video_path = video;
            params.output_dir = output.path / (std::to_string(enabled) + std::to_string(int(format)));
            params.fps = 2;
            params.end_time = 1;
            params.format = format;
            std::vector<io::VideoFrameExtractor::WrittenFrame> written;
            params.frame_written = [&](const auto& item) { written.push_back(item); EXPECT_TRUE(std::filesystem::is_regular_file(item.path)); };
            VideoMaskExtractionOptions options;
            options.enabled = enabled;
            options.circle = {0.5, 0.5, 0.5, 0.5};
            options.masking.enabled = false;
            std::string error;
            ASSERT_EQ(extract_video_frames(params, options, error), io::ExtractionOutcome::Completed) << error;
            ASSERT_EQ(written.size(), 2u);
            EXPECT_EQ(std::filesystem::is_directory(params.output_dir / "images"), enabled);
            io::MaskDirCache masks(params.output_dir);
            for (const auto& item : written) {
                EXPECT_EQ(item.path.parent_path(), video_frames_directory(params.output_dir, enabled));
                EXPECT_EQ(item.width, image_width);
                EXPECT_EQ(item.height, image_height);
                if (!enabled)
                    continue;
                const auto mask_path = masks.find(item.path.filename().string());
                ASSERT_EQ(mask_path, video_mask_path(params.output_dir, item.path.filename()));
                auto [pixels, width, height, channels] = core::load_image(mask_path);
                ASSERT_EQ(width, item.width);
                ASSERT_EQ(height, item.height);
                EXPECT_EQ(pixels[0], 0);
                EXPECT_EQ(pixels[(size_t(height / 2) * width + width / 2) * channels], 255);
                core::free_image(pixels);
            }
        }
}

TEST(VideoMaskCommand, ParsesManualSeedsAndRejectsInvalidGeometry) {
    ::testing::Test::RecordProperty("catches", "Subcommand dispatch or point validation mistakes silently start training or accept invalid masking coordinates");
    const char* valid[]{"LichtFeld-Studio", "mask-video", "frames", "--seed", "0.4,0.7", "--fisheye", "--recenter", "--no-download"};
    auto parsed = core::args::parse_args(std::size(valid), valid);
    ASSERT_TRUE(parsed.has_value());
    ASSERT_TRUE(std::holds_alternative<core::args::VideoMaskMode>(*parsed));
    const auto& params = std::get<core::args::VideoMaskMode>(*parsed).params;
    ASSERT_EQ(params.points.size(), 1u);
    EXPECT_EQ(params.points[0][2], 1);
    EXPECT_TRUE(params.recenter);
    EXPECT_TRUE(params.no_download);
    EXPECT_EQ(params.circle[2], 0.5);
    const char* invalid[]{"LichtFeld-Studio", "mask-video", "frames", "--seed", "1.1,0.7"};
    EXPECT_FALSE(core::args::parse_args(std::size(invalid), invalid).has_value());
    const char* unbounded[]{"LichtFeld-Studio", "mask-video", "frames", "--fps", "nan"};
    EXPECT_FALSE(core::args::parse_args(std::size(unbounded), unbounded).has_value());
}

TEST(VideoMaskingCells, CorrelationSeparatesFixedMovedAndFlatRealContent) {
    ::testing::Test::RecordProperty("catches", "Exposure-sensitive or unnormalized correlation, a transposed cell layout, a missing contrast gate or a scene measure that ignores motion misclassifies real image cells");
    Stream stream;
    core::CUDAStreamGuard guard(stream.value);
    auto [pixels, width, height, channels] = core::load_image(photo);
    ASSERT_EQ(channels, 3);
    std::vector<float> values(size_t(width) * height);
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = (0.299f * pixels[3 * i] + 0.587f * pixels[3 * i + 1] + 0.114f * pixels[3 * i + 2]) / 255.0f;
    core::free_image(pixels);
    const auto gray = core::Tensor::from_vector(values, {1, 1, size_t(height), size_t(width)}, core::Device::CPU)
                          .to(core::Device::CUDA, stream.value);
    const auto compare = [](const core::Tensor& first, const core::Tensor& second) {
        return preprocessing::detail::compare_cells(
            preprocessing::detail::cell_patches(first, preprocessing::detail::kCellGrid),
            preprocessing::detail::cell_patches(second, preprocessing::detail::kCellGrid),
            preprocessing::detail::cell_patches(first, preprocessing::detail::kSceneGrid),
            preprocessing::detail::cell_patches(second, preprocessing::detail::kSceneGrid));
    };
    using preprocessing::detail::CellMotion;
    const auto count = [](const preprocessing::detail::CellComparison& comparison, const CellMotion motion, const int first_column,
                          const int last_column) {
        size_t total = 0;
        for (size_t cell = 0; cell < comparison.cells.size(); ++cell) {
            const int column = int(cell % preprocessing::detail::kCellGrid);
            total += column >= first_column && column <= last_column && comparison.cells[cell] == motion;
        }
        return total;
    };
    const auto exposed = compare(gray, gray.mul(0.6f).add(0.1f));
    EXPECT_GT(exposed.textured, 200u);
    EXPECT_EQ(count(exposed, CellMotion::Fixed, 0, 47), exposed.textured);
    EXPECT_GT(exposed.scene_correlation, 0.99f);

    const size_t shift = 96;
    const auto rolled = core::Tensor::cat({gray.slice(3, shift, size_t(width)), gray.slice(3, 0, shift)}, 3);
    const auto moved = compare(gray, rolled);
    EXPECT_GT(moved.textured, 200u);
    EXPECT_GE(count(moved, CellMotion::Moving, 0, 47), moved.textured * 7 / 10);
    EXPECT_LE(count(moved, CellMotion::Fixed, 0, 47), moved.textured / 50);
    EXPECT_LE(moved.scene_correlation, preprocessing::detail::kMaximumSceneCorrelation);

    const auto half = size_t(width / 2);
    const auto mixed = compare(gray, core::Tensor::cat({gray.slice(3, 0, half), rolled.slice(3, half, size_t(width))}, 3));
    EXPECT_GT(count(mixed, CellMotion::Fixed, 0, 21), 200u);
    EXPECT_EQ(count(mixed, CellMotion::Moving, 0, 21), 0u);
    EXPECT_LE(count(mixed, CellMotion::Fixed, 26, 47), moved.textured / 50);
    EXPECT_GT(count(mixed, CellMotion::Moving, 26, 47), 10u);

    const auto flat = core::Tensor::full({1, 1, size_t(height), size_t(width)}, 0.5f, core::Device::CUDA);
    EXPECT_EQ(compare(flat, flat).textured, 0u);
}

TEST(VideoMaskingPipeline, StaticCameraWarnsAndRequireOperatorFails) {
    // Catches reporting automatic detection failure as an unqualified success or ignoring require_operator.
    OutputDirectory output;
    Stream stream;
    core::CUDAStreamGuard guard(stream.value);
    std::vector<VideoMaskFrame> frames;
    for (int index = 0; index < 38; ++index) {
        frames.push_back(frame(output.path, index));
        frames.back().time_seconds = 0.5 * index;
    }
    VideoMaskOptions options;
    options.allow_download = false;
    auto result = mask_video_sequence(frames, {}, options, stream.value);
    ASSERT_TRUE(result.has_value()) << result.error().user_message();
    EXPECT_EQ(result->detection.status, "insufficient_motion_pairs");
    EXPECT_GE(result->detection.pairs_tested, 24u);
    EXPECT_EQ(result->detection.motion_pairs, 0u);
    EXPECT_NE(result->warning.find("not enough camera motion"), std::string::npos);
    options.require_operator = true;
    auto required = mask_video_sequence(frames, {}, options, stream.value);
    ASSERT_FALSE(required.has_value());
    EXPECT_EQ(required.error().code(), ErrorCode::FailedPrecondition);
    EXPECT_EQ(required.error().user_message(), result->warning);
}

TEST(VideoMaskingGeometry, FisheyeCenterRoundTripAndRealImageSampling) {
    ::testing::Test::RecordProperty("catches", "Incorrect rotation transpose or pixel-center offset moves the operator and samples the wrong source pixels");
    Stream stream;
    core::CUDAStreamGuard guard(stream.value);
    auto [pixels, width, height, channels] = core::load_image(photo);
    ASSERT_EQ(channels, 3);
    std::vector<float> planar(size_t(width) * height * 3);
    for (size_t i = 0; i < size_t(width) * height; ++i)
        for (size_t c = 0; c < 3; ++c)
            planar[c * width * height + i] = pixels[3 * i + c] / 255.0f;
    core::free_image(pixels);
    const auto view = make_operator_view(width, height, width * 0.7f, height * 0.55f, {0.5f, 0.5f, 0.5f, 0.5f}, 200, 160, 256);
    const auto center = operator_view_coordinate(view, 127.5f, 127.5f, false);
    ASSERT_EQ(center[2], 1);
    EXPECT_NEAR(center[0], width * 0.7f, 0.001f);
    EXPECT_NEAR(center[1], height * 0.55f, 0.001f);
    const auto inverse = operator_view_coordinate(view, center[0], center[1], true);
    EXPECT_NEAR(inverse[0], 127.5f, 0.001f);
    EXPECT_NEAR(inverse[1], 127.5f, 0.001f);
    const auto input = core::Tensor::from_vector(planar, {1, 3, size_t(height), size_t(width)}, core::Device::CPU)
                           .to(core::Device::CUDA, stream.value);
    const auto output = render_operator_view(input, view).to(core::Device::CPU).to_vector();
    for (const int y : {90, 128, 166})
        for (const int x : {90, 128, 166}) {
            const auto source = operator_view_coordinate(view, float(x), float(y), false);
            ASSERT_EQ(source[2], 1);
            const int sx = int(std::floor(source[0])), sy = int(std::floor(source[1]));
            const float tx = source[0] - sx, ty = source[1] - sy;
            ASSERT_GE(sx, 0);
            ASSERT_GE(sy, 0);
            ASSERT_LT(sx + 1, width);
            ASSERT_LT(sy + 1, height);
            for (size_t c = 0; c < 3; ++c) {
                const auto at = [&](int px, int py) { return planar[c * width * height + size_t(py) * width + px]; };
                const float expected = (1 - ty) * ((1 - tx) * at(sx, sy) + tx * at(sx + 1, sy)) +
                                       ty * ((1 - tx) * at(sx, sy + 1) + tx * at(sx + 1, sy + 1));
                EXPECT_NEAR(output[c * 256 * 256 + size_t(y) * 256 + x], expected, 2e-4f);
            }
        }
    const auto restored = restore_operator_mask(render_operator_view(input, view).slice(1, 0, 1), view)
                              .to(core::Device::CPU)
                              .to_vector();
    EXPECT_LT(restored[0], 0);
    EXPECT_GE(restored[size_t(height * 0.55f) * width + size_t(width * 0.7f)], 0);
}
