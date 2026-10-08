// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/image_codecs.hpp"
#include "media/file_frame_sink.hpp"
#include "media/frame_sink.hpp"
#include <chrono>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace {
    void require(bool ok, const char* message) {
        if (!ok)
            throw std::runtime_error(message);
    }
} // namespace
int runFrameSinkUnitContracts() {
    using namespace lfs::media;
    std::vector<uint8_t> pixels{1, 2, 3, 99, 99, 4, 5, 6};
    FrameView frame{{1, 2, 5, FramePixelFormat::RGB8}, {}, pixels};
    frame.info.source_timestamp = Timestamp{-42, {1, 1000}};
    auto snapshot = FrameSurface::copyOf(frame);
    require(snapshot.has_value(), "valid snapshot accepted");
    auto surface = std::move(*snapshot);
    auto copied = surface;
    pixels.assign(8, 0);
    surface = {};
    const auto view = copied.view();
    require(view.pixels[0] == 1 && view.pixels[5] == 4 && view.pixels[3] == 0 && view.pixels[4] == 0,
            "snapshot owns visible rows and zeroes padding");
    require(view.info.source_timestamp->ticks == -42, "snapshot retains signed source timestamp");
    for (int variant = 0; variant < 5; ++variant) {
        auto invalid = frame;
        if (variant == 0)
            invalid.layout.width = 0;
        if (variant == 1)
            invalid.layout.row_stride = 2;
        if (variant == 2)
            invalid.layout.row_stride = std::numeric_limits<size_t>::max();
        if (variant == 3)
            invalid.pixels = invalid.pixels.first(2);
        if (variant == 4)
            invalid.layout.format = static_cast<FramePixelFormat>(-1);
        const auto required = invalid.requiredBytes();
        require(!required && required.error().code() == lfs::ErrorCode::InvalidArgument,
                "invalid layout returns structured error without throwing");
        const auto rejected = FrameSurface::copyOf(invalid);
        require(!rejected && rejected.error().code() == lfs::ErrorCode::InvalidArgument,
                "invalid snapshot rejected before allocation/copy");
    }
    MemoryFrameSink memory(8, 2);
    require(!memory.write(view).has_value(), "inactive write rejected");
    require(memory.begin({}).has_value(), "begin accepted");
    require(!memory.begin({}).has_value(), "overlapping begin rejected");
    auto short_frame = view;
    short_frame.pixels = short_frame.pixels.first(1);
    require(memory.write(short_frame).error().code() == lfs::ErrorCode::InvalidArgument, "invalid sink layout is structured error");
    require(memory.write(view).has_value(), "within budget accepted");
    require(!memory.write(view).has_value() && memory.payloadBytes() == 8, "budget preserves accepted data");
    require(memory.write(view).error().code() == lfs::ErrorCode::ResourceExhausted, "budget uses core error taxonomy");
    memory.abort({SinkOutcome::Cancelled, 1, {}});
    require(memory.frames().size() == 1 && memory.frames()[0].view().pixels[5] == 4, "abort retains partial snapshots");
    require(memory.begin({}).has_value() && memory.frames().empty(), "reuse resets results");
    require(memory.complete({SinkOutcome::Completed, 0, {}}).has_value(), "complete accepted");
    MemoryFrameSink limited(100, 0);
    (void)limited.begin({});
    require(!limited.write(view).has_value(), "frame count bounds allocation overhead");
    const auto directory = std::filesystem::temp_directory_path() /
                           ("lfs-padded-sink-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    FileFrameSink files({directory});
    auto file_view = view;
    file_view.info.legacy_source_frame = 1;
    require(files.begin({}).has_value(), "file sink begins");
    require(files.write(file_view).has_value(), "padded RGB file write");
    lfs::core::image_codecs::Image decoded;
    std::string error;
    require(lfs::core::image_codecs::decode(directory / "frame_1.png", decoded, error), "decode padded sink PNG");
    require(decoded.width == 1 && decoded.height == 2 && decoded.data == std::vector<uint8_t>({1, 2, 3, 4, 5, 6}),
            "file sink omits padding and preserves visible rows");
    require(!files.write(file_view).has_value(), "duplicate filenames rejected");
    files.abort({SinkOutcome::Failed, 1, {}});
    std::filesystem::remove(directory / "frame_1.png");
    for (const int quality : {0, -5, 101}) {
        const auto suffix = std::to_string(quality);
        FileFrameSink png({directory, "png_" + suffix + "_%d", FrameFileFormat::PNG, quality});
        require(png.begin({}).has_value() && png.write(file_view).has_value(), "PNG ignores JPEG quality");
        (void)png.complete({SinkOutcome::Completed, 1, {}});
        FileFrameSink jpeg({directory, "jpeg_" + suffix + "_%d", FrameFileFormat::JPEG, quality});
        require(jpeg.begin({}).has_value() && jpeg.write(file_view).has_value(), "JPEG preserves default and clamping");
        (void)jpeg.complete({SinkOutcome::Completed, 1, {}});
        const auto expected_quality = quality == 0 ? 90 : quality < 1 ? 1
                                                                      : 100;
        const auto expected = directory / ("expected_" + suffix + ".jpg");
        const std::vector<uint8_t> packed{1, 2, 3, 4, 5, 6};
        require(lfs::core::image_codecs::write_jpeg(expected, packed.data(), 1, 2, 3,
                                                    expected_quality, std::nullopt, error, expected_quality > 90),
                "reference JPEG written");
        auto read_bytes = [](const auto& path) {
            std::ifstream stream(path, std::ios::binary);
            return std::vector<char>(std::istreambuf_iterator<char>(stream), {});
        };
        require(read_bytes(directory / ("jpeg_" + suffix + "_1.jpg")) == read_bytes(expected), "JPEG normalization matches writer bytes");
        std::filesystem::remove(directory / ("png_" + suffix + "_1.png"));
        std::filesystem::remove(directory / ("jpeg_" + suffix + "_1.jpg"));
        std::filesystem::remove(expected);
    }
    const auto seed = [&](const char* name) { std::ofstream(directory / name) << "existing"; };
    seed("frame_5.png");
    seed("frame_77.exr");
    seed("photo.png");
    seed("other_9.exr");
    FileFrameSinkOptions replacement;
    replacement.output_directory = directory;
    replacement.remove_stale_frames = true;
    FileFrameSink failed_replacement(replacement);
    require(failed_replacement.begin({}).has_value(), "replacement snapshot begins");
    failed_replacement.abort({SinkOutcome::Cancelled, 0, {}});
    require(std::filesystem::exists(directory / "frame_5.png") && std::filesystem::exists(directory / "frame_77.exr"), "cancel preserves obsolete frames");
    FileFrameSink successful_replacement(replacement);
    require(successful_replacement.begin({}).has_value() && successful_replacement.write(file_view).has_value(), "replacement writes frame");
    require(std::filesystem::exists(directory / "frame_77.exr"), "stale files survive until completion");
    require(successful_replacement.complete({SinkOutcome::Completed, 1, {}}).has_value(), "replacement complete cleans stale frames");
    require(!std::filesystem::exists(directory / "frame_5.png") && !std::filesystem::exists(directory / "frame_77.exr"), "obsolete generated PNG and EXR removed");
    require(std::filesystem::exists(directory / "frame_1.png") && std::filesystem::exists(directory / "photo.png") && std::filesystem::exists(directory / "other_9.exr"), "new and unrelated files preserved");
    std::filesystem::remove_all(directory);
    return 0;
}
