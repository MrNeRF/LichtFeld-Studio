// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/image_codecs.hpp"
#include "io/media/file_frame_sink.hpp"
#include "io/media/frame_sink.hpp"
#include <chrono>
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
    auto surface = FrameSurface::copyOf(frame);
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
        bool rejected = false;
        try {
            (void)FrameSurface::copyOf(invalid);
        } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "invalid frame rejected before allocation/copy");
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
    std::filesystem::remove(directory);
    return 0;
}
