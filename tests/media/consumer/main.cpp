// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include <io/media/media_ingest.hpp>
#include <iostream>
#include <stdexcept>
void require(bool ok, const char* text) {
    if (!ok)
        throw std::runtime_error(text);
}
std::size_t accepted(const lfs::Error& error) {
    for (const auto& frame : error.frames())
        for (const auto& field : frame.fields.entries())
            if (field.key == "frames_accepted")
                return std::get<std::uint64_t>(field.value);
    throw std::runtime_error("missing accepted-frame error context");
}
int exercise(const std::filesystem::path& input, const std::filesystem::path& output) {
    using namespace lfs::media;
    const auto probe = MediaProbe::inspect(input);
    require(probe && probe->selected_video_stream && probe->streams[0].width == 64, "public probe without FFmpeg headers");
    require(!MediaIngest::capabilities().hardware_decode && !MediaIngest::capabilities().hdr_to_sdr, "CPU capabilities");
    IngestRequest request;
    request.input = input;
    request.selection.mode = SelectionMode::Interval;
    request.end_seconds = 0.35;
    MemoryFrameSink memory;
    const auto result = MediaIngest::extract(request, memory);
    require(result && result->frames_accepted == 4 && memory.frames().size() == 4, "external memory extraction");
    const auto copied = FrameSurface::copyOf(memory.frames()[0].view());
    require(copied.has_value(), "public owning snapshot Result");
    const auto retained = *copied;
    request.geometry.mode = ResizeMode::Scale;
    request.geometry.scale = 0.5f;
    require(MediaIngest::extract(request, memory).has_value() && memory.frames()[0].view().layout.width == 32, "scale uses actual source dimensions");
    require(retained.view().layout.width == 64 && retained.view().pixels.size() == 6144, "retained surface survives sink reuse");
    request.geometry = {};
    MemoryFrameSink limited(6144);
    const auto limit = MediaIngest::extract(request, limited);
    require(!limit && limit.error().code() == lfs::ErrorCode::ResourceExhausted && accepted(limit.error()) == 1 && limited.frames().size() == 1, "budget error crosses API unchanged");
    int progress = 0;
    request.progress = [&](const IngestProgress& value) { progress = value.processed; };
    request.cancelled = [&] { return progress >= 1; };
    const auto cancel = MediaIngest::extract(request, memory);
    require(!cancel && cancel.error().code() == lfs::ErrorCode::Cancelled && accepted(cancel.error()) == 1 && memory.frames().size() == 1, "cancellation retains results");
    request.progress = {};
    request.cancelled = {};
    request.convert_hdr_to_sdr = true;
    const auto hdr = MediaIngest::extract(request, memory);
    require(!hdr && hdr.error().code() == lfs::ErrorCode::Unsupported, "missing backend is explicit");
    request.convert_hdr_to_sdr = false;
    struct RejectSink : FrameSink {
        int aborts = 0;
        SinkResult write(const FrameView&) override {
            return SinkResult::failure(lfs::make_error({.code = lfs::ErrorCode::PermissionDenied, .domain = lfs::ErrorDomain::IO, .detail = "consumer denied frame", .detection = LFS_SOURCE_SITE_CURRENT(), .native = lfs::NativeError{lfs::ErrorDomain::IO, -123, "consumer status"}}));
        }
        void abort(const SinkSummary&) noexcept override { ++aborts; }
    } rejected;
    const auto rejected_result = MediaIngest::extract(request, rejected);
    require(!rejected_result && rejected_result.error().code() == lfs::ErrorCode::PermissionDenied &&
                rejected_result.error().native()->code == -123 && rejected.aborts == 1,
            "structured sink/native error preserved");
    FileExtraction files;
    files.files.output_directory = output;
    files.write_metadata = true;
    const auto file_result = MediaIngest::extractFiles(request, files);
    require(file_result && file_result->frames_accepted == 4 && std::filesystem::is_regular_file(output / "extraction_metadata.json"), "external file consumer");
    require(!std::filesystem::exists(output / "unexpected"), "no extra output policy");
    return 0;
}
#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
    try {
        require(argc == 3, "expected source and destination");
        return exercise(std::filesystem::path(argv[1]), std::filesystem::path(argv[2]));
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
