// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/path_utils.hpp"
#include "io/media/media_ingest.hpp"
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
}
#include <charconv>
#include <cmath>
#include <csignal>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace {
    using nlohmann::json;
    using namespace lfs::media;
    volatile std::sig_atomic_t cancellation = 0;
    void interrupt(int) { cancellation = 1; }
    json errorJson(const lfs::Error& error) {
        json result{{"code", lfs::to_string(error.code())}, {"domain", lfs::to_string(error.domain())}, {"message", error.detail()}};
        result["context"] = json::array();
        for (const auto& frame : error.frames()) {
            json fields = json::object();
            for (const auto& field : frame.fields.entries())
                std::visit([&](const auto& value) {
                    using T = std::decay_t<decltype(value)>;
                    if constexpr (std::is_same_v<T, std::monostate>)
                        fields[field.key] = nullptr;
                    else
                        fields[field.key] = value;
                },
                           field.value);
            result["context"].push_back({{"operation", frame.operation}, {"fields", fields}});
        }
        if (error.native())
            result["native"] = {{"domain", lfs::to_string(error.native()->domain)}, {"code", error.native()->code}, {"name", error.native()->name}};
        return result;
    }
    int failed(const lfs::Error& error) {
        std::cout << json{{"schema_version", 1}, {"success", false}, {"error", errorJson(error)}}.dump() << '\n';
        if (error.code() == lfs::ErrorCode::Cancelled)
            return 130;
        if (error.code() == lfs::ErrorCode::InvalidArgument)
            return 2;
        if (error.code() == lfs::ErrorCode::Unsupported || error.code() == lfs::ErrorCode::Unavailable ||
            error.code() == lfs::ErrorCode::NotFound || error.code() == lfs::ErrorCode::PermissionDenied || error.code() == lfs::ErrorCode::DataLoss)
            return 3;
        return 4;
    }
    template <class T>
    T number(const std::string& text) {
        T value{};
        const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
        if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
            throw std::invalid_argument("Invalid numeric argument: " + text);
        if constexpr (std::is_floating_point_v<T>)
            if (!std::isfinite(value))
                throw std::invalid_argument("Numeric arguments must be finite");
        return value;
    }
    json rational(const std::optional<Rational>& value) {
        return value ? json::array({value->numerator, value->denominator}) : json(nullptr);
    }
    json timestamp(const std::optional<Timestamp>& value) {
        return value ? json{{"ticks", value->ticks}, {"time_base", {value->time_base.numerator, value->time_base.denominator}}} : json(nullptr);
    }
    template <class T>
    json optional(const std::optional<T>& value) { return value ? json(*value) : json(nullptr); }
    const char* streamKind(StreamKind kind) {
        switch (kind) {
        case StreamKind::Video: return "video";
        case StreamKind::Audio: return "audio";
        case StreamKind::Subtitle: return "subtitle";
        case StreamKind::Data: return "data";
        case StreamKind::Attachment: return "attachment";
        default: return "unknown";
        }
    }
    const char* orientationSource(OrientationSource source) {
        switch (source) {
        case OrientationSource::RotateTag: return "rotate_tag";
        case OrientationSource::DisplayMatrix: return "display_matrix";
        default: return "none";
        }
    }
    json description(const MediaDescription& media) {
        json result{{"container", media.container}, {"stream_info_probed", media.stream_info_probed}, {"start", timestamp(media.start)}, {"duration", timestamp(media.duration)}, {"selected_video_stream", optional(media.selected_video_stream)}, {"streams", json::array()}};
        for (const auto& s : media.streams) {
            result["streams"].push_back({{"index", s.index}, {"kind", streamKind(s.kind)}, {"codec", s.codec}, {"width", optional(s.width)}, {"height", optional(s.height)}, {"pixel_format", optional(s.pixel_format)}, {"time_base", rational(s.time_base)}, {"nominal_frame_rate", rational(s.nominal_frame_rate)}, {"average_frame_rate", rational(s.average_frame_rate)}, {"sample_aspect_ratio", rational(s.sample_aspect_ratio)}, {"start", timestamp(s.start)}, {"duration", timestamp(s.duration)}, {"declared_frame_count", optional(s.declared_frame_count)}, {"attached_picture", s.attached_picture}, {"default_disposition", s.default_disposition}, {"color", {{"primaries", optional(s.color.primaries)}, {"transfer", optional(s.color.transfer)}, {"matrix", optional(s.color.matrix)}, {"range", optional(s.color.range)}, {"component_depth", optional(s.color.component_depth)}}}, {"orientation", {{"source", orientationSource(s.orientation.source)}, {"rotate_tag", optional(s.orientation.rotate_tag)}, {"display_matrix", optional(s.orientation.display_matrix)}, {"clockwise_degrees", optional(s.orientation.clockwise_degrees)}, {"reflected", optional(s.orientation.reflected)}}}});
        }
        return result;
    }
    int run(const std::vector<std::string>& args) {
        try {
            if (args.size() == 2 && (args[1] == "--help" || args[1] == "-h")) {
                std::cout << "media-ingest version\nmedia-ingest capabilities\nmedia-ingest probe INPUT [--headers-only] [--timeout-ms N]\n"
                             "media-ingest extract INPUT --output DIRECTORY [--fps N | --interval N]\n"
                             "  [--start SECONDS] [--end SECONDS] [--rotate 0|90|180|270]\n"
                             "  [--scale N | --size WIDTH HEIGHT] [--format png|jpeg] [--quality N]\n"
                             "  [--name PATTERN] [--metadata] [--sharpness THRESHOLD] [--window]\n"
                             "  [--algorithm laplacian|tenengrad|combined] [--candidates N] [--quiet]\n";
                return 0;
            }
            if (args.size() == 2 && args[1] == "version") {
                std::cout << json{{"schema_version", 1}, {"success", true}, {"version", "0.1.0"}, {"ffmpeg", av_version_info()}, {"ffmpeg_license", avcodec_license()}, {"ffmpeg_configuration", avcodec_configuration()}}.dump() << '\n';
                return 0;
            }
            if (args.size() == 2 && args[1] == "capabilities") {
                const auto c = MediaIngest::capabilities();
                std::cout << json{{"schema_version", 1}, {"success", true}, {"software_decode", c.software_decode}, {"rgb8", c.rgb8}, {"png", c.png}, {"jpeg", c.jpeg}, {"hardware_decode", c.hardware_decode}, {"hdr_to_sdr", c.hdr_to_sdr}}.dump() << '\n';
                return 0;
            }
            if (args.size() < 3 || (args[1] != "probe" && args[1] != "extract"))
                throw std::invalid_argument("Expected probe INPUT, extract INPUT --output DIRECTORY, or capabilities; use --help");
            IngestRequest request;
            request.input = lfs::core::utf8_to_path(args[2]);
            FileExtraction output;
            ProbeOptions probe;
            bool quiet = false, selection_set = false, geometry_set = false;
            for (size_t i = 3; i < args.size(); ++i) {
                const auto& flag = args[i];
                auto value = [&]() -> const std::string& { if(i+1>=args.size()) throw std::invalid_argument("Missing value for "+flag); return args[++i]; };
                if (args[1] == "probe") {
                    if (flag == "--headers-only")
                        probe.depth = ProbeDepth::Headers;
                    else if (flag == "--timeout-ms")
                        probe.timeout = std::chrono::milliseconds(number<int64_t>(value()));
                    else
                        throw std::invalid_argument("Unknown probe option: " + flag);
                    continue;
                }
                if (flag == "--output")
                    output.files.output_directory = lfs::core::utf8_to_path(value());
                else if (flag == "--fps" || flag == "--interval") {
                    if (selection_set)
                        throw std::invalid_argument("Specify FPS or interval once");
                    selection_set = true;
                    request.selection.mode = flag == "--fps" ? SelectionMode::FPS : SelectionMode::Interval;
                    if (flag == "--fps")
                        request.selection.fps = number<double>(value());
                    else
                        request.selection.interval = number<int>(value());
                } else if (flag == "--scale" || flag == "--size") {
                    if (geometry_set)
                        throw std::invalid_argument("Specify scale or size once");
                    geometry_set = true;
                    request.geometry.mode = flag == "--scale" ? ResizeMode::Scale : ResizeMode::Custom;
                    if (flag == "--scale")
                        request.geometry.scale = number<float>(value());
                    else {
                        request.geometry.width = number<int>(value());
                        request.geometry.height = number<int>(value());
                    }
                } else if (flag == "--start")
                    request.start_seconds = number<double>(value());
                else if (flag == "--end")
                    request.end_seconds = number<double>(value());
                else if (flag == "--rotate")
                    request.geometry.clockwise_rotation = number<int>(value());
                else if (flag == "--quality")
                    output.files.jpeg_quality = number<int>(value());
                else if (flag == "--name")
                    output.files.filename_pattern = value();
                else if (flag == "--format") {
                    const auto& format = value();
                    if (format != "png" && format != "jpeg" && format != "jpg")
                        throw std::invalid_argument("Format must be png or jpeg");
                    output.files.format = format == "png" ? FrameFileFormat::PNG : FrameFileFormat::JPEG;
                } else if (flag == "--metadata")
                    output.write_metadata = true;
                else if (flag == "--quiet")
                    quiet = true;
                else if (flag == "--hdr-to-sdr")
                    request.convert_hdr_to_sdr = true;
                else if (flag == "--window") {
                    request.sharpness.enabled = true;
                    request.sharpness.window = true;
                } else if (flag == "--sharpness") {
                    request.sharpness.enabled = true;
                    request.sharpness.threshold = number<double>(value());
                } else if (flag == "--candidates")
                    request.sharpness.window_candidates = number<int>(value());
                else if (flag == "--algorithm") {
                    const auto& method = value();
                    if (method == "laplacian")
                        request.sharpness.method = SharpnessMethod::Laplacian;
                    else if (method == "tenengrad")
                        request.sharpness.method = SharpnessMethod::Tenengrad;
                    else if (method == "combined")
                        request.sharpness.method = SharpnessMethod::Combined;
                    else
                        throw std::invalid_argument("Unknown sharpness algorithm");
                } else
                    throw std::invalid_argument("Unknown extract option: " + flag);
            }
            if (args[1] == "probe") {
                const auto result = MediaProbe::inspect(request.input, probe);
                if (!result)
                    return failed(result.error());
                std::cout << json{{"schema_version", 1}, {"success", true}, {"media", description(*result)}}.dump() << '\n';
                return 0;
            }
            cancellation = 0;
            std::signal(SIGINT, interrupt);
            request.cancelled = [] { return cancellation != 0; };
            if (!quiet)
                request.progress = [](const IngestProgress& p) {
                    std::cerr << json{{"event", "progress"}, {"processed", p.processed}, {"estimated", p.estimated}, {"discarded", p.discarded}}.dump() << '\n';
                };
            const auto result = MediaIngest::extractFiles(request, output);
            if (!result)
                return failed(result.error());
            std::cout << json{{"schema_version", 1}, {"success", true}, {"frames_accepted", result->frames_accepted}, {"discarded", result->discarded}}.dump() << '\n';
            return 0;
        } catch (const std::invalid_argument& error) {
            return failed(lfs::make_error({.code = lfs::ErrorCode::InvalidArgument, .domain = lfs::ErrorDomain::IO, .detail = error.what(), .detection = LFS_SOURCE_SITE_CURRENT()}));
        } catch (const std::exception& error) {
            return failed(lfs::make_error({.code = lfs::ErrorCode::Internal, .domain = lfs::ErrorDomain::IO, .detail = error.what(), .detection = LFS_SOURCE_SITE_CURRENT()}));
        }
    }
} // namespace
#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
    std::vector<std::string> args;
    for (int i = 0; i < argc; ++i)
        args.push_back(lfs::core::path_to_utf8(std::filesystem::path(argv[i])));
    return run(args);
}
#else
int main(int argc, char** argv) { return run(std::vector<std::string>(argv, argv + argc)); }
#endif
