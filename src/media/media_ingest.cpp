// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "media/media_ingest.hpp"
#include "media/media_options.hpp"
#include "media/video_frame_extractor.hpp"
#include "media_backends.hpp"
#include <cmath>
#include <exception>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
}
#ifdef __APPLE__
extern "C" {
#include <libavutil/hwcontext.h>
}
#endif

namespace lfs::media {
    namespace {
        Error failure(ErrorCode code, std::string text, std::size_t accepted) {
            return make_error({.code = code, .domain = ErrorDomain::IO, .detail = std::move(text), .detection = LFS_SOURCE_SITE_CURRENT(), .fields = SmallFields{}.add("operation", std::string_view("MediaIngest.extract")).add("frames_accepted", static_cast<uint64_t>(accepted))});
        }
        io::VideoFrameExtractor::Params parameters(const IngestRequest& request) {
            io::VideoFrameExtractor::Params p;
            p.video_path = request.input;
            p.mode = request.selection.mode == SelectionMode::FPS ? io::ExtractionMode::FPS : io::ExtractionMode::INTERVAL;
            p.fps = request.selection.fps;
            p.frame_interval = request.selection.interval;
            p.start_time = request.start_seconds;
            p.end_time = request.end_seconds;
            p.resolution_mode = request.geometry.mode == ResizeMode::Original ? io::ResolutionMode::Original
                                : request.geometry.mode == ResizeMode::Scale  ? io::ResolutionMode::Scale
                                                                              : io::ResolutionMode::Custom;
            p.scale = request.geometry.scale;
            p.custom_width = request.geometry.width;
            p.custom_height = request.geometry.height;
            p.rotation = request.geometry.clockwise_rotation;
            p.sharpness.enabled = request.sharpness.enabled;
            p.sharpness.algorithm = request.sharpness.method == SharpnessMethod::Laplacian   ? io::SharpnessAlgorithm::LAPLACIAN
                                    : request.sharpness.method == SharpnessMethod::Tenengrad ? io::SharpnessAlgorithm::TENENGRAD
                                                                                             : io::SharpnessAlgorithm::COMBINED;
            p.sharpness.threshold = request.sharpness.threshold;
            p.sharpness.window_mode = request.sharpness.window;
            p.sharpness.window_candidates_target = request.sharpness.window_candidates;
            p.cancel_requested = request.cancelled;
            p.allow_hardware_decode = request.allow_hardware_decode;
            p.convert_hdr_to_sdr = request.convert_hdr_to_sdr;
            p.output_format = request.output_format;
            p.input_color = request.input_color;
            return p;
        }
        struct TrackingSink final : FrameSink {
            FrameSink& destination;
            std::optional<Error> last_error;
            std::size_t accepted = 0;
            explicit TrackingSink(FrameSink& value) : destination(value) {}
            SinkResult track(SinkResult value) {
                if (!value)
                    last_error = value.error();
                return value;
            }
            SinkResult begin(const SinkSession& s) override { return track(destination.begin(s)); }
            SinkResult write(const FrameView& v) override {
                auto result = track(destination.write(v));
                if (result)
                    ++accepted;
                return result;
            }
            SinkResult complete(const SinkSummary& s) override {
                accepted = s.frames_accepted;
                return track(destination.complete(s));
            }
            void abort(const SinkSummary& s) noexcept override {
                accepted = s.frames_accepted;
                destination.abort(s);
            }
        };
        Result<IngestReport> execute(const IngestRequest& request, FrameSink* sink, const FileExtraction* files) {
            std::size_t accepted = 0;
            try {
                if (request.convert_hdr_to_sdr && !detail::hasHdrBackend())
                    return failure(ErrorCode::Unsupported, "HDR to SDR is unavailable in this host", 0);
                if (request.input.empty() || request.input.native().find(std::filesystem::path::value_type{}) != std::filesystem::path::string_type::npos)
                    return failure(ErrorCode::InvalidArgument, "Input path must be nonempty and contain no NUL", 0);
                if (request.selection.mode != SelectionMode::FPS && request.selection.mode != SelectionMode::Interval)
                    return failure(ErrorCode::InvalidArgument, "Invalid frame selection mode", 0);
                if (request.geometry.mode != ResizeMode::Original && request.geometry.mode != ResizeMode::Scale && request.geometry.mode != ResizeMode::Custom)
                    return failure(ErrorCode::InvalidArgument, "Invalid resize mode", 0);
                if (request.sharpness.method != SharpnessMethod::Laplacian && request.sharpness.method != SharpnessMethod::Tenengrad && request.sharpness.method != SharpnessMethod::Combined)
                    return failure(ErrorCode::InvalidArgument, "Invalid sharpness method", 0);
                if (request.output_format != FramePixelFormat::RGB8 && request.output_format != FramePixelFormat::RGBFloat32)
                    return failure(ErrorCode::Unsupported, "Ingest supports RGB8 or integer SDR to linear RGBFloat32", 0);
                if (outputOptionName(request.input_color.transfer) == "unknown" ||
                    outputOptionName(request.input_color.primaries) == "unknown" || request.input_color.alpha != AlphaMode::None)
                    return failure(ErrorCode::InvalidArgument, "Invalid input color override", 0);
                if (request.output_format == FramePixelFormat::RGB8 && (request.input_color.transfer != ColorTransfer::Unspecified || request.input_color.primaries != ColorPrimaries::Unspecified))
                    return failure(ErrorCode::InvalidArgument, "Input color overrides apply only to float SDR output", 0);
                auto p = parameters(request);
                if (files) {
                    p.output_dir = files->files.output_directory;
                    p.filename_pattern = files->files.filename_pattern;
                    p.format = files->files.format == FrameFileFormat::PNG ? io::ImageFormat::PNG : files->files.format == FrameFileFormat::EXR ? io::ImageFormat::EXR
                                                                                                                                                : io::ImageFormat::JPG;
                    p.jpg_quality = files->files.jpeg_quality;
                    p.generate_metadata = files->write_metadata || files->files.format == FrameFileFormat::EXR;
                    p.overwrite_metadata = files->files.exr.overwrite;
                    p.exr_precision = files->files.exr.precision;
                    p.exr_compression = files->files.exr.compression;
                    if ((files->files.format == FrameFileFormat::EXR) != (request.output_format == FramePixelFormat::RGBFloat32))
                        return failure(ErrorCode::InvalidArgument, "EXR file extraction requires float output; PNG/JPEG require RGB8", 0);
                    if (files->files.format == FrameFileFormat::EXR) {
                        if (auto valid = validateExrOutputOptions(files->files.exr); !valid)
                            return std::move(valid).error();
                    }
                    if (files->files.output_directory.empty() ||
                        files->files.output_directory.native().find(std::filesystem::path::value_type{}) != std::filesystem::path::string_type::npos ||
                        files->files.filename_pattern.find('\0') != std::string::npos ||
                        (files->files.format != FrameFileFormat::PNG && files->files.format != FrameFileFormat::JPEG && files->files.format != FrameFileFormat::EXR))
                        return failure(ErrorCode::InvalidArgument, "File extraction requires a nonempty output directory, supported format and NUL-free paths/names", 0);
                }
                // Request-only validation reuses the same rules as the compatibility
                // adapter; the decoder validates actual source layout/timebase later.
                io::VideoFrameExtractor::ValidatedLayout layout;
                std::string error;
                auto request_validation = p;
                if (request_validation.resolution_mode == io::ResolutionMode::Scale) {
                    if (!std::isfinite(p.scale) || p.scale <= 0)
                        return failure(ErrorCode::InvalidArgument, "Scale must be finite and positive", 0);
                    request_validation.resolution_mode = io::ResolutionMode::Original;
                }
                if (!io::VideoFrameExtractor::validateParams(request_validation, 1, 1, 1.0, layout, error))
                    return failure(ErrorCode::InvalidArgument, error, 0);
                int discarded = 0;
                p.progress_callback = [&](int current, int estimated, int skipped) {
                    discarded = skipped;
                    if (request.progress)
                        request.progress({current, estimated, skipped});
                };
                io::VideoFrameExtractor engine;
                auto file_options = files ? files->files : FileFrameSinkOptions{};
                if (files && files->files.format == FrameFileFormat::EXR) {
                    const auto sink_cancel = files->files.exr.cancelled;
                    file_options.exr.cancelled = [&, sink_cancel] { return (request.cancelled && request.cancelled()) || (sink_cancel && sink_cancel()); };
                    p.cancel_requested = file_options.exr.cancelled;
                }
                FileFrameSink file_sink(std::move(file_options));
                TrackingSink tracked(sink ? *sink : static_cast<FrameSink&>(file_sink));
                const bool ok = sink ? engine.extractToSink(p, tracked, error) : engine.extractFilesToSink(p, tracked, error);
                accepted = tracked.accepted;
                if (!ok) {
                    if (tracked.last_error)
                        return std::move(*tracked.last_error).with_context("MediaIngest.extract", LFS_SOURCE_SITE_CURRENT(), SmallFields{}.add("frames_accepted", static_cast<uint64_t>(accepted)));
                    if (auto typed = engine.lastError())
                        return std::move(*typed).with_context("MediaIngest.extract", LFS_SOURCE_SITE_CURRENT(), SmallFields{}.add("frames_accepted", static_cast<uint64_t>(accepted)));
                    return failure(engine.lastOutcome() == io::ExtractionOutcome::Cancelled ? ErrorCode::Cancelled : ErrorCode::Unavailable, error, accepted);
                }
                return IngestReport{accepted, discarded};
            } catch (const std::exception& error) {
                // LFS-CENSUS-OK(empty-catch): failure() builds a core Error with operation and accepted-frame context; exceptions never become success.
                return failure(ErrorCode::Internal, error.what(), accepted);
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): Public API converts non-standard host/callback failures into a structured Internal error.
                return failure(ErrorCode::Internal, "Unknown Media Ingest failure", accepted);
            }
        }
    } // namespace
    Result<MediaDescription> MediaIngest::probe(const std::filesystem::path& path, const ProbeOptions& options) {
        return MediaProbe::inspect(path, options);
    }
    IngestCapabilities MediaIngest::capabilities() noexcept {
        IngestCapabilities value;
        value.hardware_decode = detail::hasGpuJpegBackend();
#ifdef __APPLE__
        value.hardware_decode = av_hwdevice_find_type_by_name("videotoolbox") != AV_HWDEVICE_TYPE_NONE;
#endif
        value.hdr_to_sdr = detail::hasHdrBackend();
        return value;
    }
    Result<IngestReport> MediaIngest::extract(const IngestRequest& request, FrameSink& sink) { return execute(request, &sink, nullptr); }
    CodecBuildInfo MediaIngest::codecBuildInfo() { return {av_version_info(), avcodec_license(), avcodec_configuration()}; }
    Result<IngestReport> MediaIngest::extractFiles(const IngestRequest& request, const FileExtraction& files) { return execute(request, nullptr, &files); }
} // namespace lfs::media
