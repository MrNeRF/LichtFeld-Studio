// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "py_media.hpp"
#include "io/media/media_ingest.hpp"
#include <exception>
#include <memory>
#include <nanobind/stl/array.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
namespace nb = nanobind;
namespace lfs::python {
    namespace {
        struct Callbacks {
            nb::object progress, cancelled;
            std::exception_ptr error;
            Callbacks(nb::object p, nb::object c) : progress(std::move(p)), cancelled(std::move(c)) {}
        };
        media::IngestRequest withCallbacks(const media::IngestRequest& request, const std::shared_ptr<Callbacks>& callbacks) {
            auto copy = request;
            if (!callbacks->progress.is_none())
                copy.progress = [callbacks](const media::IngestProgress& value) {
                    nb::gil_scoped_acquire gil;
                    try {
                        callbacks->progress(value);
                    } catch (...) {
                        callbacks->error = std::current_exception();
                        throw std::runtime_error("Python progress callback failed");
                    }
                };
            if (!callbacks->cancelled.is_none())
                copy.cancelled = [callbacks] {
                    nb::gil_scoped_acquire gil;
                    try {
                        return nb::cast<bool>(callbacks->cancelled());
                    } catch (...) {
                        // LFS-CENSUS-OK(empty-catch): request native cancellation, then rethrow the captured Python exception after restoring the caller's GIL.
                        callbacks->error = std::current_exception();
                        return true;
                    }
                };
            return copy;
        }
    } // namespace
    void register_media(nb::module_& m) {
        using namespace media;
        nb::enum_<SelectionMode>(m, "SelectionMode").value("FPS", SelectionMode::FPS).value("Interval", SelectionMode::Interval);
        nb::enum_<ResizeMode>(m, "ResizeMode").value("Original", ResizeMode::Original).value("Scale", ResizeMode::Scale).value("Custom", ResizeMode::Custom);
        nb::enum_<SharpnessMethod>(m, "SharpnessMethod").value("Laplacian", SharpnessMethod::Laplacian).value("Tenengrad", SharpnessMethod::Tenengrad).value("Combined", SharpnessMethod::Combined);
        nb::enum_<FrameFileFormat>(m, "FrameFileFormat").value("PNG", FrameFileFormat::PNG).value("JPEG", FrameFileFormat::JPEG);
        nb::enum_<StreamKind>(m, "StreamKind").value("Video", StreamKind::Video).value("Audio", StreamKind::Audio).value("Subtitle", StreamKind::Subtitle).value("Data", StreamKind::Data).value("Attachment", StreamKind::Attachment).value("Unknown", StreamKind::Unknown);
        nb::enum_<OrientationSource>(m, "OrientationSource").value("Unspecified", OrientationSource::None).value("RotateTag", OrientationSource::RotateTag).value("DisplayMatrix", OrientationSource::DisplayMatrix);
        nb::enum_<ProbeDepth>(m, "ProbeDepth").value("Headers", ProbeDepth::Headers).value("StreamInfo", ProbeDepth::StreamInfo);
        nb::enum_<FramePixelFormat>(m, "FramePixelFormat").value("RGB8", FramePixelFormat::RGB8);
        nb::enum_<TimestampOrigin>(m, "TimestampOrigin").value("Missing", TimestampOrigin::Missing).value("BestEffort", TimestampOrigin::BestEffort).value("Presentation", TimestampOrigin::Presentation);
        nb::class_<Selection>(m, "Selection").def(nb::init<>()).def_rw("mode", &Selection::mode).def_rw("fps", &Selection::fps).def_rw("interval", &Selection::interval);
        nb::class_<Geometry>(m, "Geometry").def(nb::init<>()).def_rw("mode", &Geometry::mode).def_rw("scale", &Geometry::scale).def_rw("width", &Geometry::width).def_rw("height", &Geometry::height).def_rw("clockwise_rotation", &Geometry::clockwise_rotation);
        nb::class_<Sharpness>(m, "Sharpness").def(nb::init<>()).def_rw("enabled", &Sharpness::enabled).def_rw("method", &Sharpness::method).def_rw("threshold", &Sharpness::threshold).def_rw("window_candidates", &Sharpness::window_candidates).def_rw("window", &Sharpness::window);
        nb::class_<IngestRequest>(m, "IngestRequest").def(nb::init<>()).def_rw("input", &IngestRequest::input).def_rw("selection", &IngestRequest::selection).def_rw("geometry", &IngestRequest::geometry).def_rw("sharpness", &IngestRequest::sharpness).def_rw("start_seconds", &IngestRequest::start_seconds).def_rw("end_seconds", &IngestRequest::end_seconds).def_rw("convert_hdr_to_sdr", &IngestRequest::convert_hdr_to_sdr).def_rw("allow_hardware_decode", &IngestRequest::allow_hardware_decode);
        nb::class_<FileFrameSinkOptions>(m, "FileFrameSinkOptions").def(nb::init<>()).def_rw("output_directory", &FileFrameSinkOptions::output_directory).def_rw("filename_pattern", &FileFrameSinkOptions::filename_pattern).def_rw("format", &FileFrameSinkOptions::format).def_rw("jpeg_quality", &FileFrameSinkOptions::jpeg_quality);
        nb::class_<FileExtraction>(m, "FileExtraction").def(nb::init<>()).def_rw("files", &FileExtraction::files).def_rw("write_metadata", &FileExtraction::write_metadata);
        nb::class_<Rational>(m, "Rational").def_ro("numerator", &Rational::numerator).def_ro("denominator", &Rational::denominator);
        nb::class_<Timestamp>(m, "Timestamp").def_ro("ticks", &Timestamp::ticks).def_ro("time_base", &Timestamp::time_base);
        nb::class_<ColorDescription>(m, "ColorDescription").def_ro("primaries", &ColorDescription::primaries).def_ro("transfer", &ColorDescription::transfer).def_ro("matrix", &ColorDescription::matrix).def_ro("range", &ColorDescription::range).def_ro("component_depth", &ColorDescription::component_depth);
        nb::class_<Orientation>(m, "Orientation").def_ro("source", &Orientation::source).def_ro("rotate_tag", &Orientation::rotate_tag).def_ro("display_matrix", &Orientation::display_matrix).def_ro("clockwise_degrees", &Orientation::clockwise_degrees).def_ro("reflected", &Orientation::reflected);
        nb::class_<StreamDescription>(m, "StreamDescription").def_ro("index", &StreamDescription::index).def_ro("kind", &StreamDescription::kind).def_ro("codec", &StreamDescription::codec).def_ro("width", &StreamDescription::width).def_ro("height", &StreamDescription::height).def_ro("pixel_format", &StreamDescription::pixel_format).def_ro("time_base", &StreamDescription::time_base).def_ro("nominal_frame_rate", &StreamDescription::nominal_frame_rate).def_ro("average_frame_rate", &StreamDescription::average_frame_rate).def_ro("sample_aspect_ratio", &StreamDescription::sample_aspect_ratio).def_ro("start", &StreamDescription::start).def_ro("duration", &StreamDescription::duration).def_ro("declared_frame_count", &StreamDescription::declared_frame_count).def_ro("attached_picture", &StreamDescription::attached_picture).def_ro("default_disposition", &StreamDescription::default_disposition).def_ro("color", &StreamDescription::color).def_ro("orientation", &StreamDescription::orientation);
        nb::class_<MediaDescription>(m, "MediaDescription").def_ro("container", &MediaDescription::container).def_ro("stream_info_probed", &MediaDescription::stream_info_probed).def_ro("start", &MediaDescription::start).def_ro("duration", &MediaDescription::duration).def_ro("streams", &MediaDescription::streams).def_ro("selected_video_stream", &MediaDescription::selected_video_stream);
        nb::class_<FrameLayout>(m, "FrameLayout").def_ro("width", &FrameLayout::width).def_ro("height", &FrameLayout::height).def_ro("row_stride", &FrameLayout::row_stride).def_ro("format", &FrameLayout::format);
        nb::class_<FrameInfo>(m, "FrameInfo").def_ro("source_timestamp", &FrameInfo::source_timestamp).def_ro("timestamp_origin", &FrameInfo::timestamp_origin).def_ro("decode_index", &FrameInfo::decode_index).def_ro("delivery_index", &FrameInfo::delivery_index).def_ro("relative_seconds", &FrameInfo::relative_seconds).def_ro("legacy_source_frame", &FrameInfo::legacy_source_frame).def_ro("sharpness_score", &FrameInfo::sharpness_score);
        nb::class_<IngestProgress>(m, "IngestProgress").def_ro("processed", &IngestProgress::processed).def_ro("estimated", &IngestProgress::estimated).def_ro("discarded", &IngestProgress::discarded);
        nb::class_<IngestReport>(m, "IngestReport").def_ro("frames_accepted", &IngestReport::frames_accepted).def_ro("discarded", &IngestReport::discarded);
        nb::class_<IngestCapabilities>(m, "IngestCapabilities").def_ro("software_decode", &IngestCapabilities::software_decode).def_ro("rgb8", &IngestCapabilities::rgb8).def_ro("png", &IngestCapabilities::png).def_ro("jpeg", &IngestCapabilities::jpeg).def_ro("hardware_decode", &IngestCapabilities::hardware_decode).def_ro("hdr_to_sdr", &IngestCapabilities::hdr_to_sdr);
        nb::class_<CodecBuildInfo>(m, "CodecBuildInfo").def_ro("ffmpeg_version", &CodecBuildInfo::ffmpeg_version).def_ro("ffmpeg_license", &CodecBuildInfo::ffmpeg_license).def_ro("ffmpeg_configuration", &CodecBuildInfo::ffmpeg_configuration);
        nb::class_<FrameSurface>(m, "FrameSurface")
            .def_prop_ro("layout", [](const FrameSurface& frame) { return frame.view().layout; })
            .def_prop_ro("info", [](const FrameSurface& frame) { return frame.view().info; })
            .def_prop_ro("pixels", [](const FrameSurface& frame) {
                const auto pixels = frame.view().pixels;
                return nb::bytes(reinterpret_cast<const char*>(pixels.data()), pixels.size());
            });
        nb::class_<MediaIngest>(m, "MediaIngest")
            .def_static("capabilities", &MediaIngest::capabilities)
            .def_static("codec_build_info", &MediaIngest::codecBuildInfo)
            .def_static("probe", [](const std::filesystem::path& path, bool headers_only, std::int64_t timeout_ms) {
                ProbeOptions options;
                options.depth = headers_only ? ProbeDepth::Headers : ProbeDepth::StreamInfo;
                options.timeout = std::chrono::milliseconds(timeout_ms);
                std::optional<Result<MediaDescription>> result;
                { nb::gil_scoped_release release; result.emplace(MediaIngest::probe(path, options)); }
                if (!*result) throw Exception(result->error());
                return std::move(**result); }, nb::arg("path"), nb::arg("headers_only") = false, nb::arg("timeout_ms") = 10000)
            .def_static("extract_files", [](const IngestRequest& request, const FileExtraction& files, nb::object progress, nb::object cancelled) {
                const auto callbacks = std::make_shared<Callbacks>(std::move(progress), std::move(cancelled));
                const auto copy = withCallbacks(request, callbacks);
                const auto file_copy = files;
                std::optional<Result<IngestReport>> result;
                { nb::gil_scoped_release release; result.emplace(MediaIngest::extractFiles(copy, file_copy)); }
                if (callbacks->error) std::rethrow_exception(callbacks->error);
                if (!*result) throw Exception(result->error());
                return **result; }, nb::arg("request"), nb::arg("files"), nb::arg("progress") = nb::none(), nb::arg("cancelled") = nb::none())
            .def_static("extract", [](const IngestRequest& request, std::size_t payload_budget, std::size_t frame_limit, nb::object progress, nb::object cancelled) {
                const auto callbacks = std::make_shared<Callbacks>(std::move(progress), std::move(cancelled));
                const auto copy = withCallbacks(request, callbacks);
                MemoryFrameSink sink(payload_budget, frame_limit);
                std::optional<Result<IngestReport>> result;
                { nb::gil_scoped_release release; result.emplace(MediaIngest::extract(copy, sink)); }
                if (callbacks->error) std::rethrow_exception(callbacks->error);
                if (!*result) throw Exception(result->error());
                return nb::make_tuple(**result, sink.frames()); }, nb::arg("request"), nb::arg("payload_budget") = 256ULL * 1024 * 1024, nb::arg("frame_limit") = 100000, nb::arg("progress") = nb::none(), nb::arg("cancelled") = nb::none());
    }
} // namespace lfs::python
