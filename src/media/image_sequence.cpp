// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "media/image_sequence.hpp"
#include "media/media_options.hpp"
#include "media/video_frame_extractor.hpp"
#include <exception>
#include <limits>

namespace lfs::media {
    namespace {
        SinkResult sequenceError(ErrorCode code, std::string detail) {
            return SinkResult::failure(make_error({.code = code, .domain = ErrorDomain::IO, .detail = std::move(detail), .detection = LFS_SOURCE_SITE_CURRENT()}));
        }
        template <class Fn>
        SinkResult boundary(Fn&& fn) {
            try {
                return fn();
            } catch (const Exception& error) {
                return SinkResult::failure(error.error());
            } catch (const std::bad_alloc&) {
                return sequenceError(ErrorCode::ResourceExhausted, "Image sequence allocation failed");
            } catch (const std::exception& error) {
                // LFS-CENSUS-OK(empty-catch): Preserve the exception detail in a structured I/O error.
                return sequenceError(ErrorCode::Internal, std::string("Image sequence failed: ") + error.what());
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): Nonstandard callback failures terminate the sequence with a typed error.
                return sequenceError(ErrorCode::Internal, "Image sequence failed with a nonstandard exception");
            }
        }
        SinkResult validate(const ImageSequenceOptions& options) {
            const auto& files = options.files;
            const auto& layout = options.layout;
            if (files.output_directory.empty() ||
                files.output_directory.native().find(std::filesystem::path::value_type{}) != std::filesystem::path::string_type::npos ||
                (files.numbering != FrameFileNumbering::SourceFrame && files.numbering != FrameFileNumbering::DeliveryIndex) ||
                outputOptionName(files.format) == "unknown")
                return sequenceError(ErrorCode::InvalidArgument, "Invalid image sequence output directory or format");
            if (options.frame_count <= 0 || options.framerate <= 0 || options.framerate > 1000 ||
                files.first_source_frame_number < 0 ||
                options.frame_count - 1 > std::numeric_limits<int>::max() - files.first_source_frame_number)
                return sequenceError(ErrorCode::InvalidArgument, "Invalid image sequence count, rate or numbering");
            // Keep renderer cleanup independent of extraction metadata and files.
            if (files.remove_stale_frames || files.preserve_metadata)
                return sequenceError(ErrorCode::InvalidArgument, "Image sequences do not own extraction metadata or stale frames");
            if (files.filename_pattern.empty() || files.filename_pattern.find_first_of("/\\:") != std::string::npos ||
                files.filename_pattern.find('\0') != std::string::npos ||
                io::formatFrameFilenameStem(files.filename_pattern, 1) == io::formatFrameFilenameStem(files.filename_pattern, 2))
                return sequenceError(ErrorCode::InvalidArgument, "Image sequence pattern requires a numbered filename without directory components");
            const bool rgb8 = layout.format == FramePixelFormat::RGB8;
            const bool float_rgb = layout.format == FramePixelFormat::RGBFloat32;
            const bool float_rgba = layout.format == FramePixelFormat::RGBAFloat32;
            if (layout.width <= 0 || layout.height <= 0 || (!rgb8 && !float_rgb && !float_rgba) ||
                outputOptionName(layout.color.units) == "unknown" ||
                (layout.color.primaries != ColorPrimaries::Bt709 && layout.color.primaries != ColorPrimaries::Bt2020) ||
                (float_rgba ? layout.color.alpha != AlphaMode::Independent && layout.color.alpha != AlphaMode::Premultiplied : layout.color.alpha != AlphaMode::None))
                return sequenceError(ErrorCode::InvalidArgument, "Image sequence requires declared RGB/RGBA color and matching alpha");
            if (files.format != FrameFileFormat::EXR && layout.color.units != ColorUnits::Relative)
                return sequenceError(ErrorCode::Unsupported, "Only EXR sequence output supports absolute or scene-relative units");
            if (rgb8 && (layout.color.transfer != ColorTransfer::Srgb || layout.color.primaries != ColorPrimaries::Bt709))
                return sequenceError(ErrorCode::Unsupported, "RGB8 sequence input requires sRGB BT709");
            if (!rgb8 && layout.color.transfer != ColorTransfer::Linear)
                return sequenceError(ErrorCode::Unsupported, "Float sequence input requires linear color");
            if ((files.format == FrameFileFormat::PNG || files.format == FrameFileFormat::JPEG) && !rgb8)
                return sequenceError(ErrorCode::Unsupported, "PNG/JPEG sequence input requires RGB8");
            if ((files.format == FrameFileFormat::EXR || files.format == FrameFileFormat::TIFF16) && rgb8)
                return sequenceError(ErrorCode::Unsupported, "EXR/TIFF16 sequence input requires linear float samples");
            if (files.format == FrameFileFormat::JXL && files.jxl.precision == JxlPrecision::UInt16 && rgb8)
                return sequenceError(ErrorCode::Unsupported, "JXL UInt16 sequence input requires linear float samples");
            if (files.format == FrameFileFormat::AVIF && rgb8 && files.avif.bit_depth != 8)
                return sequenceError(ErrorCode::Unsupported, "High-depth AVIF sequence input requires linear float samples");
            const auto row_bytes = static_cast<std::uint64_t>(layout.width) * pixelBytes(layout.format);
            if (row_bytes > std::numeric_limits<std::size_t>::max() || layout.row_stride < row_bytes ||
                static_cast<std::uint64_t>(layout.height) > std::numeric_limits<std::size_t>::max() / layout.row_stride)
                return sequenceError(ErrorCode::InvalidArgument, "Invalid image sequence stride or byte extent");
            switch (files.format) {
            case FrameFileFormat::EXR: return validateExrOutputOptions(files.exr);
            case FrameFileFormat::TIFF16: return validateTiff16OutputOptions(files.tiff16);
            case FrameFileFormat::JXL: return validateJxlOutputOptions(files.jxl);
            case FrameFileFormat::AVIF: return validateAvifOutputOptions(files.avif);
            default: break;
            }
            return {};
        }
    } // namespace

    SinkResult validateImageSequenceOptions(const ImageSequenceOptions& options) {
        return boundary([&] { return validate(options); });
    }

    class ImageSequence::Impl {
    public:
        ImageSequenceOptions options;
        std::unique_ptr<FileFrameSink> sink;
        std::size_t accepted = 0;
        std::optional<Error> failure;
        bool busy = false;
        bool aborted = false;
        bool abort_requested = false;

        bool cancelled() {
            return abort_requested || (options.cancelled && options.cancelled()) || abort_requested;
        }

        void abort() noexcept {
            if (sink) {
                aborted = true;
                sink->abort({.outcome = SinkOutcome::Failed, .frames_accepted = accepted});
                sink.reset();
            }
        }
        SinkResult open(const ImageSequenceOptions& request) {
            if (sink)
                return sequenceError(ErrorCode::FailedPrecondition, "Image sequence already open");
            if (auto valid = validate(request); !valid)
                return valid;
            options = request;
            accepted = 0;
            failure.reset();
            aborted = false;
            abort_requested = false;
            if (cancelled())
                return sequenceError(ErrorCode::Cancelled, "Image sequence cancelled before opening");
            auto files = options.files;
            files.numbering = FrameFileNumbering::DeliveryIndex;
            // Preserve each writer's own callback while adding the job token.
            const auto combine = [this, job = options.cancelled](const std::function<bool()>& writer) {
                return [this, job, writer] { return abort_requested || (job && job()) || (writer && writer()) || abort_requested; };
            };
            files.image_output.cancelled = combine(files.image_output.cancelled);
            files.exr.cancelled = combine(files.exr.cancelled);
            files.tiff16.cancelled = combine(files.tiff16.cancelled);
            files.jxl.cancelled = combine(files.jxl.cancelled);
            files.avif.cancelled = combine(files.avif.cancelled);
            sink = std::make_unique<FileFrameSink>(std::move(files));
            return sink->begin({.output = options.layout});
        }
        SinkResult write(const FrameView& input) {
            if (!sink)
                return sequenceError(ErrorCode::FailedPrecondition, "Image sequence is not open");
            if (accepted >= static_cast<std::size_t>(options.frame_count))
                return sequenceError(ErrorCode::BoundsViolation, "Image sequence has received all requested frames");
            if (cancelled())
                return sequenceError(ErrorCode::Cancelled, "Image sequence cancelled before writing");
            const auto& actual = input.layout;
            const auto& expected = options.layout;
            if (actual.width != expected.width || actual.height != expected.height || actual.format != expected.format ||
                actual.color.transfer != expected.color.transfer || actual.color.primaries != expected.color.primaries ||
                actual.color.alpha != expected.color.alpha || actual.color.units != expected.color.units)
                return sequenceError(ErrorCode::InvalidArgument, "Image sequence frame layout or color changed");
            if (auto required = input.requiredBytes(); !required)
                return SinkResult::failure(required.error());
            auto frame = input;
            frame.info = {.delivery_index = accepted,
                          .relative_seconds = static_cast<double>(accepted) / options.framerate,
                          .origin = FrameOrigin::Rendered,
                          .output_timestamp = Timestamp{static_cast<int64_t>(accepted), {1, options.framerate}}};
            if (auto written = sink->write(frame); !written)
                return written;
            ++accepted;
            return {};
        }
        SinkResult close() {
            if (failure)
                return SinkResult::failure(*failure);
            if (aborted)
                return sequenceError(ErrorCode::FailedPrecondition, "Image sequence was aborted");
            if (!sink)
                return {};
            if (cancelled())
                return sequenceError(ErrorCode::Cancelled, "Image sequence cancelled before completion");
            if (accepted != static_cast<std::size_t>(options.frame_count))
                return sequenceError(ErrorCode::FailedPrecondition, "Image sequence is incomplete");
            const SinkSummary summary{.outcome = SinkOutcome::Completed, .frames_accepted = accepted};
            if (auto prepared = sink->prepareComplete(summary); !prepared)
                return prepared;
            if (auto completed = sink->complete(summary); !completed)
                return completed;
            sink.reset();
            return {};
        }
    };

    ImageSequence::ImageSequence() : impl_(std::make_unique<Impl>()) {}
    ImageSequence::~ImageSequence() { abort(); }
    SinkResult ImageSequence::open(const ImageSequenceOptions& options) {
        if (impl_->busy || impl_->sink)
            return sequenceError(ErrorCode::FailedPrecondition, "Image sequence already active");
        impl_->busy = true;
        auto result = boundary([&] { return impl_->open(options); });
        impl_->busy = false;
        if (!result) {
            impl_->failure = result.error();
            impl_->abort();
        }
        return result;
    }
    SinkResult ImageSequence::write(const FrameView& frame) {
        if (impl_->busy)
            return sequenceError(ErrorCode::FailedPrecondition, "Image sequence callback is reentrant");
        impl_->busy = true;
        auto result = boundary([&] { return impl_->write(frame); });
        impl_->busy = false;
        if (!result) {
            impl_->failure = result.error();
            impl_->abort();
        }
        return result;
    }
    SinkResult ImageSequence::close() {
        if (impl_->busy)
            return sequenceError(ErrorCode::FailedPrecondition, "Image sequence callback is reentrant");
        impl_->busy = true;
        auto result = boundary([&] { return impl_->close(); });
        impl_->busy = false;
        if (!result) {
            impl_->failure = result.error();
            impl_->abort();
        }
        return result;
    }
    void ImageSequence::abort() noexcept {
        if (impl_->busy)
            impl_->abort_requested = true;
        else
            impl_->abort();
    }
    bool ImageSequence::isOpen() const noexcept { return impl_->sink != nullptr; }
    std::size_t ImageSequence::framesAccepted() const noexcept { return impl_->accepted; }
} // namespace lfs::media
