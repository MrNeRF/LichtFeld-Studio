// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "resumable_file_sink.hpp"
#include "core/atomic_file.hpp"
#include "core/image_avif.hpp"
#include "core/image_codecs.hpp"
#include "core/image_jxl.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include "git_version.h"
#include "media/media_options.hpp"
#include "media/video_frame_extractor.hpp"
#include "source_metadata_internal.hpp"
#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <memory>
#include <openssl/evp.h>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
}

namespace lfs::media::detail {
    namespace {
        using Json = nlohmann::json;
        Error checkpointError(ErrorCode code, std::string message) {
            return make_error({.code = code, .domain = ErrorDomain::IO, .detail = std::move(message), .detection = LFS_SOURCE_SITE_CURRENT()});
        }
        bool emptyReservation(const std::filesystem::path& path) {
            return lfs::core::native_fs::is_regular_file(lfs::core::native_fs::symlink_status(path)) &&
                   lfs::core::native_fs::file_size(path) == 0;
        }
        SinkResult discardReservation(const std::filesystem::path& path) {
            if (!emptyReservation(path))
                return {};
            std::error_code error;
            lfs::core::native_fs::remove(path, error);
            if (error)
                return SinkResult::failure(checkpointError(ErrorCode::Unavailable, "Could not recover empty checkpoint reservation: " + core::path_to_utf8(path)));
            return {};
        }
        struct Sha256 {
            std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context{EVP_MD_CTX_new(), EVP_MD_CTX_free};
            Sha256() {
                if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
                    throw std::runtime_error("Could not initialize checkpoint SHA-256");
            }
            void update(std::span<const std::uint8_t> bytes) {
                if (EVP_DigestUpdate(context.get(), bytes.data(), bytes.size()) != 1)
                    throw std::runtime_error("Could not update checkpoint SHA-256");
            }
            std::string finish() {
                std::array<std::uint8_t, 32> digest{};
                unsigned int size = 0;
                if (EVP_DigestFinal_ex(context.get(), digest.data(), &size) != 1 || size != digest.size())
                    throw std::runtime_error("Could not finalize checkpoint SHA-256");
                constexpr char hex[] = "0123456789abcdef";
                std::string result;
                result.reserve(64);
                for (auto byte : digest) {
                    result.push_back(hex[byte >> 4]);
                    result.push_back(hex[byte & 15]);
                }
                return result;
            }
        };
        Result<std::string> fileDigest(const std::filesystem::path& path, const std::function<bool()>& cancelled) {
            if (!lfs::core::native_fs::is_regular_file(lfs::core::native_fs::symlink_status(path)))
                return checkpointError(ErrorCode::FailedPrecondition, "Checkpoint requires a regular, non-symlink file: " + core::path_to_utf8(path));
            std::ifstream input(lfs::core::native_fs::native_path(path), std::ios::binary);
            if (!input)
                return checkpointError(ErrorCode::Unavailable, "Could not read checkpoint input: " + core::path_to_utf8(path));
            Sha256 digest;
            std::array<std::uint8_t, 65536> buffer{};
            while (input) {
                if (cancelled && cancelled())
                    return checkpointError(ErrorCode::Cancelled, "Checkpoint verification cancelled");
                input.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
                digest.update(std::span(buffer.data(), static_cast<std::size_t>(input.gcount())));
            }
            if (!input.eof())
                return checkpointError(ErrorCode::Unavailable, "Checkpoint input read failed: " + core::path_to_utf8(path));
            return digest.finish();
        }
        Result<Json> readRecord(const std::filesystem::path& path) {
            if (!lfs::core::native_fs::is_regular_file(lfs::core::native_fs::symlink_status(path)) || lfs::core::native_fs::file_size(path) > 16384)
                return checkpointError(ErrorCode::FailedPrecondition, "Invalid checkpoint record");
            std::ifstream input(lfs::core::native_fs::native_path(path), std::ios::binary);
            std::array<char, 16385> buffer{};
            input.read(buffer.data(), buffer.size());
            const auto size = static_cast<std::size_t>(input.gcount());
            if (!input.eof() || size > 16384)
                return checkpointError(ErrorCode::Unavailable, "Could not read checkpoint record");
            auto record = Json::parse(buffer.data(), buffer.data() + size, nullptr, false);
            if (record.is_discarded())
                return checkpointError(ErrorCode::FailedPrecondition, "Malformed checkpoint record");
            return record;
        }
        SinkResult publishRecord(const std::filesystem::path& path, const Json& value) {
            const auto bytes = value.dump();
            if (bytes.size() > 16384)
                return SinkResult::failure(checkpointError(ErrorCode::InvalidArgument, "Checkpoint record exceeds its size limit"));
            const auto result = core::writeTextFileAtomically(path, bytes,
                                                              {.overwrite = false, .durable = false, .create_directories = false});
            if (result)
                return {};
            if (result.error().code() != ErrorCode::AlreadyExists)
                return result;
            const auto previous = readRecord(path);
            if (!previous)
                return SinkResult::failure(previous.error());
            if (*previous != value)
                return SinkResult::failure(checkpointError(ErrorCode::FailedPrecondition, "Checkpoint identity or frame changed"));
            return {};
        }
    } // namespace

    ResumableFileSink::ResumableFileSink(const IngestRequest& request, const FileExtraction& output)
        : request_(request), output_(output), cancelled_([cancelled = request.cancelled, file_cancelled = output.files.format == FrameFileFormat::AVIF ? output.files.avif.cancelled : output.files.format == FrameFileFormat::JXL  ? output.files.jxl.cancelled
                                                                                                                                                                                   : output.files.format == FrameFileFormat::EXR    ? output.files.exr.cancelled
                                                                                                                                                                                   : output.files.format == FrameFileFormat::TIFF16 ? output.files.tiff16.cancelled
                                                                                                                                                                                                                                    : output.files.image_output.cancelled] {
              return (cancelled && cancelled()) || (file_cancelled && file_cancelled());
          }),
          checkpoint_(output.files.output_directory / ".media-ingest-checkpoint") {}

    SinkResult ResumableFileSink::begin(const SinkSession& session) {
        if (output_.files.remove_stale_frames)
            return SinkResult::failure(checkpointError(ErrorCode::InvalidArgument, "Resume cannot remove stale extraction output"));
        const auto input_digest = fileDigest(request_.input, cancelled_);
        if (!input_digest)
            return SinkResult::failure(input_digest.error());
        const auto& r = request_;
        const auto& f = output_.files;
        const auto identity_path = checkpoint_ / "identity.json";
        std::string previous_provider;
        if (lfs::core::native_fs::exists(identity_path) && !emptyReservation(identity_path)) {
            const auto previous = readRecord(identity_path);
            if (!previous)
                return SinkResult::failure(previous.error());
            previous_provider = previous->value("image_encoder", std::string{});
        }
        gpu_jpeg_.reset();
        if (f.format == FrameFileFormat::JPEG && r.allow_hardware_decode && previous_provider != "cpu") {
            gpu_jpeg_ = createGpuJpegEncoder({session.output.width, session.output.height,
                                              session.output.width, session.output.height, 1, 0,
                                              false, false, false});
            if (gpu_jpeg_ && gpu_jpeg_->capacity() == 0)
                gpu_jpeg_.reset();
        }
        if (previous_provider == "nvimagecodec_cuda" && !gpu_jpeg_)
            return SinkResult::failure(checkpointError(ErrorCode::Unavailable, "Checkpoint GPU JPEG provider is unavailable; CPU reconstruction would change encoded bytes"));
        source_metadata_ = session.source_metadata;
        applied_rotation_ = session.applied_rotation;
        embedded_metadata_.reset();
        if (output_.files.embed_source_metadata) {
            if (!source_metadata_)
                return SinkResult::failure(checkpointError(ErrorCode::InvalidArgument, "Image metadata requires a source observation"));
            auto prepared = prepareImageMetadata(*source_metadata_, applied_rotation_, output_.files.format);
            if (!prepared)
                return SinkResult::failure(prepared.error());
            embedded_metadata_ = std::make_shared<const EmbeddedImageMetadata>(std::move(*prepared));
        }
        identity_ = {{"checkpoint_version", 2}, {"pipeline_version", 1}, {"application_build", GIT_TAGGED_VERSION}, {"ffmpeg", av_version_info()}, {"codec_version", avcodec_version()}, {"codec_configuration", avcodec_configuration()}, {"input_sha256", *input_digest}};
        identity_["image_encoder"] = imageEncoderBackend();
        identity_["stream"] = session.source.selected_video_stream ? Json(*session.source.selected_video_stream) : Json(nullptr);
        identity_["selection"] = {static_cast<int>(r.selection.mode), r.selection.fps, r.selection.interval, r.start_seconds, r.end_seconds};
        identity_["geometry"] = {static_cast<int>(r.geometry.mode), r.geometry.scale, r.geometry.width, r.geometry.height, r.geometry.clockwise_rotation};
        identity_["sharpness"] = {r.sharpness.enabled, static_cast<int>(r.sharpness.method), r.sharpness.threshold, r.sharpness.window_candidates, r.sharpness.window};
        identity_["color"] = {static_cast<int>(r.output_format), static_cast<int>(r.input_color.transfer), static_cast<int>(r.input_color.primaries), r.convert_hdr_to_sdr};
        identity_["files"] = {f.filename_pattern, static_cast<int>(f.format), f.jpeg_quality, static_cast<int>(f.exr.precision), static_cast<int>(f.exr.compression), f.exr.provenance};
        if (f.format == FrameFileFormat::TIFF16)
            identity_["tiff16"] = {core::image_codecs::tiff_sdr_encoding, static_cast<int>(f.tiff16.transfer), f.tiff16.provenance};
        if (f.format == FrameFileFormat::AVIF)
            identity_["avif"] = {f.avif.lossless ? "avif-sdr-v1" : "avif-sdr-ycbcr444-v2", core::image_codecs::avif_encoder_version(), f.avif.lossless, f.avif.lossless ? Json(nullptr) : Json(f.avif.quality), f.avif.speed, f.avif.bit_depth};
        if (f.format == FrameFileFormat::JXL) {
            identity_["jxl"] = {"jxl-sdr-v1", core::image_codecs::jxl_encoder_version(), f.jxl.lossless, f.jxl.lossless ? Json(nullptr) : Json(f.jxl.distance), f.jxl.effort};
            if (f.jxl.precision == JxlPrecision::UInt16)
                identity_["jxl_integer"] = {"uint16-clamp-round-half-up-v1", outputOptionName(f.jxl.transfer)};
        }
        identity_["files"].push_back(f.first_source_frame_number);
        identity_["layout"] = {session.output.width, session.output.height, static_cast<int>(session.output.format), static_cast<int>(session.output.color.transfer), static_cast<int>(session.output.color.primaries)};
        identity_["metadata"] = output_.write_metadata;
        if (f.embed_source_metadata) {
            identity_["embedded_metadata_policy"] = embeddedMetadataPolicy(f.format);
            Sha256 source_digest;
            source_digest.update({reinterpret_cast<const std::uint8_t*>(embedded_metadata_->source_properties.data()), embedded_metadata_->source_properties.size()});
            source_digest.update(embedded_metadata_->extended_markers);
            identity_["embedded_source_sha256"] = source_digest.finish();
        }
        if (output_.preserve_source_metadata || f.embed_source_metadata)
            identity_["source_metadata_policy"] = source_metadata_preservation_policy;
        if (r.float_color_profile != FloatColorProfile::Sdr)
            identity_["float_color_profile"] = static_cast<int>(r.float_color_profile);
        if (session.output.color.units != ColorUnits::Relative)
            identity_["sample_units"] = static_cast<int>(session.output.color.units);
        identity_["hardware_decode"] = r.allow_hardware_decode;
        lfs::core::native_fs::create_directories(f.output_directory);
        if (lfs::core::native_fs::is_symlink(lfs::core::native_fs::symlink_status(f.output_directory)))
            return SinkResult::failure(checkpointError(ErrorCode::FailedPrecondition, "Resume output directory must not be a symlink"));
        if (!lfs::core::native_fs::exists(checkpoint_)) {
            if (!io::generatedExtractionFiles(f.output_directory, f.filename_pattern).empty())
                return SinkResult::failure(checkpointError(ErrorCode::AlreadyExists, "Cannot adopt existing extraction output without a checkpoint"));
            lfs::core::native_fs::create_directory(checkpoint_);
        }
        if (!lfs::core::native_fs::is_directory(lfs::core::native_fs::symlink_status(checkpoint_)))
            return SinkResult::failure(checkpointError(ErrorCode::FailedPrecondition, "Checkpoint must be a non-symlink directory"));
        const bool overwrite = f.format == FrameFileFormat::AVIF ? f.avif.overwrite : f.format == FrameFileFormat::JXL  ? f.jxl.overwrite
                                                                                  : f.format == FrameFileFormat::TIFF16 ? f.tiff16.overwrite
                                                                                  : f.format == FrameFileFormat::EXR    ? f.exr.overwrite
                                                                                                                        : f.image_output.overwrite;
        if ((!lfs::core::native_fs::exists(identity_path) || emptyReservation(identity_path)) &&
            lfs::core::native_fs::exists(f.output_directory / "extraction_metadata.json") && !overwrite)
            return SinkResult::failure(checkpointError(ErrorCode::AlreadyExists, "Cannot adopt existing extraction metadata without a checkpoint identity"));
        if (!lfs::core::native_fs::exists(identity_path) || emptyReservation(identity_path)) {
            // Before identity publication no image/record can have committed.
            // Only its own interrupted atomic temporary/reservation is recoverable.
            for (const auto& entry : lfs::core::native_fs::directory_iterator(checkpoint_)) {
                auto name = core::path_to_utf8(entry.path().filename());
#ifdef __APPLE__
                // macOS may create AppleDouble companions on exFAT, including
                // for the interrupted identity temporary and reservation.
                if (name.starts_with("._"))
                    name.erase(0, 2);
#endif
                if ((name != "identity.json" && !name.starts_with("identity.json.tmp-")) ||
                    !lfs::core::native_fs::is_regular_file(lfs::core::native_fs::entry_symlink_status(checkpoint_, entry)))
                    return SinkResult::failure(checkpointError(ErrorCode::FailedPrecondition, "Cannot adopt a nonempty checkpoint directory without its identity"));
            }
            if (auto recovered = discardReservation(identity_path); !recovered)
                return recovered;
        }
        auto identity = publishRecord(identity_path, identity_);
        if (!identity)
            return identity;
        // Each record has bounded size; the sequence index is rebuilt by decoding
        // and selecting from the original start, never by guessing a seek/PTS.
        recorded_ = 0;
        for (const auto& entry : lfs::core::native_fs::directory_iterator(checkpoint_)) {
            const auto name = core::path_to_utf8(entry.path().filename());
            if (name.starts_with("frame-") && name.ends_with(".json"))
                ++recorded_;
        }
        for (std::size_t index = 0; index < recorded_; ++index) {
            const auto record_path = checkpoint_ / ("frame-" + std::to_string(index) + ".json");
            if (emptyReservation(record_path)) {
                if (index + 1 != recorded_)
                    return SinkResult::failure(checkpointError(ErrorCode::FailedPrecondition, "Empty checkpoint reservation precedes committed records"));
                if (auto recovered = discardReservation(record_path); !recovered)
                    return recovered;
                --recorded_;
                break;
            }
            const auto record = readRecord(record_path);
            if (!record)
                return SinkResult::failure(record.error());
            if (!record->is_object() || !record->contains("file") || !(*record)["file"].is_string() ||
                !record->contains("file_sha256") || !(*record)["file_sha256"].is_string())
                return SinkResult::failure(checkpointError(ErrorCode::FailedPrecondition, "Invalid checkpoint frame schema"));
            const auto name = core::utf8_to_path((*record)["file"].get<std::string>());
            if (name != name.filename() || !io::isGeneratedFrameFilename(f.filename_pattern, core::path_to_utf8(name), true))
                return SinkResult::failure(checkpointError(ErrorCode::FailedPrecondition, "Invalid checkpoint frame filename"));
            // Verify retained encoded bytes once, when their selected pixels
            // have also been regenerated in write(). No eager duplicate read.
        }
        accepted_ = 0;
        filenames_.clear();
        active_ = true;
        return {};
    }

    SinkResult ResumableFileSink::write(const FrameView& frame) {
        if (!active_)
            return SinkResult::failure(checkpointError(ErrorCode::FailedPrecondition, "Checkpoint sink is inactive"));
        if (auto size = frame.requiredBytes(); !size)
            return SinkResult::failure(size.error());
        if (frame.info.legacy_source_frame < 1 || output_.files.first_source_frame_number < 0 ||
            frame.info.legacy_source_frame - 1 >
                std::numeric_limits<int>::max() - output_.files.first_source_frame_number) {
            return SinkResult::failure(checkpointError(ErrorCode::InvalidArgument,
                                                       "Output frame number exceeds the supported range"));
        }
        const auto output_frame_number = output_.files.first_source_frame_number + frame.info.legacy_source_frame - 1;
        const auto extension = output_.files.format == FrameFileFormat::AVIF ? ".avif" : output_.files.format == FrameFileFormat::JXL  ? ".jxl"
                                                                                     : output_.files.format == FrameFileFormat::TIFF16 ? ".tif"
                                                                                     : output_.files.format == FrameFileFormat::PNG    ? ".png"
                                                                                     : output_.files.format == FrameFileFormat::JPEG   ? ".jpg"
                                                                                                                                       : ".exr";
        const auto name = core::utf8_to_path(io::formatFrameFilenameStem(output_.files.filename_pattern, output_frame_number) + extension);
        if (name != name.filename() || name.empty() || !filenames_.insert(name).second)
            return SinkResult::failure(checkpointError(ErrorCode::InvalidArgument, "Resume requires unique plain frame filenames"));
        const auto destination = output_.files.output_directory / name;
        const auto record_path = checkpoint_ / ("frame-" + std::to_string(accepted_) + ".json");
        Sha256 pixel_digest;
        const auto row_bytes = static_cast<std::size_t>(frame.layout.width) * pixelBytes(frame.layout.format);
        for (int row = 0; row < frame.layout.height; ++row)
            pixel_digest.update(frame.pixels.subspan(static_cast<std::size_t>(row) * frame.layout.row_stride, row_bytes));
        Json record = {{"file", core::path_to_utf8(name)}, {"pixels_sha256", pixel_digest.finish()}, {"source_frame", frame.info.legacy_source_frame}, {"seconds", frame.info.relative_seconds}, {"sharpness", frame.info.sharpness_score}};
        if (frame.info.source_timestamp) {
            const auto& pts = *frame.info.source_timestamp;
            record["pts"] = {pts.ticks, pts.time_base.numerator, pts.time_base.denominator};
        }
        if (output_.files.embed_source_metadata)
            record["timestamp_origin"] = static_cast<int>(frame.info.timestamp_origin);
        if (lfs::core::native_fs::exists(record_path)) {
            const auto old = readRecord(record_path);
            if (!old)
                return SinkResult::failure(old.error());
            auto expected = *old;
            expected.erase("file_sha256");
            if (record != expected || !old->contains("file_sha256") || !(*old)["file_sha256"].is_string())
                return SinkResult::failure(checkpointError(ErrorCode::FailedPrecondition, "Checkpoint frame selection or pixels changed"));
            record = *old;
            if (auto recovered = discardReservation(destination); !recovered)
                return recovered;
            if (lfs::core::native_fs::exists(destination)) {
                const auto digest = fileDigest(destination, cancelled_);
                if (!digest)
                    return SinkResult::failure(digest.error());
                if (*digest != record["file_sha256"].get<std::string>())
                    return SinkResult::failure(checkpointError(ErrorCode::FailedPrecondition, "Checkpoint output is truncated or modified"));
                ++accepted_;
                return {};
            }
        } else if (lfs::core::native_fs::exists(destination)) {
            return SinkResult::failure(checkpointError(ErrorCode::AlreadyExists, "Untracked output cannot be overwritten by resume"));
        }
        // Encode once with the existing file sink. Record the exact encoded bytes
        // before publishing the image, covering a crash between image and state.
        auto staging_options = output_.files;
        staging_options.output_directory = checkpoint_;
        staging_options.remove_stale_frames = false;
        staging_options.image_output.overwrite = true;
        if (output_.files.embed_source_metadata)
            staging_options.image_output.cancelled = cancelled_;
        staging_options.exr.overwrite = true;
        staging_options.tiff16.overwrite = true;
        staging_options.tiff16.cancelled = cancelled_;
        staging_options.avif.overwrite = true;
        staging_options.avif.cancelled = cancelled_;
        staging_options.jxl.overwrite = true;
        staging_options.jxl.cancelled = cancelled_;
        const auto staged = checkpoint_ / name;
        Result<std::string> encoded_digest = std::string{};
        if (gpu_jpeg_) {
            std::vector<std::uint8_t> packed;
            const auto* pixels = frame.pixels.data();
            if (frame.layout.row_stride != row_bytes) {
                packed.reserve(row_bytes * frame.layout.height);
                for (int row = 0; row < frame.layout.height; ++row) {
                    const auto bytes = frame.pixels.subspan(static_cast<std::size_t>(row) * frame.layout.row_stride, row_bytes);
                    packed.insert(packed.end(), bytes.begin(), bytes.end());
                }
                pixels = packed.data();
            }
            const auto quality = std::clamp(output_.files.jpeg_quality == 0 ? 90 : output_.files.jpeg_quality, 1, 100);
            auto encoded = gpu_jpeg_->encode({gpu_jpeg_->queueHost(0, pixels)}, frame.layout.width, frame.layout.height, quality);
            if (encoded.size() != 1 || encoded[0].empty())
                return SinkResult::failure(checkpointError(ErrorCode::Unavailable, "Checkpoint GPU JPEG encoder returned an incomplete frame"));
            if (!embedded_metadata_) {
                Sha256 digest;
                digest.update(encoded[0]);
                encoded_digest = digest.finish();
            }
            auto staged_result = core::writeFileAtomically(staged, [&](FILE* stream) -> SinkResult {
                if (embedded_metadata_)
                    return writeJpegMetadata(stream, encoded[0], *embedded_metadata_, frame.layout.width, frame.layout.height, frame.info);
                if (std::fwrite(encoded[0].data(), 1, encoded[0].size(), stream) != encoded[0].size())
                    return SinkResult::failure(checkpointError(ErrorCode::Unavailable, "Checkpoint GPU JPEG staging write failed"));
                return {};
            },
                                                           {.overwrite = true, .durable = false, .cancelled = cancelled_});
            if (!staged_result)
                return staged_result;
            if (embedded_metadata_)
                encoded_digest = fileDigest(staged, cancelled_);
        } else {
            FileFrameSink staging(std::move(staging_options));
            if (auto begun = staging.beginPrepared({.applied_rotation = applied_rotation_, .source_metadata = source_metadata_}, embedded_metadata_); !begun)
                return begun;
            if (auto written = staging.write(frame); !written)
                return written;
            if (auto completed = staging.complete({SinkOutcome::Completed, 1, {}}); !completed)
                return completed;
            encoded_digest = fileDigest(staged, cancelled_);
        }
        if (!encoded_digest)
            return SinkResult::failure(encoded_digest.error());
        if (record.contains("file_sha256") && record["file_sha256"].get<std::string>() != *encoded_digest)
            return SinkResult::failure(checkpointError(ErrorCode::FailedPrecondition, "Checkpoint encoder output changed"));
        record["file_sha256"] = *encoded_digest;
        if (auto saved = publishRecord(record_path, record); !saved)
            return saved;
        const auto committed = core::publishStagedFileAtomically(staged, destination, cancelled_);
        if (!committed)
            return committed;
        ++accepted_;
        return {};
    }

    SinkResult ResumableFileSink::prepareComplete(const SinkSummary&) {
        if (accepted_ < recorded_)
            return SinkResult::failure(checkpointError(ErrorCode::FailedPrecondition, "Checkpoint contains frames absent from this extraction"));
        return {};
    }
    SinkResult ResumableFileSink::complete(const SinkSummary&) {
        active_ = false;
        embedded_metadata_.reset();
        source_metadata_.reset();
        return {};
    }
    void ResumableFileSink::abort(const SinkSummary&) noexcept {
        active_ = false;
        embedded_metadata_.reset();
        source_metadata_.reset();
    }
} // namespace lfs::media::detail
