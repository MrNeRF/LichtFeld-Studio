// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "resumable_file_sink.hpp"
#include "core/atomic_file.hpp"
#include "core/path_utils.hpp"
#include "git_version.h"
#include "media/video_frame_extractor.hpp"
#include <array>
#include <fstream>
#include <memory>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/mem.h>
#include <libavutil/sha.h>
}

namespace lfs::media::detail {
    namespace {
        using Json = nlohmann::json;
        Error checkpointError(ErrorCode code, std::string message) {
            return make_error({.code = code, .domain = ErrorDomain::IO, .detail = std::move(message), .detection = LFS_SOURCE_SITE_CURRENT()});
        }
        bool emptyReservation(const std::filesystem::path& path) {
            return std::filesystem::is_regular_file(std::filesystem::symlink_status(path)) &&
                   std::filesystem::file_size(path) == 0;
        }
        SinkResult discardReservation(const std::filesystem::path& path) {
            if (!emptyReservation(path))
                return {};
            std::error_code error;
            std::filesystem::remove(path, error);
            if (error)
                return SinkResult::failure(checkpointError(ErrorCode::Unavailable, "Could not recover empty checkpoint reservation: " + core::path_to_utf8(path)));
            return {};
        }
        struct Sha256 {
            std::unique_ptr<AVSHA, decltype(&av_free)> context{av_sha_alloc(), av_free};
            Sha256() {
                if (!context || av_sha_init(context.get(), 256) != 0)
                    throw std::runtime_error("Could not initialize checkpoint SHA-256");
            }
            void update(std::span<const std::uint8_t> bytes) {
                av_sha_update(context.get(), bytes.data(), bytes.size());
            }
            std::string finish() {
                std::array<std::uint8_t, 32> digest{};
                av_sha_final(context.get(), digest.data());
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
            if (!std::filesystem::is_regular_file(std::filesystem::symlink_status(path)))
                return checkpointError(ErrorCode::FailedPrecondition, "Checkpoint requires a regular, non-symlink file: " + core::path_to_utf8(path));
            std::ifstream input(path, std::ios::binary);
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
            if (!std::filesystem::is_regular_file(std::filesystem::symlink_status(path)) || std::filesystem::file_size(path) > 16384)
                return checkpointError(ErrorCode::FailedPrecondition, "Invalid checkpoint record");
            std::ifstream input(path, std::ios::binary);
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
        : request_(request), output_(output), cancelled_([cancelled = request.cancelled, file_cancelled = output.files.format == FrameFileFormat::EXR ? output.files.exr.cancelled : output.files.image_output.cancelled] {
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
        identity_ = {{"checkpoint_version", 2}, {"pipeline_version", 1}, {"application_build", GIT_TAGGED_VERSION}, {"ffmpeg", av_version_info()}, {"codec_version", avcodec_version()}, {"codec_configuration", avcodec_configuration()}, {"input_sha256", *input_digest}};
        identity_["stream"] = session.source.selected_video_stream ? Json(*session.source.selected_video_stream) : Json(nullptr);
        identity_["selection"] = {static_cast<int>(r.selection.mode), r.selection.fps, r.selection.interval, r.start_seconds, r.end_seconds};
        identity_["geometry"] = {static_cast<int>(r.geometry.mode), r.geometry.scale, r.geometry.width, r.geometry.height, r.geometry.clockwise_rotation};
        identity_["sharpness"] = {r.sharpness.enabled, static_cast<int>(r.sharpness.method), r.sharpness.threshold, r.sharpness.window_candidates, r.sharpness.window};
        identity_["color"] = {static_cast<int>(r.output_format), static_cast<int>(r.input_color.transfer), static_cast<int>(r.input_color.primaries), r.convert_hdr_to_sdr};
        identity_["files"] = {f.filename_pattern, static_cast<int>(f.format), f.jpeg_quality, static_cast<int>(f.exr.precision), static_cast<int>(f.exr.compression), f.exr.provenance};
        identity_["layout"] = {session.output.width, session.output.height, static_cast<int>(session.output.format), static_cast<int>(session.output.color.transfer), static_cast<int>(session.output.color.primaries)};
        identity_["metadata"] = output_.write_metadata;
        identity_["hardware_decode"] = r.allow_hardware_decode;
        std::filesystem::create_directories(f.output_directory);
        if (std::filesystem::is_symlink(std::filesystem::symlink_status(f.output_directory)))
            return SinkResult::failure(checkpointError(ErrorCode::FailedPrecondition, "Resume output directory must not be a symlink"));
        if (!std::filesystem::exists(checkpoint_)) {
            if (!io::generatedExtractionFiles(f.output_directory, f.filename_pattern).empty())
                return SinkResult::failure(checkpointError(ErrorCode::AlreadyExists, "Cannot adopt existing extraction output without a checkpoint"));
            std::filesystem::create_directory(checkpoint_);
        }
        if (!std::filesystem::is_directory(std::filesystem::symlink_status(checkpoint_)))
            return SinkResult::failure(checkpointError(ErrorCode::FailedPrecondition, "Checkpoint must be a non-symlink directory"));
        const auto identity_path = checkpoint_ / "identity.json";
        if (!std::filesystem::exists(identity_path) || emptyReservation(identity_path)) {
            // Before identity publication no image/record can have committed.
            // Only its own interrupted atomic temporary/reservation is recoverable.
            for (const auto& entry : std::filesystem::directory_iterator(checkpoint_)) {
                auto name = core::path_to_utf8(entry.path().filename());
#ifdef __APPLE__
                // macOS may create AppleDouble companions on exFAT, including
                // for the interrupted identity temporary and reservation.
                if (name.starts_with("._"))
                    name.erase(0, 2);
#endif
                if ((name != "identity.json" && !name.starts_with("identity.json.tmp-")) ||
                    !std::filesystem::is_regular_file(entry.symlink_status()))
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
        for (const auto& entry : std::filesystem::directory_iterator(checkpoint_)) {
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
            if (name != name.filename() || !io::isGeneratedFrameFilename(f.filename_pattern, core::path_to_utf8(name)))
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
        const auto extension = output_.files.format == FrameFileFormat::PNG ? ".png" : output_.files.format == FrameFileFormat::JPEG ? ".jpg"
                                                                                                                                     : ".exr";
        const auto name = core::utf8_to_path(io::formatFrameFilenameStem(output_.files.filename_pattern, frame.info.legacy_source_frame) + extension);
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
        if (std::filesystem::exists(record_path)) {
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
            if (std::filesystem::exists(destination)) {
                const auto digest = fileDigest(destination, cancelled_);
                if (!digest)
                    return SinkResult::failure(digest.error());
                if (*digest != record["file_sha256"].get<std::string>())
                    return SinkResult::failure(checkpointError(ErrorCode::FailedPrecondition, "Checkpoint output is truncated or modified"));
                ++accepted_;
                return {};
            }
        } else if (std::filesystem::exists(destination)) {
            return SinkResult::failure(checkpointError(ErrorCode::AlreadyExists, "Untracked output cannot be overwritten by resume"));
        }
        // Encode once with the existing file sink. Record the exact encoded bytes
        // before publishing the image, covering a crash between image and state.
        auto staging_options = output_.files;
        staging_options.output_directory = checkpoint_;
        staging_options.remove_stale_frames = false;
        staging_options.image_output.overwrite = true;
        staging_options.exr.overwrite = true;
        FileFrameSink staging(std::move(staging_options));
        if (auto begun = staging.begin({}); !begun)
            return begun;
        if (auto written = staging.write(frame); !written)
            return written;
        if (auto completed = staging.complete({SinkOutcome::Completed, 1, {}}); !completed)
            return completed;
        const auto staged = checkpoint_ / name;
        const auto encoded_digest = fileDigest(staged, cancelled_);
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
        return {};
    }
    void ResumableFileSink::abort(const SinkSummary&) noexcept { active_ = false; }
} // namespace lfs::media::detail
