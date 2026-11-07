// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "image_metadata.hpp"
#include "core/image_codecs.hpp"
#include "media_json.hpp"
#include "metadata_file_io.hpp"
#include "xmp_metadata.hpp"
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace lfs::media::detail {
    namespace {
        constexpr char standard_signature[] = "http://ns.adobe.com/xap/1.0/";
        constexpr char extended_signature[] = "http://ns.adobe.com/xmp/extension/";
        constexpr std::size_t max_packet_bytes = core::image_codecs::max_xmp_packet_bytes;
        constexpr std::size_t max_payload_bytes = 65533;
        constexpr std::string_view tree_start =
            "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><rdf:RDF "
            "xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">"
            "<rdf:Description rdf:about=\"\" "
            "xmlns:lfs=\"https://lichtfeld.io/ns/media/1.0/\" "
            "xmlns:n=\"http://ns.adobe.com/xmp/note/\">";
        constexpr std::string_view packet_start =
            "<?xpacket begin=\"\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>";
        constexpr std::string_view packet_end = "<?xpacket end=\"w\"?>";
        constexpr std::string_view tree_end = "</rdf:Description></rdf:RDF></x:xmpmeta>";
        Error metadataError(ErrorCode code, std::string detail) {
            return make_error({.code = code, .domain = ErrorDomain::IO, .detail = std::move(detail), .detection = LFS_SOURCE_SITE_CURRENT()});
        }
        void append32(std::vector<std::uint8_t>& out, std::uint32_t value, bool little = false) {
            for (int i = 0; i < 4; ++i)
                out.push_back(static_cast<std::uint8_t>(value >> (8 * (little ? i : 3 - i))));
        }
        void append16(std::vector<std::uint8_t>& out, std::uint16_t value) {
            out.push_back(static_cast<std::uint8_t>(value));
            out.push_back(static_cast<std::uint8_t>(value >> 8));
        }
        void marker(std::vector<std::uint8_t>& out, std::span<const std::uint8_t> payload) {
            const auto size = payload.size() + 2;
            out.insert(out.end(), {0xff, 0xe1, static_cast<std::uint8_t>(size >> 8), static_cast<std::uint8_t>(size)});
            out.insert(out.end(), payload.begin(), payload.end());
        }
        void property(std::string& out, std::string_view name, std::string_view value) {
            out += "<lfs:";
            out += name;
            out += ">";
            json_detail::appendXmlText(out, value);
            out += "</lfs:";
            out += name;
            out += ">";
        }
        void frameProperties(std::string& tree, int width, int height,
                             int applied_rotation, const FrameInfo& info, std::string_view policy) {
            property(tree, "MetadataPolicy", policy);
            property(tree, "PixelWidth", std::to_string(width));
            property(tree, "PixelHeight", std::to_string(height));
            property(tree, "Orientation", "1");
            property(tree, "AppliedClockwiseRotation", std::to_string(applied_rotation));
            if (info.legacy_source_frame > 0)
                property(tree, "SourceFrame", std::to_string(info.legacy_source_frame));
            if (info.source_timestamp &&
                info.source_timestamp->time_base.denominator > 0 &&
                info.source_timestamp->time_base.numerator > 0) {
                property(tree, "SourceTimestampTicks",
                         std::to_string(info.source_timestamp->ticks));
                property(tree, "SourceTimeBaseNumerator",
                         std::to_string(info.source_timestamp->time_base.numerator));
                property(tree, "SourceTimeBaseDenominator",
                         std::to_string(info.source_timestamp->time_base.denominator));
                property(
                    tree, "TimestampOrigin",
                    info.timestamp_origin == TimestampOrigin::Presentation ? "presentation"
                    : info.timestamp_origin == TimestampOrigin::BestEffort ? "best_effort"
                                                                           : "missing");
            }
        }
        std::vector<std::uint8_t> exif(int width, int height) {
            // Classic little-endian TIFF. IFD0: orientation=1 and ExifIFD.
            // ExifIFD: dimensions of the encoded pixels, not the source video.
            std::vector<std::uint8_t> out{'E', 'x', 'i', 'f', 0, 0, 'I', 'I', 42, 0, 8, 0, 0, 0};
            const auto entry = [&](std::uint16_t tag, std::uint16_t type, std::uint32_t value) {
                append16(out, tag);
                append16(out, type);
                append32(out, 1, true);
                append32(out, value, true);
            };
            append16(out, 2);
            entry(0x0112, 3, 1);
            entry(0x8769, 4, 38);
            append32(out, 0, true);
            append16(out, 4);
            append16(out, 0x9000);
            append16(out, 7);
            append32(out, 4, true);
            append32(out, 0x32333230, true);
            entry(0xa001, 3, 65535);
            entry(0xa002, 4, static_cast<std::uint32_t>(width));
            entry(0xa003, 4, static_cast<std::uint32_t>(height));
            append32(out, 0, true);
            return out;
        }
    } // namespace
    PreviousSourceSidecar PreviousSourceSidecar::inspect(const std::filesystem::path& image) {
        PreviousSourceSidecar result;
        auto path = image;
        path += sidecar_metadata_suffix;
        std::error_code error;
        if (!core::native_fs::is_regular_file(core::native_fs::symlink_status(path, error)) || error)
            return result;
        SidecarFile file;
        const auto size = file.open(path, max_packet_bytes);
        if (!size)
            return result;
        static const auto signatures = [] {
            std::array<std::string, 2> prefixes;
            const std::array<std::string_view, 2> policies{sidecar_metadata_policy, render_sidecar_metadata_policy};
            for (std::size_t i = 0; i < policies.size(); ++i) {
                prefixes[i] = packet_start;
                prefixes[i] += tree_start;
                property(prefixes[i], "MetadataPolicy", policies[i]);
            }
            return prefixes;
        }();
        if (*size < std::min(signatures[0].size(), signatures[1].size()))
            return result;
        std::string prefix(std::min<std::uint64_t>(*size, std::max(signatures[0].size(), signatures[1].size())), '\0');
        std::size_t offset = 0;
        while (offset < prefix.size()) {
            const auto read = file.read({reinterpret_cast<std::uint8_t*>(prefix.data()) + offset, prefix.size() - offset});
            if (!read || !*read)
                return result;
            offset += *read;
        }
        if ((!prefix.starts_with(signatures[0]) && !prefix.starts_with(signatures[1])) || !file.verifyUnchanged())
            return result;
        result.path_ = std::move(path);
        result.observation_ = std::make_shared<const SidecarObservation>(file.observation());
        return result;
    }
    SinkResult PreviousSourceSidecar::removeUnchanged() {
        if (!observation_)
            return {};
        // Replacement, removal or edits by another writer relinquish our claim.
        bool unchanged = false;
        {
            SidecarFile current;
            unchanged = current.open(path_, max_packet_bytes, true) && observation_->sameVersion(current.observation());
        }
        if (unchanged) {
            std::error_code error;
            core::native_fs::remove(path_, error);
            if (error)
                return SinkResult::failure(sidecarIoError(error, "Remove superseded image XMP", path_));
        }
        observation_.reset();
        return {};
    }
    std::string_view embeddedMetadataPolicy(FrameFileFormat format) {
        return format == FrameFileFormat::JPEG     ? std::string_view(jpeg_metadata_policy)
               : format == FrameFileFormat::PNG    ? std::string_view(png_metadata_policy)
               : format == FrameFileFormat::TIFF16 ? std::string_view(tiff_metadata_policy)
               : format == FrameFileFormat::EXR    ? std::string_view(exr_metadata_policy)
               : format == FrameFileFormat::JXL    ? std::string_view(jxl_metadata_policy)
               : format == FrameFileFormat::AVIF   ? std::string_view(avif_metadata_policy)
                                                   : std::string_view{};
    }
    std::string_view imageXmpSuffix() {
        // Rewriting the packet requires updating the carrier lengths/structure.
        return "</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end=\"r\"?>";
    }
    Result<EmbeddedImageMetadata> prepareRenderedMetadata(FrameFileFormat format, std::string_view provenance) {
        if (embeddedMetadataPolicy(format).empty())
            return metadataError(ErrorCode::InvalidArgument, "Invalid render metadata format (format=" + std::to_string(static_cast<int>(format)) + ")");
        // Bound escaped output before allocating it; XML escaping can expand input.
        if (provenance.size() > max_render_provenance_bytes)
            return metadataError(ErrorCode::ResourceExhausted, "Render provenance exceeds its input byte limit (size=" + std::to_string(provenance.size()) + ", limit=" + std::to_string(max_render_provenance_bytes) + ")");
        int rejected_depth = 0;
        try {
            EmbeddedImageMetadata result;
            result.format = format;
            result.rendered = true;
            if (!provenance.empty()) {
                const auto stamp = nlohmann::json::parse(provenance, [&](int depth, nlohmann::json::parse_event_t, nlohmann::json&) {
                    if (depth > 4) {
                        rejected_depth = depth;
                        throw std::invalid_argument("Render provenance nesting exceeds its budget");
                    }
                    return true;
                });
                if (!stamp.is_object())
                    return metadataError(ErrorCode::InvalidArgument, "Render provenance must be a JSON object (type=" + std::string(stamp.type_name()) + ")");
                property(result.source_properties, "RenderProvenance", stamp.dump(-1, ' ', true));
            }
            return result;
        } catch (const std::bad_alloc&) {
            return metadataError(ErrorCode::ResourceExhausted, "Render metadata allocation failed");
        } catch (const nlohmann::json::parse_error& error) {
            return metadataError(ErrorCode::InvalidArgument, "Invalid render provenance JSON (byte=" + std::to_string(error.byte) + ", size=" + std::to_string(provenance.size()) + ")");
        } catch (const nlohmann::json::exception& error) {
            return metadataError(ErrorCode::InvalidArgument, "Invalid render provenance JSON (error=" + std::to_string(error.id) + ", size=" + std::to_string(provenance.size()) + ")");
        } catch (const std::invalid_argument&) {
            return metadataError(ErrorCode::InvalidArgument, "Render provenance nesting exceeds its budget (depth=" + std::to_string(rejected_depth) + ", limit=4)");
        }
    }
    namespace {
        Result<EmbeddedImageMetadata> packJpegMetadata(EmbeddedImageMetadata result) {
            // Reserve only the largest frame envelope this writer can produce.
            // A valid Standard packet must not depend on the MD5 provider.
            FrameInfo largest;
            largest.legacy_source_frame = std::numeric_limits<int>::max();
            largest.source_timestamp = Timestamp{
                std::numeric_limits<int64_t>::min(),
                {std::numeric_limits<int>::max(), std::numeric_limits<int>::max()}};
            largest.timestamp_origin = TimestampOrigin::Presentation;
            std::string envelope;
            frameProperties(envelope, std::numeric_limits<int>::max(),
                            std::numeric_limits<int>::max(), result.applied_rotation, largest, jpeg_metadata_policy);
            const auto overhead = sizeof(standard_signature) + packet_start.size() +
                                  tree_start.size() + envelope.size() +
                                  tree_end.size() + packet_end.size();
            if (result.source_properties.size() <= max_payload_bytes - overhead)
                return result;
            const auto tree = std::string(tree_start) + result.source_properties + std::string(tree_end);
            if (tree.size() > max_packet_bytes)
                return metadataError(ErrorCode::ResourceExhausted, "JPEG Extended XMP exceeds the export packet budget");
            const auto digest = xmpPacketDigest({reinterpret_cast<const std::uint8_t*>(tree.data()), tree.size()});
            if (!digest)
                return metadataError(ErrorCode::Unavailable, "JPEG Extended XMP MD5 is unavailable");
            result.extended_guid.assign(digest->begin(), digest->end());
            constexpr std::size_t chunk_bytes = max_payload_bytes - sizeof(extended_signature) - 32 - 8;
            static_assert(chunk_bytes == 65458);
            for (std::size_t offset = 0; offset < tree.size(); offset += chunk_bytes) {
                std::vector<std::uint8_t> payload(extended_signature, extended_signature + sizeof(extended_signature));
                payload.insert(payload.end(), result.extended_guid.begin(), result.extended_guid.end());
                append32(payload, static_cast<std::uint32_t>(tree.size()));
                append32(payload, static_cast<std::uint32_t>(offset));
                const auto count = std::min(chunk_bytes, tree.size() - offset);
                payload.insert(payload.end(), tree.begin() + offset, tree.begin() + offset + count);
                marker(result.extended_markers, payload);
            }
            std::string{}.swap(result.source_properties);
            return result;
        }
        SinkResult appendSrtMetadata(std::string& properties, const std::optional<SrtFrameMetadata>& binding, const FrameInfo& info) {
            if (!binding)
                return {};
            if (info.origin != FrameOrigin::Decoded || !info.source_timestamp || (info.timestamp_origin != TimestampOrigin::Presentation && info.timestamp_origin != TimestampOrigin::BestEffort))
                return SinkResult::failure(metadataError(ErrorCode::FailedPrecondition, "SRT frame metadata requires a decoded frame with a source timestamp (origin=" + std::to_string(static_cast<int>(info.origin)) + ", timestamp_origin=" + std::to_string(static_cast<int>(info.timestamp_origin)) + ", has_timestamp=" + std::to_string(info.source_timestamp.has_value()) + ")"));
            auto observation = json_detail::srtLookup(binding->catalog, *info.source_timestamp,
                                                      binding->offset_milliseconds, binding->point_tolerance_milliseconds);
            if (!observation)
                return SinkResult::failure(observation.error());
            (*observation)["policy"] = srt_frame_metadata_policy;
            if (properties.size() > max_packet_bytes - 1024)
                return SinkResult::failure(metadataError(ErrorCode::ResourceExhausted, "SRT frame metadata exceeds the export packet budget"));
            auto encoded = json_detail::sourceMetadataXmlProperty(*observation, max_packet_bytes - properties.size() - 1024, "SrtObservation");
            if (!encoded)
                return SinkResult::failure(metadataError(ErrorCode::ResourceExhausted, "SRT frame metadata exceeds the export packet budget"));
            properties += *encoded;
            return {};
        }
    } // namespace
    Result<std::string> imageXmpPrefix(const EmbeddedImageMetadata& metadata, int width, int height, const FrameInfo& info, bool sidecar) {
        if (((metadata.format == FrameFileFormat::JPEG && !sidecar) || embeddedMetadataPolicy(metadata.format).empty()) || width <= 0 || height <= 0)
            return metadataError(ErrorCode::InvalidArgument, "Invalid image metadata input");
        try {
            std::string tree(packet_start);
            tree += tree_start;
            if (metadata.rendered) {
                if (!sidecar || info.origin != FrameOrigin::Rendered || !info.output_timestamp ||
                    info.output_timestamp->time_base.numerator <= 0 || info.output_timestamp->time_base.denominator <= 0)
                    return metadataError(ErrorCode::InvalidArgument, "Render metadata requires a rendered output timestamp (sidecar=" + std::to_string(sidecar) + ", origin=" + std::to_string(static_cast<int>(info.origin)) + ", time_base=" + (info.output_timestamp ? std::to_string(info.output_timestamp->time_base.numerator) + "/" + std::to_string(info.output_timestamp->time_base.denominator) : std::string("missing")) + ")");
                property(tree, "MetadataPolicy", render_sidecar_metadata_policy);
                property(tree, "FrameOrigin", "rendered");
                property(tree, "PixelWidth", std::to_string(width));
                property(tree, "PixelHeight", std::to_string(height));
                property(tree, "Orientation", "1");
                property(tree, "DeliveryIndex", std::to_string(info.delivery_index));
                property(tree, "OutputTimestampTicks", std::to_string(info.output_timestamp->ticks));
                property(tree, "OutputTimeBaseNumerator", std::to_string(info.output_timestamp->time_base.numerator));
                property(tree, "OutputTimeBaseDenominator", std::to_string(info.output_timestamp->time_base.denominator));
            } else {
                frameProperties(tree, width, height, metadata.applied_rotation, info, sidecar ? std::string_view(sidecar_metadata_policy) : embeddedMetadataPolicy(metadata.format));
            }
            if (auto appended = appendSrtMetadata(tree, metadata.srt_metadata, info); !appended)
                return appended.error();
            if (tree.size() + metadata.source_properties.size() + imageXmpSuffix().size() > max_packet_bytes)
                return metadataError(ErrorCode::ResourceExhausted, "Image XMP exceeds the export packet budget");
            return tree;
        } catch (const std::bad_alloc&) {
            return metadataError(ErrorCode::ResourceExhausted, "Image frame metadata allocation failed");
        }
    }
    Result<EmbeddedImageMetadata> prepareImageMetadata(const SourceMetadataCatalog& catalog, int applied_rotation, FrameFileFormat format, bool sidecar, const std::optional<SrtFrameMetadata>& srt) {
        if (embeddedMetadataPolicy(format).empty())
            return metadataError(ErrorCode::InvalidArgument, "Embedded metadata requires JPEG, PNG, TIFF16, EXR, JXL or AVIF output");
        try {
            const auto summary = summarizeSourceMetadata(catalog);
            if (!summary)
                return summary.error();
            auto properties = json_detail::sourceMetadataXmlProperty(json_detail::sourceMetadataResult(catalog, *summary), max_packet_bytes - 1024);
            if (!properties)
                return metadataError(ErrorCode::ResourceExhausted, "Image source metadata exceeds the export packet budget");
            EmbeddedImageMetadata result;
            result.format = format;
            result.applied_rotation = applied_rotation;
            result.source_properties = std::move(*properties);
            result.srt_metadata = srt;
            if (srt && srt->point_tolerance_milliseconds && *srt->point_tolerance_milliseconds < 0)
                return metadataError(ErrorCode::InvalidArgument, "SRT point tolerance must be nonnegative");
            if (format != FrameFileFormat::JPEG || sidecar || srt)
                return result;
            return packJpegMetadata(std::move(result));
        } catch (const std::bad_alloc&) {
            return metadataError(ErrorCode::ResourceExhausted, "Image metadata allocation failed");
        }
    }
    Result<std::string> imageXmpPacket(const EmbeddedImageMetadata& metadata, int width, int height, const FrameInfo& info, bool sidecar) {
        auto packet = imageXmpPrefix(metadata, width, height, info, sidecar);
        if (!packet)
            return packet.error();
        try {
            packet->reserve(packet->size() + metadata.source_properties.size() + imageXmpSuffix().size());
            *packet += metadata.source_properties;
            *packet += imageXmpSuffix();
            return packet;
        } catch (const std::bad_alloc&) {
            return metadataError(ErrorCode::ResourceExhausted, "Image XMP packet allocation failed");
        }
    }
    SinkResult writeJpegMetadata(FILE* stream, std::span<const std::uint8_t> jpeg,
                                 const EmbeddedImageMetadata& metadata, int width, int height, const FrameInfo& info) {
        if (!stream || width <= 0 || height <= 0 || metadata.format != FrameFileFormat::JPEG)
            return SinkResult::failure(metadataError(ErrorCode::InvalidArgument, "Invalid encoded JPEG metadata input"));
        if (jpeg.size() < 4 || jpeg[0] != 0xff || jpeg[1] != 0xd8)
            return SinkResult::failure(metadataError(ErrorCode::DataLoss, "JPEG encoder returned an invalid header"));
        try {
            if (metadata.srt_metadata) {
                auto frame_metadata = metadata;
                if (auto appended = appendSrtMetadata(frame_metadata.source_properties, frame_metadata.srt_metadata, info); !appended)
                    return appended;
                frame_metadata.srt_metadata.reset();
                auto prepared = packJpegMetadata(std::move(frame_metadata));
                if (!prepared)
                    return SinkResult::failure(prepared.error());
                return writeJpegMetadata(stream, jpeg, *prepared, width, height, info);
            }
            std::string tree(packet_start);
            tree += tree_start;
            frameProperties(tree, width, height, metadata.applied_rotation, info, jpeg_metadata_policy);
            tree += metadata.source_properties;
            if (!metadata.extended_guid.empty())
                tree += "<n:HasExtendedXMP>" + metadata.extended_guid + "</n:HasExtendedXMP>";
            tree += tree_end;
            tree += packet_end;
            if (tree.size() + sizeof(standard_signature) > max_payload_bytes)
                return SinkResult::failure(metadataError(ErrorCode::ResourceExhausted, "JPEG Standard XMP exceeds APP1 capacity"));
            std::vector<std::uint8_t> markers;
            marker(markers, exif(width, height));
            std::vector<std::uint8_t> packet(standard_signature, standard_signature + sizeof(standard_signature));
            packet.insert(packet.end(), tree.begin(), tree.end());
            marker(markers, packet);
            // Preserve the encoder's JFIF APP0 immediately after SOI, if present.
            std::size_t insert = 2;
            if (jpeg.size() >= 6 && jpeg[2] == 0xff && jpeg[3] == 0xe0) {
                const std::size_t size = (static_cast<std::size_t>(jpeg[4]) << 8) | jpeg[5];
                if (size < 2 || size > jpeg.size() - 4)
                    return SinkResult::failure(metadataError(ErrorCode::DataLoss, "JPEG encoder returned a truncated APP0"));
                insert = 4 + size;
            }
            for (const auto bytes : {jpeg.first(insert), std::span<const std::uint8_t>(markers),
                                     std::span<const std::uint8_t>(metadata.extended_markers), jpeg.subspan(insert)}) {
                if (!bytes.empty() && std::fwrite(bytes.data(), 1, bytes.size(), stream) != bytes.size())
                    return SinkResult::failure(metadataError(ErrorCode::Unavailable, "JPEG metadata output write failed"));
            }
            return {};
        } catch (const std::bad_alloc&) {
            return SinkResult::failure(metadataError(ErrorCode::ResourceExhausted, "JPEG frame metadata allocation failed"));
        }
    }
} // namespace lfs::media::detail
