// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "image_metadata.hpp"
#include "media_json.hpp"
#include "xmp_metadata.hpp"
#include <algorithm>
#include <limits>

namespace lfs::media::detail {
    namespace {
        constexpr char standard_signature[] = "http://ns.adobe.com/xap/1.0/";
        constexpr char extended_signature[] = "http://ns.adobe.com/xmp/extension/";
        constexpr std::size_t max_packet_bytes = 32 * 1024 * 1024;
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
            for (char c : value) {
                if (c == '&')
                    out += "&amp;";
                else if (c == '<')
                    out += "&lt;";
                else if (c == '>')
                    out += "&gt;";
                else
                    out += c;
            }
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
    std::string_view embeddedMetadataPolicy(FrameFileFormat format) {
        return format == FrameFileFormat::JPEG     ? std::string_view(jpeg_metadata_policy)
               : format == FrameFileFormat::PNG    ? std::string_view(png_metadata_policy)
               : format == FrameFileFormat::TIFF16 ? std::string_view(tiff_metadata_policy)
               : format == FrameFileFormat::EXR    ? std::string_view(exr_metadata_policy)
                                                   : std::string_view{};
    }
    std::string_view imageXmpSuffix() {
        // Rewriting the packet requires updating the carrier lengths/structure.
        return "</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end=\"r\"?>";
    }
    Result<std::string> imageXmpPrefix(const EmbeddedImageMetadata& metadata, int width, int height, const FrameInfo& info) {
        if ((metadata.format != FrameFileFormat::PNG && metadata.format != FrameFileFormat::TIFF16 && metadata.format != FrameFileFormat::EXR) || width <= 0 || height <= 0)
            return metadataError(ErrorCode::InvalidArgument, "Invalid image metadata input");
        try {
            std::string tree(packet_start);
            tree += tree_start;
            frameProperties(tree, width, height, metadata.applied_rotation, info, embeddedMetadataPolicy(metadata.format));
            if (tree.size() + metadata.source_properties.size() + imageXmpSuffix().size() > max_packet_bytes)
                return metadataError(ErrorCode::ResourceExhausted, "Image XMP exceeds the export packet budget");
            return tree;
        } catch (const std::bad_alloc&) {
            return metadataError(ErrorCode::ResourceExhausted, "Image frame metadata allocation failed");
        }
    }
    Result<EmbeddedImageMetadata> prepareImageMetadata(const SourceMetadataCatalog& catalog, int applied_rotation, FrameFileFormat format) {
        if (embeddedMetadataPolicy(format).empty())
            return metadataError(ErrorCode::InvalidArgument, "Embedded metadata requires JPEG, PNG, TIFF16 or EXR output");
        try {
            const auto summary = summarizeSourceMetadata(catalog);
            if (!summary)
                return summary.error();
            // ASCII JSON escapes controls and non-ASCII before XML embedding.
            const auto json = json_detail::sourceMetadataResult(catalog, *summary).dump(-1, ' ', true, nlohmann::json::error_handler_t::replace);
            std::size_t escaped_bytes = json.size();
            for (char c : json) {
                if (c == '&')
                    escaped_bytes += 4;
                else if (c == '<' || c == '>')
                    escaped_bytes += 3;
            }
            if (escaped_bytes > max_packet_bytes - 1024)
                return metadataError(ErrorCode::ResourceExhausted, "Image source metadata exceeds the export packet budget");
            EmbeddedImageMetadata result;
            result.format = format;
            result.applied_rotation = applied_rotation;
            property(result.source_properties, "SourceMetadata", json);
            if (format != FrameFileFormat::JPEG)
                return result;
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
                            std::numeric_limits<int>::max(), applied_rotation, largest, jpeg_metadata_policy);
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
        } catch (const std::bad_alloc&) {
            return metadataError(ErrorCode::ResourceExhausted, "Image metadata allocation failed");
        }
    }
    Result<std::string> imageXmpPacket(const EmbeddedImageMetadata& metadata, int width, int height, const FrameInfo& info) {
        auto packet = imageXmpPrefix(metadata, width, height, info);
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
