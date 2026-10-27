// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "media/media_probe.hpp"
#include "media/metadata_observation.hpp"
#include "media/source_exif.hpp"
#include "media/source_metadata.hpp"
#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>
#include <span>
#include <type_traits>
#include <vector>
namespace lfs::media::json_detail {
    using nlohmann::json;
    inline std::string dump(const json& value) {
        // Library diagnostics can contain untrusted non-UTF-8 box names.
        // Binary originals are hex; replace invalid diagnostic text at transport.
        return value.dump(-1, ' ', false, json::error_handler_t::replace);
    }
    inline json rational(const std::optional<Rational>& value) {
        return value ? json::array({value->numerator, value->denominator}) : json(nullptr);
    }
    inline json timestamp(const std::optional<Timestamp>& value) {
        return value ? json{{"ticks", value->ticks}, {"time_base", {value->time_base.numerator, value->time_base.denominator}}} : json(nullptr);
    }
    template <class T>
    inline json optional(const std::optional<T>& value) { return value ? json(*value) : json(nullptr); }
    template <class Bytes>
    inline std::string hexBytes(const Bytes& bytes) {
        constexpr char digits[] = "0123456789abcdef";
        std::string result(bytes.size() * 2, '0');
        for (size_t i = 0; i < bytes.size(); ++i) {
            result[2 * i] = digits[static_cast<uint8_t>(bytes[i]) >> 4];
            result[2 * i + 1] = digits[static_cast<uint8_t>(bytes[i]) & 15];
        }
        return result;
    }
    inline const char* metadataStatus(core::image_codecs::MetadataStatus status) {
        using core::image_codecs::MetadataStatus;
        switch (status) {
        case MetadataStatus::Absent: return "absent";
        case MetadataStatus::Collected: return "collected";
        case MetadataStatus::Limited: return "limited";
        case MetadataStatus::Malformed: return "malformed";
        case MetadataStatus::Unsupported: return "unsupported";
        case MetadataStatus::IoError: return "io_error";
        default: return "not_examined";
        }
    }
    inline const char* exifParseStatus(ExifParseStatus status) {
        switch (status) {
        case ExifParseStatus::Parsed: return "parsed";
        case ExifParseStatus::Limited: return "limited";
        case ExifParseStatus::Malformed: return "malformed";
        default: return "unsupported";
        }
    }
    inline const char* exifEnvelope(core::image_codecs::ExifEnvelope envelope) {
        using core::image_codecs::ExifEnvelope;
        switch (envelope) {
        case ExifEnvelope::JpegApp1: return "jpeg_app1";
        case ExifEnvelope::HeifOffset: return "heif_offset";
        case ExifEnvelope::PngExif: return "png_exif";
        case ExifEnvelope::JxlOffset: return "jxl_offset";
        default: return "unknown";
        }
    }
    inline json sourceExif(const ImageExifDescription& description) {
        const auto& original = description.original;
        json result{{"status", metadataStatus(original.status)}, {"scope", original.scope}, {"diagnostic", original.diagnostic}, {"binary_encoding", "hex"}, {"blocks", json::array()}};
        for (size_t i = 0; i < original.blocks.size(); ++i) {
            const auto& block = original.blocks[i];
            json item{{"envelope", exifEnvelope(block.envelope)},
                      {"image_item", optional(block.image_item)},
                      {"metadata_item", optional(block.metadata_item)},
                      {"bytes", hexBytes(block.bytes)},
                      {"interpretation", nullptr}};
            if (block.jxl_box) {
                item["representation"] = "libjxl_decoded_bytes";
                item["jxl_box"] = {{"ordinal", block.jxl_box->ordinal}, {"brotli_compressed", block.jxl_box->brotli_compressed}};
            }
            if (i < description.interpretations.size()) {
                const auto& parsed = description.interpretations[i];
                json interpretation{{"status", exifParseStatus(parsed.status)}, {"diagnostic", parsed.diagnostic}, {"entries", json::array()}};
                for (const auto& entry : parsed.entries) {
                    json rationals = json::array(), floating = json::array();
                    for (const auto& r : entry.rational_values)
                        rationals.push_back({{"numerator", r.numerator}, {"denominator", r.denominator}});
                    for (const auto value : entry.floating_values)
                        floating.push_back(std::isfinite(value) ? json(value) : json(std::isnan(value) ? "nan" : value < 0 ? "-inf"
                                                                                                                           : "+inf"));
                    interpretation["entries"].push_back({{"directory_path", entry.directory_path}, {"directory_offset", entry.directory_offset}, {"entry_index", entry.entry_index}, {"id", entry.id}, {"type", entry.type}, {"count", entry.count}, {"directory", entry.directory}, {"unsigned_values", entry.unsigned_values}, {"signed_values", entry.signed_values}, {"floating_values", floating}, {"rational_values", rationals}, {"bytes", hexBytes(entry.bytes)}});
                }
                item["interpretation"] = std::move(interpretation);
            }
            result["blocks"].push_back(std::move(item));
        }
        if (original.tiff) {
            const auto& tiff = *original.tiff;
            json archive{{"representation", "libtiff-decoded-tags-and-observed-source-ranges"},
                         {"complete_metadata_archive", false},
                         {"relocatable_exif", false},
                         {"big_tiff", tiff.big_tiff},
                         {"big_endian", tiff.big_endian},
                         {"source_ranges", json::array()},
                         {"directories", json::array()}};
            for (const auto& range : tiff.source_ranges)
                archive["source_ranges"].push_back({{"offset", range.offset}, {"bytes", hexBytes(range.bytes)}});
            for (const auto& directory : tiff.directories) {
                json node{{"offset", directory.offset}, {"parent", optional(directory.parent)}, {"kind", directory.kind}, {"tags", json::array()}};
                for (const auto& tag : directory.tags) {
                    json floating = json::array();
                    for (const auto value : tag.floating_values)
                        floating.push_back(std::isfinite(value) ? json(value) : json(std::isnan(value) ? "nan" : value < 0 ? "-inf"
                                                                                                                           : "+inf"));
                    node["tags"].push_back({{"id", tag.id}, {"type", tag.type}, {"count", tag.count}, {"unsigned_values", tag.unsigned_values}, {"signed_values", tag.signed_values}, {"floating_values", floating}, {"bytes", hexBytes(tag.bytes)}, {"decoded_storage_bytes", tag.decoded_storage_bytes}});
                }
                archive["directories"].push_back(std::move(node));
            }
            result["tiff"] = std::move(archive);
        }
        return result;
    }
    inline json description(const MediaDescription& media);
    inline json metadataText(const MetadataTextEntry& entry) {
        // Text presentation follows the existing UTF-8 replacement policy;
        // the hex fields retain the exact demuxer-exposed key/value bytes.
        return {{"key", entry.key}, {"value", entry.value}, {"key_bytes", hexBytes(entry.key)}, {"value_bytes", hexBytes(entry.value)}};
    }
    inline json metadataReference(const SourceMetadataReference& reference) {
        if (const auto* xmp = std::get_if<XmpBlockReference>(&reference))
            return {{"kind", "xmp_block"}, {"block", xmp->block}};
        if (const auto* exif = std::get_if<ExifEntryReference>(&reference))
            return {{"kind", "exif_entry"}, {"block", exif->block}, {"entry", exif->entry}};
        if (const auto* tiff = std::get_if<TiffTagReference>(&reference))
            return {{"kind", "tiff_tag"}, {"directory", tiff->directory}, {"tag", tiff->tag}};
        if (const auto* tag = std::get_if<ContainerTagReference>(&reference))
            return {{"kind", "container_tag"}, {"entry", tag->entry}};
        if (const auto* stream = std::get_if<StreamDescriptionReference>(&reference))
            return {{"kind", "stream_description"}, {"stream", stream->stream}};
        if (const auto* tag = std::get_if<StreamTagReference>(&reference))
            return {{"kind", "stream_tag"}, {"stream", tag->stream}, {"entry", tag->entry}};
        const auto& block = std::get<StreamSideDataReference>(reference);
        return {{"kind", "stream_side_data"}, {"stream", block.stream}, {"block", block.block}};
    }
    inline json sourceMetadata(const SourceMetadataCatalog& catalog) {
        const char* status = "unknown";
        switch (catalog.indexStatus()) {
        case MetadataIndexStatus::Complete: status = "complete"; break;
        case MetadataIndexStatus::Limited: status = "limited"; break;
        }
        json result{{"scope", catalog.container() ? "container_stream" : "static_image_exif"}, {"index_status", status}, {"records", json::array()}};
        if (!catalog.container()) {
            result["image_exif"] = sourceExif(catalog.imageExif());
            const auto& xmp = catalog.imageXmp();
            json retained{{"status", metadataStatus(xmp.status)}, {"scope", xmp.scope}, {"diagnostic", xmp.diagnostic}, {"binary_encoding", "hex"}, {"xml_validation", "not_examined"}, {"extended_assembly", "not_performed"}, {"blocks", json::array()}};
            for (std::size_t index = 0; index < xmp.blocks.size(); ++index) {
                const auto& block = xmp.blocks[index];
                const auto original = catalog.xmpBytes({index});
                using core::image_codecs::XmpEnvelope;
                const char* envelope = block.envelope == XmpEnvelope::JpegApp1 ? "jpeg_app1" : block.envelope == XmpEnvelope::JpegExtendedApp1   ? "jpeg_extended_app1"
                                                                                           : block.envelope == XmpEnvelope::PngInternationalText ? "png_itxt"
                                                                                           : block.envelope == XmpEnvelope::TiffTag              ? "tiff_tag700"
                                                                                           : block.envelope == XmpEnvelope::HeifMime             ? "heif_mime"
                                                                                                                                                 : "jxl_xml";
                json payload = nullptr;
                if (const auto view = catalog.xmpPayloadBytes({index}))
                    payload = hexBytes(*view);
                retained["blocks"].push_back({{"envelope", envelope}, {"bytes", original ? json(hexBytes(*original)) : json(nullptr)}, {"status", metadataStatus(block.status)}, {"diagnostic", block.diagnostic}, {"crc_valid", optional(block.crc_valid)}, {"payload_offset", block.payload_offset}, {"payload_kind", block.envelope == XmpEnvelope::JpegExtendedApp1 ? "extended_segment" : "packet"}, {"compressed", block.compressed}, {"payload_bytes", std::move(payload)}});
                if (block.heif_item) {
                    const auto& item = *block.heif_item;
                    retained["blocks"].back()["representation"] = "libheif_decoded_bytes";
                    retained["blocks"].back()["heif_item"] = {{"image_item", item.image_item}, {"metadata_item", item.metadata_item}, {"content_type", item.content_type}, {"content_type_bytes", hexBytes(item.content_type)}, {"content_encoding", item.content_encoding}, {"content_encoding_bytes", hexBytes(item.content_encoding)}};
                }
                if (block.jxl_box) {
                    retained["blocks"].back()["representation"] = "libjxl_decoded_bytes";
                    retained["blocks"].back()["jxl_box"] = {{"ordinal", block.jxl_box->ordinal}, {"brotli_compressed", block.jxl_box->brotli_compressed}};
                }
                if (block.tiff_tag) {
                    retained["blocks"].back()["representation"] = "libtiff_decoded_bytes";
                    const auto reference = *block.tiff_tag;
                    retained["blocks"].back()["tiff_tag"] = {{"directory", reference.directory}, {"tag", reference.tag}, {"directory_offset", catalog.imageExif().original.tiff->directories[reference.directory].offset}};
                }
            }
            result["image_xmp"] = std::move(retained);
        }
        if (catalog.container()) {
            result["description"] = description(*catalog.container());
            result["source_metadata"] = nullptr;
            if (catalog.container()->source_metadata) {
                const auto& metadata = *catalog.container()->source_metadata;
                json source{{"limited", metadata.limited}, {"binary_encoding", "hex"}, {"tags", json::array()}, {"streams", json::array()}};
                source["provider"] = metadata.provider ? json{{"name", metadata.provider->name}, {"version", metadata.provider->version}, {"codec_api_version", metadata.provider->codec_api_version}} : json(nullptr);
                for (const auto& tag : metadata.tags)
                    source["tags"].push_back(metadataText(tag));
                for (const auto& stream : metadata.streams) {
                    json node{{"stream_index", stream.stream_index}, {"tags", json::array()}, {"coded_side_data", json::array()}};
                    for (const auto& tag : stream.tags)
                        node["tags"].push_back(metadataText(tag));
                    for (const auto& block : stream.coded_side_data)
                        node["coded_side_data"].push_back({{"ffmpeg_type", block.type}, {"bytes", hexBytes(block.bytes)}});
                    source["streams"].push_back(std::move(node));
                }
                result["source_metadata"] = std::move(source);
            }
        }
        const auto& locator = catalog.sourceLocator();
        result["source_locator"] = locator ? json(locator->display) : json(nullptr);
        result["source_locator_bytes"] = locator ? json(hexBytes(locator->native_bytes)) : json(nullptr);
        const char* encoding = "unknown";
        if (locator) {
            switch (locator->encoding) {
            case MetadataPathEncoding::PosixBytes: encoding = "posix_bytes"; break;
            case MetadataPathEncoding::Utf16LE: encoding = "utf16le"; break;
            }
        }
        result["source_locator_encoding"] = locator ? json(encoding) : json(nullptr);
        result["identity_scope"] = "in_memory_observation";
        for (const auto& reference : catalog.records())
            result["records"].push_back(metadataReference(reference));
        return result;
    }
    inline json metadataSummary(const MetadataSummary& summary) {
        const auto scope = summary.scope == MetadataSummaryScope::StreamProperties ? "static_stream_properties" : summary.scope == MetadataSummaryScope::ImageProperties ? "static_image_properties"
                                                                                                                                                                         : "unknown";
        json result{{"scope", scope}, {"limited", summary.limited}, {"facts", json::array()}};
        std::vector<const XmpPacketAssembly*> assemblies;
        const auto add_assembly = [&](const std::shared_ptr<const XmpPacketAssembly>& value) {
            if (!value || std::find(assemblies.begin(), assemblies.end(), value.get()) != assemblies.end())
                return;
            assemblies.push_back(value.get());
            if (!result.contains("xmp_assemblies"))
                result["xmp_assemblies"] = json::array();
            json item{{"guid", value->guid}, {"declared_bytes", value->declared_bytes}, {"md5_verified", value->md5_verified}, {"standard_source", value->standard_source ? metadataReference(value->standard_source->reference) : json(nullptr)}, {"fragments", json::array()}};
            for (const auto& fragment : value->fragments)
                item["fragments"].push_back({{"source", metadataReference(fragment.source.reference)},
                                             {"offset", fragment.offset},
                                             {"bytes", fragment.bytes}});
            result["xmp_assemblies"].push_back(std::move(item));
        };
        for (const auto& fact : summary.facts) {
            json value;
            std::visit([&](const auto& scalar) {
                using T = std::decay_t<decltype(scalar)>;
                if constexpr (std::is_same_v<T, ExifRational>)
                    value = {{"numerator", scalar.numerator}, {"denominator", scalar.denominator}};
                else if constexpr (std::is_same_v<T, double>)
                    value = std::isfinite(scalar) ? json(scalar) : json(std::isnan(scalar) ? "nan" : scalar < 0 ? "-inf"
                                                                                                                : "+inf");
                else
                    value = scalar;
            },
                       fact.value);
            const char* state = "unknown";
            switch (fact.state) {
            case MetadataValueState::Present: state = "present"; break;
            case MetadataValueState::Absent: state = "absent"; break;
            case MetadataValueState::Invalid: state = "invalid"; break;
            case MetadataValueState::Unsupported: state = "unsupported"; break;
            }
            const char* origin = "unknown";
            switch (fact.origin) {
            case MetadataValueOrigin::InterpretedSource: origin = "interpreted_source"; break;
            case MetadataValueOrigin::DemuxerObservation: origin = "demuxer_observation"; break;
            case MetadataValueOrigin::Derived: origin = "derived"; break;
            }
            result["facts"].push_back({{"semantic", fact.semantic}, {"value", std::move(value)}, {"unit", metadataUnitId(fact.unit)}, {"state", state}, {"origin", origin}, {"temporal_scope", "static"}, {"source", metadataReference(fact.source.reference)}});
            if (summary.scope == MetadataSummaryScope::ImageProperties) {
                auto& item = result["facts"].back();
                item["inputs"] = json::array();
                for (const auto& input : fact.inputs)
                    item["inputs"].push_back(metadataReference(input.reference));
                item["derivation"] = fact.derivation;
                item["diagnostic"] = fact.diagnostic;
                item["ambiguous"] = fact.ambiguous;
                const auto location = [](const XmpPropertyLocation& value) {
                    json encoded{{"namespace", value.field ? value.field->namespace_uri : std::string_view{}}, {"name", value.field ? value.field->name : std::string_view{}}, {"description", value.description}, {"ordinal", value.ordinal}};
                    if (!value.subject.empty())
                        encoded["rdf_about"] = value.subject;
                    return encoded;
                };
                if (fact.xmp) {
                    item["xmp_property"] = location(fact.xmp->property);
                    item["xmp_inputs"] = json::array();
                    for (std::size_t i = 0; i < fact.xmp->inputs.size(); ++i) {
                        auto input_location = location(fact.xmp->inputs[i]);
                        if (i < fact.inputs.size()) {
                            const auto* own = std::get_if<XmpBlockReference>(&fact.source.reference);
                            const auto* input = std::get_if<XmpBlockReference>(&fact.inputs[i].reference);
                            if (!own || !input || own->block != input->block || fact.source.observation != fact.inputs[i].observation)
                                input_location["source"] = metadataReference(fact.inputs[i].reference);
                        }
                        item["xmp_inputs"].push_back(std::move(input_location));
                    }
                    if (fact.xmp->assembly) {
                        const auto& assembly = *fact.xmp->assembly;
                        add_assembly(fact.xmp->assembly);
                        item["xmp_packet_kind"] = fact.xmp->extended_packet ? "jpeg_extended_assembly" : "retained_packet";
                        item["xmp_assembly"] = {{"guid", assembly.guid},
                                                {"source", assembly.fragments.empty() ? json(nullptr) : metadataReference(assembly.fragments.front().source.reference)}};
                    }
                }
            }
        }
        if (!summary.xmp_packets.empty()) {
            result["xmp_packets"] = json::array();
            for (const auto& packet : summary.xmp_packets) {
                const char* state = packet.state == XmpPacketState::Parsed      ? "parsed"
                                    : packet.state == XmpPacketState::Malformed ? "malformed"
                                    : packet.state == XmpPacketState::Limited   ? "limited"
                                                                                : "unsupported";
                result["xmp_packets"].push_back({{"source", metadataReference(packet.source.reference)},
                                                 {"state", state},
                                                 {"diagnostic", packet.diagnostic}});
                if (packet.assembly) {
                    add_assembly(packet.assembly);
                    result["xmp_packets"].back()["extended_xmp"] = packet.assembly->guid;
                }
            }
        }
        return result;
    }
    inline const char* streamKind(StreamKind kind) {
        switch (kind) {
        case StreamKind::Video: return "video";
        case StreamKind::Audio: return "audio";
        case StreamKind::Subtitle: return "subtitle";
        case StreamKind::Data: return "data";
        case StreamKind::Attachment: return "attachment";
        default: return "unknown";
        }
    }
    inline const char* orientationSource(OrientationSource source) {
        switch (source) {
        case OrientationSource::RotateTag: return "rotate_tag";
        case OrientationSource::DisplayMatrix: return "display_matrix";
        default: return "none";
        }
    }
    inline json description(const MediaDescription& media) {
        json result{{"container", media.container}, {"stream_info_probed", media.stream_info_probed}, {"start", timestamp(media.start)}, {"duration", timestamp(media.duration)}, {"selected_video_stream", optional(media.selected_video_stream)}, {"streams", json::array()}};
        for (const auto& s : media.streams) {
            result["streams"].push_back({{"index", s.index}, {"kind", streamKind(s.kind)}, {"codec", s.codec}, {"width", optional(s.width)}, {"height", optional(s.height)}, {"pixel_format", optional(s.pixel_format)}, {"time_base", rational(s.time_base)}, {"nominal_frame_rate", rational(s.nominal_frame_rate)}, {"average_frame_rate", rational(s.average_frame_rate)}, {"sample_aspect_ratio", rational(s.sample_aspect_ratio)}, {"start", timestamp(s.start)}, {"duration", timestamp(s.duration)}, {"declared_frame_count", optional(s.declared_frame_count)}, {"attached_picture", s.attached_picture}, {"default_disposition", s.default_disposition}, {"color", {{"primaries", optional(s.color.primaries)}, {"transfer", optional(s.color.transfer)}, {"matrix", optional(s.color.matrix)}, {"range", optional(s.color.range)}, {"component_depth", optional(s.color.component_depth)}}}, {"orientation", {{"source", orientationSource(s.orientation.source)}, {"rotate_tag", optional(s.orientation.rotate_tag)}, {"display_matrix", optional(s.orientation.display_matrix)}, {"clockwise_degrees", optional(s.orientation.clockwise_degrees)}, {"reflected", optional(s.orientation.reflected)}}}});
            if (s.orientation.rotate_tag_limited)
                result["streams"].back()["orientation"]["rotate_tag_limited"] = true;
        }
        return result;
    }
    inline json sourceMetadataResult(const SourceMetadataCatalog& catalog, const MetadataSummary& summary) {
        return {{"schema_version", 1}, {"success", true}, {"metadata", sourceMetadata(catalog)}, {"summary", metadataSummary(summary)}};
    }
} // namespace lfs::media::json_detail
