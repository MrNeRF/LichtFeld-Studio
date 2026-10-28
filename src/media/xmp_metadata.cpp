// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "xmp_metadata.hpp"
#include "metadata_summary_common.hpp"
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <expat.h>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <string_view>

namespace lfs::media {
    namespace {
        constexpr std::string_view rdf = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
        constexpr std::string_view tiff = "http://ns.adobe.com/tiff/1.0/";
        constexpr std::string_view exif = "http://ns.adobe.com/exif/1.0/";
        constexpr std::string_view exif_ex = "http://cipa.jp/exif/1.0/";
        constexpr std::string_view aux = "http://ns.adobe.com/exif/1.0/aux/";
        constexpr std::string_view xmp = "http://ns.adobe.com/xap/1.0/";
        constexpr std::string_view note = "http://ns.adobe.com/xmp/note/";
        constexpr std::string_view dji = "http://www.dji.com/drone-dji/1.0/";
        constexpr std::string_view dji_subject = "DJI Meta Data";
        enum class Kind { Text,
                          Number,
                          Orientation,
                          Latitude,
                          Longitude,
                          Altitude,
                          AltitudeRef,
                          SignedNumber,
                          DecimalLatitude,
                          DecimalLongitude,
                          SourceFlag,
                          Date };
        struct Field {
            XmpPropertyName property;
            std::string_view semantic;
            Kind kind = Kind::Text;
            MetadataUnit unit = MetadataUnit::Unspecified;
        };
        constexpr Field fields[]{
            {{tiff, "Make"}, summary_detail::semantic::make},
            {{tiff, "Model"}, summary_detail::semantic::model},
            {{tiff, "Orientation"}, summary_detail::semantic::orientation, Kind::Orientation, MetadataUnit::Unitless},
            {{aux, "Lens"}, summary_detail::semantic::lens_model},
            {{exif_ex, "LensModel"}, summary_detail::semantic::lens_model},
            {{exif_ex, "LensMake"}, summary_detail::semantic::lens_make},
            {{exif, "ExposureTime"}, summary_detail::semantic::exposure, Kind::Number, MetadataUnit::Seconds},
            {{exif, "FNumber"}, summary_detail::semantic::f_number, Kind::Number, MetadataUnit::Unitless},
            {{exif, "FocalLength"}, summary_detail::semantic::focal_length, Kind::Number, MetadataUnit::Millimetres},
            {{exif, "GPSLatitude"}, summary_detail::semantic::latitude, Kind::Latitude, MetadataUnit::Degrees},
            {{exif, "GPSLongitude"}, summary_detail::semantic::longitude, Kind::Longitude, MetadataUnit::Degrees},
            {{exif, "GPSAltitude"}, summary_detail::semantic::altitude, Kind::Altitude, MetadataUnit::Metres},
            {{exif, "GPSAltitudeRef"}, "gps.altitude_reference", Kind::AltitudeRef, MetadataUnit::Unitless},
            {{exif, "GPSMapDatum"}, summary_detail::semantic::datum},
            {{exif, "DateTimeOriginal"}, summary_detail::semantic::original, Kind::Date},
            {{exif, "DateTimeDigitized"}, summary_detail::semantic::digitized, Kind::Date},
            {{xmp, "CreateDate"}, summary_detail::semantic::digitized, Kind::Date},
            {{xmp, "ModifyDate"}, summary_detail::semantic::modified, Kind::Date}};
        // Vendor observations do not establish a camera optical center, a
        // vertical datum or an exposure epoch comparable with standard GPS.
        constexpr Field dji_fields[]{
            {{dji, "GpsLatitude"}, "vendor.dji.position.latitude", Kind::DecimalLatitude, MetadataUnit::Degrees},
            {{dji, "GpsLongitude"}, "vendor.dji.position.longitude", Kind::DecimalLongitude, MetadataUnit::Degrees},
            {{dji, "GpsLongtitude"}, "vendor.dji.position.longitude", Kind::DecimalLongitude, MetadataUnit::Degrees},
            {{dji, "AbsoluteAltitude"}, "vendor.dji.altitude.absolute", Kind::SignedNumber, MetadataUnit::Metres},
            {{dji, "RelativeAltitude"}, "vendor.dji.altitude.relative", Kind::SignedNumber, MetadataUnit::Metres},
            {{dji, "FlightRollDegree"}, "vendor.dji.flight.roll", Kind::SignedNumber, MetadataUnit::Degrees},
            {{dji, "FlightPitchDegree"}, "vendor.dji.flight.pitch", Kind::SignedNumber, MetadataUnit::Degrees},
            {{dji, "FlightYawDegree"}, "vendor.dji.flight.yaw", Kind::SignedNumber, MetadataUnit::Degrees},
            {{dji, "GimbalRollDegree"}, "vendor.dji.gimbal.roll", Kind::SignedNumber, MetadataUnit::Degrees},
            {{dji, "GimbalPitchDegree"}, "vendor.dji.gimbal.pitch", Kind::SignedNumber, MetadataUnit::Degrees},
            {{dji, "GimbalYawDegree"}, "vendor.dji.gimbal.yaw", Kind::SignedNumber, MetadataUnit::Degrees},
            {{dji, "GpsStatus"}, "vendor.dji.gps.status"},
            {{dji, "AltitudeType"}, "vendor.dji.altitude.type"},
            {{dji, "SurveyingMode"}, "vendor.dji.surveying_mode", Kind::SourceFlag, MetadataUnit::Unitless},
            {{dji, "CamReverse"}, "vendor.dji.camera.reverse", Kind::SourceFlag, MetadataUnit::Unitless},
            {{dji, "Version"}, "vendor.dji.version"}};
        bool named(std::string_view expanded, std::string_view uri, std::string_view name) {
            return expanded.size() == uri.size() + name.size() + 1 && expanded.starts_with(uri) &&
                   expanded[uri.size()] == '\x1f' && expanded.substr(uri.size() + 1) == name;
        }
        const Field* field(std::string_view name) {
            if (name.size() > dji.size() && name.starts_with(dji) && name[dji.size()] == '\x1f') {
                for (const auto& candidate : dji_fields)
                    if (named(name, candidate.property.namespace_uri, candidate.property.name))
                        return &candidate;
                return nullptr;
            }
            for (const auto& candidate : fields)
                if (named(name, candidate.property.namespace_uri, candidate.property.name))
                    return &candidate;
            return nullptr;
        }
        std::string_view trim(std::string_view value) {
            const auto begin = value.find_first_not_of(" \r\n\t");
            return begin == value.npos ? std::string_view{} : value.substr(begin, value.find_last_not_of(" \r\n\t") - begin + 1);
        }
        struct Property {
            const Field* field;
            std::string value;
            std::size_t description, ordinal;
            bool structured = false;
            std::string_view subject;
            XmpPropertyLocation location() const {
                return {&field->property, description, ordinal, subject};
            }
        };
        enum class Role { Wrapper,
                          Rdf,
                          Description,
                          Property,
                          ExtendedLink,
                          Other };
        struct Frame {
            Role role;
            std::optional<std::size_t> property;
        };
        struct Packet {
            XML_Parser parser = nullptr;
            std::vector<Frame> stack;
            std::vector<Property> properties;
            XmpPacketState state = XmpPacketState::Parsed;
            const char* diagnostic = "";
            bool oom = false;
            std::size_t elements = 0, descriptions = 0, ordinal = 0, rdf_roots = 0, text_bytes = 0;
            std::size_t max_properties, max_text;
            bool collect_links = false, invalid_link = false;
            std::size_t link_count = 0;
            std::string extended_guid;
            std::string_view subject;
            bool named_subject = false;
            void linkText(std::string_view value) {
                if (invalid_link || link_count != 1)
                    return;
                if (value.size() > 64 - extended_guid.size()) {
                    invalid_link = true;
                    extended_guid.clear();
                    return;
                }
                extended_guid.append(value);
            }
            void beginLink(bool qualified) {
                ++link_count;
                invalid_link |= qualified || link_count != 1;
            }
            void stop(XmpPacketState reason, const char* message) noexcept {
                if (state == XmpPacketState::Parsed) {
                    state = reason;
                    diagnostic = message;
                    XML_StopParser(parser, XML_FALSE);
                }
            }
            template <class Action>
            void guard(Action&& action) noexcept {
                if (state != XmpPacketState::Parsed)
                    return;
                try {
                    action();
                } catch (const std::bad_alloc&) {
                    oom = true;
                    stop(XmpPacketState::Limited, "XMP allocation failed");
                } catch (...) {
                    // LFS-CENSUS-OK(empty-catch): never unwind a C++ exception through Expat's C callback stack.
                    stop(XmpPacketState::Malformed, "XMP callback failed");
                }
            }
            std::optional<std::size_t> add(const Field* field, std::string_view value) {
                if (!field || (!subject.empty() && field->property.namespace_uri != dji))
                    return std::nullopt;
                if (properties.size() >= max_properties || value.size() > max_text - text_bytes) {
                    stop(XmpPacketState::Limited, "XMP property or text budget exhausted");
                    return std::nullopt;
                }
                text_bytes += value.size();
                properties.push_back({field, std::string(value), descriptions - 1, ordinal, false, subject});
                return properties.size() - 1;
            }
            void start(std::string_view name, const XML_Char** attrs) {
                if (++elements > 16384 || stack.size() >= 32 || name.size() > 1024) {
                    stop(XmpPacketState::Limited, "XMP XML structure budget exhausted");
                    return;
                }
                std::size_t attributes = 0;
                for (; attrs[attributes * 2]; ++attributes) {
                    if (attributes >= 128 || std::string_view(attrs[attributes * 2]).size() > 1024) {
                        stop(XmpPacketState::Limited, "XMP XML attribute budget exhausted");
                        return;
                    }
                }
                Role role = Role::Other;
                std::optional<std::size_t> selected;
                const auto parent = stack.empty() ? Role::Other : stack.back().role;
                if (stack.empty() && named(name, "adobe:ns:meta/", "xmpmeta")) {
                    role = Role::Wrapper;
                } else if ((stack.empty() || parent == Role::Wrapper) && named(name, rdf, "RDF")) {
                    role = Role::Rdf;
                    if (++rdf_roots > 1) {
                        stop(XmpPacketState::Unsupported, "Multiple RDF roots are not interpreted");
                        return;
                    }
                } else if (parent == Role::Rdf && named(name, rdf, "Description")) {
                    role = Role::Description;
                    ++descriptions;
                    subject = {};
                    // Inspect the subject before collecting any attributes:
                    // XML attribute order must not change which fields survive.
                    for (std::size_t i = 0; i < attributes; ++i)
                        if (named(attrs[i * 2], rdf, "about") && *attrs[i * 2 + 1]) {
                            if (std::string_view(attrs[i * 2 + 1]) != dji_subject) {
                                stop(XmpPacketState::Unsupported, "RDF subject or description qualifier is not interpreted");
                                return;
                            }
                            subject = dji_subject;
                            named_subject = true;
                        }
                    for (std::size_t i = 0; i < attributes; ++i) {
                        const std::string_view key(attrs[i * 2]), value(attrs[i * 2 + 1]);
                        if (named(key, rdf, "about"))
                            continue;
                        if (key.starts_with(rdf) || key.starts_with("http://www.w3.org/XML/1998/namespace\x1f")) {
                            stop(XmpPacketState::Unsupported, "RDF subject or description qualifier is not interpreted");
                            return;
                        }
                        ++ordinal;
                        if (collect_links && named(key, note, "HasExtendedXMP")) {
                            beginLink(!subject.empty());
                            linkText(value);
                        } else {
                            add(field(key), value);
                        }
                        if (state != XmpPacketState::Parsed)
                            return;
                    }
                } else if (parent == Role::Description) {
                    ++ordinal;
                    if (collect_links && named(name, note, "HasExtendedXMP")) {
                        role = Role::ExtendedLink;
                        beginLink(attributes != 0 || !subject.empty());
                    } else {
                        role = Role::Property;
                        selected = add(field(name), {});
                        if (selected && attributes)
                            properties[*selected].structured = true;
                        if (!selected)
                            role = Role::Other;
                    }
                } else if (parent == Role::ExtendedLink) {
                    invalid_link = true;
                } else if (parent == Role::Property && stack.back().property) {
                    properties[*stack.back().property].structured = true;
                }
                if (role == Role::Rdf && attributes) {
                    stop(XmpPacketState::Unsupported, "Qualified RDF root is not interpreted");
                    return;
                }
                if (role == Role::Wrapper)
                    for (std::size_t i = 0; i < attributes; ++i)
                        if (named(attrs[i * 2], "http://www.w3.org/XML/1998/namespace", "base") ||
                            named(attrs[i * 2], "http://www.w3.org/XML/1998/namespace", "lang")) {
                            stop(XmpPacketState::Unsupported, "Inherited XML qualifier is not interpreted");
                            return;
                        }
                stack.push_back({role, selected});
            }
            void text(std::string_view value) {
                if (!stack.empty() && stack.back().role == Role::ExtendedLink) {
                    linkText(value);
                } else if (!stack.empty() && stack.back().role == Role::Property && stack.back().property) {
                    auto& property = properties[*stack.back().property];
                    if (value.size() > max_text - text_bytes) {
                        stop(XmpPacketState::Limited, "XMP property text budget exhausted");
                        return;
                    }
                    property.value.append(value);
                    text_bytes += value.size();
                } else if (!trim(value).empty() && !stack.empty() && stack.back().role != Role::Other) {
                    stop(XmpPacketState::Unsupported, "RDF mixed content is not interpreted");
                }
            }
        };
        Packet parse(std::span<const uint8_t> bytes, std::size_t max_properties, std::size_t max_text, bool collect_links = false) {
            Packet packet{.max_properties = max_properties, .max_text = max_text, .collect_links = collect_links};
            std::unique_ptr<XML_ParserStruct, decltype(&XML_ParserFree)> parser(XML_ParserCreateNS(nullptr, '\x1f'), XML_ParserFree);
            if (!parser)
                throw std::bad_alloc();
            packet.parser = parser.get();
            XML_SetUserData(parser.get(), &packet);
            XML_SetParamEntityParsing(parser.get(), XML_PARAM_ENTITY_PARSING_NEVER);
            XML_SetNamespaceDeclHandler(parser.get(), [](void* context, const XML_Char* prefix, const XML_Char* uri) {
                auto& packet = *static_cast<Packet*>(context);
                if ((prefix && std::string_view(prefix).size() > 1024) || (uri && std::string_view(uri).size() > 1024))
                    packet.stop(XmpPacketState::Limited, "XMP namespace name budget exhausted"); }, nullptr);
            XML_SetStartDoctypeDeclHandler(parser.get(), [](void* context, const XML_Char*, const XML_Char*, const XML_Char*, int) {
                static_cast<Packet*>(context)->stop(XmpPacketState::Unsupported, "XMP DTDs are disabled");
            });
            XML_SetExternalEntityRefHandler(parser.get(), [](XML_Parser, const XML_Char*, const XML_Char*, const XML_Char*, const XML_Char*) -> int { return XML_STATUS_ERROR; });
            XML_SetElementHandler(parser.get(), [](void* context, const XML_Char* name, const XML_Char** attrs) {
                auto& packet = *static_cast<Packet*>(context); packet.guard([&] { packet.start(name, attrs); }); }, [](void* context, const XML_Char*) {
                auto& packet = *static_cast<Packet*>(context); packet.guard([&] { packet.stack.pop_back(); }); });
            XML_SetCharacterDataHandler(parser.get(), [](void* context, const XML_Char* value, int size) {
                auto& packet = *static_cast<Packet*>(context);
                packet.guard([&] { packet.text({value, static_cast<std::size_t>(size)}); });
            });
            const auto status = XML_Parse(parser.get(), reinterpret_cast<const char*>(bytes.data()), static_cast<int>(bytes.size()), XML_TRUE);
            if (packet.oom || XML_GetErrorCode(parser.get()) == XML_ERROR_NO_MEMORY)
                throw std::bad_alloc();
            if (packet.state == XmpPacketState::Parsed && status != XML_STATUS_OK) {
                packet.state = XmpPacketState::Malformed;
                packet.diagnostic = "Malformed XMP XML";
            }
            if (packet.state == XmpPacketState::Parsed && packet.rdf_roots != 1) {
                packet.state = XmpPacketState::Unsupported;
                packet.diagnostic = "XML packet has no supported RDF root";
            }
            packet.parser = nullptr;
            return packet;
        }
        Packet preparePacket(const SourceMetadataCatalog& catalog, const XmpBlockReference& reference,
                             std::size_t max_properties, std::size_t max_text, std::size_t& input_bytes) {
            using core::image_codecs::MetadataStatus;
            Packet packet{.max_properties = max_properties, .max_text = max_text};
            const auto* block = catalog.xmpBlock(reference);
            const auto bytes = catalog.xmpPayloadBytes(reference);
            if (!bytes) {
                packet.state = block->status == MetadataStatus::Limited     ? XmpPacketState::Limited
                               : block->status == MetadataStatus::Malformed ? XmpPacketState::Malformed
                                                                            : XmpPacketState::Unsupported;
                packet.diagnostic = "XMP packet payload is unavailable for interpretation";
            } else if (block->tiff_tag && block->tiff_tag->directory != 0) {
                packet.state = XmpPacketState::Unsupported;
                packet.diagnostic = "XMP belongs to a non-primary TIFF directory";
            } else if (bytes->size() > 1024 * 1024 || bytes->size() > 4 * 1024 * 1024 - input_bytes) {
                packet.state = XmpPacketState::Limited;
                packet.diagnostic = "XMP interpretation input budget exhausted";
            } else {
                input_bytes += bytes->size();
                return parse(*bytes, max_properties, max_text, block->envelope == core::image_codecs::XmpEnvelope::JpegApp1);
            }
            return packet;
        }
        bool integer(std::string_view value, int64_t& result) {
            if (value.starts_with('+')) {
                value.remove_prefix(1);
                if (value.starts_with('-'))
                    return false;
            }
            if (value.empty())
                return false;
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
            return !value.empty() && parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size();
        }
        bool real(std::string_view value, double& result) {
            if (value.starts_with('+')) {
                value.remove_prefix(1);
                if (value.starts_with('-'))
                    return false;
            }
            if (value.empty())
                return false;
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result, std::chars_format::general);
            return !value.empty() && parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() && std::isfinite(result);
        }
        bool number(std::string_view value, double& result, MetadataScalar& stored) {
            const auto slash = value.find('/');
            if (slash != value.npos) {
                int64_t n = 0, d = 0;
                if (!integer(value.substr(0, slash), n) || !integer(value.substr(slash + 1), d) || n < 0 || d <= 0)
                    return false;
                result = double(n) / double(d);
                stored = ExifRational{n, d};
                return true;
            }
            if (!real(value, result) || result < 0)
                return false;
            stored = result;
            return true;
        }
        bool coordinate(std::string_view value, bool latitude, double& result) {
            if (value.empty())
                return false;
            const char direction = value.back();
            value.remove_suffix(1);
            if (latitude ? direction != 'N' && direction != 'S' : direction != 'E' && direction != 'W')
                return false;
            const auto decimal_component = [](std::string_view text, bool fraction) {
                const auto dot = text.find('.');
                if (dot != text.npos && (!fraction || dot == 0 || dot + 1 == text.size()))
                    return false;
                for (std::size_t i = 0; i < text.size(); ++i)
                    if ((text[i] < '0' || text[i] > '9') && i != dot)
                        return false;
                return !text.empty();
            };
            const auto comma = value.find(',');
            if (comma == value.npos || !decimal_component(value.substr(0, comma), false))
                return false;
            int64_t degrees = 0;
            if (!integer(value.substr(0, comma), degrees) || degrees < 0)
                return false;
            value.remove_prefix(comma + 1);
            double minutes = 0, seconds = 0;
            const auto second_comma = value.find(',');
            if (second_comma == value.npos) {
                if (!decimal_component(value, true) || !real(value, minutes))
                    return false;
            } else {
                int64_t whole_minutes = 0;
                if (!decimal_component(value.substr(0, second_comma), false) || !decimal_component(value.substr(second_comma + 1), true) ||
                    !integer(value.substr(0, second_comma), whole_minutes) || !real(value.substr(second_comma + 1), seconds))
                    return false;
                minutes = double(whole_minutes);
            }
            return summary_detail::coordinate(double(degrees), minutes, seconds, latitude, direction, result);
        }

        MetadataValueState date(std::string_view value) {
            const auto digits = [](std::string_view part) { return !part.empty() && std::all_of(part.begin(), part.end(), [](char c) { return c >= '0' && c <= '9'; }); };
            const auto decimal = [](std::string_view part) { int n = 0; for (char c : part) n = n * 10 + c - '0'; return n; };
            if (value.size() < 4 || !digits(value.substr(0, 4)) || decimal(value.substr(0, 4)) == 0)
                return MetadataValueState::Invalid;
            if (value.size() == 4)
                return MetadataValueState::Present;
            if (value.size() < 7 || value[4] != '-' || !digits(value.substr(5, 2)))
                return MetadataValueState::Invalid;
            const auto month = decimal(value.substr(5, 2));
            if (month < 1 || month > 12)
                return MetadataValueState::Invalid;
            if (value.size() == 7)
                return MetadataValueState::Present;
            if (value.size() < 10 || value[7] != '-' || !digits(value.substr(8, 2)))
                return MetadataValueState::Invalid;
            if (!(std::chrono::year(decimal(value.substr(0, 4))) / unsigned(month) / unsigned(decimal(value.substr(8, 2)))).ok())
                return MetadataValueState::Invalid;
            if (value.size() == 10)
                return MetadataValueState::Present;
            if (value.size() < 16 || value[10] != 'T' || value[13] != ':' || !digits(value.substr(11, 2)) || !digits(value.substr(14, 2)) ||
                decimal(value.substr(11, 2)) > 23 || decimal(value.substr(14, 2)) > 59)
                return MetadataValueState::Invalid;
            std::size_t pos = 16;
            bool leap = false;
            if (pos < value.size() && value[pos] == ':') {
                if (value.size() < 19 || !digits(value.substr(17, 2)) || decimal(value.substr(17, 2)) > 60)
                    return MetadataValueState::Invalid;
                leap = value.substr(17, 2) == "60";
                pos = 19;
                if (pos < value.size() && value[pos] == '.') {
                    const auto begin = ++pos;
                    while (pos < value.size() && value[pos] >= '0' && value[pos] <= '9')
                        ++pos;
                    if (pos == begin)
                        return MetadataValueState::Invalid;
                }
            }
            const auto zone = value.substr(pos);
            if (!zone.empty() && zone != "Z" && !(zone.size() == 6 && (zone[0] == '+' || zone[0] == '-') && zone[3] == ':' && digits(zone.substr(1, 2)) && digits(zone.substr(4, 2)) && decimal(zone.substr(1, 2)) < 24 && decimal(zone.substr(4, 2)) < 60))
                return MetadataValueState::Invalid;
            return leap ? MetadataValueState::Unsupported : MetadataValueState::Present;
        }
        bool project(Packet& packet, const MetadataRecordHandle& handle, MetadataSummary& summary,
                     std::size_t max_facts, std::size_t max_text_bytes, std::size_t& charged,
                     std::shared_ptr<const XmpPacketAssembly> assembly = {}, bool extended = false,
                     const Packet* related = nullptr, const MetadataRecordHandle* related_source = nullptr) {
            if (summary.xmp_packets.size() >= std::min(max_facts, std::size_t{64})) {
                summary.limited = true;
                return false;
            }
            if (!assembly && packet.state == XmpPacketState::Parsed &&
                (packet.properties.size() > max_facts - summary.facts.size() || packet.text_bytes > max_text_bytes - charged)) {
                packet.state = XmpPacketState::Limited;
                packet.diagnostic = "XMP property or text budget exhausted";
            }
            const std::string_view diagnostic(packet.diagnostic);
            const auto packet_cost = diagnostic.size() + (extended && assembly ? assembly->guid.size() : 0);
            if (packet_cost > max_text_bytes - charged) {
                summary.limited = true;
                return false;
            }
            charged += packet_cost;
            summary.xmp_packets.push_back({handle, packet.state, std::string(diagnostic), extended ? assembly : nullptr});
            summary.limited |= packet.state == XmpPacketState::Limited;
            // Never publish fields from a malformed, unsupported or partial packet.
            if (packet.state != XmpPacketState::Parsed)
                return true;
            struct AltitudeRef {
                const Property* property = nullptr;
                const MetadataRecordHandle* source = nullptr;
                bool duplicate = false;
            };
            std::map<std::size_t, AltitudeRef> altitude_refs;
            const auto add_refs = [&](const Packet& values, const MetadataRecordHandle& source) {
                for (const auto& candidate : values.properties)
                    if (candidate.field->kind == Kind::AltitudeRef) {
                        auto& ref = altitude_refs[candidate.description];
                        ref.duplicate |= ref.property != nullptr;
                        ref.property = &candidate;
                        ref.source = &source;
                    }
            };
            add_refs(packet, handle);
            // Both portions must describe a single unqualified image subject
            // before an operand can cross the Standard/Extended boundary.
            if (related && related_source && related->state == XmpPacketState::Parsed &&
                packet.descriptions == 1 && related->descriptions == 1 && !packet.named_subject && !related->named_subject)
                add_refs(*related, *related_source);
            for (const auto& property : packet.properties) {
                const auto& field = *property.field;
                if (summary.facts.size() >= max_facts || field.semantic.size() + field.property.namespace_uri.size() + field.property.name.size() > max_text_bytes - charged) {
                    summary.limited = true;
                    return false;
                }
                const auto value = trim(property.value);
                MetadataFact fact{handle, std::string(field.semantic), std::string{}, field.unit,
                                  MetadataValueState::Present, MetadataValueOrigin::InterpretedSource};
                XmpFactProvenance provenance{property.location(), {}};
                provenance.assembly = extended ? assembly : nullptr;
                provenance.extended_packet = extended;
                double numeric = 0;
                int64_t whole = 0;
                if (property.structured) {
                    fact.state = MetadataValueState::Unsupported;
                    fact.diagnostic = "Structured or qualified XMP property is not interpreted";
                    fact.value = property.value;
                } else if (value.empty()) {
                    fact.state = MetadataValueState::Absent;
                    fact.value = property.value;
                } else if (field.kind == Kind::SignedNumber || field.kind == Kind::DecimalLatitude || field.kind == Kind::DecimalLongitude) {
                    if (!real(value, numeric) ||
                        (field.kind == Kind::DecimalLatitude && std::abs(numeric) > 90) ||
                        (field.kind == Kind::DecimalLongitude && std::abs(numeric) > 180))
                        fact.state = MetadataValueState::Invalid;
                    else
                        fact.value = numeric;
                } else if (field.kind == Kind::SourceFlag) {
                    if (!integer(value, whole) || whole < 0)
                        fact.state = MetadataValueState::Invalid;
                    else if (whole > 1) {
                        fact.state = MetadataValueState::Unsupported;
                        fact.value = property.value;
                        fact.diagnostic = "Unknown DJI flag value";
                    } else
                        fact.value = whole;
                } else if (field.kind == Kind::Orientation || field.kind == Kind::AltitudeRef) {
                    if (!integer(value, whole) || whole < (field.kind == Kind::Orientation ? 1 : 0) || whole > (field.kind == Kind::Orientation ? 8 : 1))
                        fact.state = MetadataValueState::Invalid;
                    else
                        fact.value = whole;
                } else if (field.kind == Kind::Number || field.kind == Kind::Altitude) {
                    if (!number(value, numeric, fact.value) || (field.kind == Kind::Number && field.property.name == "ExposureTime" && numeric == 0))
                        fact.state = MetadataValueState::Invalid;
                    else if (field.kind == Kind::Number && numeric == 0)
                        fact.state = MetadataValueState::Absent;
                    if (field.kind == Kind::Altitude && fact.state == MetadataValueState::Present) {
                        const auto found = altitude_refs.find(property.description);
                        const auto* ref = found == altitude_refs.end() ? nullptr : found->second.property;
                        const bool duplicate = found != altitude_refs.end() && found->second.duplicate;
                        if (!ref || duplicate || ref->structured || !integer(trim(ref->value), whole) || (whole != 0 && whole != 1)) {
                            fact.state = MetadataValueState::Unsupported;
                            fact.diagnostic = "Unique valid XMP altitude reference required";
                            fact.value = property.value;
                            fact.ambiguous |= duplicate;
                        } else if (double signed_altitude = 0; !summary_detail::altitude(numeric, static_cast<int>(whole), signed_altitude)) {
                            fact.state = MetadataValueState::Invalid;
                        } else {
                            fact.value = signed_altitude;
                            fact.origin = MetadataValueOrigin::Derived;
                            fact.inputs.push_back(*found->second.source);
                            provenance.inputs.push_back(ref->location());
                            if (found->second.source != &handle) {
                                provenance.assembly = assembly;
                            }
                            fact.derivation = "xmp_altitude_reference_0_1";
                        }
                    }
                } else if (field.kind == Kind::Latitude || field.kind == Kind::Longitude) {
                    if (!coordinate(value, field.kind == Kind::Latitude, numeric))
                        fact.state = MetadataValueState::Invalid;
                    else {
                        fact.value = numeric;
                        fact.origin = MetadataValueOrigin::Derived;
                        fact.derivation = "xmp_coordinate_degrees";
                    }
                } else if (field.kind == Kind::Date) {
                    fact.state = date(value);
                    fact.value = std::string(value);
                    if (fact.state == MetadataValueState::Unsupported)
                        fact.diagnostic = "Leap-second interpretation unavailable";
                }
                if (field.kind == Kind::Text)
                    fact.value = property.value;
                if (fact.state == MetadataValueState::Invalid) {
                    fact.value = property.value;
                    fact.diagnostic = "Invalid XMP field value";
                }
                std::size_t provenance_bytes = field.property.namespace_uri.size() + field.property.name.size() + property.subject.size();
                for (const auto& input : provenance.inputs)
                    provenance_bytes += input.field->namespace_uri.size() + input.field->name.size() + input.subject.size();
                if (provenance.assembly)
                    provenance_bytes += provenance.assembly->guid.size();
                const auto cost = summary_detail::factCost(summary, max_facts, max_text_bytes, charged,
                                                           fact.semantic, fact.value, fact.diagnostic, fact.derivation, provenance_bytes);
                if (!cost)
                    return false;
                fact.xmp = std::make_shared<const XmpFactProvenance>(std::move(provenance));
                charged += *cost;
                summary.facts.push_back(std::move(fact));
            }
            return true;
        }

        bool guid(std::string_view value) {
            return value.size() == 32 && std::all_of(value.begin(), value.end(), [](char c) {
                       return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F');
                   });
        }
        enum class DigestState { Match,
                                 Mismatch,
                                 Unavailable };
        DigestState digestMatches(std::span<const uint8_t> bytes, std::string_view expected) {
            const auto digest = xmpPacketDigest(bytes);
            if (!digest)
                return DigestState::Unavailable;
            return std::string_view(digest->data(), digest->size()) == expected ? DigestState::Match : DigestState::Mismatch;
        }
        struct PreparedPacket {
            Packet packet{.max_properties = 0, .max_text = 0};
            MetadataRecordHandle source;
            std::shared_ptr<const XmpPacketAssembly> assembly;
            std::optional<std::size_t> related;
            std::size_t order = 0;
            bool extended = false;
        };
        struct ExtendedGroup {
            struct Segment {
                MetadataRecordHandle source;
                core::image_codecs::JpegExtendedXmpSegment view;
            };
            std::vector<Segment> segments;
            std::vector<std::size_t> anchors;
            std::size_t order = 64;
        };
        void appendExtended(const SourceMetadataCatalog& catalog, MetadataSummary& summary,
                            std::size_t max_facts, std::size_t max_text_bytes, std::size_t& charged) {
            using core::image_codecs::MetadataStatus;
            using core::image_codecs::XmpEnvelope;
            if (summary.xmp_packets.size() >= std::min(max_facts, std::size_t{64}) || charged == max_text_bytes) {
                summary.limited = true;
                return;
            }
            std::vector<PreparedPacket> prepared;
            std::map<std::string, ExtendedGroup> groups;
            std::size_t records = 0, input_bytes = 0;
            bool bad_header = false, bad_link = false, limited_segment = false, incomplete_standard = false;
            const auto property_limit = std::min(std::size_t{1024}, max_facts - summary.facts.size());
            const auto text_limit = std::min(std::size_t{64 * 1024}, max_text_bytes - charged);
            for (const auto& record : catalog.records()) {
                const auto* reference = std::get_if<XmpBlockReference>(&record);
                if (!reference)
                    continue;
                if (records == 64)
                    break;
                const auto order = records++;
                const auto handle = catalog.bind(record);
                if (!handle)
                    continue;
                const auto* block = catalog.xmpBlock(*reference);
                if (block->envelope == XmpEnvelope::JpegExtendedApp1) {
                    const auto segment = core::image_codecs::source_extended_xmp_segment(*block);
                    if (segment && block->status == MetadataStatus::Collected &&
                        block->payload_offset == block->bytes.size() - segment->bytes.size()) {
                        auto& group = groups[std::string(segment->guid)];
                        group.order = std::min(group.order, order);
                        group.segments.push_back({*handle, *segment});
                        continue;
                    }
                    // Its unknown/invalid association cannot safely be assigned
                    // to one group while accepting a different group as complete.
                    bad_header = true;
                    limited_segment |= block->status == MetadataStatus::Limited;
                    PreparedPacket item;
                    item.source = *handle;
                    item.order = order;
                    item.packet.state = block->status == MetadataStatus::Limited ? XmpPacketState::Limited : XmpPacketState::Malformed;
                    item.packet.diagnostic = "Invalid or unavailable Extended XMP segment";
                    prepared.push_back(std::move(item));
                    continue;
                }
                PreparedPacket item;
                item.source = *handle;
                item.order = order;
                item.packet = preparePacket(catalog, *reference, property_limit, text_limit, input_bytes);
                // Any unparsed Standard remainder may hide a duplicate link,
                // regardless of why its interpretation stopped.
                incomplete_standard |= block->envelope == XmpEnvelope::JpegApp1 && item.packet.state != XmpPacketState::Parsed;
                if (item.packet.state == XmpPacketState::Parsed && item.packet.link_count) {
                    const auto value = trim(item.packet.extended_guid);
                    if (!item.packet.invalid_link && item.packet.link_count == 1 && guid(value)) {
                        auto& group = groups[std::string(value)];
                        group.anchors.push_back(prepared.size());
                        group.order = std::min(group.order, order);
                    } else {
                        bad_link = true;
                    }
                }
                prepared.push_back(std::move(item));
            }
            const bool incomplete = records < catalog.imageXmp().blocks.size() || catalog.imageXmp().status != MetadataStatus::Collected || limited_segment || incomplete_standard;
            summary.limited |= incomplete;
            for (auto& [name, group] : groups) {
                PreparedPacket item;
                item.extended = true;
                item.order = group.order;
                item.packet.max_properties = property_limit;
                item.packet.max_text = text_limit;
                auto assembly = std::make_shared<XmpPacketAssembly>();
                assembly->guid = name;
                std::sort(group.segments.begin(), group.segments.end(), [](const auto& a, const auto& b) {
                    return a.view.offset != b.view.offset ? a.view.offset < b.view.offset
                                                          : std::get<XmpBlockReference>(a.source.reference).block < std::get<XmpBlockReference>(b.source.reference).block;
                });
                for (const auto& segment : group.segments)
                    assembly->fragments.push_back({segment.source, segment.view.offset, segment.view.bytes.size()});
                if (!group.anchors.empty())
                    assembly->standard_source = prepared[group.anchors.front()].source;
                item.source = group.segments.empty() ? *assembly->standard_source : group.segments.front().source;
                item.assembly = assembly;
                const auto fail = [&](XmpPacketState state, const char* message) {
                    item.packet.state = state;
                    item.packet.diagnostic = message;
                };
                if (!group.segments.empty())
                    assembly->declared_bytes = group.segments.front().view.full_length;
                if (incomplete) {
                    fail(XmpPacketState::Limited, "Extended XMP association scan is incomplete");
                } else if (bad_header) {
                    fail(XmpPacketState::Malformed, "Extended XMP has an unidentified malformed segment");
                } else if (bad_link || group.anchors.size() != 1) {
                    fail(XmpPacketState::Unsupported, "Extended XMP requires one unambiguous standard packet reference");
                } else if (group.segments.empty()) {
                    fail(XmpPacketState::Malformed, "Referenced Extended XMP has no retained segments");
                } else if (assembly->declared_bytes > 1024 * 1024 || assembly->declared_bytes > 4 * 1024 * 1024 - input_bytes) {
                    fail(XmpPacketState::Limited, "Extended XMP interpretation input budget exhausted");
                } else {
                    std::size_t cursor = 0;
                    for (const auto& segment : group.segments) {
                        if (segment.view.full_length != assembly->declared_bytes || segment.view.offset != cursor || segment.view.bytes.empty()) {
                            fail(XmpPacketState::Malformed, "Extended XMP has inconsistent lengths, missing, empty or overlapping segments");
                            break;
                        }
                        cursor += segment.view.bytes.size();
                    }
                    if (item.packet.state == XmpPacketState::Parsed && cursor != assembly->declared_bytes)
                        fail(XmpPacketState::Malformed, "Extended XMP has missing segments");
                    if (item.packet.state == XmpPacketState::Parsed) {
                        // Allocation follows range/coverage checks, never the
                        // untrusted declared length alone. One buffer per group.
                        std::vector<uint8_t> bytes;
                        bytes.reserve(cursor);
                        for (const auto& segment : group.segments)
                            bytes.insert(bytes.end(), segment.view.bytes.begin(), segment.view.bytes.end());
                        input_bytes += bytes.size();
                        const auto digest = digestMatches(bytes, name);
                        if (digest == DigestState::Unavailable) {
                            fail(XmpPacketState::Unsupported, "MD5 is unavailable for Extended XMP integrity validation");
                        } else if (digest == DigestState::Mismatch) {
                            fail(XmpPacketState::Malformed, "Extended XMP digest does not match its GUID");
                        } else {
                            assembly->md5_verified = true;
                            item.packet = parse(bytes, property_limit, text_limit);
                            if (item.packet.state == XmpPacketState::Parsed) {
                                item.related = group.anchors.front();
                                prepared[*item.related].related = prepared.size();
                                prepared[*item.related].assembly = assembly;
                            }
                        }
                    }
                }
                prepared.push_back(std::move(item));
            }
            std::vector<std::size_t> order;
            order.reserve(prepared.size());
            for (std::size_t i = 0; i < prepared.size(); ++i)
                order.push_back(i);
            std::sort(order.begin(), order.end(), [&](auto a, auto b) {
                if (prepared[a].order != prepared[b].order)
                    return prepared[a].order < prepared[b].order;
                return prepared[a].extended != prepared[b].extended ? prepared[a].extended < prepared[b].extended : a < b;
            });
            for (const auto index : order) {
                auto& item = prepared[index];
                const auto* related = item.related ? &prepared[*item.related] : nullptr;
                if (!project(item.packet, item.source, summary, max_facts, max_text_bytes, charged, item.assembly, item.extended,
                             related ? &related->packet : nullptr, related ? &related->source : nullptr))
                    break;
            }
        }
    } // namespace

    void appendXmpMetadata(const SourceMetadataCatalog& catalog, MetadataSummary& summary, std::size_t max_facts,
                           std::size_t max_text_bytes, std::size_t& charged) {
        using core::image_codecs::MetadataStatus;
        summary.limited |= catalog.imageXmp().status == MetadataStatus::Limited;
        if (catalog.imageXmp().blocks.empty())
            return;
        std::size_t input_bytes = 0;
        const auto first_xmp_fact = summary.facts.size();
        const auto first_xmp_packet = summary.xmp_packets.size();
        const auto first_charged = charged;
        const auto initial_limited = summary.limited;
        const auto mark_ambiguous = [&] {
            if (summary.facts.size() == first_xmp_fact)
                return;
            struct Group {
                const MetadataFact* first = nullptr;
                const MetadataFact* exact = nullptr;
                bool xmp = false, conflict = false;
            };
            std::map<std::string_view, Group> groups;
            for (const auto& fact : summary.facts) {
                auto& group = groups[fact.semantic];
                if (group.first)
                    group.conflict |= !summary_detail::equivalent(*group.first, fact);
                else
                    group.first = &fact;
                // Floating equality alone is not transitive between two exact
                // rationals that round to the same double. Compare those too,
                // independently of which carrier was encountered first.
                if (fact.state == MetadataValueState::Present &&
                    (std::holds_alternative<int64_t>(fact.value) || std::holds_alternative<ExifRational>(fact.value))) {
                    if (group.exact)
                        group.conflict |= !summary_detail::equivalent(*group.exact, fact);
                    else
                        group.exact = &fact;
                }
                group.xmp |= bool(fact.xmp);
            }
            for (auto& fact : summary.facts) {
                const auto& group = groups.at(fact.semantic);
                if (group.xmp && group.conflict)
                    fact.ambiguous = true;
            }
        };
        if (std::any_of(catalog.imageXmp().blocks.begin(), catalog.imageXmp().blocks.end(), [](const auto& block) {
                return block.envelope == core::image_codecs::XmpEnvelope::JpegExtendedApp1;
            })) {
            appendExtended(catalog, summary, max_facts, max_text_bytes, charged);
            mark_ambiguous();
            return;
        }
        for (const auto& record : catalog.records()) {
            const auto* reference = std::get_if<XmpBlockReference>(&record);
            if (!reference)
                continue;
            const auto handle = catalog.bind(record);
            if (!handle)
                continue;
            if (summary.xmp_packets.size() >= std::min(max_facts, std::size_t{64})) {
                summary.limited = true;
                break;
            }
            auto packet = preparePacket(catalog, *reference, std::min(std::size_t{1024}, max_facts - summary.facts.size()),
                                        std::min(std::size_t{64 * 1024}, max_text_bytes - charged), input_bytes);
            if (packet.state == XmpPacketState::Parsed && packet.link_count) {
                // A reference without any retained fragments still needs a
                // missing/limited assembly result. Roll back only this
                // projection, retaining the original catalog and EXIF.
                summary.facts.resize(first_xmp_fact);
                summary.xmp_packets.resize(first_xmp_packet);
                charged = first_charged;
                summary.limited = initial_limited;
                appendExtended(catalog, summary, max_facts, max_text_bytes, charged);
                mark_ambiguous();
                return;
            }
            if (!project(packet, *handle, summary, max_facts, max_text_bytes, charged))
                break;
        }
        // Retain all occurrences. Do not reconcile EXIF and XMP automatically.
        mark_ambiguous();
    }
    std::optional<std::array<char, 32>> xmpPacketDigest(std::span<const uint8_t> bytes) {
        // Preserve a caller's preexisting OpenSSL error queue.
        struct Errors {
            bool marked = ERR_set_mark() != 0;
            ~Errors() {
                if (marked)
                    ERR_pop_to_mark();
                else
                    ERR_clear_error();
            }
        } errors;
        std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
        unsigned int size = 0;
        const auto* algorithm = EVP_get_digestbyname("MD5");
        if (!algorithm)
            return std::nullopt;
        if (EVP_Digest(bytes.data(), bytes.size(), digest.data(), &size, algorithm, nullptr) != 1) {
            if (ERR_GET_REASON(ERR_peek_last_error()) == ERR_R_MALLOC_FAILURE)
                throw std::bad_alloc();
            return std::nullopt;
        }
        if (size != 16)
            return std::nullopt;
        constexpr char hex[] = "0123456789ABCDEF";
        std::array<char, 32> result{};
        for (std::size_t i = 0; i < 16; ++i) {
            result[i * 2] = hex[digest[i] >> 4];
            result[i * 2 + 1] = hex[digest[i] & 15];
        }
        return result;
    }
} // namespace lfs::media
