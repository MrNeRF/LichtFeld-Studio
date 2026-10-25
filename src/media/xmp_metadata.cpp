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
#include <string_view>

namespace lfs::media {
    namespace {
        constexpr std::string_view rdf = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
        constexpr std::string_view tiff = "http://ns.adobe.com/tiff/1.0/";
        constexpr std::string_view exif = "http://ns.adobe.com/exif/1.0/";
        constexpr std::string_view exif_ex = "http://cipa.jp/exif/1.0/";
        constexpr std::string_view aux = "http://ns.adobe.com/exif/1.0/aux/";
        constexpr std::string_view xmp = "http://ns.adobe.com/xap/1.0/";
        enum class Kind { Text,
                          Number,
                          Orientation,
                          Latitude,
                          Longitude,
                          Altitude,
                          AltitudeRef,
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
        bool named(std::string_view expanded, std::string_view uri, std::string_view name) {
            return expanded.size() == uri.size() + name.size() + 1 && expanded.starts_with(uri) &&
                   expanded[uri.size()] == '\x1f' && expanded.substr(uri.size() + 1) == name;
        }
        const Field* field(std::string_view name) {
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
            XmpPropertyLocation location() const {
                return {&field->property, description, ordinal};
            }
        };
        enum class Role { Wrapper,
                          Rdf,
                          Description,
                          Property,
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
                if (!field)
                    return std::nullopt;
                if (properties.size() >= max_properties || value.size() > max_text - text_bytes) {
                    stop(XmpPacketState::Limited, "XMP property or text budget exhausted");
                    return std::nullopt;
                }
                text_bytes += value.size();
                properties.push_back({field, std::string(value), descriptions - 1, ordinal});
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
                    for (std::size_t i = 0; i < attributes; ++i) {
                        const std::string_view key(attrs[i * 2]), value(attrs[i * 2 + 1]);
                        if (named(key, rdf, "about") && value.empty())
                            continue;
                        if (key.starts_with(rdf) || key.starts_with("http://www.w3.org/XML/1998/namespace\x1f")) {
                            stop(XmpPacketState::Unsupported, "RDF subject or description qualifier is not interpreted");
                            return;
                        }
                        ++ordinal;
                        add(field(key), value);
                        if (state != XmpPacketState::Parsed)
                            return;
                    }
                } else if (parent == Role::Description) {
                    role = Role::Property;
                    ++ordinal;
                    selected = add(field(name), {});
                    if (selected && attributes)
                        properties[*selected].structured = true;
                    if (!selected)
                        role = Role::Other;
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
                if (!stack.empty() && stack.back().role == Role::Property && stack.back().property) {
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
        Packet parse(std::span<const uint8_t> bytes, std::size_t max_properties, std::size_t max_text) {
            Packet packet{.max_properties = max_properties, .max_text = max_text};
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
        bool integer(std::string_view value, int64_t& result) {
            if (value.starts_with('+'))
                value.remove_prefix(1);
            if (value.empty())
                return false;
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
            return !value.empty() && parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size();
        }
        bool real(std::string_view value, double& result) {
            if (value.starts_with('+'))
                value.remove_prefix(1);
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
    } // namespace

    void appendXmpMetadata(const SourceMetadataCatalog& catalog, MetadataSummary& summary, std::size_t max_facts,
                           std::size_t max_text_bytes, std::size_t& charged) {
        using core::image_codecs::MetadataStatus;
        summary.limited |= catalog.imageXmp().status == MetadataStatus::Limited;
        if (catalog.imageXmp().blocks.empty())
            return;
        std::size_t input_bytes = 0;
        const auto first_xmp_fact = summary.facts.size();
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
        for (const auto& record : catalog.records()) {
            const auto* reference = std::get_if<XmpBlockReference>(&record);
            if (!reference)
                continue;
            const auto bytes = catalog.xmpPayloadBytes(*reference);
            const auto handle = catalog.bind(record);
            if (!handle)
                continue;
            if (summary.xmp_packets.size() >= std::min(max_facts, std::size_t{64})) {
                summary.limited = true;
                break;
            }
            Packet packet{.max_properties = std::min(std::size_t{1024}, max_facts - summary.facts.size()),
                          .max_text = std::min(std::size_t{64 * 1024}, max_text_bytes - charged)};
            const auto* block = catalog.xmpBlock(*reference);
            if (!bytes) {
                packet.state = block->status == MetadataStatus::Limited     ? XmpPacketState::Limited
                               : block->status == MetadataStatus::Malformed ? XmpPacketState::Malformed
                                                                            : XmpPacketState::Unsupported;
                packet.diagnostic = "XMP packet payload is unavailable for interpretation";
            } else if (block->envelope == core::image_codecs::XmpEnvelope::JpegExtendedApp1) {
                packet.state = XmpPacketState::Unsupported;
                packet.diagnostic = "Extended XMP requires explicit segment assembly";
            } else if (block->tiff_tag && block->tiff_tag->directory != 0) {
                packet.state = XmpPacketState::Unsupported;
                packet.diagnostic = "XMP belongs to a non-primary TIFF directory";
            } else if (bytes->size() > 1024 * 1024 || bytes->size() > 4 * 1024 * 1024 - input_bytes) {
                packet.state = XmpPacketState::Limited;
                packet.diagnostic = "XMP interpretation input budget exhausted";
            } else {
                input_bytes += bytes->size();
                packet = parse(*bytes, packet.max_properties, packet.max_text);
            }
            const std::string_view diagnostic(packet.diagnostic);
            if (diagnostic.size() > max_text_bytes - charged) {
                summary.limited = true;
                break;
            }
            charged += diagnostic.size();
            summary.xmp_packets.push_back({*handle, packet.state, std::string(diagnostic)});
            summary.limited |= packet.state == XmpPacketState::Limited;
            // Never publish fields from a malformed, unsupported or partial packet.
            if (packet.state != XmpPacketState::Parsed)
                continue;
            std::map<std::size_t, std::pair<const Property*, bool>> altitude_refs;
            for (const auto& candidate : packet.properties)
                if (candidate.field->kind == Kind::AltitudeRef) {
                    auto& ref = altitude_refs[candidate.description];
                    ref.second |= ref.first != nullptr;
                    ref.first = &candidate;
                }
            std::map<std::pair<std::size_t, std::string_view>, std::size_t> occurrences;
            for (const auto& property : packet.properties)
                ++occurrences[{property.description, property.field->semantic}];
            for (const auto& property : packet.properties) {
                const auto& field = *property.field;
                if (summary.facts.size() >= max_facts || field.semantic.size() + field.property.namespace_uri.size() + field.property.name.size() > max_text_bytes - charged) {
                    summary.limited = true;
                    mark_ambiguous();
                    return;
                }
                const auto value = trim(property.value);
                MetadataFact fact{*handle, std::string(field.semantic), std::string{}, field.unit,
                                  MetadataValueState::Present, MetadataValueOrigin::InterpretedSource};
                fact.ambiguous = occurrences.at({property.description, field.semantic}) > 1;
                XmpFactProvenance provenance{property.location(), {}};
                double numeric = 0;
                int64_t whole = 0;
                if (property.structured) {
                    fact.state = MetadataValueState::Unsupported;
                    fact.diagnostic = "Structured or qualified XMP property is not interpreted";
                    fact.value = property.value;
                } else if (value.empty()) {
                    fact.state = MetadataValueState::Absent;
                    fact.value = property.value;
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
                        const auto* ref = found == altitude_refs.end() ? nullptr : found->second.first;
                        const bool duplicate = found != altitude_refs.end() && found->second.second;
                        if (!ref || duplicate || ref->structured || !integer(trim(ref->value), whole) || (whole != 0 && whole != 1)) {
                            fact.state = MetadataValueState::Unsupported;
                            fact.diagnostic = "Unique valid XMP altitude reference required";
                            fact.value = property.value;
                            fact.ambiguous |= duplicate;
                        } else {
                            double signed_altitude = 0;
                            summary_detail::altitude(numeric, static_cast<int>(whole), signed_altitude);
                            fact.value = signed_altitude;
                            fact.origin = MetadataValueOrigin::Derived;
                            fact.inputs.push_back(*handle);
                            provenance.inputs.push_back(ref->location());
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
                std::size_t provenance_bytes = field.property.namespace_uri.size() + field.property.name.size();
                for (const auto& input : provenance.inputs)
                    provenance_bytes += input.field->namespace_uri.size() + input.field->name.size();
                const auto cost = summary_detail::factCost(summary, max_facts, max_text_bytes, charged,
                                                           fact.semantic, fact.value, fact.diagnostic, fact.derivation, provenance_bytes);
                if (!cost) {
                    mark_ambiguous();
                    return;
                }
                fact.xmp = std::make_shared<const XmpFactProvenance>(std::move(provenance));
                charged += *cost;
                summary.facts.push_back(std::move(fact));
            }
        }
        // Retain all occurrences. Do not reconcile EXIF and XMP automatically.
        mark_ambiguous();
    }
} // namespace lfs::media
