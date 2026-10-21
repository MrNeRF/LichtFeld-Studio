// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "media/metadata_observation.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <new>
#include <span>
#include <tuple>

namespace lfs::media {
    namespace {
        enum class Role { Image,
                          Exif,
                          Gps,
                          Other };
        enum class TextKind { General,
                              CivilTime,
                              Offset,
                              Subseconds };
        struct TagView {
            const SourceMetadataReference* reference;
            uint32_t id;
            uint16_t type;
            uint64_t count;
            bool directory;
            uint32_t ordinal;
            std::span<const uint64_t> integers;
            std::span<const double> floating;
            std::span<const ExifRational> ratios;
            std::span<const uint8_t> bytes;
        };
        // EXIF: envelope ordinal + full IFD path. TIFF: directory ordinal.
        // No compound value is assembled across these keys.
        using GroupKey = std::tuple<bool, std::size_t, std::vector<uint32_t>>;
        struct Group {
            Role role = Role::Other;
            bool complete = true;
            std::vector<TagView> tags;
        };
        bool number(const TagView& tag, std::size_t index, double& result) {
            if (index >= tag.count)
                return false;
            if (tag.type == 5 && tag.ratios.size() == tag.count) {
                const auto value = tag.ratios[index];
                if (value.numerator < 0 || value.denominator <= 0)
                    return false;
                result = double(value.numerator) / double(value.denominator);
            } else if (tag.type == 5 && tag.floating.size() == tag.count) {
                // libtiff has already rounded rational tags. Do not manufacture
                // an exact numerator/denominator from its decoded floating value.
                result = tag.floating[index];
            } else if ((tag.type == 3 || tag.type == 4) && tag.integers.size() == tag.count) {
                result = double(tag.integers[index]);
            } else if (tag.type == 1 && tag.bytes.size() == tag.count) {
                result = tag.bytes[index];
            } else {
                return false;
            }
            return std::isfinite(result) && result >= 0;
        }
        std::optional<std::string_view> text(const TagView& tag) {
            if (tag.type != 2 || tag.bytes.empty() || tag.bytes.size() != tag.count || tag.bytes.back() != 0)
                return std::nullopt;
            const auto end = std::find(tag.bytes.begin(), tag.bytes.end(), 0);
            // Accept trailing NUL padding, but never discard another string or
            // nonzero payload after the first terminator. Originals stay intact.
            if (std::any_of(end, tag.bytes.end(), [](auto c) { return c != 0; }))
                return std::nullopt;
            const auto value = tag.bytes.first(static_cast<std::size_t>(end - tag.bytes.begin()));
            if (std::any_of(value.begin(), value.end(), [](auto c) { return c > 127; }))
                return std::nullopt;
            return std::string_view(reinterpret_cast<const char*>(value.data()), value.size());
        }
        const TagView* unique(const Group& group, uint32_t id, bool& ambiguous) {
            const TagView* found = nullptr;
            for (const auto& tag : group.tags) {
                if (tag.id != id || tag.directory)
                    continue;
                if (found) {
                    ambiguous = true;
                    return nullptr;
                }
                found = &tag;
            }
            return found;
        }
        bool digits(std::string_view value) {
            return !value.empty() && std::all_of(value.begin(), value.end(), [](auto c) { return c >= '0' && c <= '9'; });
        }
        int decimal(std::string_view value) {
            int result = 0;
            for (const auto c : value)
                result = result * 10 + c - '0';
            return result;
        }
        bool civilTime(std::string_view value, bool allow_leap_second = false) {
            if (value.size() != 19 || value[4] != ':' || value[7] != ':' || value[10] != ' ' || value[13] != ':' || value[16] != ':')
                return false;
            for (const auto [offset, count] : std::array<std::pair<std::size_t, std::size_t>, 6>{{{0, 4}, {5, 2}, {8, 2}, {11, 2}, {14, 2}, {17, 2}}})
                if (!digits(value.substr(offset, count)))
                    return false;
            const auto date = std::chrono::year(decimal(value.substr(0, 4))) / unsigned(decimal(value.substr(5, 2))) / unsigned(decimal(value.substr(8, 2)));
            return decimal(value.substr(0, 4)) > 0 && date.ok() && decimal(value.substr(11, 2)) < 24 &&
                   decimal(value.substr(14, 2)) < 60 && decimal(value.substr(17, 2)) <= (allow_leap_second ? 60 : 59);
        }
        bool leapSecond(std::string_view value) {
            return civilTime(value, true) && value.substr(17, 2) == "60";
        }
        bool timeOffset(std::string_view value) {
            return value.size() == 6 && (value[0] == '+' || value[0] == '-') && value[3] == ':' &&
                   digits(value.substr(1, 2)) && digits(value.substr(4, 2)) &&
                   decimal(value.substr(1, 2)) < 24 && decimal(value.substr(4, 2)) < 60;
        }
        std::string_view subseconds(std::string_view value) {
            const auto last = value.find_last_not_of(' ');
            return last == std::string_view::npos ? std::string_view{} : value.substr(0, last + 1);
        }
        bool unknownText(std::string_view value, TextKind kind) {
            if (kind == TextKind::Subseconds)
                return subseconds(value).empty();
            if (value.size() != (kind == TextKind::CivilTime ? 19 : 6))
                return false;
            for (std::size_t i = 0; i < value.size(); ++i) {
                const bool separator = kind == TextKind::CivilTime ? i == 4 || i == 7 || i == 13 || i == 16 : i == 3;
                if (value[i] != ' ' && !(separator && value[i] == ':'))
                    return false;
            }
            return kind != TextKind::General;
        }
        struct SummaryBuilder {
            const SourceMetadataCatalog& catalog;
            MetadataSummary result;
            std::size_t facts, bytes, charged = 0;
            bool append(const TagView& tag, std::string_view semantic, MetadataScalar value, MetadataUnit unit,
                        MetadataValueState state, std::string_view diagnostic = {}, bool ambiguous = false,
                        std::span<const TagView* const> operands = {}, std::string_view derivation = {}) {
                const auto* string = std::get_if<std::string>(&value);
                std::size_t cost = 0;
                for (const auto count : {semantic.size(), diagnostic.size(), derivation.size(), string ? string->size() : 0}) {
                    if (count > bytes - charged - cost) {
                        result.limited = true;
                        return false;
                    }
                    cost += count;
                }
                if (result.facts.size() >= facts) {
                    result.limited = true;
                    return false;
                }
                const auto source = catalog.bind(*tag.reference);
                if (!source)
                    return false;
                MetadataFact fact{*source, std::string(semantic), std::move(value), unit, state, MetadataValueOrigin::InterpretedSource};
                fact.diagnostic = diagnostic;
                fact.ambiguous = ambiguous;
                fact.derivation = derivation;
                if (!derivation.empty())
                    fact.origin = MetadataValueOrigin::Derived;
                for (const auto* operand : operands) {
                    const auto handle = catalog.bind(*operand->reference);
                    if (handle)
                        fact.inputs.push_back(*handle);
                }
                charged += cost;
                result.facts.push_back(std::move(fact));
                return true;
            }
            bool string(const TagView& tag, std::string_view semantic, bool ambiguous, TextKind validation = TextKind::General) {
                auto value = text(tag);
                if (value && validation == TextKind::Subseconds)
                    value = subseconds(*value);
                bool valid = value && !value->empty();
                if (valid && validation == TextKind::CivilTime)
                    valid = civilTime(*value);
                if (valid && validation == TextKind::Offset)
                    valid = timeOffset(*value);
                if (valid && validation == TextKind::Subseconds)
                    valid = digits(*value);
                if (value && value->size() > bytes - charged) {
                    result.limited = true;
                    return false;
                }
                const bool unknown = value && (validation == TextKind::General ? value->find_first_not_of(' ') == std::string_view::npos : unknownText(*value, validation));
                const bool leap = validation == TextKind::CivilTime && value && leapSecond(*value);
                return append(tag, semantic, value ? std::string(*value) : std::string{}, MetadataUnit::Unspecified,
                              unknown ? MetadataValueState::Absent : leap ? MetadataValueState::Unsupported
                                                                 : valid  ? MetadataValueState::Present
                                                                          : MetadataValueState::Invalid,
                              unknown ? "Source declares unknown value" : leap ? "Leap-second interpretation unavailable"
                                                                      : valid  ? ""
                                                                               : "Invalid standard ASCII field",
                              ambiguous);
            }
            bool scalar(const TagView& tag, std::string_view semantic, MetadataUnit unit, uint16_t type, double minimum, double maximum, bool ambiguous,
                        std::optional<double> unknown = std::nullopt, std::optional<double> extension = std::nullopt) {
                if (tag.id == 0x8827 && tag.type == 3 && tag.count > 1 && tag.integers.size() == tag.count)
                    return append(tag, semantic, std::string{}, unit, MetadataValueState::Unsupported,
                                  "Multiple photographic sensitivity values require explicit interpretation", ambiguous);
                double value = 0;
                const bool valid = tag.type == type && tag.count == 1 && number(tag, 0, value) && value >= minimum && value <= maximum;
                MetadataScalar stored = value;
                if (tag.ratios.size() == 1)
                    stored = tag.ratios.front();
                else if (tag.integers.size() == 1 && tag.integers.front() <= uint64_t(INT64_MAX))
                    stored = int64_t(tag.integers.front());
                const auto state = !valid ? MetadataValueState::Invalid : unknown && value == *unknown   ? MetadataValueState::Absent
                                                                      : extension && value == *extension ? MetadataValueState::Unsupported
                                                                                                         : MetadataValueState::Present;
                return append(tag, semantic, std::move(stored), unit, state,
                              !valid ? "Invalid field type, count or numeric range" : state == MetadataValueState::Absent    ? "Source declares unknown value"
                                                                                  : state == MetadataValueState::Unsupported ? "Value requires extended sensitivity fields"
                                                                                                                             : "",
                              ambiguous);
            }
            bool coordinate(const Group& group, const TagView& tag, bool ambiguous) {
                const bool latitude = tag.id == 2;
                bool duplicate = false;
                const auto* ref = unique(group, tag.id - 1, duplicate);
                const auto direction = ref ? text(*ref) : std::nullopt;
                std::array<double, 3> dms{};
                const double limit = latitude ? 90 : 180;
                bool valid = tag.type == 5 && tag.count == 3;
                for (std::size_t i = 0; i < dms.size(); ++i)
                    valid = number(tag, i, dms[i]) && valid;
                valid = valid && dms[0] <= limit && dms[1] < 60 && dms[2] < 60 &&
                        (dms[0] < limit || (dms[1] == 0 && dms[2] == 0));
                const double magnitude = dms[0] + dms[1] / 60 + dms[2] / 3600;
                valid = valid && magnitude <= limit;
                const bool valid_ref = direction && direction->size() == 1 &&
                                       (latitude ? (*direction == "N" || *direction == "S") : (*direction == "E" || *direction == "W"));
                auto state = !valid ? MetadataValueState::Invalid : duplicate ? MetadataValueState::Unsupported
                                                                : !ref        ? MetadataValueState::Absent
                                                                : !valid_ref  ? MetadataValueState::Invalid
                                                                              : MetadataValueState::Present;
                if (!group.complete)
                    state = MetadataValueState::Unsupported;
                double value = magnitude;
                if (valid_ref && (*direction == "S" || *direction == "W"))
                    value = -value;
                const std::array<const TagView*, 1> inputs{ref};
                return append(tag, latitude ? "gps.latitude" : "gps.longitude", state == MetadataValueState::Present ? MetadataScalar(value) : MetadataScalar(std::string{}),
                              MetadataUnit::Degrees, state, !group.complete ? "Directory metadata is incomplete" : state == MetadataValueState::Present ? ""
                                                                                                                                                        : "Coordinate requires valid DMS and one hemisphere reference",
                              ambiguous || duplicate, ref ? std::span<const TagView* const>(inputs) : std::span<const TagView* const>{}, "exif_dms_hemisphere");
            }
            bool altitude(const Group& group, const TagView& tag, bool ambiguous) {
                bool duplicate = false;
                const auto* ref = unique(group, 5, duplicate);
                double magnitude = 0, sign = 0;
                const bool valid = tag.type == 5 && tag.count == 1 && number(tag, 0, magnitude);
                const bool valid_ref = ref && ref->type == 1 && ref->count == 1 && number(*ref, 0, sign);
                auto state = !valid ? MetadataValueState::Invalid : duplicate ? MetadataValueState::Unsupported
                                                                : !ref        ? MetadataValueState::Absent
                                                                : !valid_ref  ? MetadataValueState::Invalid
                                                                : sign > 1    ? MetadataValueState::Unsupported
                                                                              : MetadataValueState::Present;
                if (!group.complete)
                    state = MetadataValueState::Unsupported;
                const std::array<const TagView*, 1> inputs{ref};
                return append(tag, "gps.altitude_sea_level", state == MetadataValueState::Present ? MetadataScalar(sign == 1 ? -magnitude : magnitude) : MetadataScalar(std::string{}),
                              MetadataUnit::Metres, state, !group.complete ? "Directory metadata is incomplete" : state == MetadataValueState::Present ? ""
                                                                                                                                                       : "Altitude requires one supported altitude reference",
                              ambiguous || duplicate, ref ? std::span<const TagView* const>(inputs) : std::span<const TagView* const>{}, "exif_altitude_reference_0_1");
            }
            bool captureTime(const Group& group, const TagView& tag, bool ambiguous, uint32_t offset_id, uint32_t subsec_id, std::string_view semantic) {
                bool duplicate = false;
                const auto* offset = unique(group, offset_id, duplicate);
                const auto* subsec = unique(group, subsec_id, duplicate);
                const auto base = text(tag);
                auto zone = offset ? text(*offset) : std::nullopt;
                auto fraction = subsec ? text(*subsec) : std::nullopt;
                if (fraction)
                    fraction = subseconds(*fraction);
                const bool unknown_zone = zone && unknownText(*zone, TextKind::Offset);
                const bool unknown_fraction = fraction && unknownText(*fraction, TextKind::Subseconds);
                const bool leap = base && leapSecond(*base);
                const bool valid = base && (civilTime(*base) || leap) && (!offset || unknown_zone || (zone && timeOffset(*zone))) &&
                                   (!subsec || unknown_fraction || (fraction && digits(*fraction)));
                auto state = base && unknownText(*base, TextKind::CivilTime) ? MetadataValueState::Absent : !valid  ? MetadataValueState::Invalid
                                                                                                        : duplicate ? MetadataValueState::Unsupported
                                                                                                        : leap      ? MetadataValueState::Unsupported
                                                                                                                    : MetadataValueState::Present;
                if (!group.complete)
                    state = MetadataValueState::Unsupported;
                if (unknown_zone)
                    zone.reset();
                if (unknown_fraction)
                    fraction.reset();
                std::string value;
                if (state == MetadataValueState::Present) {
                    const auto extra = (fraction ? fraction->size() + 1 : 0) + (zone ? zone->size() : 0);
                    if (extra > bytes - charged || base->size() > bytes - charged - extra) {
                        result.limited = true;
                        return false;
                    }
                    value = *base;
                    value[4] = value[7] = '-';
                    value[10] = 'T';
                    if (fraction) {
                        value += '.';
                        value += *fraction;
                    }
                    if (zone)
                        value += *zone;
                }
                std::array<const TagView*, 2> inputs{};
                std::size_t count = 0;
                if (offset)
                    inputs[count++] = offset;
                if (subsec)
                    inputs[count++] = subsec;
                return append(tag, semantic, std::move(value), MetadataUnit::Unspecified, state,
                              !group.complete ? "Directory metadata is incomplete" : leap && valid                      ? "Leap-second interpretation unavailable"
                                                                                 : state == MetadataValueState::Absent  ? "Source declares unknown date/time"
                                                                                 : state != MetadataValueState::Present ? "Invalid or ambiguous civil-time components"
                                                                                 : !zone                                ? "Timezone unspecified"
                                                                                                                        : "",
                              ambiguous || duplicate, std::span<const TagView* const>(inputs.data(), count), "exif_civil_time_components");
            }
            bool gpsTime(const TagView& tag, bool ambiguous) {
                std::array<double, 3> hms{};
                bool valid = tag.type == 5 && tag.count == 3;
                for (std::size_t i = 0; i < hms.size(); ++i)
                    valid = number(tag, i, hms[i]) && valid;
                const double seconds = hms[0] * 3600 + hms[1] * 60 + hms[2];
                const bool leap = valid && hms[0] == 23 && hms[1] == 59 && hms[2] == 60;
                valid = valid && hms[0] < 24 && hms[1] < 60 && hms[2] < 60 && seconds < 86400;
                return append(tag, "gps.utc_seconds_of_day", valid ? MetadataScalar(seconds) : MetadataScalar(std::string{}),
                              MetadataUnit::Seconds, leap ? MetadataValueState::Unsupported : valid ? MetadataValueState::Present
                                                                                                    : MetadataValueState::Invalid,
                              leap ? "Leap-second interpretation unavailable" : valid ? ""
                                                                                      : "Invalid GPS UTC time components",
                              ambiguous, {}, "exif_gps_hms_to_double_seconds");
            }
            bool gpsDate(const TagView& tag, bool ambiguous) {
                const auto date = text(tag);
                bool valid = date && date->size() == 10;
                if (valid) {
                    std::string candidate(*date);
                    candidate += " 00:00:00";
                    valid = civilTime(candidate);
                }
                std::string value = valid ? std::string(*date) : std::string{};
                if (valid)
                    value[4] = value[7] = '-';
                return append(tag, "gps.utc_date", std::move(value), MetadataUnit::Unspecified,
                              valid ? MetadataValueState::Present : MetadataValueState::Invalid,
                              valid ? "" : "Invalid GPS UTC date", ambiguous, {}, "exif_gps_date");
            }
        };
    } // namespace

    Result<MetadataSummary> summarizeImageMetadata(const SourceMetadataCatalog& catalog, std::size_t max_facts, std::size_t max_text_bytes) {
        if (catalog.container())
            return make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::IO, .detail = "Image summary requires an image metadata catalog", .detection = LFS_SOURCE_SITE_CURRENT()});
        try {
            SummaryBuilder builder{catalog, {}, max_facts, max_text_bytes};
            builder.result.scope = MetadataSummaryScope::ImageProperties;
            builder.result.limited = catalog.indexStatus() == MetadataIndexStatus::Limited ||
                                     catalog.imageExif().original.status == core::image_codecs::MetadataStatus::Limited ||
                                     std::any_of(catalog.imageExif().interpretations.begin(), catalog.imageExif().interpretations.end(), [](const auto& item) { return item.status == ExifParseStatus::Limited; });
            if ((!max_facts || !max_text_bytes) && !catalog.records().empty()) {
                builder.result.limited = true;
                return std::move(builder.result);
            }
            std::map<GroupKey, Group> groups;
            for (const auto& reference : catalog.records()) {
                if (const auto* position = std::get_if<ExifEntryReference>(&reference)) {
                    if (catalog.imageExif().interpretations[position->block].status != ExifParseStatus::Parsed)
                        continue;
                    const auto* tag = catalog.exifEntry(*position);
                    if (!tag || tag->directory_path.empty())
                        continue;
                    groups[{false, position->block, tag->directory_path}].tags.push_back(
                        {&reference, tag->id, tag->type, tag->count, tag->directory, tag->entry_index, tag->unsigned_values, tag->floating_values, tag->rational_values, tag->bytes});
                } else if (const auto* position = std::get_if<TiffTagReference>(&reference)) {
                    const auto* tag = catalog.tiffTag(*position);
                    if (!tag)
                        continue;
                    auto& group = groups[{true, position->directory, {}}];
                    const auto& kind = catalog.imageExif().original.tiff->directories[position->directory].kind;
                    group.role = kind == "image" || kind == "subimage" ? Role::Image : kind == "exif" ? Role::Exif
                                                                                   : kind == "gps"    ? Role::Gps
                                                                                                      : Role::Other;
                    group.tags.push_back({&reference, tag->id, tag->type, tag->count, false, 0, tag->unsigned_values, tag->floating_values, {}, tag->bytes});
                }
            }
            for (auto& [key, group] : groups) {
                const auto& [tiff, block, path] = key;
                // A retained prefix cannot establish uniqueness or absence of
                // compound operands. Parsed EXIF blocks remain self-contained;
                // the TIFF reader may stop within an observed directory.
                group.complete = catalog.indexStatus() == MetadataIndexStatus::Complete &&
                                 (!tiff || catalog.imageExif().original.status == core::image_codecs::MetadataStatus::Collected);
                if (!tiff) {
                    if (path.size() == 1) {
                        group.role = Role::Image;
                    } else {
                        auto parent_path = path;
                        const auto ordinal = parent_path.back();
                        parent_path.pop_back();
                        const auto parent = groups.find({false, block, parent_path});
                        if (parent != groups.end() && parent->second.role == Role::Image) {
                            const TagView* pointer = nullptr;
                            bool duplicate = false;
                            for (const auto& tag : parent->second.tags)
                                if (tag.directory && tag.ordinal == ordinal) {
                                    duplicate = pointer != nullptr;
                                    if (duplicate)
                                        break;
                                    pointer = &tag;
                                }
                            if (pointer && !duplicate && pointer->count == 1 && (pointer->type == 4 || pointer->type == 13))
                                group.role = pointer->id == 0x8769 ? Role::Exif : pointer->id == 0x8825 ? Role::Gps
                                                                                                        : Role::Other;
                        }
                    }
                }
                if (group.role == Role::Other)
                    continue;
                std::map<uint32_t, std::size_t> counts;
                for (const auto& tag : group.tags)
                    if (!tag.directory)
                        ++counts[tag.id];
                for (const auto& tag : group.tags) {
                    if (tag.directory)
                        continue;
                    const bool ambiguous = counts[tag.id] > 1;
                    bool more = true;
                    if (group.role == Role::Image) {
                        switch (tag.id) {
                        case 0x010f: more = builder.string(tag, "camera.make", ambiguous); break;
                        case 0x0110: more = builder.string(tag, "camera.model", ambiguous); break;
                        case 0x0112: more = builder.scalar(tag, "orientation.exif", MetadataUnit::Unitless, 3, 1, 8, ambiguous); break;
                        case 0x0132: more = builder.string(tag, "time.modified.civil", ambiguous, TextKind::CivilTime); break;
                        }
                    } else if (group.role == Role::Exif) {
                        constexpr double positive = std::numeric_limits<double>::min();
                        constexpr double maximum = std::numeric_limits<double>::max();
                        switch (tag.id) {
                        case 0x829a: more = builder.scalar(tag, "exposure.duration", MetadataUnit::Seconds, 5, positive, maximum, ambiguous); break;
                        case 0x829d: more = builder.scalar(tag, "lens.f_number", MetadataUnit::Unitless, 5, positive, maximum, ambiguous); break;
                        case 0x8827: more = builder.scalar(tag, "sensitivity.photographic", MetadataUnit::Unitless, 3, 0, 65535, ambiguous, 0, 65535); break;
                        case 0x8830: more = builder.scalar(tag, "sensitivity.type", MetadataUnit::Unitless, 3, 0, 7, ambiguous, 0); break;
                        case 0x8831: more = builder.scalar(tag, "sensitivity.standard_output", MetadataUnit::Unitless, 4, 0, UINT32_MAX, ambiguous, 0); break;
                        case 0x8832: more = builder.scalar(tag, "sensitivity.recommended_exposure_index", MetadataUnit::Unitless, 4, 0, UINT32_MAX, ambiguous, 0); break;
                        case 0x8833: more = builder.scalar(tag, "sensitivity.iso_speed", MetadataUnit::Unitless, 4, 0, UINT32_MAX, ambiguous, 0); break;
                        case 0x920a: more = builder.scalar(tag, "lens.focal_length", MetadataUnit::Millimetres, 5, positive, maximum, ambiguous); break;
                        case 0xa405: more = builder.scalar(tag, "lens.focal_length_35mm", MetadataUnit::Millimetres, 3, 0, 65535, ambiguous, 0); break;
                        case 0xa433: more = builder.string(tag, "lens.make", ambiguous); break;
                        case 0xa434: more = builder.string(tag, "lens.model", ambiguous); break;
                        case 0x9003: more = builder.string(tag, "time.original.civil", ambiguous, TextKind::CivilTime) && builder.captureTime(group, tag, ambiguous, 0x9011, 0x9291, "time.original"); break;
                        case 0x9004: more = builder.string(tag, "time.digitized.civil", ambiguous, TextKind::CivilTime) && builder.captureTime(group, tag, ambiguous, 0x9012, 0x9292, "time.digitized"); break;
                        case 0x9010: more = builder.string(tag, "time.modified.offset", ambiguous, TextKind::Offset); break;
                        case 0x9011: more = builder.string(tag, "time.original.offset", ambiguous, TextKind::Offset); break;
                        case 0x9012: more = builder.string(tag, "time.digitized.offset", ambiguous, TextKind::Offset); break;
                        case 0x9290: more = builder.string(tag, "time.modified.subseconds", ambiguous, TextKind::Subseconds); break;
                        case 0x9291: more = builder.string(tag, "time.original.subseconds", ambiguous, TextKind::Subseconds); break;
                        case 0x9292: more = builder.string(tag, "time.digitized.subseconds", ambiguous, TextKind::Subseconds); break;
                        }
                    } else if (group.role == Role::Gps) {
                        switch (tag.id) {
                        case 2:
                        case 4: more = builder.coordinate(group, tag, ambiguous); break;
                        case 6: more = builder.altitude(group, tag, ambiguous); break;
                        case 7: more = builder.gpsTime(tag, ambiguous); break;
                        case 18: more = builder.string(tag, "gps.map_datum", ambiguous); break;
                        case 29: more = builder.gpsDate(tag, ambiguous); break;
                        }
                    }
                    if (!more)
                        return std::move(builder.result);
                }
            }
            return std::move(builder.result);
        } catch (const std::bad_alloc&) {
            return make_error({.code = ErrorCode::ResourceExhausted, .domain = ErrorDomain::IO, .detail = "Image metadata summary allocation failed", .detection = LFS_SOURCE_SITE_CURRENT()});
        }
    }
} // namespace lfs::media
