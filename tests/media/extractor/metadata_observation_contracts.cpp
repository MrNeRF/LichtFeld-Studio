// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/path_utils.hpp"
#include "media/media_ingest.hpp"
#include "media/media_json.hpp"
#include "media/metadata_observation.hpp"
#include "media/metadata_summary_common.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <future>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {
    using namespace lfs::media;
    void require(bool value, const char* message) {
        if (!value)
            throw std::runtime_error(message);
    }
    SourceMetadataCatalog fixture() {
        MediaDescription description;
        StreamDescription stream;
        stream.index = 7;
        stream.width = 63;
        stream.height = 47;
        stream.nominal_frame_rate = Rational{30000, 1001};
        stream.color.primaries = "bt709";
        description.streams.push_back(stream);
        description.source_metadata.emplace();
        description.source_metadata->streams.push_back({7, {{"sensor", "test"}}, {}});
        auto catalog = catalogContainerMetadata(std::move(description));
        require(bool(catalog), "create observation fixture");
        return std::move(*catalog);
    }
} // namespace

namespace {
    ImageExifDescription cameraFixture() {
        ImageExifDescription image;
        image.original.status = lfs::core::image_codecs::MetadataStatus::Collected;
        image.original.blocks.emplace_back();
        image.interpretations.emplace_back();
        image.interpretations.front().status = ExifParseStatus::Parsed;
        auto& entries = image.interpretations.front().entries;
        const auto add = [&](uint16_t id, uint16_t type, std::vector<uint32_t> path) -> ExifEntry& {
            ExifEntry tag;
            tag.id = id;
            tag.type = type;
            tag.directory_path = std::move(path);
            tag.entry_index = static_cast<uint32_t>(std::count_if(entries.begin(), entries.end(), [&](const auto& e) { return e.directory_path == tag.directory_path; }));
            entries.push_back(std::move(tag));
            return entries.back();
        };
        const auto ascii = [&](uint16_t id, std::vector<uint32_t> path, const std::string& value) {
            auto& tag = add(id, 2, std::move(path));
            tag.bytes.assign(value.begin(), value.end());
            tag.bytes.push_back(0);
            tag.count = static_cast<uint32_t>(tag.bytes.size());
        };
        ascii(0x010f, {0}, "Test Camera");
        auto& exif = add(0x8769, 4, {0});
        exif.directory = true;
        exif.count = 1;
        auto& gps = add(0x8825, 4, {0});
        gps.directory = true;
        gps.count = 1;
        ascii(0x9003, {0, 1}, "2024:02:29 12:34:56");
        ascii(0x9011, {0, 1}, "+02:30");
        ascii(0x9291, {0, 1}, "000123456789");
        auto& focal = add(0x920a, 5, {0, 1});
        focal.count = 1;
        focal.rational_values = {{350, 10}};
        ascii(1, {0, 2}, "S");
        auto& latitude = add(2, 5, {0, 2});
        latitude.count = 3;
        latitude.rational_values = {{45, 1}, {30, 1}, {0, 1}};
        ascii(3, {0, 2}, "W");
        auto& longitude = add(4, 5, {0, 2});
        longitude.count = 3;
        longitude.rational_values = {{73, 1}, {15, 1}, {0, 1}};
        auto& altitude_ref = add(5, 1, {0, 2});
        altitude_ref.count = 1;
        altitude_ref.bytes = {1};
        auto& altitude = add(6, 5, {0, 2});
        altitude.count = 1;
        altitude.rational_values = {{125, 10}};
        ascii(18, {0, 2}, "WGS-84");
        // Same tag number under a root IFD must not become GPS latitude.
        auto& decoy = add(2, 5, {1});
        decoy.count = 3;
        decoy.rational_values = {{1, 1}, {2, 1}, {3, 1}};
        ascii(0x010f, {1}, "Thumbnail Camera");
        ascii(0x0110, {0}, "Primary Model");
        ascii(0x0132, {0}, "2024:02:29 12:34:56");
        auto& orientation = add(0x0112, 3, {0});
        orientation.count = 1;
        orientation.unsigned_values = {6};
        ascii(0x0110, {1}, "Thumbnail Model");
        ascii(0x0132, {1}, "2000:01:01 00:00:00");
        auto& thumbnail_orientation = add(0x0112, 3, {1});
        thumbnail_orientation.count = 1;
        thumbnail_orientation.unsigned_values = {1};
        auto& thumbnail_exif = add(0x8769, 4, {1});
        thumbnail_exif.directory = true;
        thumbnail_exif.count = 1;
        const auto thumbnail_exif_ordinal = thumbnail_exif.entry_index;
        auto& thumbnail_focal = add(0x920a, 5, {1, thumbnail_exif_ordinal});
        thumbnail_focal.count = 1;
        thumbnail_focal.rational_values = {{99, 1}};
        return image;
    }
    const MetadataFact& fact(const MetadataSummary& summary, std::string_view semantic) {
        const auto found = std::find_if(summary.facts.begin(), summary.facts.end(), [&](const auto& item) { return item.semantic == semantic; });
        require(found != summary.facts.end(), "expected semantic fact");
        return *found;
    }
    void cameraContracts() {
        auto image = cameraFixture();
        const auto original = catalogSourceMetadata(image);
        require(bool(original), "camera catalog fixture");
        const auto normalized = summarizeImageMetadata(*original);
        require(bool(normalized) && !normalized->limited && normalized->scope == MetadataSummaryScope::ImageProperties, "complete image summary");
        require(std::get<std::string>(fact(*normalized, "camera.make").value) == "Test Camera", "camera ASCII preserved");
        for (const auto semantic : {"camera.make", "camera.model", "orientation.exif", "time.modified.civil", "lens.focal_length"})
            require(std::count_if(normalized->facts.begin(), normalized->facts.end(), [&](const auto& item) { return item.semantic == semantic; }) == 1,
                    "secondary image and its EXIF fields cannot become primary image facts");
        require(std::get<int64_t>(fact(*normalized, "orientation.exif").value) == 6, "primary orientation is not replaced by thumbnail orientation");
        for (const auto& item : normalized->facts) {
            const auto* ref = std::get_if<ExifEntryReference>(&item.source.reference);
            const auto* entry = ref ? original->exifEntry(*ref) : nullptr;
            require(entry && entry->directory_path.front() == 0, "every image summary fact belongs to IFD0 or its direct metadata children");
        }
        require(std::get<ExifRational>(fact(*normalized, "lens.focal_length").value).denominator == 10 &&
                    fact(*normalized, "lens.focal_length").unit == MetadataUnit::Millimetres,
                "exact EXIF focal rational and units");
        const auto& time = fact(*normalized, "time.original");
        require(std::get<std::string>(time.value) == "2024-02-29T12:34:56.000123456789+02:30" && time.inputs.size() == 2 &&
                    time.origin == MetadataValueOrigin::Derived,
                "civil date, exact subseconds and explicit offset");
        require(original->resolve(time.source).has_value() && original->resolve(time.inputs[0]).has_value(), "derived operands retain source provenance");
        require(std::get<double>(fact(*normalized, "gps.latitude").value) == -45.5 &&
                    std::get<double>(fact(*normalized, "gps.longitude").value) == -73.25,
                "hemisphere sign and DMS conversion");
        require(std::get<double>(fact(*normalized, "gps.altitude_sea_level").value) == -12.5, "below sea-level reference");
        require(std::count_if(normalized->facts.begin(), normalized->facts.end(), [](const auto& item) { return item.semantic == "gps.latitude"; }) == 1,
                "thumbnail root tag cannot become GPS metadata");
        const auto serialized = json_detail::metadataSummary(*normalized);
        require(serialized["scope"] == "static_image_properties" && serialized["facts"][0].contains("ambiguous"), "image JSON carries interpretation status");
        const auto summarize = [](ImageExifDescription changed, std::size_t records = 4096) {
            const auto catalog = catalogSourceMetadata(std::move(changed), {records});
            require(bool(catalog), "changed camera fixture");
            const auto summary = summarizeImageMetadata(*catalog);
            require(bool(summary), "changed camera summary");
            return *summary;
        };
        const auto edit = [](ImageExifDescription& target, uint16_t id) -> ExifEntry& {
            auto& entries = target.interpretations[0].entries;
            const auto found = std::find_if(entries.begin(), entries.end(), [&](const auto& entry) { return entry.id == id; });
            require(found != entries.end(), "editable fixture tag");
            return *found;
        };
        auto changed = image;
        for (const auto id : {uint16_t(0x920a), uint16_t(0x829d)}) {
            changed = image;
            auto& optic = edit(changed, 0x920a);
            optic.id = id;
            optic.rational_values = {{0, 10}};
            const auto semantic = id == 0x920a ? "lens.focal_length" : "lens.f_number";
            const auto unknown = summarize(changed);
            require(fact(unknown, semantic).state == MetadataValueState::Absent &&
                        fact(unknown, semantic).diagnostic == "Source declares unknown value" &&
                        std::get<ExifRational>(fact(unknown, semantic).value).denominator == 10,
                    "zero manual-lens optics are unknown and retain their exact rational");
            optic.rational_values = {{0, 0}};
            require(fact(summarize(changed), semantic).state == MetadataValueState::Invalid, "zero denominator never becomes unknown optics");
            optic.type = 4;
            optic.rational_values.clear();
            optic.unsigned_values = {0};
            require(fact(summarize(changed), semantic).state == MetadataValueState::Invalid, "wrong-type zero is not interpreted as an unknown optical rational");
        }
        for (const auto id : {uint16_t(1), uint16_t(3)}) {
            for (const auto direction : id == 1 ? std::string_view("NS") : std::string_view("EW")) {
                changed = image;
                auto& ref = edit(changed, id);
                ref.bytes = {static_cast<uint8_t>(direction)};
                ref.count = 1;
                const auto accepted = summarize(changed);
                const double magnitude = id == 1 ? 45.5 : 73.25;
                require(std::get<double>(fact(accepted, id == 1 ? "gps.latitude" : "gps.longitude").value) ==
                            (direction == 'S' || direction == 'W' ? -magnitude : magnitude),
                        "one-byte unterminated ASCII hemisphere reference is interpreted narrowly");
            }
        }
        for (const auto invalid_ref : {std::string("NN"), std::string("n"), std::string("E"), std::string(" "), std::string("\xff")}) {
            changed = image;
            auto& ref = edit(changed, 1);
            ref.bytes.assign(invalid_ref.begin(), invalid_ref.end());
            ref.count = static_cast<uint32_t>(ref.bytes.size());
            require(fact(summarize(changed), "gps.latitude").state == MetadataValueState::Invalid, "hemisphere tolerance does not accept malformed values or multiple characters");
        }
        for (const auto wrong_type : {false, true}) {
            changed = image;
            auto& ref = edit(changed, 1);
            ref.bytes = {'N'};
            ref.count = wrong_type ? 1 : 2;
            ref.type = wrong_type ? 1 : 2;
            require(fact(summarize(changed), "gps.latitude").state == MetadataValueState::Invalid, "one-byte reference requires both ASCII type and matching declared count");
        }
        changed = image;
        edit(changed, 0x010f).bytes.pop_back();
        --edit(changed, 0x010f).count;
        require(fact(summarize(changed), "camera.make").state == MetadataValueState::Invalid, "unterminated ASCII tolerance is not generalized to camera text");
        changed = image;
        edit(changed, 0x9011).bytes.pop_back();
        --edit(changed, 0x9011).count;
        const auto invalid_zone = summarize(changed);
        require(fact(invalid_zone, "time.original.offset").state == MetadataValueState::Invalid &&
                    fact(invalid_zone, "time.original").state == MetadataValueState::Present &&
                    std::get<std::string>(fact(invalid_zone, "time.original").value) == "2024-02-29T12:34:56.000123456789" &&
                    fact(invalid_zone, "time.original").diagnostic == "Timezone unspecified" &&
                    fact(invalid_zone, "time.original").inputs.size() == 2,
                "malformed offset retains a valid local timestamp and the original component provenance");
        edit(changed, 0x9003).id = 0x9004;
        edit(changed, 0x9011).id = 0x9012;
        edit(changed, 0x9291).id = 0x9292;
        const auto digitized_zone = summarize(changed);
        require(fact(digitized_zone, "time.digitized.offset").state == MetadataValueState::Invalid &&
                    fact(digitized_zone, "time.digitized").state == MetadataValueState::Present &&
                    std::get<std::string>(fact(digitized_zone, "time.digitized").value) == "2024-02-29T12:34:56.000123456789" &&
                    fact(digitized_zone, "time.digitized").diagnostic == "Timezone unspecified",
                "digitized civil time uses the same conservative invalid-offset fallback");
        changed = image;
        auto& padded_make = edit(changed, 0x010f);
        padded_make.bytes.resize(32, 0);
        padded_make.count = 32;
        const auto padded = summarize(changed);
        require(fact(padded, "camera.make").state == MetadataValueState::Present &&
                    std::get<std::string>(fact(padded, "camera.make").value) == "Test Camera",
                "trailing NUL padding can be projected without changing original bytes");
        changed = image;
        auto& padded_fraction = edit(changed, 0x9291);
        padded_fraction.bytes.pop_back();
        padded_fraction.bytes.insert(padded_fraction.bytes.end(), 5, ' ');
        padded_fraction.bytes.push_back(0);
        padded_fraction.count = static_cast<uint32_t>(padded_fraction.bytes.size());
        require(std::get<std::string>(fact(summarize(changed), "time.original").value) == std::get<std::string>(time.value),
                "standard blank-padded subseconds retain the declared significant digits");
        changed = image;
        const std::string unknown_civil = "    :  :     :  :  ";
        edit(changed, 0x9003).bytes.assign(unknown_civil.begin(), unknown_civil.end());
        edit(changed, 0x9003).bytes.push_back(0);
        require(fact(summarize(changed), "time.original").state == MetadataValueState::Absent &&
                    fact(summarize(changed), "time.original.civil").state == MetadataValueState::Absent,
                "standard colon-preserving unknown date is not corruption");
        changed = image;
        edit(changed, 0x9011).bytes = {' ', ' ', ' ', ':', ' ', ' ', 0};
        require(fact(summarize(changed), "time.original").diagnostic == "Timezone unspecified" &&
                    fact(summarize(changed), "time.original.offset").state == MetadataValueState::Absent,
                "standard colon-preserving unknown UTC offset remains unknown");
        changed = image;
        edit(changed, 0x010f).bytes = {'A', 0, 'B', 0};
        edit(changed, 0x010f).count = 4;
        require(fact(summarize(changed), "camera.make").state == MetadataValueState::Invalid, "nonzero bytes after a terminator are not hidden by NUL-padding handling");
        changed = image;
        edit(changed, 0x010f).bytes = {'A', '\n', 'B', 0};
        edit(changed, 0x010f).count = 4;
        require(fact(summarize(changed), "camera.make").state == MetadataValueState::Present, "ASCII text is not restricted to printable characters");
        changed = image;
        edit(changed, 0x010f).bytes = {0};
        edit(changed, 0x010f).count = 1;
        require(fact(summarize(changed), "camera.make").state == MetadataValueState::Absent, "empty but valid ASCII camera field is absent rather than corrupt");
        changed = image;
        const auto partial_time = summarize(changed, 4);
        require(partial_time.limited && fact(partial_time, "time.original").state == MetadataValueState::Unsupported,
                "index truncation does not imply missing timezone or subseconds in the source");
        const auto partial_coordinate = summarize(changed, 9);
        require(partial_coordinate.limited && fact(partial_coordinate, "gps.latitude").state == MetadataValueState::Unsupported,
                "retained DMS and hemisphere cannot establish uniqueness outside a limited index");
        changed.original.status = lfs::core::image_codecs::MetadataStatus::Limited;
        require(fact(summarize(changed), "time.original").state == MetadataValueState::Present,
                "a fully parsed EXIF block remains complete when other envelopes were not collected");
        changed = image;
        auto& reference = edit(changed, 1);
        reference.bytes = {'N', 0};
        require(std::get<double>(fact(summarize(changed), "gps.latitude").value) == 45.5, "north sign");
        edit(changed, 2).rational_values = {{90, 1}, {0, 1}, {1, 1}};
        require(fact(summarize(changed), "gps.latitude").state == MetadataValueState::Invalid, "latitude cannot exceed pole through seconds");
        edit(changed, 2).rational_values = {{899, 10}, {59, 1}, {59, 1}};
        require(fact(summarize(changed), "gps.latitude").state == MetadataValueState::Invalid, "fractional degree component cannot overflow the coordinate range");
        changed = image;
        edit(changed, 2).rational_values[1].denominator = 0;
        require(fact(summarize(changed), "gps.latitude").state == MetadataValueState::Invalid, "zero coordinate denominator");
        changed = image;
        edit(changed, 1).directory_path = {1};
        require(fact(summarize(changed), "gps.latitude").state == MetadataValueState::Absent, "hemisphere not borrowed from thumbnail");
        changed = image;
        changed.interpretations[0].entries.push_back(edit(changed, 1));
        const auto duplicate = summarize(changed);
        require(fact(duplicate, "gps.latitude").state == MetadataValueState::Unsupported && fact(duplicate, "gps.latitude").ambiguous,
                "duplicate hemisphere never chooses precedence");
        changed = image;
        edit(changed, 5).bytes = {2};
        require(fact(summarize(changed), "gps.altitude_sea_level").state == MetadataValueState::Unsupported, "unqualified altitude reference is explicit");
        changed = image;
        edit(changed, 0x9011).directory_path = {1};
        const auto unknown_zone = summarize(changed);
        require(std::get<std::string>(fact(unknown_zone, "time.original").value) == "2024-02-29T12:34:56.000123456789" &&
                    fact(unknown_zone, "time.original").diagnostic == "Timezone unspecified",
                "no inferred UTC or cross-directory offset");
        for (const auto* bad_date : {"2023:02:29 12:00:00", "2024:04:31 12:00:00", "0000:01:01 00:00:00", "2024:01:01 24:00:00"}) {
            changed = image;
            auto& tag = edit(changed, 0x9003);
            tag.bytes.assign(bad_date, bad_date + 19);
            tag.bytes.push_back(0);
            require(fact(summarize(changed), "time.original").state == MetadataValueState::Invalid, "invalid civil date/time");
        }
        changed = image;
        edit(changed, 0x9011).bytes = {'+', '2', '4', ':', '0', '1', 0};
        require(fact(summarize(changed), "time.original").state == MetadataValueState::Present &&
                    fact(summarize(changed), "time.original").diagnostic == "Timezone unspecified" &&
                    fact(summarize(changed), "time.original.offset").state == MetadataValueState::Invalid,
                "invalid offset range does not invalidate usable local date and subseconds");
        edit(changed, 0x9011).bytes = {'+', '1', '4', ':', '0', '1', 0};
        require(fact(summarize(changed), "time.original").state == MetadataValueState::Present, "offset validation does not invent a geographic timezone bound");
        changed = image;
        const std::string leap_civil = "2016:12:31 23:59:60";
        edit(changed, 0x9003).bytes.assign(leap_civil.begin(), leap_civil.end());
        edit(changed, 0x9003).bytes.push_back(0);
        require(fact(summarize(changed), "time.original").state == MetadataValueState::Unsupported &&
                    fact(summarize(changed), "time.original.civil").state == MetadataValueState::Unsupported,
                "leap-second representation is unsupported rather than necessarily corrupt");
        changed = image;
        auto& gps_clock = edit(changed, 0x920a);
        gps_clock.id = 7;
        gps_clock.directory_path = {0, 2};
        gps_clock.count = 3;
        gps_clock.rational_values = {{23, 1}, {59, 1}, {60, 1}};
        require(fact(summarize(changed), "gps.utc_seconds_of_day").state == MetadataValueState::Unsupported, "GPS leap second is not rolled into the next day");
        gps_clock.rational_values[2] = {61, 1};
        require(fact(summarize(changed), "gps.utc_seconds_of_day").state == MetadataValueState::Invalid, "out-of-range GPS second is invalid");
        gps_clock.rational_values[2] = {119, 2};
        require(std::get<double>(fact(summarize(changed), "gps.utc_seconds_of_day").value) == 86399.5, "last fractional GPS second stays within its day");
        changed = image;
        edit(changed, 0x9011).bytes = {' ', ' ', ' ', ' ', ' ', ' ', 0};
        const auto blank_zone = summarize(changed);
        require(fact(blank_zone, "time.original.offset").state == MetadataValueState::Absent &&
                    fact(blank_zone, "time.original").diagnostic == "Timezone unspecified",
                "declared unknown offset is not UTC or corruption");
        changed = image;
        changed.interpretations[0].entries.push_back(edit(changed, 0x9011));
        require(fact(summarize(changed), "time.original").state == MetadataValueState::Unsupported, "duplicate offset is not chosen arbitrarily");
        changed = image;
        edit(changed, 0x9003).bytes.assign(20, ' ');
        edit(changed, 0x9003).bytes.back() = 0;
        require(fact(summarize(changed), "time.original").state == MetadataValueState::Absent, "unknown date is not a fabricated timestamp");
        changed = image;
        edit(changed, 0x920a).rational_values = {{1, 0}};
        require(fact(summarize(changed), "lens.focal_length").state == MetadataValueState::Invalid, "invalid focal rational preserved with invalid state");
        changed = image;
        edit(changed, 0x920a).id = 0xa405;
        edit(changed, 0xa405).type = 3;
        edit(changed, 0xa405).rational_values.clear();
        edit(changed, 0xa405).unsigned_values = {0};
        require(fact(summarize(changed), "lens.focal_length_35mm").state == MetadataValueState::Absent, "unknown equivalent focal length is not zero millimetres");
        changed = image;
        edit(changed, 0x920a).id = 0x8827;
        edit(changed, 0x8827).type = 3;
        edit(changed, 0x8827).rational_values.clear();
        edit(changed, 0x8827).unsigned_values = {65535};
        require(fact(summarize(changed), "sensitivity.photographic").state == MetadataValueState::Unsupported, "extended sensitivity marker not reported as ISO 65535");
        edit(changed, 0x8827).count = 2;
        edit(changed, 0x8827).unsigned_values = {100, 200};
        require(fact(summarize(changed), "sensitivity.photographic").state == MetadataValueState::Unsupported, "valid multi-value sensitivity is unsupported rather than corrupt");
        changed = image;
        changed.interpretations[0].status = ExifParseStatus::Malformed;
        require(summarize(changed).facts.empty(), "failed caller-built interpretation is not trusted");
        changed = image;
        edit(changed, 0x010f).bytes[0] = 255;
        require(fact(summarize(changed), "camera.make").state == MetadataValueState::Invalid, "non-ASCII never silently replaced");
        const auto no_facts = summarizeImageMetadata(*original, 0);
        const auto no_text = summarizeImageMetadata(*original, 4096, 0);
        require(no_facts && no_facts->limited && no_facts->facts.empty() && no_text && no_text->limited && no_text->facts.empty(), "zero summary budgets");
        require(summarize(image, 1).limited, "limited reference index remains visible");
        for (std::size_t budget = 0; budget < 300; ++budget) {
            const auto limited = summarizeImageMetadata(*original, 4096, budget);
            require(bool(limited), "every small text budget is safe");
            std::size_t used = 0;
            for (const auto& item : limited->facts) {
                used += item.semantic.size() + item.derivation.size() + item.diagnostic.size();
                if (const auto* value = std::get_if<std::string>(&item.value))
                    used += value->size();
            }
            require(used <= budget, "semantic text storage stays in aggregate budget");
        }
        // A TIFF snapshot uses decoded floating rationals and explicit IFD kinds.
        ImageExifDescription tiff;
        tiff.original.status = lfs::core::image_codecs::MetadataStatus::Collected;
        tiff.original.tiff.emplace();
        for (const auto kind : {"image", "exif", "gps"})
            tiff.original.tiff->directories.push_back({0, {}, kind, {}});
        tiff.original.tiff->directories[0].offset = 8;
        for (std::size_t index = 1; index < 3; ++index) {
            tiff.original.tiff->directories[index].offset = 128 * index;
            tiff.original.tiff->directories[index].parent = 8;
        }
        for (const auto& entry : image.interpretations[0].entries) {
            if (entry.directory || entry.directory_path[0] != 0)
                continue;
            const auto directory = entry.directory_path.size() == 1 ? 0 : entry.directory_path.back();
            lfs::core::image_codecs::TiffSourceTag tag;
            tag.id = entry.id;
            tag.type = entry.type;
            tag.count = entry.count;
            tag.bytes = entry.bytes;
            tag.unsigned_values = entry.unsigned_values;
            for (const auto ratio : entry.rational_values)
                tag.floating_values.push_back(double(ratio.numerator) / double(ratio.denominator));
            tiff.original.tiff->directories[directory].tags.push_back(std::move(tag));
        }
        const auto primary_directories = tiff.original.tiff->directories;
        tiff.original.tiff->directories.push_back({512, {}, "image", primary_directories[0].tags});
        tiff.original.tiff->directories.push_back({768, 8, "subimage", primary_directories[0].tags});
        tiff.original.tiff->directories.push_back({1024, 512, "exif", primary_directories[1].tags});
        tiff.original.tiff->directories.push_back({1280, 768, "gps", primary_directories[2].tags});
        const auto decoded_tiff = summarize(tiff);
        require(decoded_tiff.facts.size() == normalized->facts.size(), "TIFF secondary pages/subimages and their metadata cannot add primary facts");
        for (const auto& item : decoded_tiff.facts) {
            const auto* ref = std::get_if<TiffTagReference>(&item.source.reference);
            require(ref && ref->directory < 3, "all TIFF facts retain provenance in the primary image or its own metadata children");
        }
        require(std::get<double>(fact(decoded_tiff, "lens.focal_length").value) == 35 &&
                    std::get<double>(fact(decoded_tiff, "gps.latitude").value) == -45.5 &&
                    std::get<std::string>(fact(decoded_tiff, "time.original").value) == std::get<std::string>(time.value),
                "TIFF decoded values agree without fabricated rational precision");
        auto unknown_tiff = tiff;
        auto& optics = unknown_tiff.original.tiff->directories[1].tags;
        const auto focal = std::find_if(optics.begin(), optics.end(), [](const auto& tag) { return tag.id == 0x920a; });
        require(focal != optics.end(), "TIFF focal fixture exists");
        focal->floating_values = {0};
        auto f_number = *focal;
        f_number.id = 0x829d;
        optics.push_back(std::move(f_number));
        const auto unknown_tiff_summary = summarize(unknown_tiff);
        require(fact(unknown_tiff_summary, "lens.focal_length").state == MetadataValueState::Absent &&
                    fact(unknown_tiff_summary, "lens.f_number").state == MetadataValueState::Absent,
                "libtiff floating zero optics have the same unknown semantics as exact EXIF rationals");
        tiff.original.status = lfs::core::image_codecs::MetadataStatus::Limited;
        const auto partial_tiff = summarize(tiff);
        require(partial_tiff.limited && fact(partial_tiff, "gps.latitude").state == MetadataValueState::Unsupported &&
                    fact(partial_tiff, "gps.altitude_sea_level").state == MetadataValueState::Unsupported &&
                    fact(partial_tiff, "time.original").state == MetadataValueState::Unsupported,
                "partial TIFF observations cannot establish uniqueness of compound operands");
        tiff.original.status = lfs::core::image_codecs::MetadataStatus::Collected;
        tiff.original.tiff->directories[2].tags[1].floating_values[0] = std::numeric_limits<double>::infinity();
        require(fact(summarize(tiff), "gps.latitude").state == MetadataValueState::Invalid, "non-finite TIFF coordinates rejected");
        auto saved = summarize(image);
        require(std::get<std::string>(fact(saved, "camera.make").value) == "Test Camera" && fact(saved, "camera.make").source.observation,
                "facts own values and keep observation alive after catalog destruction");
        auto parallel = std::async(std::launch::async, [&] { return summarizeImageMetadata(*original); });
        require(parallel.get().has_value(), "immutable image summary can run concurrently");
    }
} // namespace

namespace {
    SourceMetadataCatalog xmpFixture(std::string_view xml, ImageExifDescription image = {}) {
        using namespace lfs::core::image_codecs;
        SourceXmp xmp;
        xmp.status = MetadataStatus::Collected;
        XmpBlock block;
        block.envelope = XmpEnvelope::JxlXml;
        block.jxl_box = JxlXmpBox{2, false};
        block.bytes.assign(xml.begin(), xml.end());
        xmp.blocks.push_back(std::move(block));
        auto result = catalogSourceMetadata(std::move(image), {}, {}, std::move(xmp));
        require(bool(result), "create XMP catalog");
        return std::move(*result);
    }
    void xmpContracts() {
        const std::string begin = R"(<r:RDF xmlns:r="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><r:Description r:about="" xmlns:t="http://ns.adobe.com/tiff/1.0/" xmlns:e="http://ns.adobe.com/exif/1.0/" )";
        const std::string end = "</r:Description></r:RDF>";
        const auto catalog = xmpFixture(begin + R"(t:Make="Camera &amp; Co" t:Model="body"><e:FNumber>28/10</e:FNumber><e:GPSLatitude>45,30.0N</e:GPSLatitude><e:GPSLongitude>90,10,30W</e:GPSLongitude><e:GPSAltitude>25/2</e:GPSAltitude><e:GPSAltitudeRef>1</e:GPSAltitudeRef><e:DateTimeOriginal>2024-02-29T10:20:30.120+02:00</e:DateTimeOriginal>)" + end);
        const auto result = summarizeImageMetadata(catalog);
        require(result && !result->limited && result->xmp_packets.size() == 1 && result->xmp_packets[0].state == XmpPacketState::Parsed && result->facts.size() == 8, "standard XMP scalar fields parsed");
        require(std::get<std::string>(result->facts[0].value) == "Camera & Co", "namespace-aware abbreviated RDF and predefined entity");
        require(std::get<ExifRational>(result->facts[2].value).denominator == 10, "XMP rational remains exact");
        require(std::get<double>(result->facts[3].value) == 45.5 && std::abs(std::get<double>(result->facts[4].value) + 90.175) < 1e-12, "XMP coordinate forms and hemisphere");
        require(std::get<double>(result->facts[5].value) == -12.5 && result->facts[5].xmp->inputs.size() == 1 && result->facts[5].xmp->inputs[0].ordinal == 7, "altitude qualifier identified within description");
        const auto json = json_detail::metadataSummary(*result);
        require(json["facts"][0]["xmp_property"]["namespace"] == "http://ns.adobe.com/tiff/1.0/" && json["facts"][0]["source"]["kind"] == "xmp_block", "property and source provenance serialized");
        auto retained = result->facts[0].source;
        require(catalog.resolve(retained).has_value(), "XMP fact binds immutable packet occurrence");
        const auto invalid = [&](std::string xml, XmpPacketState expected) {
            const auto summary = summarizeImageMetadata(xmpFixture(xml));
            require(summary && summary->facts.empty() && summary->xmp_packets.size() == 1 && summary->xmp_packets[0].state == expected, "packet failure discards speculative fields");
        };
        invalid(begin + R"(t:Make="prefix">)" + end.substr(0, end.size() - 1), XmpPacketState::Malformed);
        invalid(R"(<!DOCTYPE x [<!ENTITY a "expanded">]>)" + begin + R"(><t:Make>&a;</t:Make>)" + end, XmpPacketState::Unsupported);
        invalid(R"(<!DOCTYPE x SYSTEM "file:///must-not-open">)" + begin + ">" + end, XmpPacketState::Unsupported);
        invalid("<opaque/>", XmpPacketState::Unsupported);
        invalid(R"(<r:RDF xmlns:r="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><r:Description r:about="other"/></r:RDF>)", XmpPacketState::Unsupported);
        invalid(std::string(1024 * 1024 + 1, 'x'), XmpPacketState::Limited);
        const auto aliases = summarizeImageMetadata(xmpFixture(begin + R"(t:Make="one"><t:Make>two</t:Make><x:Make xmlns:x="urn:wrong">ignored</x:Make><t:Model><r:Seq><r:li>structured</r:li></r:Seq></t:Model>)" + end));
        require(aliases && aliases->facts.size() == 3 && aliases->facts[0].ambiguous && aliases->facts[1].ambiguous && aliases->facts[2].state == MetadataValueState::Unsupported, "duplicates retained, spoof namespace ignored, structure unsupported");
        for (const std::string lens : {"Lens One", "Lens Two"}) {
            const auto agreeing_aliases = summarizeImageMetadata(xmpFixture(begin + R"(xmlns:x="http://ns.adobe.com/xap/1.0/" xmlns:a="http://ns.adobe.com/exif/1.0/aux/" xmlns:c="http://cipa.jp/exif/1.0/" t:Make="Same"><t:Make>Same</t:Make><e:DateTimeDigitized>2024-02-29T12:34:56</e:DateTimeDigitized><x:CreateDate>2024-02-29T12:34:56</x:CreateDate><a:Lens>Lens One</a:Lens><c:LensModel>)" + lens + "</c:LensModel>" + end));
            require(agreeing_aliases && agreeing_aliases->facts.size() == 6, "alias summary");
            for (const auto& item : agreeing_aliases->facts)
                require(item.ambiguous == (item.semantic == "lens.model" && lens != "Lens One"), "equal aliases in one description agree; a differing alias flags only its semantic");
        }
        const auto limited = summarizeImageMetadata(catalog, 1);
        require(limited && limited->limited && limited->facts.empty(), "property limit does not project partial packet");
        const auto text_limited = summarizeImageMetadata(catalog, 4096, 1);
        require(text_limited && text_limited->limited && text_limited->facts.empty(), "summary text budget bounds XMP decoding");
        const auto conflicts = summarizeImageMetadata(xmpFixture(begin + R"(t:Make="different">)" + end, cameraFixture()));
        require(conflicts && std::count_if(conflicts->facts.begin(), conflicts->facts.end(), [](const auto& f) { return f.semantic == "camera.make" && f.ambiguous; }) == 2, "EXIF/XMP values retained as unresolved occurrences");
        const auto agreeing = summarizeImageMetadata(xmpFixture(begin + R"(t:Make="Test Camera" t:Orientation="6"><e:FocalLength>35/1</e:FocalLength><e:GPSLatitude>45,30S</e:GPSLatitude><e:GPSAltitude>12.5</e:GPSAltitude><e:GPSAltitudeRef>1</e:GPSAltitudeRef>)" + end, cameraFixture()));
        for (const auto semantic : {"camera.make", "orientation.exif", "lens.focal_length", "gps.latitude", "gps.altitude_sea_level"}) {
            std::size_t count = 0;
            for (const auto& item : agreeing->facts)
                if (item.semantic == semantic) {
                    ++count;
                    require(!item.ambiguous, "agreeing EXIF/XMP facts retain both sources without conflict");
                }
            require(count == 2, "both carriers contribute an agreeing value");
        }
        const auto orientation_conflict = summarizeImageMetadata(xmpFixture(begin + R"(t:Orientation="3">)" + end, cameraFixture()));
        require(orientation_conflict && std::count_if(orientation_conflict->facts.begin(), orientation_conflict->facts.end(), [](const auto& item) { return item.semantic == "orientation.exif" && item.ambiguous; }) == 2,
                "orientation disagreement marks both sources");
        const auto invalid_orientation = summarizeImageMetadata(xmpFixture(begin + R"(t:Orientation="bad">)" + end, cameraFixture()));
        require(invalid_orientation && std::count_if(invalid_orientation->facts.begin(), invalid_orientation->facts.end(), [](const auto& item) { return item.semantic == "orientation.exif" && item.ambiguous; }) == 2,
                "invalid scalar is not equivalent to a valid scalar");
        const auto shared_name = agreeing->facts.back().xmp->property.field;
        auto copied_summary = *agreeing;
        require(shared_name && copied_summary.facts.back().xmp->property.field == shared_name,
                "copies share immutable field descriptors");
        const auto again = summarizeImageMetadata(xmpFixture(begin + "><e:GPSAltitudeRef>1</e:GPSAltitudeRef>" + end));
        require(again && again->facts[0].xmp->property.field == shared_name,
                "field descriptor lifetime is independent of the parser and catalog");
        using namespace lfs::core::image_codecs;
        const auto independent = [&](std::string_view later, MetadataStatus status, ImageExifDescription exif = {}, bool reverse = false) {
            SourceXmp xmp;
            xmp.status = status;
            const auto good = begin + R"(t:Make="Unique">)" + end;
            for (const auto xml : {std::string_view(good), later}) {
                XmpBlock block;
                block.envelope = XmpEnvelope::JxlXml;
                block.jxl_box = JxlXmpBox{static_cast<uint32_t>(xmp.blocks.size()), false};
                block.bytes.assign(xml.begin(), xml.end());
                xmp.blocks.push_back(std::move(block));
            }
            if (reverse)
                std::reverse(xmp.blocks.begin(), xmp.blocks.end());
            auto catalog = catalogSourceMetadata(std::move(exif), {}, {}, std::move(xmp));
            require(bool(catalog), "mixed packet fixture");
            return summarizeImageMetadata(*catalog);
        };
        const auto rejected = begin + "><u>" + std::string(1024 * 1024, 'x') + "</u>" + end;
        for (const bool reverse : {false, true}) {
            const auto partial = independent(rejected, MetadataStatus::Collected, {}, reverse);
            require(partial && partial->limited && partial->facts.size() == 1 && !partial->facts[0].ambiguous,
                    "unrelated input omission never changes a valid fact's ambiguity in either order");
        }
        const auto collection_limited = independent("<broken/>", MetadataStatus::Limited);
        require(collection_limited && collection_limited->limited && collection_limited->facts.size() == 1 && !collection_limited->facts[0].ambiguous,
                "collector incompleteness stays separate from value conflicts");
        auto exif_limited = cameraFixture();
        exif_limited.original.status = MetadataStatus::Limited;
        auto& exif_entries = exif_limited.interpretations[0].entries;
        std::erase_if(exif_entries, [](const auto& entry) { return entry.id == 0x010f; });
        const auto partial_exif = independent("<broken/>", MetadataStatus::Collected, std::move(exif_limited));
        require(partial_exif && partial_exif->limited && !fact(*partial_exif, "camera.make").ambiguous,
                "EXIF limits do not make an independent XMP make ambiguous");
        const auto same = independent(begin + R"(t:Make="Unique">)" + end, MetadataStatus::Collected);
        require(same && same->facts.size() == 2 && !same->facts[0].ambiguous && !same->facts[1].ambiguous,
                "equal independent packets retain their sources without conflict");
        const auto different = independent(begin + R"(t:Make="Other">)" + end, MetadataStatus::Collected);
        require(different && different->facts[0].ambiguous && different->facts[1].ambiguous,
                "independent conflicting packets remain ambiguous");
        std::array<std::string_view, 3> rounded_values{"0.6666666666666666", "2/3", "6004799503160661/9007199254740992"};
        std::sort(rounded_values.begin(), rounded_values.end());
        do {
            SourceXmp packets;
            packets.status = MetadataStatus::Collected;
            for (const auto value : rounded_values) {
                const auto xml = begin + "><e:ExposureTime>" + std::string(value) + "</e:ExposureTime>" + end;
                XmpBlock block;
                block.envelope = XmpEnvelope::JxlXml;
                block.jxl_box = JxlXmpBox{static_cast<uint32_t>(packets.blocks.size()), false};
                block.bytes.assign(xml.begin(), xml.end());
                packets.blocks.push_back(std::move(block));
            }
            auto rounding_catalog = catalogSourceMetadata({}, {}, {}, std::move(packets));
            require(bool(rounding_catalog), "rounding conflict catalog");
            const auto summary = summarizeImageMetadata(*rounding_catalog);
            require(summary && summary->facts.size() == 3 && std::all_of(summary->facts.begin(), summary->facts.end(), [](const auto& item) { return item.ambiguous; }),
                    "distinct exact rationals remain conflicting in every order even if they round to the same double");
        } while (std::next_permutation(rounded_values.begin(), rounded_values.end()));
        require(summary_detail::equalValue(ExifRational{350, 10}, ExifRational{35, 1}) &&
                    summary_detail::equalValue(ExifRational{1, 125}, 0.008) &&
                    !summary_detail::equalValue(ExifRational{1, 3}, std::nextafter(1.0 / 3, 1.0)) &&
                    !summary_detail::equalValue(int64_t{9007199254740993}, 9007199254740992.0) &&
                    !summary_detail::equalValue(ExifRational{1, 0}, 0.0) &&
                    !summary_detail::equalValue(std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()),
                "numeric equivalence has no epsilon, overflow or invalid-value equality");
        for (const auto& value : {"2024", "2024-02", "2024-02-29", "2024-02-29T01:02", "2024-02-29T01:02Z", "2024-02-29T01:02:03.25-05:30"}) {
            const auto summary = summarizeImageMetadata(xmpFixture(begin + "><e:DateTimeOriginal>" + value + "</e:DateTimeOriginal>" + end));
            require(summary && summary->facts.size() == 1 && summary->facts[0].state == MetadataValueState::Present && std::get<std::string>(summary->facts[0].value) == value, "date precision and timezone preserved without inference");
        }
        for (const auto& value : {"2023-02-29", "2024-13", "2024-01-01T24:00", "2024-01-01T01:02:03.", "2024-01-01T01:02+24:00"}) {
            const auto summary = summarizeImageMetadata(xmpFixture(begin + "><e:DateTimeOriginal>" + value + "</e:DateTimeOriginal>" + end));
            require(summary && summary->facts.size() == 1 && summary->facts[0].state == MetadataValueState::Invalid, "malformed date never converted to a valid zero");
        }
        const auto leap = summarizeImageMetadata(xmpFixture(begin + "><e:DateTimeOriginal>2016-12-31T23:59:60Z</e:DateTimeOriginal>" + end));
        require(leap && leap->facts[0].state == MetadataValueState::Unsupported, "leap second not silently normalized");
        const auto missing_ref = summarizeImageMetadata(xmpFixture(begin + "><e:GPSAltitude>5/1</e:GPSAltitude>" + end));
        require(missing_ref && missing_ref->facts[0].state == MetadataValueState::Unsupported, "altitude sign cannot be inferred");
        const auto duplicate_ref = summarizeImageMetadata(xmpFixture(begin + "><e:GPSAltitude>5/1</e:GPSAltitude><e:GPSAltitudeRef>0</e:GPSAltitudeRef><e:GPSAltitudeRef>1</e:GPSAltitudeRef>" + end));
        require(duplicate_ref && duplicate_ref->facts[0].state == MetadataValueState::Unsupported && duplicate_ref->facts[0].ambiguous, "ambiguous altitude operand cannot be chosen");
        const auto complete = begin + R"(t:Make="valid">)" + end;
        for (std::size_t i = 0; i < complete.size(); ++i) {
            const auto prefix = summarizeImageMetadata(xmpFixture(std::string_view(complete).substr(0, i)));
            require(prefix && prefix->facts.empty() && prefix->xmp_packets.size() == 1 && prefix->xmp_packets[0].state == XmpPacketState::Malformed,
                    "every truncated XML prefix discards all speculative properties");
        }
        for (const auto size : {std::size_t{65536}, std::size_t{65537}}) {
            const auto summary = summarizeImageMetadata(xmpFixture(begin + "><t:Make>" + std::string(size, 'x') + "</t:Make>" + end), 4096, 1024 * 1024);
            require(summary && summary->xmp_packets[0].state == (size == 65536 ? XmpPacketState::Parsed : XmpPacketState::Limited), "decoded text exact boundary and overflow");
            require(summary->facts.size() == (size == 65536 ? 1 : 0), "text overflow never leaks a partial property");
        }
        for (const auto count : {1024, 1025}) {
            std::string properties;
            for (int i = 0; i < count; ++i)
                properties += "<t:Make>x</t:Make>";
            const auto summary = summarizeImageMetadata(xmpFixture(begin + ">" + properties + end), 4096, 1024 * 1024);
            require(summary && summary->xmp_packets[0].state == (count == 1024 ? XmpPacketState::Parsed : XmpPacketState::Limited), "property count boundary");
            require(summary->facts.size() == (count == 1024 ? 1024 : 0), "property overflow discards packet");
        }
        auto padded = complete;
        padded.resize(1024 * 1024, ' ');
        const auto exact_input = summarizeImageMetadata(xmpFixture(padded));
        require(exact_input && exact_input->xmp_packets[0].state == XmpPacketState::Parsed && exact_input->facts.size() == 1, "exact input byte budget accepted");
        padded += ' ';
        invalid(padded, XmpPacketState::Limited);
        // Namespace declarations are not ordinary attributes. Exercise a full
        // input budget with many bindings without depending on a timing limit.
        std::string namespace_heavy = R"(<r:RDF xmlns:r="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:t="http://ns.adobe.com/tiff/1.0/")";
        constexpr std::string_view namespace_tail = R"(><r:Description t:Make="namespace stress"/></r:RDF>)";
        std::size_t bindings = 0;
        for (;;) {
            const auto declaration = " xmlns:n" + std::to_string(bindings) + "=\"urn:stress:" + std::to_string(bindings) + "\"";
            if (declaration.size() + namespace_tail.size() > 1024 * 1024 - namespace_heavy.size())
                break;
            namespace_heavy += declaration;
            ++bindings;
        }
        require(bindings > 16384, "namespace stress exercises many bindings on one element");
        namespace_heavy += namespace_tail;
        namespace_heavy.resize(1024 * 1024, ' ');
        const auto namespace_catalog = xmpFixture(namespace_heavy);
        const auto namespace_summary = summarizeImageMetadata(namespace_catalog);
        require(namespace_summary && !namespace_summary->limited && namespace_summary->facts.size() == 1 &&
                    namespace_summary->xmp_packets.size() == 1 && namespace_summary->xmp_packets[0].state == XmpPacketState::Parsed &&
                    std::get<std::string>(namespace_summary->facts[0].value) == "namespace stress",
                "namespace-heavy packet at the byte limit retains the correct scalar");
        const auto namespace_payload = namespace_catalog.xmpPayloadBytes({0});
        require(namespace_payload && namespace_payload->size() == namespace_heavy.size() &&
                    std::equal(namespace_payload->begin(), namespace_payload->end(), namespace_heavy.begin()),
                "namespace stress preserves original payload");
        namespace_heavy += ' ';
        invalid(namespace_heavy, XmpPacketState::Limited);
        const auto recovered = summarizeImageMetadata(catalog);
        require(recovered && recovered->facts.size() == result->facts.size(), "fresh parser recovers after namespace stress and byte rejection");
        for (const auto count : {126, 127}) {
            std::string attrs = R"(t:Make="valid" )";
            for (int i = 0; i < count; ++i)
                attrs += "a" + std::to_string(i) + R"(="x" )";
            const auto summary = summarizeImageMetadata(xmpFixture(begin + attrs + ">" + end));
            require(summary && summary->xmp_packets[0].state == (count == 126 ? XmpPacketState::Parsed : XmpPacketState::Limited), "attribute count includes rdf about but excludes namespace declarations");
        }
        const auto split_altitude = summarizeImageMetadata(xmpFixture(begin + "><e:GPSAltitude>5/1</e:GPSAltitude></r:Description><r:Description xmlns:e=\"http://ns.adobe.com/exif/1.0/\"><e:GPSAltitudeRef>0</e:GPSAltitudeRef>" + end));
        require(split_altitude && split_altitude->facts[0].state == MetadataValueState::Unsupported, "altitude reference never borrowed across RDF descriptions");
        auto future = std::async(std::launch::async, [&] { return summarizeImageMetadata(catalog); });
        require(future.get()->facts.size() == result->facts.size(), "independent Expat parser supports concurrent catalog readers");
    }
} // namespace

int runMetadataObservationContracts() {
    xmpContracts();
    cameraContracts();
    require(!MediaIngest::sourceMetadata({}), "empty source path rejected before collection");
    require(!MediaIngest::sourceMetadata(std::filesystem::path(std::string("a\0b", 3))), "embedded NUL cannot redirect metadata collection");
    static_assert(std::is_const_v<std::remove_reference_t<decltype(std::declval<SourceMetadataCatalog&>().imageExif())>>);
    static_assert(std::is_const_v<std::remove_reference_t<decltype(std::declval<SourceMetadataCatalog&>().container())>>);
    static_assert(std::is_const_v<std::remove_reference_t<decltype(std::declval<SourceMetadataCatalog&>().records())>>);
#ifdef _WIN32
    const auto native_name = std::filesystem::path(std::wstring{wchar_t(0xe9), wchar_t(0x4e00)});
    const auto located = MediaIngest::sourceMetadata(native_name);
    require(bool(located) && located->sourceLocator() && located->sourceLocator()->encoding == MetadataPathEncoding::Utf16LE &&
                json_detail::sourceMetadata(*located)["source_locator_bytes"] == "e900004e",
            "Windows source location preserves original UTF16 code units");
#else
    // The locator contract does not assume the host filesystem accepts every
    // byte sequence (the real non-UTF8 file-open integration runs on Linux).
    const auto native_name = std::filesystem::path(std::string("missing-metadata-source-\xff.bin"));
    require(lfs::core::path_to_native_utf8(native_name) == native_name.native(), "I/O filename helper preserves non-UTF8 POSIX bytes");
    require(lfs::core::path_to_utf8(native_name) != native_name.native(), "display filename remains sanitized independently of I/O");
    const auto located = MediaIngest::sourceMetadata(native_name);
    require(bool(located) && located->sourceLocator() && located->sourceLocator()->encoding == MetadataPathEncoding::PosixBytes &&
                located->sourceLocator()->native_bytes == native_name.native() && located->sourceLocator()->display != native_name.native(),
            "non-UTF8 POSIX path remains exact independently of sanitized display text");
    require(json_detail::sourceMetadata(*located)["source_locator_bytes"] == json_detail::hexBytes(native_name.native()), "native path bytes survive JSON exactly");
#endif
    require(!catalogSourceMetadata({}, {}, MetadataSourceLocator{"display", "native", static_cast<MetadataPathEncoding>(-1)}), "unknown native path encoding is never treated as POSIX");
    require(!catalogSourceMetadata({}, {}, MetadataSourceLocator{"display", "x", MetadataPathEncoding::Utf16LE}), "truncated UTF16 path bytes rejected");
    require(!catalogSourceMetadata({}, {}, MetadataSourceLocator{"display", std::string("x\0y", 3), MetadataPathEncoding::PosixBytes}), "NUL bytes cannot masquerade as a native POSIX filename");
    auto catalog = fixture();
    const auto handle = catalog.bind(StreamDescriptionReference{0});
    require(bool(handle) && catalog.resolve(*handle).has_value(), "handle resolves only in its observation");
    require(!summarizeSourceMetadata(SourceMetadataCatalog{}), "image-only catalog is not advertised as a supported stream summary");
    auto bad_identity = *catalog.container();
    bad_identity.source_metadata->streams[0].stream_index = 42;
    require(!catalogContainerMetadata(std::move(bad_identity)), "mismatched source stream association rejected");
    bad_identity = *catalog.container();
    bad_identity.streams.push_back(bad_identity.streams[0]);
    require(!catalogContainerMetadata(std::move(bad_identity)), "duplicate native stream identities rejected");
    auto missing_stream_metadata = *catalog.container();
    missing_stream_metadata.source_metadata->streams.clear();
    require(!catalogContainerMetadata(missing_stream_metadata), "omitted stream nodes cannot claim a complete observation");
    missing_stream_metadata.source_metadata->limited = true;
    const auto partial_catalog = catalogContainerMetadata(std::move(missing_stream_metadata));
    require(bool(partial_catalog) && summarizeSourceMetadata(*partial_catalog)->limited,
            "explicitly limited stream prefixes retain descriptions and propagate the partial status");
    auto different = fixture();
    require(!different.resolve(*handle), "same ordinals in another source cannot be confused");
    require(!catalog.bind(StreamDescriptionReference{99}) && !catalog.resolve({}), "invalid and unbound handles are rejected");
    auto copy = catalog;
    require(copy.resolve(*handle).has_value(), "copy keeps the original observation identity");
    auto moved = std::move(copy);
    require(moved.resolve(*handle).has_value() && !copy.resolve(*handle), "move transfers observation identity");
    copy = fixture();
    require(!copy.resolve(*handle), "reassignment invalidates handles from the previous observation");

    const auto summary = summarizeSourceMetadata(catalog);
    require(bool(summary) && summary->facts.size() == 4 && !summary->limited, "known stream properties produce typed static facts");
    require(summary->facts[0].unit == MetadataUnit::Pixels && summary->facts[2].unit == MetadataUnit::FramesPerSecond &&
                std::get<ExifRational>(summary->facts[2].value).denominator == 1001,
            "stream units and rational frame rate remain explicit and exact");
    require(catalog.resolve(summary->facts[0].source).has_value(), "summary fact traces back to the same source");
    auto zero = summarizeSourceMetadata(catalog, 0);
    require(bool(zero) && zero->limited && zero->facts.empty(), "zero fact budget never invents a complete summary");
    require(summarizeSourceMetadata(catalog, 4096, 0)->limited, "summary text budget enforced before copying text");
    const auto encoded = json_detail::metadataSummary(*summary);
    require(encoded["facts"][0]["temporal_scope"] == "static" && encoded["facts"][0]["origin"] == "demuxer_observation" &&
                encoded["facts"][2]["unit"] == "frames_per_second",
            "summary transport distinguishes scope, units and origin");
    auto invalid_description = *catalog.container();
    invalid_description.streams[0].orientation.clockwise_degrees = std::numeric_limits<double>::quiet_NaN();
    auto invalid_catalog = catalogContainerMetadata(std::move(invalid_description));
    require(bool(invalid_catalog), "create independent invalid-property observation");
    auto invalid_summary = summarizeSourceMetadata(*invalid_catalog);
    require(bool(invalid_summary) && invalid_summary->facts[3].state == MetadataValueState::Invalid,
            "invalid angle is flagged, never silently zeroed");
    require(json_detail::metadataSummary(*invalid_summary)["facts"][3]["value"] == "nan", "invalid numeric class survives JSON transport instead of becoming an absent value");

    auto first = std::async(std::launch::async, [&] { return summarizeSourceMetadata(catalog); });
    auto second = std::async(std::launch::async, [&] { return summarizeSourceMetadata(different); });
    require(bool(first.get()) && bool(second.get()), "independent observations support parallel read-only summaries");
    return 0;
}
