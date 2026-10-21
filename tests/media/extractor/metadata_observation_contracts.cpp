// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/path_utils.hpp"
#include "media/media_ingest.hpp"
#include "media/media_json.hpp"
#include "media/metadata_observation.hpp"
#include <algorithm>
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
        require(bool(normalized) && !normalized->limited && normalized->scope == "static_image_properties", "complete image summary");
        require(std::get<std::string>(fact(*normalized, "camera.make").value) == "Test Camera", "camera ASCII preserved");
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
        edit(changed, 0x9011).bytes = {'+', '1', '4', ':', '0', '1', 0};
        require(fact(summarize(changed), "time.original").state == MetadataValueState::Invalid, "offset range");
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
        const auto decoded_tiff = summarize(tiff);
        require(std::get<double>(fact(decoded_tiff, "lens.focal_length").value) == 35 &&
                    std::get<double>(fact(decoded_tiff, "gps.latitude").value) == -45.5 &&
                    std::get<std::string>(fact(decoded_tiff, "time.original").value) == std::get<std::string>(time.value),
                "TIFF decoded values agree without fabricated rational precision");
        tiff.original.tiff->directories[2].tags[1].floating_values[0] = std::numeric_limits<double>::infinity();
        require(fact(summarize(tiff), "gps.latitude").state == MetadataValueState::Invalid, "non-finite TIFF coordinates rejected");
        auto saved = summarize(image);
        require(std::get<std::string>(fact(saved, "camera.make").value) == "Test Camera" && fact(saved, "camera.make").source.observation,
                "facts own values and keep observation alive after catalog destruction");
        auto parallel = std::async(std::launch::async, [&] { return summarizeImageMetadata(*original); });
        require(parallel.get().has_value(), "immutable image summary can run concurrently");
    }
} // namespace

int runMetadataObservationContracts() {
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
