// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/path_utils.hpp"
#include "media/media_ingest.hpp"
#include "media/media_json.hpp"
#include "media/metadata_observation.hpp"
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

int runMetadataObservationContracts() {
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
