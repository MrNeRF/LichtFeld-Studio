// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
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
    MetadataNumericSeries series(const SourceMetadataCatalog& catalog) {
        MetadataNumericSeries result;
        result.source = *catalog.bind(StreamTagReference{0, 0});
        result.clock = {"sensor-clock", 7};
        result.channels = {{"gyro.x", MetadataUnit::RadiansPerSecond, std::nullopt},
                           {"gyro.y", MetadataUnit::RadiansPerSecond, 0.001}};
        result.timing = {{{-10, {1, 1000}}, 0, std::nullopt},
                         {{0, {1, 1000}}, 0, std::nullopt},
                         {{0, {1, 1000}}, 0, std::nullopt},
                         {{-100, {1, 1000}}, 1, std::nullopt}};
        result.values = {1, 2, 3, 4, 5, 6, 7, 8};
        result.states.assign(4, MetadataValueState::Present);
        return result;
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

    auto samples = series(catalog);
    require(bool(validateMetadataSeries(catalog, samples)), "negative timestamps, duplicate sample times and explicit clock resets are valid");
    require(!validateMetadataSeries(different, samples), "sample provenance cannot be rebound to a different catalog");
    MetadataSampleWindow window{{"sensor-clock", 7}, 0, {-1, {1, 1000}}, {1, {1, 1000}}};
    auto range = metadataSampleRange(catalog, samples, window);
    require(bool(range) && range->first == 1 && range->count == 2, "window keeps both high-frequency samples with equal timestamps");
    auto prepared_input = series(catalog);
    const auto* input_values = prepared_input.values.data();
    auto prepared = prepareMetadataSeries(catalog, std::move(prepared_input));
    require(bool(prepared) && prepared->data().values.data() == input_values, "prepare validates once and moves sample storage without copying");
    static_assert(std::is_const_v<std::remove_reference_t<decltype(std::declval<ValidatedMetadataSeries&>().data())>>);
    for (int i = 0; i < 100; ++i) {
        const auto repeated = prepared->range(window);
        require(bool(repeated) && repeated->first == 1 && repeated->count == 2, "immutable prepared view supports repeated logarithmic window queries");
    }
    const auto orphaned = [&] {
        auto local = fixture();
        return prepareMetadataSeries(local, series(local));
    }();
    require(bool(orphaned) && bool(orphaned->range(window)), "prepared view remains valid after the source catalog owner is destroyed");
    auto preserved = std::move(*prepared);
    require(!prepared->range(window) && prepared->data().values.empty() && bool(preserved.range(window)), "moved-from prepared view is rejected safely");
    window.segment = 1;
    window.begin = {-1, {1, 10}};
    window.end = {0, {1, 1}};
    range = metadataSampleRange(catalog, samples, window);
    require(bool(range) && range->first == 3 && range->count == 1, "equivalent rational timebases are compared by FFmpeg without clock conversion");
    window.segment = 99;
    range = metadataSampleRange(catalog, samples, window);
    require(bool(range) && range->count == 0, "missing segment yields no manufactured samples");
    window.clock.domain = "another-clock";
    require(!metadataSampleRange(catalog, samples, window), "different clock requires an explicit mapping outside this API");
    window.clock = samples.clock;
    window.begin = {1, {1, 1}};
    window.end = {0, {1, 1}};
    require(!metadataSampleRange(catalog, samples, window), "reversed window rejected");

    auto extremes = series(catalog);
    extremes.timing = {{{std::numeric_limits<int64_t>::min() + 1, {1, 1000000000}}, 0, std::nullopt},
                       {{std::numeric_limits<int64_t>::max() - 1, {1, 1000000000}}, 0, std::nullopt}};
    extremes.values.resize(4);
    extremes.states.resize(2);
    auto extreme_range = metadataSampleRange(catalog, extremes,
                                             {extremes.clock, 0, {std::numeric_limits<int64_t>::max() - 1, {1, 1000000000}}, {std::numeric_limits<int64_t>::max(), {1, 1000000000}}});
    require(bool(extreme_range) && extreme_range->first == 1 && extreme_range->count == 1,
            "large timestamp comparisons do not subtract or overflow ticks");
    const auto rejected = [&](auto change) {
        auto altered = series(catalog);
        change(altered);
        require(!validateMetadataSeries(catalog, altered), "malformed series must be rejected");
    };
    rejected([](auto& s) { s.timing[0].position.ticks = std::numeric_limits<int64_t>::min(); });
    rejected([](auto& s) { s.clock.domain.clear(); });
    rejected([](auto& s) { s.clock.stream_index = 42; });
    rejected([](auto& s) { s.clock.stream_index = -1; });
    rejected([](auto& s) { s.channels.clear(); });
    rejected([](auto& s) { s.channels[0].semantic.clear(); });
    rejected([](auto& s) { s.channels[0].unit = static_cast<MetadataUnit>(-1); });
    rejected([](auto& s) { s.channels[0].resolution = 0; });
    rejected([](auto& s) { s.channels[0].resolution = std::numeric_limits<double>::infinity(); });
    rejected([](auto& s) { s.timing[0].position.time_base.denominator = 0; });
    rejected([](auto& s) { s.timing[0].position.time_base.numerator = -1; });
    rejected([](auto& s) { s.timing[1].position.ticks = -11; });
    rejected([](auto& s) { s.timing[3].segment = 0; });
    rejected([](auto& s) { s.values.pop_back(); });
    rejected([](auto& s) { s.states.pop_back(); });
    rejected([](auto& s) { s.states[0] = static_cast<MetadataValueState>(-1); });
    rejected([](auto& s) { s.values[0] = std::numeric_limits<double>::quiet_NaN(); });
    rejected([](auto& s) { s.scope = MetadataTemporalScope::Static; });
    rejected([](auto& s) { s.scope = static_cast<MetadataTemporalScope>(-1); });
    auto associated = series(catalog);
    for (auto& time : associated.timing)
        time.frame_index = 12;
    require(bool(validateMetadataSeries(catalog, associated)), "multiple high-frequency samples may retain the same explicitly supplied video frame association");
    rejected([](auto& s) { s.origin = MetadataValueOrigin::Derived; });
    rejected([](auto& s) { s.origin = static_cast<MetadataValueOrigin>(-1); });
    rejected([](auto& s) { s.derivation = "unmarked-derived"; });
    samples.values[0] = std::numeric_limits<double>::quiet_NaN();
    samples.states[0] = MetadataValueState::Invalid;
    require(bool(validateMetadataSeries(catalog, samples)), "explicitly invalid sample retains its original nonfinite value");
    samples.origin = MetadataValueOrigin::Derived;
    samples.derivation = "test scale conversion";
    require(bool(validateMetadataSeries(catalog, samples)), "derived values must declare their transformation");
    samples = series(catalog);
    samples.scope = MetadataTemporalScope::PerFrame;
    for (std::size_t i = 0; i < samples.timing.size(); ++i)
        samples.timing[i].frame_index = i;
    require(bool(validateMetadataSeries(catalog, samples)), "per-frame associations require explicit frame indices");
    samples.timing[1].frame_index = 0;
    require(!validateMetadataSeries(catalog, samples), "duplicate per-frame identities rejected within a segment");
    auto extra_description = *catalog.container();
    auto another_stream = extra_description.streams.front();
    another_stream.index = 42;
    extra_description.streams.push_back(another_stream);
    extra_description.source_metadata->limited = true;
    auto extra_catalog = catalogContainerMetadata(std::move(extra_description));
    require(bool(extra_catalog), "create independent extra-stream observation");
    auto wrong_clock = series(*extra_catalog);
    wrong_clock.clock.stream_index = 42;
    require(!validateMetadataSeries(*extra_catalog, wrong_clock), "clock cannot silently refer to another existing source stream");
    samples = series(catalog);
    require(!validateMetadataSeries(catalog, samples, {.channels = 1}), "channel budget enforced");
    require(!validateMetadataSeries(catalog, samples, {.channels = 2, .samples = 3}), "sample budget enforced");
    require(!validateMetadataSeries(catalog, samples, {.channels = 2, .samples = 4, .value_bytes = 63}), "value byte budget enforced");
    require(bool(validateMetadataSeries(catalog, samples, {.channels = 2, .samples = 4, .value_bytes = 64})), "exact value byte boundary accepted");
    require(!validateMetadataSeries(catalog, samples, {.text_bytes = 0}), "aggregate text budget enforced");
    samples.timing.clear();
    samples.states.clear();
    samples.values.clear();
    require(bool(validateMetadataSeries(catalog, samples)), "empty series with declared channels and clock is valid");
    const auto prepared_empty = prepareMetadataSeries(catalog, samples);
    require(bool(prepared_empty), "prepare empty declared series");
    const auto empty_range = prepared_empty->range({samples.clock, 0, {0, {1, 1}}, {1, {1, 1}}});
    require(bool(empty_range) && empty_range->count == 0, "valid empty prepared series returns an empty window");
    // Independent linear oracle in millisecond units for mixed rational bases,
    // repeated timestamps and clock-reset segments; queries must remain half-open.
    auto many = series(catalog);
    many.timing.clear();
    many.values.clear();
    many.states.clear();
    for (std::uint64_t segment = 0; segment < 4; ++segment) {
        for (int i = 0; i < 32; ++i) {
            const int scale = i % 2 + 1;
            many.timing.push_back({{(i / 3 - 4) * scale, {1, 1000 * scale}}, segment, std::nullopt});
            many.values.insert(many.values.end(), {double(i), double(segment)});
            many.states.push_back(MetadataValueState::Present);
        }
    }
    const auto prepared_many = prepareMetadataSeries(catalog, std::move(many));
    require(bool(prepared_many), "prepare mixed-timebase series for independent window oracle");
    for (std::uint64_t segment = 0; segment < 5; ++segment) {
        for (int begin = -6; begin <= 8; ++begin) {
            for (int end = begin; end <= 9; ++end) {
                std::size_t expected_first = segment < 4 ? static_cast<std::size_t>(segment) * 32 : 128;
                std::size_t expected_count = 0;
                if (segment < 4) {
                    for (int i = 0; i < 32; ++i) {
                        const int millis = i / 3 - 4;
                        if (millis < begin)
                            ++expected_first;
                        else if (millis < end)
                            ++expected_count;
                    }
                }
                const auto selected = prepared_many->range({samples.clock, segment, {begin, {1, 1000}}, {end, {1, 1000}}});
                require(bool(selected) && selected->first == expected_first && selected->count == expected_count,
                        "prepared binary windows agree with independent linear oracle");
            }
        }
    }
    auto first = std::async(std::launch::async, [&] { return summarizeSourceMetadata(catalog); });
    auto second = std::async(std::launch::async, [&] { return summarizeSourceMetadata(different); });
    require(bool(first.get()) && bool(second.get()), "independent observations support parallel read-only summaries");
    return 0;
}
