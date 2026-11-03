// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "media/media_ingest.hpp"
#include <chrono>
#include <fstream>
#include <future>
#include <limits>
#include <stdexcept>

namespace {
    using namespace lfs::media;
    void require(bool condition, const char* message) {
        if (!condition)
            throw std::runtime_error(message);
    }
    constexpr auto packet =
        "1\r\n00:00:00,000 --> 00:00:01,000\r\n"
        "<font size=\"28\">FrameCnt: 1, DiffTime: 33ms\r\n"
        "[latitude: 0] [longtitude: -73.25] [rel_alt: 12.5 abs_alt: -2] [unknown: kept]</font>\r\n\r\n"
        "2\r\n00:00:02,000 --> 00:00:03,000\r\n"
        "FrameCnt: 2 [latitude: 45.5] [longitude: 180]\r\n";
} // namespace
int runSrtMetadataContracts() {
    const auto root = std::filesystem::temp_directory_path() / ("lfs-srt-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::error_code error;
            std::filesystem::remove_all(path, error);
        }
    } cleanup{root};
    const auto path = root / "telemetry.srt";
    const auto write = [&](std::string_view text) {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        file.write(text.data(), text.size());
        require(file.good(), "write SRT fixture");
    };
    write(packet);
    const auto before = std::filesystem::last_write_time(path);
    auto result = MediaIngest::sourceSrtMetadata(path, {.bytes = std::string_view(packet).size()});
    if (!result)
        throw std::runtime_error(std::string(result.error().detail()));
    auto catalog = *result;
    require(catalog.cues().size() == 2, "FFmpeg SRT cue count");
    const auto raw = catalog.originalBytes();
    require(std::string_view(reinterpret_cast<const char*>(raw.data()), raw.size()) == packet && std::filesystem::last_write_time(path) == before,
            "SRT collection preserves every original CRLF byte and input mtime");
    const auto& first = catalog.cues()[0];
    require(first.start_milliseconds == 0 && first.end_milliseconds == 1000 && first.source_offset >= 0,
            "SRT interval timing and source position");
    require(first.dji_bracket_profile && first.facts.size() == 4 && first.facts[0].value == 0.0 && first.facts[1].value == -73.25 &&
                first.facts[2].value == 12.5 && first.facts[3].value == -2.0 && first.facts[2].unit == "unspecified",
            "DJI bracket facts retain zero, signs, alias and distinct altitude declarations");
    for (const auto& fact : first.facts) {
        require(fact.offset <= first.text.size() && fact.bytes <= first.text.size() - fact.offset,
                "fact byte provenance stays inside the owned cue text");
    }
    const auto expect = [&](Timestamp timestamp, std::int64_t offset, std::optional<std::size_t> index) {
        const auto match = catalog.lookup(timestamp, offset);
        require(match && *match == index, "exact interval/clock lookup");
    };
    expect({0, {1, 1000}}, 0, 0);
    expect({999999, {1, 1000000}}, 0, 0);
    expect({1000000, {1, 1000000}}, 0, std::nullopt);
    expect({-1, {1, 90000}}, 0, std::nullopt);
    expect({-1, {1, 90000}}, 1, 0);
    expect({0, {1001, 30000}}, 2000, 1);
    expect({3000, {1, 1000}}, 0, std::nullopt);
    expect({1500, {1, 1000}}, 0, std::nullopt);
    require(!catalog.lookup({1, {0, 1}}, 0), "invalid time base rejected");
    require(!catalog.lookup({std::numeric_limits<std::int64_t>::max(), {1, 1000}}, 1), "offset overflow rejected");
    require(!catalog.lookup({std::numeric_limits<std::int64_t>::max(), {std::numeric_limits<int>::max(), 1}}, 0), "rescale overflow rejected");
    for (const auto limits : {SrtMetadataLimits{.bytes = std::string_view(packet).size() - 1}, SrtMetadataLimits{.cues = 1},
                              SrtMetadataLimits{.line_bytes = 10}, SrtMetadataLimits{.lines = 2}, SrtMetadataLimits{.facts = 1}, SrtMetadataLimits{.cue_bytes = 10}}) {
        const auto limited = MediaIngest::sourceSrtMetadata(path, limits);
        require(!limited && limited.error().code() == lfs::ErrorCode::ResourceExhausted, "each SRT collection budget is enforced");
    }
    const auto invalid_budget = MediaIngest::sourceSrtMetadata(path, {.bytes = 0});
    require(!invalid_budget && std::string(invalid_budget.error().detail()).find("bytes=0/33554432") != std::string::npos,
            "invalid SRT budget reports the observed value and maximum");
    std::filesystem::remove(path);
    require(catalog.cues().size() == 2 && catalog.originalBytes().size() == std::string_view(packet).size(), "catalog owns bytes and intervals after file removal");
    auto copy = catalog;
    auto parallel = std::async(std::launch::async, [copy] {
        for (int i = 0; i < 1000; ++i) {
            const auto match = copy.lookup({i, {1, 1000}}, 0);
            if (!match || *match != std::optional<std::size_t>(0))
                return false;
        }
        return true;
    });
    expect({500, {1, 1000}}, 0, 0);
    require(parallel.get(), "immutable lookup supports concurrent callers");
    write("1\n00:00:00,000 --> 00:00:01,000\nFrameCnt: 1 [latitude: nan] [longitude: 181] [latitude: 1] [longitude: 2]\n");
    const auto invalid = MediaIngest::sourceSrtMetadata(path);
    require(invalid && invalid->cues()[0].facts.size() == 4, "invalid and duplicate fields retained");
    for (const auto& fact : invalid->cues()[0].facts)
        require(fact.ambiguous, "duplicate semantics are not silently selected");
    require(!invalid->cues()[0].facts[0].value && !invalid->cues()[0].facts[1].value, "nonfinite and out-of-range coordinates are invalid, not zero");
    write("1\n00:00:00,000 --> 00:00:01,000\nordinary subtitle [latitude: 40]\n");
    const auto ordinary = MediaIngest::sourceSrtMetadata(path);
    require(ordinary && !ordinary->cues()[0].dji_bracket_profile && ordinary->cues()[0].facts.empty(), "ordinary subtitle does not acquire a DJI identity from one key");
    write("1\n00:00:00,000 --> 00:00:01,000\na\n\n2\n00:00:00,500 --> 00:00:01,500\nb\n");
    const auto overlapping = MediaIngest::sourceSrtMetadata(path);
    require(overlapping.has_value(), "overlapping original cues are retained");
    const auto ambiguous = overlapping->lookup({750, {1, 1000}}, 0);
    require(!ambiguous && ambiguous.error().code() == lfs::ErrorCode::FailedPrecondition, "overlap is ambiguous rather than arbitrary nearest GPS");
    require(overlapping->lookup({1200, {1, 1000}}, 0).value() == std::optional<std::size_t>(1), "single remaining overlap interval is usable");
    write("1\n00:00:00,000 --> 00:00:02,000\nlong\n\n2\n00:00:00,000 --> 00:00:01,000\nshort\n");
    const auto tied = MediaIngest::sourceSrtMetadata(path);
    require(tied && tied->cues().size() == 2, "equal-start cues are overlaps, not clock resets");
    require(!tied->lookup({500, {1, 1000}}, 0), "equal-start overlap is ambiguous");
    const auto surviving = tied->lookup({1500, {1, 1000}}, 0);
    require(surviving && *surviving && tied->cues()[**surviving].end_milliseconds == 2000, "long tied cue survives short cue expiry");
    write("1\n00:00:01,000 --> 00:00:02,000\na\n\n2\n00:00:00,000 --> 00:00:00,500\nb\n");
    const auto reset = MediaIngest::sourceSrtMetadata(path);
    require(!reset && reset.error().code() == lfs::ErrorCode::Unsupported, "FFmpeg sorting cannot hide a source clock reset");
    write("1\n00:00:00,000 --> 00:00:00,000\na\n\n2\n00:00:00,002 --> 00:00:00,002\nb\n");
    const auto points = MediaIngest::sourceSrtMetadata(path);
    require(points && points->cues().size() == 2, "zero-duration camera cues are retained as point samples");
    require(points->lookup({0, {1, 1000}}, 0).value() == std::optional<std::size_t>(0), "point lookup matches exact source time");
    require(!points->lookup({1, {1, 2000}}, 0).value(), "fractional time does not become an exact point match by rounding");
    require(points->lookup({1, {1, 2000}}, 0, 1).value() == std::optional<std::size_t>(0), "explicit point tolerance accepts the nearer first sample");
    require(points->lookup({3, {1, 2000}}, 0, 1).value() == std::optional<std::size_t>(1), "fractional nearest-point distance selects the later sample");
    require(!points->lookup({1, {1, 1000}}, 0, 1), "equal-distance point tie is ambiguous");
    require(!points->lookup({1, {1, 2000}}, 0, 0).value(), "zero tolerance cannot invent coverage");
    require(points->lookup({-1, {1, 90000}}, 0, 1).value() == std::optional<std::size_t>(0), "negative fractional timestamp maps to the first point only with tolerance");
    require(points->lookup({3000000, {1, 1000000000}}, 0, 1).value() == std::optional<std::size_t>(1), "point tolerance includes its exact boundary");
    require(!points->lookup({3000001, {1, 1000000000}}, 0, 1).value(), "point tolerance refuses a fraction beyond its boundary");
    require(points->lookup({std::numeric_limits<std::int64_t>::max(), {1, 1000}}, 0, std::numeric_limits<std::int64_t>::max()).value() == std::optional<std::size_t>(1),
            "nearest lookup does not overflow distance arithmetic at extreme timestamps");
    require(!points->lookup({0, {1, 1000}}, 0, -1), "negative tolerance rejected");
    write("1\n00:00:00,000 --> 00:00:00,000\na\n\n2\n00:00:00,000 --> 00:00:00,000\nb\n");
    const auto duplicate_points = MediaIngest::sourceSrtMetadata(path);
    require(duplicate_points && duplicate_points->cues().size() == 2, "distinct point payloads at the same time are retained");
    require(!duplicate_points->lookup({0, {1, 1000}}, 0) && !duplicate_points->lookup({1, {1, 2000}}, 0, 1),
            "exact and nearest lookup refuse duplicate point samples");
    for (const auto& malformed : {std::string(), std::string("no cue"), std::string("1\n00:00:01,000 --> 00:00:00,000\nx\n")}) {
        write(malformed);
        require(!MediaIngest::sourceSrtMetadata(path), "missing/empty timed input refused");
    }
    write(std::string_view("\xff\xfe"
                           "1\0",
                           4));
    const auto binary = MediaIngest::sourceSrtMetadata(path);
    require(!binary && binary.error().code() == lfs::ErrorCode::Unsupported &&
                std::string(binary.error().detail()).find("offset=3, size=4") != std::string::npos,
            "UTF16/binary SRT is refused with its observed NUL position");
    return 0;
}
