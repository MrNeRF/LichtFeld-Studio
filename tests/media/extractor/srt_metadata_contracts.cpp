// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "media/media_ingest.hpp"
#include "media/media_json.hpp"
#include <chrono>
#include <fstream>
#include <future>
#include <limits>
#include <random>
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
            try {
                std::error_code error;
                std::filesystem::remove_all(path, error);
            } catch (const std::exception&) {
                // Best-effort cleanup must not replace the test's own failure.
            }
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
    require(catalog.cues().size() == 2, "SRT cue count");
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
        require(fact.offset >= static_cast<std::size_t>(first.source_offset) && fact.offset <= raw.size() && fact.bytes <= raw.size() - fact.offset,
                "fact byte provenance stays inside the original input");
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
                              SrtMetadataLimits{.cue_facts = 1}, SrtMetadataLimits{.lines = 2}, SrtMetadataLimits{.facts = 1}, SrtMetadataLimits{.cue_bytes = 10}}) {
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
    require(!reset && reset.error().code() == lfs::ErrorCode::Unsupported, "source clock reset requires explicit segmentation");
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
    for (const auto text : {
             "1\n00:00:01,000 --> 00:00:00,000\nx\n\n2\n00:00:02,000 --> 00:00:03,000\ny\n",
             "1\n999999999999999999999:00:00,000 --> 00:00:01,000\nx\n",
             "1\n00:60:00,000 --> 01:01:00,000\nx\n",
             "1\n00:00:00,000 --> 00:00:01,000 X1:99999999999999999999 X2:2 Y1:3 Y2:4\nx\n"}) {
        write(text);
        require(!MediaIngest::sourceSrtMetadata(path), "malformed or unrepresentable timing is refused");
    }
    write("1\n00:00:00,000 --> 596:31:23,647\nx\n");
    const auto longest = MediaIngest::sourceSrtMetadata(path);
    require(longest && longest->cues()[0].end_milliseconds == 2147483647, "long durations retain their exact millisecond value");
    write("1\n00:00:00,000 --> 596:31:23,648\nx\n");
    require(MediaIngest::sourceSrtMetadata(path)->cues()[0].end_milliseconds == 2147483648LL, "duration is not narrowed to a signed 32-bit value");
    write("1\n00:00:00,000 --> 2562047788015:12:55,807\nx\n");
    require(MediaIngest::sourceSrtMetadata(path)->cues()[0].end_milliseconds == std::numeric_limits<std::int64_t>::max(), "maximum signed 64-bit millisecond timestamp is retained");
    write("1\n00:00:00,000 --> 2562047788015:12:55,808\nx\n");
    require(!MediaIngest::sourceSrtMetadata(path), "timestamp overflow beyond the exact 64-bit boundary is refused");
    write("1\n00:00:00.000 --> 00:00:01.000 X1:-1 X2:+2 Y1:3 Y2:4\nx\n");
    require(MediaIngest::sourceSrtMetadata(path).has_value(), "supported decimal separator and bounded subtitle coordinates remain accepted");
    write("1\n00:00:59,000 --> 00:00:60,000\nx\n\n2\n00:01:00,000 --> 00:01:01,000\ny\n");
    const auto carried_second = MediaIngest::sourceSrtMetadata(path);
    require(carried_second && carried_second->cues().size() == 2 && carried_second->cues()[0].end_milliseconds == 60000 &&
                carried_second->lookup({60000, {1, 1000}}, 0).value() == std::optional<std::size_t>(1),
            "camera :60 second carry preserves an exact minute transition without overlap");
    for (const auto end : {"00:00:00,000", "00:00:01,000"}) {
        const auto entry = std::string("00:00:00,000 --> ") + end + "\nx\n\n";
        write("1\n" + entry + "2\n" + entry);
        const auto duplicate = MediaIngest::sourceSrtMetadata(path);
        require(duplicate && duplicate->cues().size() == 2 && !duplicate->lookup({0, {1, 1000}}, 0), "identical occurrences are retained and lookup refuses ambiguity");
    }
    for (const auto marker : {"NotFrameCnt:", "_SrtCnt:", "FrameCntExtra:", "FrameCnt"}) {
        write(std::string("1\n00:00:00,000 --> 00:00:01,000\n") + marker + " [latitude: 12]\n");
        const auto caption = MediaIngest::sourceSrtMetadata(path);
        require(caption && !caption->cues()[0].dji_bracket_profile && caption->cues()[0].facts.empty(), "marker requires an entire supported token");
    }
    write("1\n00:00:00,000 --> 00:00:01,000\nNotFrameCnt: ignored <font>FrameCnt : 1 [latitude: +12] [longitude: -12]</font>\n");
    const auto later_marker = MediaIngest::sourceSrtMetadata(path);
    require(later_marker && later_marker->cues()[0].facts.size() == 2 && later_marker->cues()[0].facts[0].value == 12 && later_marker->cues()[0].facts[1].value == -12,
            "later valid marker and single signs remain supported");
    for (const auto number : {"+-12", "++12", "--12", "-+12", "+", "-", "1e9999"}) {
        for (const auto field : {"latitude", "longitude", "longtitude", "rel_alt", "abs_alt"}) {
            write(std::string("1\n00:00:00,000 --> 00:00:01,000\nFrameCnt: 1 [") + field + ": " + number + "]\n");
            const auto numeric = MediaIngest::sourceSrtMetadata(path);
            require(numeric && numeric->cues()[0].facts.size() == 1 && !numeric->cues()[0].facts[0].value, "malformed numeric token is retained but never interpreted as a value");
        }
    }
    const std::string heading = "1\n00:00:00,000 --> 00:00:01,000\n";
    for (const auto length : {4094u, 4095u, 4096u}) {
        write(heading + std::string(length, 'x') + "\n");
        const auto line = MediaIngest::sourceSrtMetadata(path);
        require(line && line->cues()[0].text == std::string(length, 'x'),
                "long physical lines retain their exact original bytes");
    }
    write(heading + "FrameCnt: 1 " + std::string(4071, 'x') + "[latitude: 12]\n");
    const auto long_line = MediaIngest::sourceSrtMetadata(path);
    require(long_line && long_line->cues()[0].facts[0].value == 12, "numeric token after a long physical line prefix remains intact");
    write("1\r00:00:00,000 --> 00:00:01,000\rFrameCnt: 1 [latitude: 12]\r");
    const auto cr = MediaIngest::sourceSrtMetadata(path);
    require(cr && cr->cues()[0].facts[0].value == 12, "CR-only line endings are supported without inventing text");
    const auto cr_budget = MediaIngest::sourceSrtMetadata(path, {.lines = 2});
    require(!cr_budget && cr_budget.error().code() == lfs::ErrorCode::ResourceExhausted, "CR-only input cannot bypass the line quota");
    for (const auto separator : {"\n", "\r", "\r\n"}) {
        write(std::string("1") + separator + "00:00:00,000 --> 00:00:01,000" + separator + "x" + separator);
        require(MediaIngest::sourceSrtMetadata(path, {.lines = 3}).has_value(), "line quota accepts its exact physical line count without counting a phantom trailing line");
        const auto exceeded = MediaIngest::sourceSrtMetadata(path, {.lines = 2});
        require(!exceeded && exceeded.error().code() == lfs::ErrorCode::ResourceExhausted, "physical line quota plus one is refused for each line-ending style");
    }
    write(packet);
    const auto early_cues = MediaIngest::sourceSrtMetadata(path, {.cues = 1});
    require(!early_cues && std::string(early_cues.error().detail()).find("SRT cue budget exceeded") != std::string::npos,
            "cue quota stops collection at the first excessive cue");
    write("1\n\v00:00:00,000 --> 00:00:01,000\nx\n\n2\n\f00:00:01,000 --> 00:00:02,000\ny\n");
    const auto spaced_cues = MediaIngest::sourceSrtMetadata(path, {.cues = 1});
    require(!spaced_cues && std::string(spaced_cues.error().detail()).find("SRT cue budget exceeded") != std::string::npos,
            "ASCII whitespace cannot bypass the cue quota");
    write("\xef\xbb\xbf"
          "00:00:00,000 --> 00:00:01,000\nx\n");
    require(MediaIngest::sourceSrtMetadata(path).has_value(), "UTF8 BOM before a timestamp is recognized by the parser");
    {
        constexpr std::size_t maximum = 32u * 1024 * 1024;
        const std::string cue = heading + "x\n";
        {
            std::string padded(maximum - cue.size(), ' ');
            for (std::size_t i = 4095; i < padded.size(); i += 4096)
                padded[i] = '\n';
            padded.back() = '\n';
            padded += cue;
            write(padded);
        }
        {
            const auto full = MediaIngest::sourceSrtMetadata(path);
            require(full && full->originalBytes().size() == maximum && full->cues().size() == 1,
                    "exact maximum original size is accepted with bounded physical lines");
        }
        {
            std::ofstream extra(path, std::ios::binary | std::ios::app);
            extra.put('\n');
            require(extra.good(), "append one byte beyond SRT input limit");
        }
        const auto too_large = MediaIngest::sourceSrtMetadata(path);
        require(!too_large && too_large.error().code() == lfs::ErrorCode::ResourceExhausted, "maximum original size plus one is refused before parsing");
    }
    write("00:00:00,000 --> 00:00:01,000\r\n12.5 m\r\n\r\nFrameCnt: 1 [longtitude: -73] [altitude: 12]\r\n123\r\nkept\r\n\r\n2\r\n00:00:01,000 --> 00:00:02,000\r\n\r\n3\r\n00:00:02,000 --> 00:00:03,000\r\n456");
    const auto native = MediaIngest::sourceSrtMetadata(path);
    require(native && native->cues().size() == 3, "empty cue between valid cues is retained");
    require(native->cues()[0].text == "12.5 m\r\n\r\nFrameCnt: 1 [longtitude: -73] [altitude: 12]\r\n123\r\nkept" && native->cues()[1].text.empty() && native->cues()[2].text == "456",
            "indexless digit-leading text, paragraphs, numeric payload and original CRLF survive");
    require(native->cues()[0].facts.size() == 2 && native->cues()[0].facts[1].semantic == "dji.srt.altitude" && native->cues()[0].facts[1].unit == "unspecified" && native->cues()[0].facts[1].value == 12,
            "altitude is a distinct declaration without an inferred datum");
    for (const auto separator : {"\n", "\r\n", "\r"}) {
        const std::string newline(separator);
        const auto payload = std::string("12.5 m") + newline + newline + "FrameCnt: 1 [latitude: 12]";
        const auto original = std::string("1") + newline + "00:00:00,000 --> 00:00:01,000" + newline + newline + " \t" + newline + payload + newline + newline + "2" + newline + "00:00:01,000 --> 00:00:02,000" + newline + newline;
        write(original);
        const auto blank_prefix = MediaIngest::sourceSrtMetadata(path, {.cue_bytes = payload.size()});
        require(blank_prefix && blank_prefix->cues().size() == 2, "leading separators do not consume the cue text budget or drop an empty cue");
        const auto& cue = blank_prefix->cues()[0];
        const auto expected_offset = original.find("12.5 m");
        require(cue.text == payload && cue.source_offset == static_cast<std::int64_t>(expected_offset), "text starts at its first nonblank line and retains internal paragraphs");
        const auto retained = blank_prefix->originalBytes();
        require(std::string_view(reinterpret_cast<const char*>(retained.data()), retained.size()) == original && cue.text.data() == reinterpret_cast<const char*>(retained.data()) + expected_offset, "cue trimming preserves original bytes and shares their storage");
        require(cue.facts.size() == 1 && cue.facts[0].offset == original.find("12]") && cue.facts[0].value == 12 && blank_prefix->cues()[1].text.empty(), "absolute fact provenance and genuinely empty cues survive leading separators");
    }
    write(heading + "12345");
    require(MediaIngest::sourceSrtMetadata(path, {.cue_bytes = 5}).has_value() && !MediaIngest::sourceSrtMetadata(path, {.cue_bytes = 4}), "numeric final text obeys the exact cue byte budget");
    for (const auto limit : {SrtMetadataLimits{.cue_facts = 0}, SrtMetadataLimits{.cue_facts = 65}})
        require(!MediaIngest::sourceSrtMetadata(path, limit), "per-cue fact limits must be positive and bounded");
    std::mt19937 random(73561);
    for (unsigned i = 0; i < 128; ++i) {
        std::string mutated(packet);
        const auto position = random() % mutated.size();
        if (i % 3 == 0)
            mutated.resize(position);
        else if (i % 3 == 1)
            mutated[position] = static_cast<char>(random() % 256);
        else
            for (unsigned j = 0; j < 8; ++j)
                mutated.insert(mutated.begin() + position, static_cast<char>(random() % 256));
        write(mutated);
        const auto observed = MediaIngest::sourceSrtMetadata(path);
        if (!observed)
            continue;
        const auto original = observed->originalBytes();
        require(std::string_view(reinterpret_cast<const char*>(original.data()), original.size()) == mutated, "accepted corrupted input retains its exact original bytes");
        for (const auto& cue : observed->cues()) {
            require(cue.start_milliseconds >= 0 && cue.end_milliseconds >= cue.start_milliseconds && cue.source_offset >= 0 &&
                        static_cast<std::uint64_t>(cue.source_offset) <= original.size() &&
                        cue.text.size() <= original.size() - static_cast<std::size_t>(cue.source_offset) &&
                        cue.text.data() == reinterpret_cast<const char*>(original.data()) + cue.source_offset,
                    "accepted corrupted input retains valid intervals and source offsets");
            for (const auto& fact : cue.facts)
                require(fact.offset >= static_cast<std::size_t>(cue.source_offset) && fact.offset <= static_cast<std::size_t>(cue.source_offset) + cue.text.size() && fact.bytes <= static_cast<std::size_t>(cue.source_offset) + cue.text.size() - fact.offset, "corrupted fact ranges remain inside the original cue slice");
        }
        const auto serialized = json_detail::srtMetadataJson(*observed);
        require(serialized && !serialized->empty(), "corrupted non-UTF8 input remains serializable with retained raw bytes");
        const auto lookup = observed->lookup({250, {1, 1000}}, 0, 1);
        require(!lookup || !*lookup || **lookup < observed->cues().size(), "corrupted input cannot produce an out-of-range lookup index");
    }
    return 0;
}
