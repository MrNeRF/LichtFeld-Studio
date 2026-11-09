// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "media/media_ingest.hpp"
#include "metadata_file_io.hpp"
#include "source_metadata_internal.hpp"
extern "C" {
#include <libavutil/mathematics.h>
}
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <limits>

namespace lfs::media {
    struct SrtMetadataStorage {
        MetadataSourceLocator source;
        std::vector<std::uint8_t> original;
        std::vector<SrtCue> cues;
        // For every start-time prefix, retain the two greatest interval ends.
        // Lookup can detect zero/one/multiple covering cues in O(log n), even
        // with a long early cue followed by many already expired short cues.
        std::vector<std::array<std::size_t, 2>> candidates;
        std::vector<std::size_t> points;
    };
    namespace {
        Error srtError(ErrorCode code, std::string detail, core::SourceSite site = LFS_SOURCE_SITE_CURRENT()) {
            return make_error({.code = code, .domain = ErrorDomain::IO, .detail = std::move(detail), .detection = site});
        }
        std::string_view trim(std::string_view value) {
            const auto space = [](unsigned char c) { return c == ' ' || (c >= '\t' && c <= '\r'); };
            while (!value.empty() && space(value.front()))
                value.remove_prefix(1);
            while (!value.empty() && space(value.back()))
                value.remove_suffix(1);
            return value;
        }
        bool takeInteger(std::string_view& text, std::int64_t& value) {
            if (text.empty() || text.front() < '0' || text.front() > '9')
                return false;
            const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
            if (parsed.ec != std::errc{})
                return false;
            text.remove_prefix(static_cast<std::size_t>(parsed.ptr - text.data()));
            return true;
        }
        bool takeTime(std::string_view& text, std::int64_t& value) {
            std::int64_t hours = 0, minutes = 0, seconds = 0, milliseconds = 0;
            const auto separator = [&](char c) {
                if (text.empty() || text.front() != c)
                    return false;
                text.remove_prefix(1);
                return true;
            };
            if (!takeInteger(text, hours) || !separator(':') ||
                !takeInteger(text, minutes) || minutes > 59 || !separator(':') ||
                !takeInteger(text, seconds) || seconds > 60 || text.empty() || (text.front() != ',' && text.front() != '.'))
                return false;
            text.remove_prefix(1);
            if (!takeInteger(text, milliseconds) || milliseconds > 999)
                return false;
            // Matrice camera sidecars use :60 at a minute boundary. Preserve
            // that arithmetic carry, rather than clamping or dropping the cue.
            const auto remainder = (minutes * 60 + seconds) * 1000 + milliseconds;
            if (hours > (std::numeric_limits<std::int64_t>::max() - remainder) / 3600000)
                return false;
            value = hours * 3600000 + remainder;
            return true;
        }
        bool validTimingSuffix(std::string_view text) {
            if (text.empty())
                return true;
            // A token glued to the final timestamp is not a supported suffix.
            if (trim(text).size() == text.size())
                return false;
            text = trim(text);
            if (text.empty())
                return true;
            for (const auto label : {"X1:", "X2:", "Y1:", "Y2:"}) {
                if (!text.starts_with(label))
                    return false;
                text.remove_prefix(3);
                text = trim(text);
                if (text.starts_with('+')) {
                    text.remove_prefix(1);
                    if (text.empty() || text.front() < '0' || text.front() > '9')
                        return false;
                }
                std::int32_t coordinate = 0;
                const auto parsed = std::from_chars(text.data(), text.data() + text.size(), coordinate);
                if (parsed.ec != std::errc{})
                    return false;
                text.remove_prefix(static_cast<std::size_t>(parsed.ptr - text.data()));
                if (!text.empty() && trim(text).size() == text.size())
                    return false;
                text = trim(text);
            }
            return text.empty();
        }
        Result<void> parseSrt(SrtMetadataStorage& storage, const SrtMetadataLimits& limits) {
            const std::string_view text(reinterpret_cast<const char*>(storage.original.data()), storage.original.size());
            std::size_t offset = 0, lines = 0, text_begin = 0, text_end = 0;
            std::optional<std::size_t> possible_index;
            const auto finish = [&]() -> Result<void> {
                if (storage.cues.empty())
                    return {};
                auto& cue = storage.cues.back();
                const auto end = possible_index.value_or(text_end);
                if (end - text_begin > limits.cue_bytes)
                    return Result<void>::failure(srtError(ErrorCode::ResourceExhausted, "SRT cue text exceeds its byte budget (bytes=" + std::to_string(end - text_begin) + ", limit=" + std::to_string(limits.cue_bytes) + ")"));
                cue.source_offset = static_cast<std::int64_t>(text_begin);
                cue.text = text.substr(text_begin, end - text_begin);
                return {};
            };
            while (offset < text.size()) {
                if (++lines > limits.lines)
                    return Result<void>::failure(srtError(ErrorCode::ResourceExhausted, "SRT line budget exceeded (lines=" + std::to_string(lines) + ", limit=" + std::to_string(limits.lines) + ")"));
                const auto delimiter = text.find_first_of("\r\n", offset);
                const auto end = delimiter == std::string_view::npos ? text.size() : delimiter;
                auto next = end;
                if (next < text.size() && text[next++] == '\r' && next < text.size() && text[next] == '\n')
                    ++next;
                auto line = text.substr(offset, end - offset);
                if (const auto nul = line.find('\0'); nul != std::string_view::npos)
                    return Result<void>::failure(srtError(ErrorCode::Unsupported, "SRT input contains a NUL byte; UTF-16/32 and binary inputs are unsupported (offset=" + std::to_string(offset + nul) + ", size=" + std::to_string(text.size()) + ")"));
                if (!offset && line.starts_with("\xef\xbb\xbf"))
                    line.remove_prefix(3);
                line = trim(line);
                const auto arrow = line.find("-->");
                if (!line.empty() && ((line.front() >= '0' && line.front() <= '9') || line.front() == '+' || line.front() == '-') &&
                    arrow != std::string_view::npos && line.substr(0, arrow).find(':') != std::string_view::npos) {
                    auto begin = trim(line.substr(0, arrow));
                    auto ending = trim(line.substr(arrow + 3));
                    std::int64_t start = 0, stop = 0;
                    if (!takeTime(begin, start) || !trim(begin).empty() || !takeTime(ending, stop) || !validTimingSuffix(ending) || stop < start)
                        return Result<void>::failure(srtError(ErrorCode::DataLoss, "SRT timing is malformed or outside the supported range (line=" + std::to_string(lines) + ", offset=" + std::to_string(offset) + ")"));
                    if (storage.cues.size() >= limits.cues)
                        return Result<void>::failure(srtError(ErrorCode::ResourceExhausted, "SRT cue budget exceeded (count=" + std::to_string(storage.cues.size() + 1) + ", limit=" + std::to_string(limits.cues) + ")"));
                    if (!storage.cues.empty() && start < storage.cues.back().start_milliseconds)
                        return Result<void>::failure(srtError(ErrorCode::Unsupported, "SRT timeline reorders source cues; explicit segmentation is required (line=" + std::to_string(lines) + ", start_ms=" + std::to_string(start) + ")"));
                    if (const auto completed = finish(); !completed)
                        return Result<void>::failure(completed.error());
                    storage.cues.push_back(SrtCue{.start_milliseconds = start, .end_milliseconds = stop});
                    text_begin = text_end = next;
                    possible_index.reset();
                } else if (!storage.cues.empty() && !line.empty()) {
                    if (text_end == text_begin)
                        text_begin = text_end = offset;
                    // Only an all-digit line immediately preceding the next timing
                    // can be an index. Preserve every other digit-leading payload.
                    possible_index = std::all_of(line.begin(), line.end(), [](char c) { return c >= '0' && c <= '9'; })
                                         ? std::optional(text_end)
                                         : std::nullopt;
                    text_end = end;
                    if (text_end - text_begin > limits.cue_bytes && !possible_index)
                        return Result<void>::failure(srtError(ErrorCode::ResourceExhausted, "SRT cue text exceeds its byte budget (bytes=" + std::to_string(text_end - text_begin) + ", limit=" + std::to_string(limits.cue_bytes) + ")"));
                }
                offset = next;
            }
            // Without a following timing line, a final all-digit line is text.
            possible_index.reset();
            return finish();
        }
        Result<void> interpret(SrtCue& cue, std::size_t remaining_facts, const SrtMetadataLimits& limits) {
            const auto marker = [&](std::string_view name) {
                std::size_t cursor = 0;
                while ((cursor = cue.text.find(name, cursor)) != std::string_view::npos) {
                    const auto previous = cursor ? static_cast<unsigned char>(cue.text[cursor - 1]) : 0;
                    const auto identifier = previous >= 128 || (previous >= 'a' && previous <= 'z') || (previous >= 'A' && previous <= 'Z') ||
                                            (previous >= '0' && previous <= '9') || previous == '_';
                    if (!identifier && trim(cue.text.substr(cursor + name.size())).starts_with(':'))
                        return true;
                    cursor += name.size();
                }
                return false;
            };
            if (!marker("FrameCnt") && !marker("SrtCnt"))
                return {};
            cue.dji_bracket_profile = true;
            struct Field {
                std::string_view key, semantic, unit;
                double low, high;
            };
            static constexpr Field fields[] = {
                {"latitude", "dji.srt.latitude", "degrees", -90, 90},
                {"longitude", "dji.srt.longitude", "degrees", -180, 180},
                {"longtitude", "dji.srt.longitude", "degrees", -180, 180},
                {"altitude", "dji.srt.altitude", "unspecified", -std::numeric_limits<double>::max(), std::numeric_limits<double>::max()},
                {"rel_alt", "dji.srt.relative_altitude", "unspecified", -std::numeric_limits<double>::max(), std::numeric_limits<double>::max()},
                {"abs_alt", "dji.srt.absolute_altitude", "unspecified", -std::numeric_limits<double>::max(), std::numeric_limits<double>::max()},
            };
            const auto text = cue.text;
            std::size_t cursor = 0;
            while ((cursor = text.find('[', cursor)) != std::string_view::npos) {
                const auto end = text.find(']', cursor + 1);
                if (end == std::string_view::npos)
                    break;
                auto group = text.substr(cursor + 1, end - cursor - 1);
                while (!group.empty()) {
                    group = trim(group);
                    const auto colon = group.find(':');
                    if (colon == std::string_view::npos)
                        break;
                    const auto key = trim(group.substr(0, colon));
                    group.remove_prefix(colon + 1);
                    group = trim(group);
                    const auto whitespace = group.find_first_of(" \t\r\n");
                    const auto raw = group.substr(0, whitespace);
                    for (const auto& field : fields) {
                        if (key != field.key)
                            continue;
                        double value = 0;
                        auto number = raw;
                        const auto leading_plus = number.starts_with('+');
                        if (leading_plus)
                            number.remove_prefix(1);
                        const auto parsed = std::from_chars(number.data(), number.data() + number.size(), value);
                        DjiSrtFact fact{field.semantic, field.unit, static_cast<std::size_t>(cue.source_offset) + static_cast<std::size_t>(raw.data() - text.data()), raw.size()};
                        if (!number.empty() && !(leading_plus && number.starts_with('-')) && parsed.ec == std::errc{} && parsed.ptr == number.data() + number.size() &&
                            std::isfinite(value) && value >= field.low && value <= field.high)
                            fact.value = value;
                        for (auto& previous : cue.facts)
                            if (previous.semantic == fact.semantic)
                                previous.ambiguous = fact.ambiguous = true;
                        if (cue.facts.size() >= limits.cue_facts || cue.facts.size() >= remaining_facts)
                            return Result<void>::failure(srtError(ErrorCode::ResourceExhausted, "SRT fact budget exceeded (cue_facts=" + std::to_string(cue.facts.size() + 1) + ", max_cue_facts=" + std::to_string(limits.cue_facts) + ", remaining=" + std::to_string(remaining_facts) + ")"));
                        cue.facts.push_back(fact);
                        break;
                    }
                    if (whitespace == std::string_view::npos)
                        break;
                    group.remove_prefix(whitespace);
                }
                cursor = end + 1;
            }
            return {};
        }
    } // namespace
    const MetadataSourceLocator& SrtMetadataCatalog::source() const noexcept { return storage_->source; }
    std::span<const std::uint8_t> SrtMetadataCatalog::originalBytes() const noexcept { return storage_->original; }
    std::span<const SrtCue> SrtMetadataCatalog::cues() const noexcept { return storage_->cues; }

    Result<SrtMetadataCatalog> MediaIngest::sourceSrtMetadata(const std::filesystem::path& path, const SrtMetadataLimits& limits) {
        if (path.empty() || path.native().find(std::filesystem::path::value_type{}) != std::filesystem::path::string_type::npos)
            return srtError(ErrorCode::InvalidArgument, "SRT path must be nonempty and contain no NUL bytes (native_units=" + std::to_string(path.native().size()) + ", contains_nul=" + std::to_string(path.native().find(std::filesystem::path::value_type{}) != std::filesystem::path::string_type::npos) + ")");
        if (!limits.bytes || limits.bytes > srt_metadata_maxima.bytes ||
            !limits.cues || limits.cues > srt_metadata_maxima.cues ||
            !limits.lines || limits.lines > srt_metadata_maxima.lines ||
            !limits.facts || limits.facts > srt_metadata_maxima.facts ||
            !limits.cue_bytes || limits.cue_bytes > srt_metadata_maxima.cue_bytes ||
            !limits.cue_facts || limits.cue_facts > srt_metadata_maxima.cue_facts)
            return srtError(ErrorCode::InvalidArgument, "SRT limits must be positive and within maxima (bytes=" + std::to_string(limits.bytes) + "/" + std::to_string(srt_metadata_maxima.bytes) + ", cues=" + std::to_string(limits.cues) + "/" + std::to_string(srt_metadata_maxima.cues) + ", lines=" + std::to_string(limits.lines) + "/" + std::to_string(srt_metadata_maxima.lines) + ", facts=" + std::to_string(limits.facts) + "/" + std::to_string(srt_metadata_maxima.facts) + ", cue_bytes=" + std::to_string(limits.cue_bytes) + "/" + std::to_string(srt_metadata_maxima.cue_bytes) + ", cue_facts=" + std::to_string(limits.cue_facts) + "/" + std::to_string(srt_metadata_maxima.cue_facts) + ")");
        try {
            detail::SidecarFile file;
            const auto size = file.open(path, limits.bytes);
            if (!size)
                return size.error();
            auto bytes = detail::readSidecarBytes(file, *size);
            if (!bytes)
                return Error(bytes.error()).with_context("Read SRT metadata (path=" + core::path_to_utf8(path) + ")", LFS_SOURCE_SITE_CURRENT());
            auto storage = std::make_shared<SrtMetadataStorage>();
            storage->source = detail::sourceMetadataLocator(path);
            storage->original = std::move(*bytes);
            if (const auto parsed = parseSrt(*storage, limits); !parsed)
                return Error(parsed.error()).with_context("Read SRT metadata (path=" + core::path_to_utf8(path) + ")", LFS_SOURCE_SITE_CURRENT());
            std::size_t facts = 0;
            for (std::size_t index = 0; index < storage->cues.size(); ++index) {
                auto& cue = storage->cues[index];
                if (const auto interpreted = interpret(cue, limits.facts - facts, limits); !interpreted)
                    return interpreted.error();
                facts += cue.facts.size();
                if (cue.start_milliseconds == cue.end_milliseconds)
                    storage->points.push_back(index);
                constexpr auto missing = std::numeric_limits<std::size_t>::max();
                auto best = storage->candidates.empty() ? std::array{missing, missing} : storage->candidates.back();
                if (best[0] == missing || cue.end_milliseconds > storage->cues[best[0]].end_milliseconds) {
                    best[1] = best[0];
                    best[0] = index;
                } else if (best[1] == missing || cue.end_milliseconds > storage->cues[best[1]].end_milliseconds) {
                    best[1] = index;
                }
                storage->candidates.push_back(best);
            }
            if (storage->cues.empty())
                return srtError(ErrorCode::DataLoss, "SRT input contains no timed cues (path=" + core::path_to_utf8(path) + ", bytes=" + std::to_string(storage->original.size()) + ")");
            return SrtMetadataCatalog(std::move(storage));
        } catch (const std::bad_alloc&) {
            return srtError(ErrorCode::ResourceExhausted, "SRT metadata allocation failed (path=" + core::path_to_utf8(path) + ")");
        }
    }
    Result<std::optional<std::size_t>> SrtMetadataCatalog::lookup(Timestamp timestamp, std::int64_t offset, std::optional<std::int64_t> tolerance) const {
        if (tolerance && *tolerance < 0)
            return srtError(ErrorCode::InvalidArgument, "SRT point tolerance must be nonnegative (tolerance_ms=" + std::to_string(*tolerance) + ")");
        if (timestamp.time_base.numerator <= 0 || timestamp.time_base.denominator <= 0)
            return srtError(ErrorCode::InvalidArgument, "SRT lookup requires a positive time base (numerator=" + std::to_string(timestamp.time_base.numerator) + ", denominator=" + std::to_string(timestamp.time_base.denominator) + ")");
        const auto milliseconds = av_rescale_q_rnd(timestamp.ticks, {timestamp.time_base.numerator, timestamp.time_base.denominator}, {1, 1000}, AV_ROUND_DOWN);
        if (milliseconds == std::numeric_limits<std::int64_t>::min() ||
            (offset > 0 && milliseconds > std::numeric_limits<std::int64_t>::max() - offset) ||
            (offset < 0 && milliseconds < std::numeric_limits<std::int64_t>::min() - offset))
            return srtError(ErrorCode::InvalidArgument, "SRT lookup clock mapping overflows milliseconds (ticks=" + std::to_string(timestamp.ticks) + ", time_base=" + std::to_string(timestamp.time_base.numerator) + "/" + std::to_string(timestamp.time_base.denominator) + ", offset_ms=" + std::to_string(offset) + ")");
        const auto time = milliseconds + offset;
        const auto& entries = storage_->cues;
        auto end = std::upper_bound(entries.begin(), entries.end(), time, [](auto value, const SrtCue& cue) { return value < cue.start_milliseconds; });
        std::optional<std::size_t> found;
        if (end != entries.begin()) {
            const auto prefix = static_cast<std::size_t>(end - entries.begin() - 1);
            for (const auto candidate : storage_->candidates[prefix]) {
                if (candidate == std::numeric_limits<std::size_t>::max() || entries[candidate].end_milliseconds <= time)
                    continue;
                if (found)
                    return srtError(ErrorCode::FailedPrecondition, "SRT lookup matches overlapping cues (time_ms=" + std::to_string(time) + ", first=" + std::to_string(*found) + ", second=" + std::to_string(candidate) + ")");
                found = candidate;
            }
        }
        // Compute the sub-millisecond remainder exactly without multiplying
        // the full timestamp. Both modular operands are < INT_MAX, so their
        // product fits int64 even for extreme caller timestamps/time bases.
        const std::int64_t denominator = timestamp.time_base.denominator;
        auto ticks_mod = timestamp.ticks % denominator;
        if (ticks_mod < 0)
            ticks_mod += denominator;
        const auto factor_mod = (std::int64_t(timestamp.time_base.numerator) * 1000) % denominator;
        const auto remainder = (ticks_mod * factor_mod) % denominator;
        const auto& points = storage_->points;
        const auto start = [&](std::size_t index) { return entries[index].start_milliseconds; };
        auto right = std::upper_bound(points.begin(), points.end(), time, [&](auto value, auto index) { return value < start(index); });
        if (!remainder && right != points.begin() && start(*(right - 1)) == time) {
            const auto point = *(right - 1);
            if (found || (right - points.begin() >= 2 && start(*(right - 2)) == time))
                return srtError(ErrorCode::FailedPrecondition, "SRT lookup has ambiguous point/interval matches (time_ms=" + std::to_string(time) + ")");
            found = point;
        }
        if (found || !tolerance || points.empty())
            return found;
        // Ordered signed endpoints can span the full uint64 distance range.
        // Unsigned subtraction below avoids signed overflow.
        const auto budget = static_cast<std::uint64_t>(*tolerance);
        std::optional<std::size_t> left_position, right_position;
        std::uint64_t left_distance = 0, right_distance = 0;
        if (right != points.begin()) {
            const auto position = static_cast<std::size_t>(right - points.begin() - 1);
            left_distance = std::uint64_t(time) - std::uint64_t(start(points[position]));
            if (left_distance < budget || (left_distance == budget && !remainder))
                left_position = position;
        }
        if (right != points.end()) {
            const auto position = static_cast<std::size_t>(right - points.begin());
            right_distance = std::uint64_t(start(points[position])) - std::uint64_t(time);
            if (right_distance <= budget)
                right_position = position;
        }
        std::optional<std::size_t> selected;
        if (left_position && right_position) {
            if (left_distance > right_distance)
                selected = right_position;
            else if (left_distance == right_distance) {
                if (!remainder)
                    return srtError(ErrorCode::FailedPrecondition, "SRT nearest-point lookup has an equal-distance tie (time_ms=" + std::to_string(time) + ")");
                selected = right_position;
            } else if (right_distance - left_distance > 1 || remainder * 2 < denominator)
                selected = left_position;
            else if (remainder * 2 > denominator)
                selected = right_position;
            else
                return srtError(ErrorCode::FailedPrecondition, "SRT nearest-point lookup has an equal-distance tie (time_ms=" + std::to_string(time) + ")");
        } else {
            selected = left_position ? left_position : right_position;
        }
        if (!selected)
            return found;
        const auto position = *selected;
        const auto point_time = start(points[position]);
        if ((position && start(points[position - 1]) == point_time) ||
            (position + 1 < points.size() && start(points[position + 1]) == point_time))
            return srtError(ErrorCode::FailedPrecondition, "SRT nearest-point lookup has duplicate samples (time_ms=" + std::to_string(point_time) + ")");
        return std::optional<std::size_t>(points[position]);
    }
} // namespace lfs::media
