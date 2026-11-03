// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "media/media_ingest.hpp"
#include "metadata_file_io.hpp"
#include "source_metadata_internal.hpp"
extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/mathematics.h>
}
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstring>
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
        Error srtError(ErrorCode code, std::string detail, int native = 0, core::SourceSite site = LFS_SOURCE_SITE_CURRENT()) {
            return make_error({.code = code, .domain = ErrorDomain::IO, .detail = std::move(detail), .detection = site, .native = native ? std::optional{NativeError{ErrorDomain::IO, native, "ffmpeg"}} : std::nullopt});
        }
        struct MemoryInput {
            std::span<const std::uint8_t> bytes;
            std::size_t offset = 0;
            static int read(void* opaque, std::uint8_t* target, int capacity) noexcept {
                auto& input = *static_cast<MemoryInput*>(opaque);
                if (capacity <= 0)
                    return AVERROR(EINVAL);
                const auto count = std::min(input.bytes.size() - input.offset, static_cast<std::size_t>(capacity));
                if (!count)
                    return AVERROR_EOF;
                std::memcpy(target, input.bytes.data() + input.offset, count);
                input.offset += count;
                return static_cast<int>(count);
            }
        };
        struct Demux {
            AVFormatContext* format = nullptr;
            AVIOContext* io = nullptr;
            AVPacket* packet = nullptr;
            ~Demux() {
                av_packet_free(&packet);
                avformat_close_input(&format);
                if (io)
                    av_freep(&io->buffer);
                avio_context_free(&io);
            }
        };
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
            if (!takeInteger(text, hours) || hours > std::numeric_limits<int>::max() || !separator(':') ||
                !takeInteger(text, minutes) || minutes > 59 || !separator(':') ||
                !takeInteger(text, seconds) || seconds > 60 || text.empty() || (text.front() != ',' && text.front() != '.'))
                return false;
            text.remove_prefix(1);
            if (!takeInteger(text, milliseconds) || milliseconds > 999)
                return false;
            // Matrice camera sidecars use :60 at a minute boundary. Preserve
            // that arithmetic carry, rather than clamping or dropping the cue.
            value = ((hours * 60 + minutes) * 60 + seconds) * 1000 + milliseconds;
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
        // This is a guard for the provider's representable input, not a second
        // subtitle parser. FFmpeg still identifies and produces every cue. Bound
        // its queue before opening and reject timing that it would repair/wrap.
        Result<std::size_t> preflightSrt(std::span<const std::uint8_t> bytes, const SrtMetadataLimits& limits) {
            const std::string_view text(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            std::size_t offset = 0, lines = 0, cues = 0;
            while (offset < text.size()) {
                if (++lines > limits.lines)
                    return srtError(ErrorCode::ResourceExhausted, "SRT line budget exceeded before demux (lines=" + std::to_string(lines) + ", limit=" + std::to_string(limits.lines) + ")");
                const auto delimiter = text.find_first_of("\r\n", offset);
                const auto end = delimiter == std::string_view::npos ? text.size() : delimiter;
                auto line = text.substr(offset, end - offset);
                if (const auto nul = line.find('\0'); nul != std::string_view::npos)
                    return srtError(ErrorCode::Unsupported, "SRT input contains a NUL byte; UTF-16/32 and binary inputs are unsupported (offset=" + std::to_string(offset + nul) + ", size=" + std::to_string(bytes.size()) + ")");
                if (line.size() > limits.line_bytes)
                    return srtError(ErrorCode::ResourceExhausted, "SRT text budget exceeded before demux (lines=" + std::to_string(lines) + ", line_bytes=" + std::to_string(line.size()) + ", max_lines=" + std::to_string(limits.lines) + ", max_line_bytes=" + std::to_string(limits.line_bytes) + ")");
                if (!offset && line.starts_with("\xef\xbb\xbf"))
                    line.remove_prefix(3);
                line = trim(line);
                const auto arrow = line.find("-->");
                if (!line.empty() && ((line.front() >= '0' && line.front() <= '9') || line.front() == '+' || line.front() == '-') &&
                    arrow != std::string_view::npos && line.substr(0, arrow).find(':') != std::string_view::npos) {
                    auto begin = trim(line.substr(0, arrow));
                    auto finish = trim(line.substr(arrow + 3));
                    std::int64_t start = 0, stop = 0;
                    if (!takeTime(begin, start) || !trim(begin).empty() || !takeTime(finish, stop) || !validTimingSuffix(finish) ||
                        stop < start || stop - start > std::numeric_limits<std::int32_t>::max())
                        return srtError(ErrorCode::DataLoss, "SRT timing is malformed or outside the provider range before demux (line=" + std::to_string(lines) + ", offset=" + std::to_string(offset) + ", max_duration_ms=2147483647)");
                    if (++cues > limits.cues)
                        return srtError(ErrorCode::ResourceExhausted, "SRT cue budget exceeded before demux (count=" + std::to_string(cues) + ", limit=" + std::to_string(limits.cues) + ")");
                }
                offset = end;
                if (offset < text.size()) {
                    const auto first = text[offset++];
                    if (first == '\r' && offset < text.size() && text[offset] == '\n')
                        ++offset;
                }
            }
            return cues;
        }
        Result<void> interpret(SrtCue& cue, std::size_t remaining_facts) {
            const auto marker = [&](std::string_view name) {
                std::size_t cursor = 0;
                while ((cursor = cue.text.find(name, cursor)) != std::string::npos) {
                    const auto previous = cursor ? static_cast<unsigned char>(cue.text[cursor - 1]) : 0;
                    const auto identifier = previous >= 128 || (previous >= 'a' && previous <= 'z') || (previous >= 'A' && previous <= 'Z') ||
                                            (previous >= '0' && previous <= '9') || previous == '_';
                    if (!identifier && trim(std::string_view(cue.text).substr(cursor + name.size())).starts_with(':'))
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
                {"rel_alt", "dji.srt.relative_altitude", "unspecified", -std::numeric_limits<double>::max(), std::numeric_limits<double>::max()},
                {"abs_alt", "dji.srt.absolute_altitude", "unspecified", -std::numeric_limits<double>::max(), std::numeric_limits<double>::max()},
            };
            const std::string_view text(cue.text);
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
                        DjiSrtFact fact{field.semantic, field.unit, static_cast<std::size_t>(raw.data() - text.data()), raw.size()};
                        if (!number.empty() && !(leading_plus && number.starts_with('-')) && parsed.ec == std::errc{} && parsed.ptr == number.data() + number.size() &&
                            std::isfinite(value) && value >= field.low && value <= field.high)
                            fact.value = value;
                        for (auto& previous : cue.facts)
                            if (previous.semantic == fact.semantic)
                                previous.ambiguous = fact.ambiguous = true;
                        if (cue.facts.size() >= 64 || cue.facts.size() >= remaining_facts)
                            return Result<void>::failure(srtError(ErrorCode::ResourceExhausted, "SRT fact budget exceeded (cue_facts=" + std::to_string(cue.facts.size() + 1) + ", max_cue_facts=64, remaining=" + std::to_string(remaining_facts) + ")"));
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
        if (!limits.bytes || limits.bytes > 32u * 1024 * 1024 || !limits.cues || limits.cues > 100000 ||
            !limits.line_bytes || limits.line_bytes > 4095 || !limits.lines || limits.lines > 1000000 || !limits.facts || limits.facts > 500000 || !limits.cue_bytes || limits.cue_bytes > 1024u * 1024)
            return srtError(ErrorCode::InvalidArgument, "SRT limits must be positive and within maxima (bytes=" + std::to_string(limits.bytes) + "/33554432, cues=" + std::to_string(limits.cues) + "/100000, line_bytes=" + std::to_string(limits.line_bytes) + "/4095, lines=" + std::to_string(limits.lines) + "/1000000, facts=" + std::to_string(limits.facts) + "/500000, cue_bytes=" + std::to_string(limits.cue_bytes) + "/1048576)");
        try {
            detail::SidecarFile file;
            const auto size = file.open(path, limits.bytes);
            if (!size)
                return size.error();
            auto bytes = detail::readSidecarBytes(file, *size);
            if (!bytes)
                return Error(bytes.error()).with_context("Read SRT metadata (path=" + core::path_to_utf8(path) + ")", LFS_SOURCE_SITE_CURRENT());
            const auto expected_cues = preflightSrt(*bytes, limits);
            if (!expected_cues)
                return Error(expected_cues.error()).with_context("Validate SRT metadata (path=" + core::path_to_utf8(path) + ")", LFS_SOURCE_SITE_CURRENT());
            const auto* input_format = av_find_input_format("srt");
            if (!input_format)
                return srtError(ErrorCode::Unavailable, "FFmpeg SRT demuxer is unavailable");
            auto storage = std::make_shared<SrtMetadataStorage>();
            storage->source = detail::sourceMetadataLocator(path);
            storage->original = std::move(*bytes);
            MemoryInput input{storage->original};
            Demux demux;
            auto* buffer = static_cast<std::uint8_t*>(av_malloc(4096));
            if (!buffer)
                throw std::bad_alloc();
            demux.io = avio_alloc_context(buffer, 4096, 0, &input, MemoryInput::read, nullptr, nullptr);
            if (!demux.io) {
                av_free(buffer);
                throw std::bad_alloc();
            }
            demux.format = avformat_alloc_context();
            if (!demux.format)
                throw std::bad_alloc();
            demux.format->pb = demux.io;
            demux.format->flags |= AVFMT_FLAG_CUSTOM_IO;
            const auto opened = avformat_open_input(&demux.format, nullptr, input_format, nullptr);
            if (opened < 0)
                return srtError(opened == AVERROR(ENOMEM) ? ErrorCode::ResourceExhausted : ErrorCode::DataLoss, "FFmpeg could not read SRT input (path=" + core::path_to_utf8(path) + ", native=" + std::to_string(opened) + ")", opened);
            demux.packet = av_packet_alloc();
            if (!demux.packet)
                throw std::bad_alloc();
            std::int64_t previous_position = -1, previous_pts = -1, group_floor = -1;
            std::size_t text_bytes = 0, facts = 0;
            int status = 0;
            while ((status = av_read_frame(demux.format, demux.packet)) >= 0) {
                const auto& packet = *demux.packet;
                if (storage->cues.size() >= limits.cues)
                    return srtError(ErrorCode::ResourceExhausted, "SRT cue budget exceeded (count=" + std::to_string(storage->cues.size() + 1) + ", limit=" + std::to_string(limits.cues) + ")");
                if (packet.stream_index < 0 || static_cast<unsigned>(packet.stream_index) >= demux.format->nb_streams ||
                    packet.pts == AV_NOPTS_VALUE || packet.pts < 0 || packet.duration < 0 || packet.size < 0 ||
                    packet.pos < 0 || static_cast<std::uint64_t>(packet.pos) >= storage->original.size() || (packet.size && !packet.data))
                    return srtError(ErrorCode::DataLoss, "SRT cue has missing/invalid timing or packet position (cue=" + std::to_string(storage->cues.size()) + ", pts=" + std::to_string(packet.pts) + ", duration=" + std::to_string(packet.duration) + ", position=" + std::to_string(packet.pos) + ", previous_position=" + std::to_string(previous_position) + ")");
                if (packet.pts != previous_pts) {
                    group_floor = previous_position;
                    previous_pts = packet.pts;
                }
                if (packet.pos < group_floor)
                    return srtError(ErrorCode::Unsupported, "SRT timeline reorders source cues; explicit segmentation is required (cue=" + std::to_string(storage->cues.size()) + ", pts=" + std::to_string(packet.pts) + ", position=" + std::to_string(packet.pos) + ", previous_group_position=" + std::to_string(group_floor) + ")");
                const auto time_base = demux.format->streams[packet.stream_index]->time_base;
                if (time_base.num != 1 || time_base.den != 1000 || packet.duration > std::numeric_limits<std::int64_t>::max() - packet.pts)
                    return srtError(ErrorCode::DataLoss, "SRT cue requires finite millisecond timing (time_base=" + std::to_string(time_base.num) + "/" + std::to_string(time_base.den) + ")");
                if (static_cast<std::size_t>(packet.size) > limits.cue_bytes)
                    return srtError(ErrorCode::ResourceExhausted, "SRT cue text exceeds its byte budget (cue=" + std::to_string(storage->cues.size()) + ", bytes=" + std::to_string(packet.size) + ", limit=" + std::to_string(limits.cue_bytes) + ")");
                if (static_cast<std::size_t>(packet.size) > limits.bytes - text_bytes)
                    return srtError(ErrorCode::ResourceExhausted, "SRT demuxed text exceeds the input byte budget (retained=" + std::to_string(text_bytes) + ", next=" + std::to_string(packet.size) + ", limit=" + std::to_string(limits.bytes) + ")");
                SrtCue cue;
                cue.start_milliseconds = packet.pts;
                cue.end_milliseconds = packet.pts + packet.duration;
                cue.source_offset = packet.pos;
                if (packet.size)
                    cue.text.assign(reinterpret_cast<const char*>(packet.data), packet.size);
                if (const auto interpreted = interpret(cue, limits.facts - facts); !interpreted)
                    return interpreted.error();
                facts += cue.facts.size();
                text_bytes += packet.size;
                previous_position = std::max(previous_position, packet.pos);
                storage->cues.push_back(std::move(cue));
                const auto index = storage->cues.size() - 1;
                if (storage->cues[index].start_milliseconds == storage->cues[index].end_milliseconds)
                    storage->points.push_back(index);
                constexpr auto missing = std::numeric_limits<std::size_t>::max();
                auto best = storage->candidates.empty() ? std::array{missing, missing} : storage->candidates.back();
                if (best[0] == missing || storage->cues[index].end_milliseconds > storage->cues[best[0]].end_milliseconds) {
                    best[1] = best[0];
                    best[0] = index;
                } else if (best[1] == missing || storage->cues[index].end_milliseconds > storage->cues[best[1]].end_milliseconds) {
                    best[1] = index;
                }
                storage->candidates.push_back(best);
                av_packet_unref(demux.packet);
            }
            if (status != AVERROR_EOF)
                return srtError(status == AVERROR(ENOMEM) ? ErrorCode::ResourceExhausted : ErrorCode::DataLoss, "Read SRT packet failed (path=" + core::path_to_utf8(path) + ", native=" + std::to_string(status) + ")", status);
            if (storage->cues.empty())
                return srtError(ErrorCode::DataLoss, "SRT input contains no timed cues (path=" + core::path_to_utf8(path) + ", bytes=" + std::to_string(storage->original.size()) + ")");
            if (storage->cues.size() != *expected_cues)
                return srtError(ErrorCode::Unsupported, "FFmpeg normalized or removed SRT occurrences; ambiguous/dropped input is unsupported (path=" + core::path_to_utf8(path) + ", expected=" + std::to_string(*expected_cues) + ", observed=" + std::to_string(storage->cues.size()) + ")");
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
