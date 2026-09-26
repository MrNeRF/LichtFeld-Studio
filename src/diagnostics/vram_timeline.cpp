/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "diagnostics/vram_timeline.hpp"

#include <algorithm>
#include <format>
#include <limits>
#include <unordered_map>

namespace lfs::diagnostics {
    namespace {
        constexpr std::size_t kRawLimit = 1800;
        constexpr std::size_t kBinLimit = 4320;
        constexpr std::size_t kMarkerLimit = 2048;
        constexpr std::size_t kLargeAllocation = 64ull * 1024ull * 1024ull;

        [[nodiscard]] std::string row_key(const VramOwnerRow& row) {
            return row.scope + "\x1f" + row.label;
        }

        [[nodiscard]] std::string escape_csv(std::string_view text) {
            std::string out = "\"";
            for (const char c : text) {
                if (c == '"')
                    out += "\"\"";
                else if (c == '\r' || c == '\n')
                    out += ' ';
                else
                    out += c;
            }
            out += '"';
            return out;
        }
    } // namespace

    bool isVramProcessSpike(std::size_t previous, std::size_t current) {
        if (current <= previous)
            return false;
        return current - previous >= std::max(kLargeAllocation, previous / 50);
    }

    std::size_t vramTimelineAxisMax(std::span<const VramTimelinePoint> points, bool device_scale) {
        std::size_t highest = 0;
        for (const auto& point : points)
            highest = std::max(highest, device_scale ? point.device_bytes : point.process_bytes);
        constexpr auto MiB = 1024ull * 1024ull;
        const double desired_step = std::max(1.0, static_cast<double>(highest) * 1.12 / 4.0);
        std::size_t magnitude = MiB;
        while (magnitude < desired_step / 2.0 && magnitude <= std::numeric_limits<std::size_t>::max() / 2)
            magnitude *= 2;
        for (const double multiple : {1.0, 1.25, 1.5, 2.0}) {
            const auto step = static_cast<std::size_t>(magnitude * multiple);
            if (step >= desired_step && step <= std::numeric_limits<std::size_t>::max() / 4)
                return 4 * step;
        }
        return std::numeric_limits<std::size_t>::max();
    }

    void VramTimeline::clear() {
        recent_.clear();
        older_.clear();
        markers_.clear();
        previous_rows_.clear();
        previous_raw_rows_.clear();
        last_profiler_marker_id_ = 0;
    }

    void VramTimeline::push(const VramProfilerSnapshot& snapshot, std::int64_t epoch_ms) {
        if (!snapshot.enabled)
            return;
        const auto breakdown = buildVramOwnerBreakdown(snapshot,
                                                       snapshot.process.shared_scratch_bytes > 0);
        VramTimelinePoint p;
        p.epoch_ms = epoch_ms;
        p.iteration = snapshot.iteration;
        p.splats = snapshot.training_state.live_splats;
        p.bytes = displayVramOwnerBytes(breakdown);
        p.process_bytes = snapshot.process.process_used;
        p.device_bytes = snapshot.process.total_used;
        p.capacity_bytes = snapshot.process.total;

        for (const auto& marker : snapshot.markers) {
            if (marker.id > last_profiler_marker_id_) {
                markers_.push_back(marker);
                last_profiler_marker_id_ = marker.id;
            }
        }

        if (!recent_.empty() && isVramProcessSpike(recent_.back().process_bytes, p.process_bytes)) {
            std::unordered_map<std::string, std::size_t> previous;
            for (const auto& row : previous_rows_)
                previous[row_key(row)] = row.bytes;
            std::vector<std::pair<std::size_t, std::string>> growth;
            for (const auto& row : breakdown.rows) {
                const auto before = previous[row_key(row)];
                if (row.bytes > before)
                    growth.emplace_back(row.bytes - before, row.scope + "/" + row.label);
            }
            std::sort(growth.begin(), growth.end(), std::greater<>());
            std::string label;
            for (std::size_t i = 0; i < std::min<std::size_t>(3, growth.size()); ++i) {
                if (i)
                    label += ", ";
                label += growth[i].second;
            }
            markers_.push_back({0, epoch_ms, p.iteration, "spike", label,
                                p.process_bytes - recent_.back().process_bytes, 0.0});
        }

        if (!recent_.empty()) {
            std::unordered_map<std::string, std::size_t> next_raw;
            for (const auto& row : snapshot.rows) {
                if (row.live_bytes < kLargeAllocation ||
                    (row.kind != VramRowKind::Hooked && row.kind != VramRowKind::Sampled)) {
                    next_raw[row.scope + "\x1f" + row.label] = row.live_bytes;
                    continue;
                }
                const auto key = row.scope + "\x1f" + row.label;
                const auto before = previous_raw_rows_[key];
                if (row.live_bytes > before && row.live_bytes - before >= kLargeAllocation)
                    markers_.push_back({0, epoch_ms, p.iteration, "allocation",
                                        row.scope + "/" + row.label, row.live_bytes - before, 0.0});
                next_raw[key] = row.live_bytes;
            }
            previous_raw_rows_ = std::move(next_raw);
        } else {
            for (const auto& row : snapshot.rows)
                previous_raw_rows_[row.scope + "\x1f" + row.label] = row.live_bytes;
        }
        previous_rows_ = breakdown.rows;
        recent_.push_back(std::move(p));
        if (recent_.size() > kRawLimit) {
            auto old = std::move(recent_.front());
            recent_.pop_front();
            const auto slot = old.epoch_ms / 10000;
            old.resolution_seconds = 10;
            if (older_.empty() || older_.back().slot != slot) {
                older_.push_back({old, old, old, old, slot});
            } else {
                auto& bin = older_.back();
                bin.last = old;
                if (old.process_bytes < bin.low.process_bytes)
                    bin.low = old;
                if (old.process_bytes > bin.high.process_bytes)
                    bin.high = old;
            }
            if (older_.size() > kBinLimit)
                older_.pop_front();
        }
        while (markers_.size() > kMarkerLimit)
            markers_.pop_front();
    }

    std::vector<VramTimelinePoint> VramTimeline::points() const {
        std::vector<VramTimelinePoint> out;
        out.reserve(older_.size() * 4 + recent_.size());
        for (const auto& bin : older_) {
            auto append = [&](VramTimelinePoint point, std::string role) {
                point.role = std::move(role);
                out.push_back(std::move(point));
            };
            append(bin.first, "first");
            if (bin.low.epoch_ms != bin.first.epoch_ms && bin.low.epoch_ms != bin.last.epoch_ms)
                append(bin.low, "min");
            if (bin.high.epoch_ms != bin.first.epoch_ms && bin.high.epoch_ms != bin.last.epoch_ms &&
                bin.high.epoch_ms != bin.low.epoch_ms)
                append(bin.high, "max");
            if (bin.last.epoch_ms != bin.first.epoch_ms)
                append(bin.last, "last");
        }
        out.insert(out.end(), recent_.begin(), recent_.end());
        std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
            return a.epoch_ms < b.epoch_ms;
        });
        return out;
    }

    std::string VramTimeline::csv() const {
        std::string out = "epoch_ms,iteration,splats,resolution_seconds,sample_role";
        for (std::size_t i = 0; i < kVramOwnerCount; ++i)
            out += std::format(",{}", escape_csv(vramOwnerName(static_cast<VramOwner>(i))));
        out += ",process_bytes,device_bytes,capacity_bytes\n";
        for (const auto& p : points()) {
            out += std::format("{},{},{},{},{}", p.epoch_ms, p.iteration, p.splats,
                               p.resolution_seconds, escape_csv(p.role));
            for (const auto bytes : p.bytes)
                out += std::format(",{}", bytes);
            out += std::format(",{},{},{}\n", p.process_bytes, p.device_bytes, p.capacity_bytes);
        }
        out += "\nevent_epoch_ms,iteration,kind,text,bytes,pause_ms\n";
        for (const auto& m : markers_)
            out += std::format("{},{},{},{},{},{:.3f}\n", m.epoch_ms, m.iteration,
                               escape_csv(m.kind), escape_csv(m.text), m.bytes, m.pause_ms);
        return out;
    }
} // namespace lfs::diagnostics
