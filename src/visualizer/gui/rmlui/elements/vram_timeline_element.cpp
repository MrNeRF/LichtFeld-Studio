/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/elements/vram_timeline_element.hpp"

#include "diagnostics/vram_profiler.hpp"
#include "theme/theme.hpp"

#include <RmlUi/Core/RenderManager.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>

namespace lfs::vis::gui {
    namespace {
        [[nodiscard]] Rml::ColourbPremultiplied color(const ThemeColor& c, float alpha = 1.f) {
            const auto byte = [alpha](float v) {
                return static_cast<Rml::byte>(std::clamp(v * alpha * 255.f, 0.f, 255.f));
            };
            return {byte(c.x), byte(c.y), byte(c.z), byte(1.f)};
        }

        void quad(Rml::Mesh& mesh, float x0, float y0, float x1, float y1,
                  Rml::ColourbPremultiplied c) {
            const auto base = static_cast<int>(mesh.vertices.size());
            mesh.vertices.push_back({{x0, y0}, c, {0, 0}});
            mesh.vertices.push_back({{x1, y0}, c, {0, 0}});
            mesh.vertices.push_back({{x1, y1}, c, {0, 0}});
            mesh.vertices.push_back({{x0, y1}, c, {0, 0}});
            mesh.indices.insert(mesh.indices.end(), {base, base + 1, base + 2,
                                                     base, base + 2, base + 3});
        }
    } // namespace

    VramTimelineElement::VramTimelineElement(const Rml::String& tag) : Element(tag) {}

    void VramTimelineElement::setData(std::vector<lfs::diagnostics::VramTimelinePoint> points) {
        points_ = std::move(points);
        dirty_ = true;
    }

    void VramTimelineElement::setMarkers(std::vector<lfs::diagnostics::VramMarker> markers) {
        markers_ = std::move(markers);
        dirty_ = true;
    }

    void VramTimelineElement::setVisibleMask(std::uint16_t mask) {
        if (mask_ != mask) {
            mask_ = mask;
            dirty_ = true;
        }
    }

    void VramTimelineElement::setWindowSeconds(int seconds) {
        if (window_seconds_ != seconds) {
            window_seconds_ = seconds;
            dirty_ = true;
        }
    }

    void VramTimelineElement::setIterationAxis(bool iteration) {
        if (iteration_axis_ != iteration) {
            iteration_axis_ = iteration;
            dirty_ = true;
        }
    }

    void VramTimelineElement::setDeviceScale(bool device) {
        if (device_scale_ != device) {
            device_scale_ = device;
            dirty_ = true;
        }
    }

    void VramTimelineElement::setHighlightedOwner(int owner) {
        if (highlighted_owner_ != owner) {
            highlighted_owner_ = owner;
            dirty_ = true;
        }
    }

    void VramTimelineElement::setCompact(bool compact) {
        if (compact_ != compact) {
            compact_ = compact;
            dirty_ = true;
        }
    }

    bool VramTimelineElement::GetIntrinsicDimensions(Rml::Vector2f& dimensions, float& ratio) {
        dimensions = {1.f, 180.f};
        ratio = 0.f;
        return true;
    }

    void VramTimelineElement::OnResize() { dirty_ = true; }

    void VramTimelineElement::rebuild() {
        auto* rm = GetRenderManager();
        if (!rm)
            return;
        const auto size = GetBox().GetSize(Rml::BoxArea::Content);
        if (size.x < 2.f || size.y < 2.f)
            return;
        const auto& pal = theme().palette;
        const std::array<ThemeColor, lfs::diagnostics::kVramOwnerCount> colors{
            pal.primary, pal.secondary, pal.info, pal.success, pal.warning,
            pal.primary_dim, pal.error, pal.text_dim, pal.border, pal.warning};

        Rml::Mesh grid_mesh;
        Rml::Mesh area_mesh;
        Rml::Mesh line_mesh;
        if (!compact_) {
            for (int i = 0; i <= 4; ++i) {
                const auto y = std::floor(size.y * static_cast<float>(i) / 4.f) + .5f;
                quad(grid_mesh, 0.f, y, size.x, y + 1.f, color(pal.border, .48f));
            }
        }
        if (points_.empty()) {
            grid_ = rm->MakeGeometry(std::move(grid_mesh));
            areas_ = rm->MakeGeometry(std::move(area_mesh));
            lines_ = rm->MakeGeometry(std::move(line_mesh));
            dirty_ = false;
            return;
        }
        const auto end_ms = points_.back().epoch_ms;
        const auto start_ms = window_seconds_ > 0
                                  ? end_ms - static_cast<std::int64_t>(window_seconds_) * 1000
                                  : points_.front().epoch_ms;
        auto begin = std::lower_bound(points_.begin(), points_.end(), start_ms,
                                      [](const auto& p, std::int64_t ms) { return p.epoch_ms < ms; });
        if (begin == points_.end())
            begin = points_.end() - 1;
        const auto x0 = iteration_axis_ ? begin->iteration : begin->epoch_ms;
        const auto x1 = iteration_axis_ ? points_.back().iteration : points_.back().epoch_ms;
        const auto span = std::max<std::int64_t>(1, static_cast<std::int64_t>(x1) - x0);
        const auto max_bytes = lfs::diagnostics::vramTimelineAxisMax(
            {&*begin, static_cast<std::size_t>(points_.end() - begin)}, device_scale_);
        const auto y_for = [&](std::size_t bytes) {
            return std::clamp(size.y - size.y * static_cast<float>(bytes) /
                                           static_cast<float>(max_bytes),
                              0.f, size.y);
        };
        const auto x_for = [&](const auto& point) {
            const auto value = iteration_axis_ ? point.iteration : point.epoch_ms;
            return size.x * static_cast<float>(static_cast<std::int64_t>(value) - x0) /
                   static_cast<float>(span);
        };

        // Draw one coherent sample across each bucket, then its extrema at
        // their actual x positions. A narrow process spike is not skipped when
        // a long run contains more samples than horizontal pixels.
        const auto count = static_cast<std::size_t>(points_.end() - begin);
        const auto stride = std::max<std::size_t>(
            1, (count + static_cast<std::size_t>(std::max(1.f, size.x)) - 1) /
                   static_cast<std::size_t>(std::max(1.f, size.x)));
        const auto draw_sample = [&](const lfs::diagnostics::VramTimelinePoint& p,
                                     float x, float next) {
            std::size_t base = 0;
            for (std::size_t owner = 0; owner < lfs::diagnostics::kVramOwnerCount; ++owner) {
                if (!(mask_ & (1u << owner)) || p.bytes[owner] <= 0)
                    continue;
                const auto top = base + static_cast<std::size_t>(p.bytes[owner]);
                quad(area_mesh, x, y_for(top), next, y_for(base),
                     color(colors[owner], highlighted_owner_ >= 0 &&
                                                  highlighted_owner_ != static_cast<int>(owner)
                                              ? .23f
                                              : .88f));
                base = top;
            }
            if (!compact_) {
                const auto process_y = y_for(p.process_bytes);
                quad(line_mesh, x, process_y, next, process_y + 1.f, color(pal.text, .8f));
                if (p.device_bytes < max_bytes) {
                    const auto device_y = y_for(p.device_bytes);
                    for (float dash = x; dash < next; dash += 7.f)
                        quad(line_mesh, dash, device_y, std::min(dash + 3.f, next),
                             device_y + 1.f, color(pal.text_dim, .8f));
                }
            }
        };
        for (std::size_t i = 0; i < count; i += stride) {
            const auto& p = *(begin + static_cast<std::ptrdiff_t>(i));
            const float x = x_for(p);
            const auto last = std::min(count, i + stride);
            const float next = last < count
                                   ? x_for(*(begin + static_cast<std::ptrdiff_t>(last)))
                                   : size.x;
            draw_sample(p, x, std::max(x + 1.f, next));
        }
        if (!compact_) {
            float last_marker_x = -40.f;
            const auto marker_spacing = std::max(6.f, size.x / 10.f);
            for (const auto& marker : markers_) {
                if (marker.epoch_ms < start_ms || marker.epoch_ms > end_ms)
                    continue;
                const auto value = iteration_axis_ ? marker.iteration : marker.epoch_ms;
                const auto x = size.x * static_cast<float>(static_cast<std::int64_t>(value) - x0) /
                               static_cast<float>(span);
                if (x - last_marker_x < marker_spacing)
                    continue;
                last_marker_x = x;
                quad(line_mesh, x, 3.f, x + 1.f, size.y, color(pal.warning, .18f));
                quad(line_mesh, x - 2.f, 0.f, x + 3.f, 3.f, color(pal.warning, .72f));
            }
        }
        grid_ = rm->MakeGeometry(std::move(grid_mesh));
        areas_ = rm->MakeGeometry(std::move(area_mesh));
        lines_ = rm->MakeGeometry(std::move(line_mesh));
        dirty_ = false;
    }

    void VramTimelineElement::OnRender() {
        const bool measure = GetId() == "vram-hud-timeline" &&
                             lfs::diagnostics::VramProfiler::instance().enabled();
        const auto start = measure ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
        if (dirty_)
            rebuild();
        const auto offset = GetAbsoluteOffset(Rml::BoxArea::Content);
        grid_.Render(offset);
        areas_.Render(offset);
        lines_.Render(offset);
        if (measure) {
            const auto elapsed = std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - start)
                                     .count();
            render_total_ms_ += elapsed;
            render_max_ms_ = std::max(render_max_ms_, elapsed);
            if (++render_samples_ >= 60) {
                auto& profiler = lfs::diagnostics::VramProfiler::instance();
                profiler.setGauge("vram.hud.chart_render_mean_ms", render_total_ms_ / render_samples_);
                profiler.setGauge("vram.hud.chart_render_max_ms", render_max_ms_);
                render_total_ms_ = 0.0;
                render_max_ms_ = 0.0;
                render_samples_ = 0;
            }
        }
    }
} // namespace lfs::vis::gui
