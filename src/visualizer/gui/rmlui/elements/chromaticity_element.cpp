/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/elements/chromaticity_element.hpp"

#include <RmlUi/Core/ComputedValues.h>
#include <RmlUi/Core/Event.h>
#include <RmlUi/Core/ID.h>
#include <RmlUi/Core/Property.h>
#include <RmlUi/Core/RenderManager.h>
#include <RmlUi/Core/StyleTypes.h>
#include <algorithm>
#include <cassert>
#include <cmath>

namespace lfs::vis::gui {

    ChromaticityElement::ChromaticityElement(const Rml::String& tag) : Rml::Element(tag) {
        SetProperty(Rml::PropertyId::Drag, Rml::Property(Rml::Style::Drag::Drag));
        AddEventListener(Rml::EventId::Dragend, &drag_end_listener_);
    }

    void ChromaticityElement::DragEndListener::ProcessEvent(Rml::Event&) { owner.endDrag(); }

    void ChromaticityElement::endDrag() {
        if (dragging_ >= 0) {
            dragging_ = -1;
            points_dirty_ = true;
        }
    }

    bool ChromaticityElement::GetIntrinsicDimensions(Rml::Vector2f& dimensions, float& ratio) {
        dimensions = {140.f, 140.f};
        ratio = 1.0f;
        return true;
    }

    void ChromaticityElement::OnResize() {
        grid_dirty_ = true;
        points_dirty_ = true;
    }

    void ChromaticityElement::OnAttributeChange(const Rml::ElementAttributes& changed) {
        Element::OnAttributeChange(changed);
        readAttributes();
        points_dirty_ = true;
    }

    void ChromaticityElement::readAttributes() {
        auto get = [this](const Rml::String& name, float fallback) {
            auto* attr = GetAttribute(name);
            return attr ? attr->Get(fallback) : fallback;
        };
        red_x_ = get("red-x", 0.f);
        red_y_ = get("red-y", 0.f);
        green_x_ = get("green-x", 0.f);
        green_y_ = get("green-y", 0.f);
        blue_x_ = get("blue-x", 0.f);
        blue_y_ = get("blue-y", 0.f);
        wb_temp_ = get("wb-temp", 0.f);
        wb_tint_ = get("wb-tint", 0.f);
    }

    void ChromaticityElement::dispatchChange() {
        Rml::Dictionary params;
        params["red_x"] = Rml::Variant(red_x_);
        params["red_y"] = Rml::Variant(red_y_);
        params["green_x"] = Rml::Variant(green_x_);
        params["green_y"] = Rml::Variant(green_y_);
        params["blue_x"] = Rml::Variant(blue_x_);
        params["blue_y"] = Rml::Variant(blue_y_);
        params["wb_temp"] = Rml::Variant(wb_temp_);
        params["wb_tint"] = Rml::Variant(wb_tint_);
        DispatchEvent("change", params);
    }

    void ChromaticityElement::rebuildGrid() {
        auto* rm = GetRenderManager();
        if (!rm)
            return;

        const float w = GetBox().GetSize(Rml::BoxArea::Content).x;
        const float h = GetBox().GetSize(Rml::BoxArea::Content).y;
        if (w <= 0.f || h <= 0.f)
            return;

        const float cell_w = w / GRID_RES;
        const float cell_h = h / GRID_RES;

        Rml::Mesh mesh;
        mesh.vertices.reserve(GRID_RES * GRID_RES * 4);
        mesh.indices.reserve(GRID_RES * GRID_RES * 6);

        for (int iy = 0; iy < GRID_RES; ++iy) {
            for (int ix = 0; ix < GRID_RES; ++ix) {
                const float r_chrom = static_cast<float>(ix) / (GRID_RES - 1);
                const float g_chrom = 1.0f - static_cast<float>(iy) / (GRID_RES - 1);
                const float b_chrom = std::max(0.0f, 1.0f - r_chrom - g_chrom);

                const float intensity = 0.7f;
                float r = r_chrom * intensity + 0.15f;
                float g = g_chrom * intensity + 0.15f;
                float b = b_chrom * intensity + 0.15f;
                const float max_val = std::max({r, g, b});
                if (max_val > 1.0f) {
                    r /= max_val;
                    g /= max_val;
                    b /= max_val;
                }

                const auto rb = static_cast<Rml::byte>(r * 255.f);
                const auto gb = static_cast<Rml::byte>(g * 255.f);
                const auto bb = static_cast<Rml::byte>(b * 255.f);
                Rml::ColourbPremultiplied col(rb, gb, bb, 255);

                const float x0 = ix * cell_w;
                const float y0 = iy * cell_h;
                const float x1 = x0 + cell_w + 0.5f;
                const float y1 = y0 + cell_h + 0.5f;

                const int base = static_cast<int>(mesh.vertices.size());
                mesh.vertices.push_back({{x0, y0}, col, {0, 0}});
                mesh.vertices.push_back({{x1, y0}, col, {0, 0}});
                mesh.vertices.push_back({{x1, y1}, col, {0, 0}});
                mesh.vertices.push_back({{x0, y1}, col, {0, 0}});
                mesh.indices.push_back(base);
                mesh.indices.push_back(base + 1);
                mesh.indices.push_back(base + 2);
                mesh.indices.push_back(base);
                mesh.indices.push_back(base + 2);
                mesh.indices.push_back(base + 3);
            }
        }

        grid_geom_ = rm->MakeGeometry(std::move(mesh));

        // Border
        Rml::Mesh border_mesh;
        const Rml::ColourbPremultiplied border_col(88, 91, 112, 255);
        const float bw = 1.5f;

        auto addBorderQuad = [&](float x0, float y0, float x1, float y1) {
            const int base = static_cast<int>(border_mesh.vertices.size());
            border_mesh.vertices.push_back({{x0, y0}, border_col, {0, 0}});
            border_mesh.vertices.push_back({{x1, y0}, border_col, {0, 0}});
            border_mesh.vertices.push_back({{x1, y1}, border_col, {0, 0}});
            border_mesh.vertices.push_back({{x0, y1}, border_col, {0, 0}});
            border_mesh.indices.push_back(base);
            border_mesh.indices.push_back(base + 1);
            border_mesh.indices.push_back(base + 2);
            border_mesh.indices.push_back(base);
            border_mesh.indices.push_back(base + 2);
            border_mesh.indices.push_back(base + 3);
        };

        addBorderQuad(0.f, 0.f, w, bw);
        addBorderQuad(0.f, h - bw, w, h);
        addBorderQuad(0.f, bw, bw, h - bw);
        addBorderQuad(w - bw, bw, w, h - bw);

        border_geom_ = rm->MakeGeometry(std::move(border_mesh));
        grid_dirty_ = false;
    }

    void ChromaticityElement::rebuildPoints() {
        auto* rm = GetRenderManager();
        if (!rm)
            return;

        const float w = GetBox().GetSize(Rml::BoxArea::Content).x;
        const float h = GetBox().GetSize(Rml::BoxArea::Content).y;
        if (w <= 0.f || h <= 0.f)
            return;

        const float center_x = w * 0.5f;
        const float center_y = h * 0.5f;

        struct PointDef {
            float* x;
            float* y;
            Rml::ColourbPremultiplied fill;
            Rml::ColourbPremultiplied outline;
        };

        PointDef points[4] = {
            {&red_x_, &red_y_, {255, 80, 80, 255}, {180, 0, 0, 255}},
            {&green_x_, &green_y_, {80, 220, 80, 255}, {0, 150, 0, 255}},
            {&blue_x_, &blue_y_, {80, 120, 255, 255}, {0, 0, 180, 255}},
            {&wb_temp_, &wb_tint_, {200, 200, 200, 255}, {80, 80, 80, 255}},
        };

        Rml::Mesh mesh;
        const float radius = POINT_RADIUS;
        const float outline_w = 2.0f;

        for (int pi = 0; pi < 4; ++pi) {
            const float px = center_x + (*points[pi].x / RANGE) * center_x;
            const float py = center_y - (*points[pi].y / RANGE) * center_y;
            const float r = (pi == dragging_) ? radius * 1.3f : radius;

            // Filled circle (triangle fan)
            const int center_idx = static_cast<int>(mesh.vertices.size());
            mesh.vertices.push_back({{px, py}, points[pi].fill, {0, 0}});
            for (int s = 0; s <= CIRCLE_SEGMENTS; ++s) {
                const float angle = 2.0f * 3.14159265f * static_cast<float>(s) / CIRCLE_SEGMENTS;
                mesh.vertices.push_back(
                    {{px + std::cos(angle) * r, py + std::sin(angle) * r}, points[pi].fill, {0, 0}});
                if (s > 0) {
                    mesh.indices.push_back(center_idx);
                    mesh.indices.push_back(center_idx + s);
                    mesh.indices.push_back(center_idx + s + 1);
                }
            }

            // Outline ring
            const float ro = r + outline_w;
            for (int s = 0; s < CIRCLE_SEGMENTS; ++s) {
                const float a0 = 2.0f * 3.14159265f * static_cast<float>(s) / CIRCLE_SEGMENTS;
                const float a1 = 2.0f * 3.14159265f * static_cast<float>(s + 1) / CIRCLE_SEGMENTS;

                const float ix0 = px + std::cos(a0) * r;
                const float iy0 = py + std::sin(a0) * r;
                const float ox0 = px + std::cos(a0) * ro;
                const float oy0 = py + std::sin(a0) * ro;
                const float ix1 = px + std::cos(a1) * r;
                const float iy1 = py + std::sin(a1) * r;
                const float ox1 = px + std::cos(a1) * ro;
                const float oy1 = py + std::sin(a1) * ro;

                const int base = static_cast<int>(mesh.vertices.size());
                mesh.vertices.push_back({{ix0, iy0}, points[pi].outline, {0, 0}});
                mesh.vertices.push_back({{ox0, oy0}, points[pi].outline, {0, 0}});
                mesh.vertices.push_back({{ox1, oy1}, points[pi].outline, {0, 0}});
                mesh.vertices.push_back({{ix1, iy1}, points[pi].outline, {0, 0}});
                mesh.indices.push_back(base);
                mesh.indices.push_back(base + 1);
                mesh.indices.push_back(base + 2);
                mesh.indices.push_back(base);
                mesh.indices.push_back(base + 2);
                mesh.indices.push_back(base + 3);
            }
        }

        points_geom_ = rm->MakeGeometry(std::move(mesh));
        points_dirty_ = false;
    }

    void ChromaticityElement::OnRender() {
        auto offset = GetAbsoluteOffset(Rml::BoxArea::Content);

        if (grid_dirty_)
            rebuildGrid();
        if (points_dirty_)
            rebuildPoints();

        grid_geom_.Render(offset);
        border_geom_.Render(offset);
        points_geom_.Render(offset);
    }

    void ChromaticityElement::ProcessDefaultAction(Rml::Event& event) {
        Element::ProcessDefaultAction(event);

        const auto type = event.GetId();
        const float w = GetBox().GetSize(Rml::BoxArea::Content).x;
        const float h = GetBox().GetSize(Rml::BoxArea::Content).y;
        if (w <= 0.f || h <= 0.f)
            return;

        const float center_x = w * 0.5f;
        const float center_y = h * 0.5f;

        float* xs[4] = {&red_x_, &green_x_, &blue_x_, &wb_temp_};
        float* ys[4] = {&red_y_, &green_y_, &blue_y_, &wb_tint_};

        auto getLocalMouse = [&](const Rml::Event& e) -> Rml::Vector2f {
            const auto abs_offset = GetAbsoluteOffset(Rml::BoxArea::Content);
            return {e.GetParameter("mouse_x", 0.f) - abs_offset.x, e.GetParameter("mouse_y", 0.f) - abs_offset.y};
        };

        if (type == Rml::EventId::Mousedown) {
            if (event.GetParameter("button", 0) != 0)
                return;
            const auto mouse = getLocalMouse(event);
            float best_dist = HIT_RADIUS;
            dragging_ = -1;

            for (int i = 0; i < 4; ++i) {
                const float px = center_x + (*xs[i] / RANGE) * center_x;
                const float py = center_y - (*ys[i] / RANGE) * center_y;
                const float dx = mouse.x - px;
                const float dy = mouse.y - py;
                const float dist = std::sqrt(dx * dx + dy * dy);
                if (dist < best_dist) {
                    best_dist = dist;
                    dragging_ = i;
                }
            }
            if (dragging_ >= 0)
                points_dirty_ = true;
        } else if (type == Rml::EventId::Drag && dragging_ >= 0) {
            const auto mouse = getLocalMouse(event);
            *xs[dragging_] = ((mouse.x / w) - 0.5f) * 2.0f * RANGE;
            *ys[dragging_] = -((mouse.y / h) - 0.5f) * 2.0f * RANGE;
            *xs[dragging_] = std::clamp(*xs[dragging_], -RANGE, RANGE);
            *ys[dragging_] = std::clamp(*ys[dragging_], -RANGE, RANGE);
            points_dirty_ = true;
            dispatchChange();
        } else if (type == Rml::EventId::Dblclick) {
            red_x_ = red_y_ = green_x_ = green_y_ = blue_x_ = blue_y_ = wb_temp_ = wb_tint_ = 0.f;
            points_dirty_ = true;
            dispatchChange();
        }
    }

} // namespace lfs::vis::gui
