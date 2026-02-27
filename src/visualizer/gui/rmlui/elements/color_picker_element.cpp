/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/elements/color_picker_element.hpp"

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

    namespace {

        void hsvToRgb(float h, float s, float v, float& r, float& g, float& b) {
            if (s == 0.0f) {
                r = g = b = v;
                return;
            }
            h = std::fmod(h, 1.0f) * 6.0f;
            const int i = static_cast<int>(h);
            const float f = h - static_cast<float>(i);
            const float p = v * (1.0f - s);
            const float q = v * (1.0f - s * f);
            const float t = v * (1.0f - s * (1.0f - f));
            switch (i) {
            case 0:
                r = v;
                g = t;
                b = p;
                break;
            case 1:
                r = q;
                g = v;
                b = p;
                break;
            case 2:
                r = p;
                g = v;
                b = t;
                break;
            case 3:
                r = p;
                g = q;
                b = v;
                break;
            case 4:
                r = t;
                g = p;
                b = v;
                break;
            default:
                r = v;
                g = p;
                b = q;
                break;
            }
        }

        void rgbToHsv(float r, float g, float b, float& h, float& s, float& v) {
            const float mx = std::max({r, g, b});
            const float mn = std::min({r, g, b});
            const float d = mx - mn;
            v = mx;
            s = (mx > 0.0f) ? d / mx : 0.0f;
            if (d == 0.0f) {
                h = 0.0f;
            } else if (mx == r) {
                h = std::fmod((g - b) / d + 6.0f, 6.0f) / 6.0f;
            } else if (mx == g) {
                h = ((b - r) / d + 2.0f) / 6.0f;
            } else {
                h = ((r - g) / d + 4.0f) / 6.0f;
            }
        }

        Rml::ColourbPremultiplied hsvToColor(float h, float s, float v) {
            float r, g, b;
            hsvToRgb(h, s, v, r, g, b);
            return {static_cast<Rml::byte>(r * 255.f), static_cast<Rml::byte>(g * 255.f),
                    static_cast<Rml::byte>(b * 255.f), 255};
        }

        void addQuad(Rml::Mesh& mesh, float x0, float y0, float x1, float y1,
                     Rml::ColourbPremultiplied c0, Rml::ColourbPremultiplied c1,
                     Rml::ColourbPremultiplied c2, Rml::ColourbPremultiplied c3) {
            const int base = static_cast<int>(mesh.vertices.size());
            mesh.vertices.push_back({{x0, y0}, c0, {0, 0}});
            mesh.vertices.push_back({{x1, y0}, c1, {0, 0}});
            mesh.vertices.push_back({{x1, y1}, c2, {0, 0}});
            mesh.vertices.push_back({{x0, y1}, c3, {0, 0}});
            mesh.indices.push_back(base);
            mesh.indices.push_back(base + 1);
            mesh.indices.push_back(base + 2);
            mesh.indices.push_back(base);
            mesh.indices.push_back(base + 2);
            mesh.indices.push_back(base + 3);
        }

        void addQuadSolid(Rml::Mesh& mesh, float x0, float y0, float x1, float y1,
                          Rml::ColourbPremultiplied col) {
            addQuad(mesh, x0, y0, x1, y1, col, col, col, col);
        }

    } // namespace

    ColorPickerElement::ColorPickerElement(const Rml::String& tag) : Rml::Element(tag) {
        SetProperty(Rml::PropertyId::Drag, Rml::Property(Rml::Style::Drag::Drag));
        AddEventListener(Rml::EventId::Dragend, &drag_end_listener_);
    }

    void ColorPickerElement::DragEndListener::ProcessEvent(Rml::Event&) { owner.endDrag(); }

    void ColorPickerElement::endDrag() {
        if (drag_target_ >= 0) {
            drag_target_ = -1;
            cursor_dirty_ = true;
        }
    }

    bool ColorPickerElement::GetIntrinsicDimensions(Rml::Vector2f& dimensions, float& ratio) {
        dimensions = {200.f, 160.f};
        ratio = 0.f;
        return true;
    }

    void ColorPickerElement::OnResize() {
        sv_dirty_ = true;
        hue_dirty_ = true;
        cursor_dirty_ = true;
    }

    void ColorPickerElement::OnAttributeChange(const Rml::ElementAttributes& changed) {
        Element::OnAttributeChange(changed);
        readAttributes();
        sv_dirty_ = true;
        cursor_dirty_ = true;
    }

    void ColorPickerElement::readAttributes() {
        auto get = [this](const Rml::String& name, float fallback) {
            auto* attr = GetAttribute(name);
            return attr ? attr->Get(fallback) : fallback;
        };
        float r = get("red", 1.f);
        float g = get("green", 0.f);
        float b = get("blue", 0.f);
        rgbToHsv(r, g, b, h_, s_, v_);
    }

    void ColorPickerElement::dispatchChange() {
        float r, g, b;
        hsvToRgb(h_, s_, v_, r, g, b);
        Rml::Dictionary params;
        params["red"] = Rml::Variant(r);
        params["green"] = Rml::Variant(g);
        params["blue"] = Rml::Variant(b);
        DispatchEvent("change", params);
    }

    void ColorPickerElement::rebuildSVPlane() {
        auto* rm = GetRenderManager();
        if (!rm)
            return;

        const float w = GetBox().GetSize(Rml::BoxArea::Content).x;
        const float h = GetBox().GetSize(Rml::BoxArea::Content).y;
        if (w <= 0.f || h <= 0.f)
            return;

        const float sv_w = w - HUE_BAR_WIDTH - GAP;
        if (sv_w <= 0.f)
            return;

        const float cell_w = sv_w / GRID_RES;
        const float cell_h = h / GRID_RES;

        Rml::Mesh mesh;
        mesh.vertices.reserve(GRID_RES * GRID_RES * 4);
        mesh.indices.reserve(GRID_RES * GRID_RES * 6);

        for (int row = 0; row < GRID_RES; ++row) {
            for (int col = 0; col < GRID_RES; ++col) {
                const float s0 = static_cast<float>(col) / GRID_RES;
                const float s1 = static_cast<float>(col + 1) / GRID_RES;
                const float v0 = 1.f - static_cast<float>(row) / GRID_RES;
                const float v1 = 1.f - static_cast<float>(row + 1) / GRID_RES;

                const float x0 = col * cell_w;
                const float y0 = row * cell_h;
                const float x1 = x0 + cell_w + 0.5f;
                const float y1 = y0 + cell_h + 0.5f;

                auto tl = hsvToColor(h_, s0, v0);
                auto tr = hsvToColor(h_, s1, v0);
                auto br = hsvToColor(h_, s1, v1);
                auto bl = hsvToColor(h_, s0, v1);
                addQuad(mesh, x0, y0, x1, y1, tl, tr, br, bl);
            }
        }

        sv_geom_ = rm->MakeGeometry(std::move(mesh));
        sv_dirty_ = false;
    }

    void ColorPickerElement::rebuildHueBar() {
        auto* rm = GetRenderManager();
        if (!rm)
            return;

        const float w = GetBox().GetSize(Rml::BoxArea::Content).x;
        const float h = GetBox().GetSize(Rml::BoxArea::Content).y;
        if (w <= 0.f || h <= 0.f)
            return;

        const float bar_x = w - HUE_BAR_WIDTH;
        const float cell_h = h / GRID_RES;

        Rml::Mesh mesh;
        mesh.vertices.reserve(GRID_RES * 4);
        mesh.indices.reserve(GRID_RES * 6);

        for (int row = 0; row < GRID_RES; ++row) {
            const float hue0 = static_cast<float>(row) / GRID_RES;
            const float hue1 = static_cast<float>(row + 1) / GRID_RES;
            const float y0 = row * cell_h;
            const float y1 = y0 + cell_h + 0.5f;

            auto top = hsvToColor(hue0, 1.f, 1.f);
            auto bot = hsvToColor(hue1, 1.f, 1.f);
            addQuad(mesh, bar_x, y0, w, y1, top, top, bot, bot);
        }

        hue_geom_ = rm->MakeGeometry(std::move(mesh));
        hue_dirty_ = false;
    }

    void ColorPickerElement::rebuildCursors() {
        auto* rm = GetRenderManager();
        if (!rm)
            return;

        const float w = GetBox().GetSize(Rml::BoxArea::Content).x;
        const float h = GetBox().GetSize(Rml::BoxArea::Content).y;
        if (w <= 0.f || h <= 0.f)
            return;

        const float sv_w = w - HUE_BAR_WIDTH - GAP;
        const float bar_x = w - HUE_BAR_WIDTH;

        Rml::Mesh mesh;

        // SV crosshair
        const float cx = s_ * sv_w;
        const float cy = (1.f - v_) * h;
        const float r = CURSOR_SIZE;
        const float t = CURSOR_THICKNESS;
        const Rml::ColourbPremultiplied white(255, 255, 255, 255);
        const Rml::ColourbPremultiplied black(0, 0, 0, 200);

        // Outer shadow
        addQuadSolid(mesh, cx - r - 1.f, cy - t - 1.f, cx + r + 1.f, cy + t + 1.f, black);
        addQuadSolid(mesh, cx - t - 1.f, cy - r - 1.f, cx + t + 1.f, cy + r + 1.f, black);
        // Inner white cross
        addQuadSolid(mesh, cx - r, cy - t * 0.5f, cx + r, cy + t * 0.5f, white);
        addQuadSolid(mesh, cx - t * 0.5f, cy - r, cx + t * 0.5f, cy + r, white);

        // Hue cursor - horizontal bar across the hue strip
        const float hy = h_ * h;
        addQuadSolid(mesh, bar_x - 2.f, hy - 2.f, w + 2.f, hy + 2.f, black);
        addQuadSolid(mesh, bar_x - 1.f, hy - 1.f, w + 1.f, hy + 1.f, white);

        cursor_geom_ = rm->MakeGeometry(std::move(mesh));
        cursor_dirty_ = false;
    }

    void ColorPickerElement::OnRender() {
        auto offset = GetAbsoluteOffset(Rml::BoxArea::Content);

        if (sv_dirty_)
            rebuildSVPlane();
        if (hue_dirty_)
            rebuildHueBar();
        if (cursor_dirty_)
            rebuildCursors();

        sv_geom_.Render(offset);
        hue_geom_.Render(offset);
        cursor_geom_.Render(offset);
    }

    void ColorPickerElement::ProcessDefaultAction(Rml::Event& event) {
        Element::ProcessDefaultAction(event);

        const auto type = event.GetId();
        const float w = GetBox().GetSize(Rml::BoxArea::Content).x;
        const float h = GetBox().GetSize(Rml::BoxArea::Content).y;
        if (w <= 0.f || h <= 0.f)
            return;

        const float sv_w = w - HUE_BAR_WIDTH - GAP;
        const float bar_x = w - HUE_BAR_WIDTH;

        auto getLocalMouse = [&](const Rml::Event& e) -> Rml::Vector2f {
            const auto abs_offset = GetAbsoluteOffset(Rml::BoxArea::Content);
            return {e.GetParameter("mouse_x", 0.f) - abs_offset.x,
                    e.GetParameter("mouse_y", 0.f) - abs_offset.y};
        };

        if (type == Rml::EventId::Mousedown) {
            if (event.GetParameter("button", 0) != 0)
                return;
            const auto mouse = getLocalMouse(event);

            if (mouse.x >= bar_x && mouse.x <= w) {
                drag_target_ = 1;
                h_ = std::clamp(mouse.y / h, 0.f, 1.f);
                sv_dirty_ = true;
                cursor_dirty_ = true;
                dispatchChange();
            } else if (mouse.x >= 0.f && mouse.x <= sv_w) {
                drag_target_ = 0;
                s_ = std::clamp(mouse.x / sv_w, 0.f, 1.f);
                v_ = std::clamp(1.f - mouse.y / h, 0.f, 1.f);
                cursor_dirty_ = true;
                dispatchChange();
            }
        } else if (type == Rml::EventId::Drag && drag_target_ >= 0) {
            const auto mouse = getLocalMouse(event);
            if (drag_target_ == 1) {
                h_ = std::clamp(mouse.y / h, 0.f, 1.f);
                sv_dirty_ = true;
                cursor_dirty_ = true;
            } else {
                s_ = std::clamp(mouse.x / sv_w, 0.f, 1.f);
                v_ = std::clamp(1.f - mouse.y / h, 0.f, 1.f);
                cursor_dirty_ = true;
            }
            dispatchChange();
        }
    }

} // namespace lfs::vis::gui
