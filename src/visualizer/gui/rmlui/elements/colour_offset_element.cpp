/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "colour_offset_element.hpp"

#include "gui/rmlui/rml_theme.hpp"
#include "theme/theme.hpp"

#include <RmlUi/Core/Context.h>
#include <RmlUi/Core/Event.h>
#include <RmlUi/Core/Input.h>
#include <RmlUi/Core/RenderManager.h>
#include <algorithm>
#include <cmath>
#include <numbers>

namespace lfs::vis::gui {
    namespace {
        constexpr float kSqrt3 = 1.73205080757f;
        Rml::ColourbPremultiplied rgb(const float r, const float g, const float b) {
            return {static_cast<Rml::byte>(std::clamp(r, 0.0f, 1.0f) * 255),
                    static_cast<Rml::byte>(std::clamp(g, 0.0f, 1.0f) * 255),
                    static_cast<Rml::byte>(std::clamp(b, 0.0f, 1.0f) * 255), 255};
        }
        void disc(Rml::Mesh& mesh, const Rml::Vector2f centre, const float radius,
                  const Rml::ColourbPremultiplied colour) {
            const int base = static_cast<int>(mesh.vertices.size());
            mesh.vertices.push_back({centre, colour, {}});
            for (int i = 0; i <= 48; ++i) {
                const float angle = i * 2.0f * std::numbers::pi_v<float> / 48;
                mesh.vertices.push_back({centre + Rml::Vector2f{std::cos(angle), std::sin(angle)} * radius, colour, {}});
                if (i > 0)
                    mesh.indices.insert(mesh.indices.end(), {base, base + i, base + i + 1});
            }
        }
    } // namespace

    ColourOffsetElement::ColourOffsetElement(const Rml::String& tag) : Element(tag) {
        SetProperty("drag", "drag");
        SetProperty("focus", "auto");
        AddEventListener("dragend", this);
        AddEventListener("drag", this);
        AddEventListener("mousemove", this);
    }

    void ColourOffsetElement::OnResize() {
        Element::OnResize();
        dirty_ = true;
    }

    void ColourOffsetElement::OnAttributeChange(const Rml::ElementAttributes& changed) {
        Element::OnAttributeChange(changed);
        value_ = {GetAttribute<float>("red", 0), GetAttribute<float>("green", 0), GetAttribute<float>("blue", 0)};
        dirty_ = true;
    }

    void ColourOffsetElement::change(Rml::Vector2f offset, const bool reset) {
        const float length = std::hypot(offset.x, offset.y);
        if (length > 1)
            offset = offset / length;
        const float mean = reset ? 0 : (value_[0] + value_[1] + value_[2]) / 3;
        value_ = reset ? std::array<float, 3>{} : std::array<float, 3>{std::clamp(mean + offset.x, -1.0f, 1.0f), std::clamp(mean - offset.x * 0.5f - offset.y * kSqrt3 * 0.5f, -1.0f, 1.0f), std::clamp(mean - offset.x * 0.5f + offset.y * kSqrt3 * 0.5f, -1.0f, 1.0f)};
        dirty_ = true;
        Rml::Dictionary parameters;
        parameters["red"] = value_[0];
        parameters["green"] = value_[1];
        parameters["blue"] = value_[2];
        DispatchEvent("change", parameters);
    }

    void ColourOffsetElement::ProcessEvent(Rml::Event& event) {
        if (event.GetType() == "dragend")
            dragging_ = false;
        else if (dragging_)
            updatePointer(event);
    }

    void ColourOffsetElement::updatePointer(const Rml::Event& event) {
        const auto size = GetBox().GetSize(Rml::BoxArea::Content);
        const float radius = std::min(size.x, size.y) * 0.5f - 4 * GetContext()->GetDensityIndependentPixelRatio();
        if (radius <= 0)
            return;
        const auto origin = GetAbsoluteOffset(Rml::BoxArea::Content) + size * 0.5f;
        change({(event.GetParameter("mouse_x", origin.x) - origin.x) / radius,
                (event.GetParameter("mouse_y", origin.y) - origin.y) / radius},
               false);
    }

    void ColourOffsetElement::ProcessDefaultAction(Rml::Event& event) {
        Element::ProcessDefaultAction(event);
        const auto& type = event.GetType();
        if (type == "dblclick") {
            change({}, true);
        } else if (type == "mouseup" || (type == "keydown" && event.GetParameter("key_identifier", 0) == Rml::Input::KI_ESCAPE)) {
            dragging_ = false;
        } else if (type == "mousedown" && event.GetParameter("button", 0) == 0) {
            dragging_ = true;
            Focus();
            updatePointer(event);
        }
    }

    void ColourOffsetElement::rebuild() {
        const auto size = GetBox().GetSize(Rml::BoxArea::Content);
        const float dp = GetContext()->GetDensityIndependentPixelRatio();
        const auto centre = size * 0.5f;
        const float radius = std::max(0.0f, std::min(size.x, size.y) * 0.5f - 4 * dp);
        const auto& palette = theme().palette;
        Rml::Mesh mesh;
        disc(mesh, centre, radius + dp, rgb(palette.border.x, palette.border.y, palette.border.z));
        const int base = static_cast<int>(mesh.vertices.size());
        mesh.vertices.push_back({centre, rgb(0.5f, 0.5f, 0.5f), {}});
        for (int i = 0; i <= 96; ++i) {
            const float angle = i * 2.0f * std::numbers::pi_v<float> / 96;
            const Rml::Vector2f direction{std::cos(angle), std::sin(angle)};
            const auto colour = rgb(0.5f + 0.5f * direction.x,
                                    0.5f - 0.25f * direction.x - kSqrt3 * 0.25f * direction.y,
                                    0.5f - 0.25f * direction.x + kSqrt3 * 0.25f * direction.y);
            mesh.vertices.push_back({centre + direction * radius, colour, {}});
            if (i > 0)
                mesh.indices.insert(mesh.indices.end(), {base, base + i, base + i + 1});
        }
        disc(mesh, centre, dp, rgb(palette.text_dim.x, palette.text_dim.y, palette.text_dim.z));
        const float mean = (value_[0] + value_[1] + value_[2]) / 3;
        Rml::Vector2f dot{value_[0] - mean, (value_[2] - value_[1]) / kSqrt3};
        const float length = std::hypot(dot.x, dot.y);
        if (length > 1)
            dot = dot / length;
        const auto position = centre + dot * radius;
        disc(mesh, position, 4.0f * dp, rgb(0.08f, 0.08f, 0.08f));
        disc(mesh, position, 2.5f * dp, rgb(0.95f, 0.95f, 0.95f));
        geometry_ = GetRenderManager()->MakeGeometry(std::move(mesh));
        dirty_ = false;
        theme_signature_ = rml_theme::currentThemeSignature();
    }

    void ColourOffsetElement::OnRender() {
        if (dirty_ || theme_signature_ != rml_theme::currentThemeSignature())
            rebuild();
        geometry_.Render(GetAbsoluteOffset(Rml::BoxArea::Content));
    }
} // namespace lfs::vis::gui
