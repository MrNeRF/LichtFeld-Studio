/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "node_curve_element.hpp"
#include "core/nodes/curve.hpp"
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
        using Json = nlohmann::json;
        Json editablePoints(const Json& input, bool ramp) {
            Json result = Json::array();
            if (input.is_array())
                for (const auto& point : input) {
                    const size_t columns = ramp ? 5 : 2;
                    if (!point.is_array() || point.size() != columns)
                        continue;
                    if (!std::ranges::all_of(point, [](const auto& value) { return value.is_number() && std::isfinite(value.template get<float>()); }))
                        continue;
                    Json clean = Json::array();
                    for (const auto& value : point)
                        clean.push_back(std::clamp(value.get<float>(), 0.0f, 1.0f));
                    result.push_back(std::move(clean));
                }
            std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) { return a[0].template get<float>() < b[0].template get<float>(); });
            auto end = std::unique(result.begin(), result.end(), [](const auto& a, const auto& b) { return std::abs(a[0].template get<float>() - b[0].template get<float>()) < 0.0001f; });
            result.erase(end, result.end());
            if (result.size() < 2)
                return ramp ? Json::array({{0, 0, 0, 0, 1}, {1, 1, 1, 1, 1}}) : Json::array({{0, 0}, {1, 1}});
            return result;
        }
        Rml::ColourbPremultiplied colour(float r, float g, float b) {
            return {Rml::byte(std::clamp(r, 0.0f, 1.0f) * 255), Rml::byte(std::clamp(g, 0.0f, 1.0f) * 255), Rml::byte(std::clamp(b, 0.0f, 1.0f) * 255), 255};
        }
        void quad(Rml::Mesh& mesh, float x, float y, float w, float h, Rml::ColourbPremultiplied c) {
            const int i = int(mesh.vertices.size());
            mesh.vertices.insert(mesh.vertices.end(), {{{x, y}, c, {}}, {{x + w, y}, c, {}}, {{x + w, y + h}, c, {}}, {{x, y + h}, c, {}}});
            mesh.indices.insert(mesh.indices.end(), {i, i + 1, i + 2, i, i + 2, i + 3});
        }
        void line(Rml::Mesh& mesh, Rml::Vector2f a, Rml::Vector2f b, float width, Rml::ColourbPremultiplied c) {
            const auto delta = b - a;
            const float length = std::hypot(delta.x, delta.y);
            if (length < 1e-6f)
                return;
            const Rml::Vector2f normal{-delta.y * width / (2 * length), delta.x * width / (2 * length)};
            const int i = int(mesh.vertices.size());
            mesh.vertices.insert(mesh.vertices.end(), {{a + normal, c, {}}, {b + normal, c, {}}, {b - normal, c, {}}, {a - normal, c, {}}});
            mesh.indices.insert(mesh.indices.end(), {i, i + 1, i + 2, i, i + 2, i + 3});
        }
        void disc(Rml::Mesh& mesh, Rml::Vector2f centre, float radius, Rml::ColourbPremultiplied c) {
            const int base = int(mesh.vertices.size());
            mesh.vertices.push_back({centre, c, {}});
            for (int i = 0; i <= 20; ++i) {
                const float a = i * 2 * std::numbers::pi_v<float> / 20;
                mesh.vertices.push_back({centre + Rml::Vector2f{std::cos(a), std::sin(a)} * radius, c, {}});
                if (i)
                    mesh.indices.insert(mesh.indices.end(), {base, base + i, base + i + 1});
            }
        }
        std::array<float, 4> rampSample(const Json& stops, float x, const std::string& interpolation) {
            std::array<float, 4> result{};
            if (stops.empty())
                return result;
            size_t right = 0;
            while (right < stops.size() && stops[right][0].get<float>() <= x)
                ++right;
            const size_t a = right ? right - 1 : 0, b = std::min(right, stops.size() - 1);
            const float width = stops[b][0].get<float>() - stops[a][0].get<float>();
            float t = width > 0 ? std::clamp((x - stops[a][0].get<float>()) / width, 0.0f, 1.0f) : 0;
            if (interpolation == "constant")
                t = 0;
            else if (interpolation == "ease")
                t = t * t * (3 - 2 * t);
            for (int c = 0; c < 4; ++c)
                result[c] = std::lerp(stops[a][c + 1].get<float>(), stops[b][c + 1].get<float>(), t);
            return result;
        }
    } // namespace
    NodeCurveElement::NodeCurveElement(const Rml::String& tag) : Element(tag) {
        SetProperty("drag", "drag");
        SetProperty("focus", "auto");
        AddEventListener("drag", this);
        AddEventListener("dragend", this);
        AddEventListener("mousemove", this);
    }
    void NodeCurveElement::OnResize() {
        Element::OnResize();
        dirty_ = true;
    }
    void NodeCurveElement::OnAttributeChange(const Rml::ElementAttributes& changed) {
        Element::OnAttributeChange(changed);
        if (changed.find("data-values") != changed.end()) {
            auto parsed = Json::parse(GetAttribute<Rml::String>("data-values", "{}"), nullptr, false);
            if (parsed.is_object())
                values_ = std::move(parsed);
        }
        const auto mode = GetAttribute<Rml::String>("mode", "curve");
        if (mode == "ramp") {
            property_ = "stops";
            if (selected_ < 0)
                selected_ = 0;
        } else if (mode == "rgb" && property_ == "points")
            property_ = "combined";
        if (changed.find("data-selected") != changed.end())
            selected_ = GetAttribute<int>("data-selected", mode == "ramp" ? 0 : -1);
        if (mode == "rgb" && changed.find("data-channel") != changed.end()) {
            const auto channel = GetAttribute<Rml::String>("data-channel", "combined");
            if (channel == "combined" || channel == "r" || channel == "g" || channel == "b")
                property_ = channel;
        }
        if (!values_.is_object())
            values_ = Json::object();
        for (const char* name : {"points", "combined", "r", "g", "b", "stops"})
            values_[name] = editablePoints(values_.value(name, Json::array()), std::string_view(name) == "stops");
        dirty_ = true;
    }
    Rml::Vector2f NodeCurveElement::coordinates(const Rml::Event& event) {
        const auto origin = GetAbsoluteOffset(Rml::BoxArea::Content), size = GetBox().GetSize(Rml::BoxArea::Content);
        return {(event.GetParameter("mouse_x", origin.x) - origin.x) * 200 / std::max(1.0f, size.x),
                (event.GetParameter("mouse_y", origin.y) - origin.y) * 120 / std::max(1.0f, size.y)};
    }
    void NodeCurveElement::emit(const char* event) {
        Rml::Dictionary parameters;
        parameters["property"] = property_;
        parameters["value"] = values_[property_].dump();
        parameters["selected"] = selected_;
        DispatchEvent(event, parameters);
    }
    void NodeCurveElement::pointer(const Rml::Event& event) {
        auto& points = values_[property_];
        if (selected_ < 0 || size_t(selected_) >= points.size())
            return;
        const auto p = coordinates(event);
        const bool ramp = property_ == "stops";
        if (channel_ >= 0)
            points[selected_][channel_ + 1] = std::clamp((p.x - 20) / 170, 0.0f, 1.0f);
        else {
            const float left = selected_ ? points[selected_ - 1][0].get<float>() + 0.0001f : 0;
            const float right = size_t(selected_ + 1) < points.size() ? points[selected_ + 1][0].get<float>() - 0.0001f : 1;
            points[selected_][0] = std::clamp((p.x - 8) / 184, std::min(left, right), right);
            if (!ramp)
                points[selected_][1] = std::clamp((112 - p.y) / (property_ == "points" ? 104 : 88), 0.0f, 1.0f);
        }
        dirty_ = true;
        emit("curvechange");
    }
    void NodeCurveElement::removePoint() {
        auto& points = values_[property_];
        if (selected_ < 0 || points.size() <= 2)
            return;
        emit("curvebegin");
        points.erase(points.begin() + selected_);
        selected_ = -1;
        dirty_ = true;
        emit("curvechange");
        emit("curveend");
    }
    void NodeCurveElement::ProcessEvent(Rml::Event& event) {
        if (dragging_ && (event.GetType() == "mousemove" || event.GetType() == "drag"))
            pointer(event);
        if (dragging_ && event.GetType() == "dragend") {
            dragging_ = false;
            emit("curveend");
        }
    }
    void NodeCurveElement::ProcessDefaultAction(Rml::Event& event) {
        Element::ProcessDefaultAction(event);
        if (!GetAttribute<int>("editing", 1))
            return;
        const auto type = event.GetType();
        if (type == "dblclick") {
            if (coordinates(event).y >= 21 || property_ == "points")
                removePoint();
            event.StopPropagation();
            return;
        }
        if (type == "keydown") {
            const int key = event.GetParameter("key_identifier", 0);
            if (key == Rml::Input::KI_DELETE || key == Rml::Input::KI_BACK) {
                removePoint();
                event.StopPropagation();
            }
            if (key == Rml::Input::KI_ESCAPE && dragging_) {
                dragging_ = false;
                emit("curvecancel");
                event.StopPropagation();
            }
        }
        if (type == "mouseup" && dragging_) {
            pointer(event);
            dragging_ = false;
            emit("curveend");
            event.StopPropagation();
        }
        if (type != "mousedown" || event.GetParameter("button", 0) != 0)
            return;
        Focus();
        const auto p = coordinates(event);
        const auto mode = GetAttribute<Rml::String>("mode", "curve");
        if (mode == "rgb" && p.y < 21) {
            const char* names[] = {"combined", "r", "g", "b"};
            property_ = names[std::clamp(int(p.x / 50), 0, 3)];
            selected_ = -1;
            dirty_ = true;
            emit("curvefocus");
            event.StopPropagation();
            return;
        }
        auto& points = values_[property_];
        const bool ramp = mode == "ramp";
        if (!points.is_array() || points.size() < 2)
            points = ramp ? Json::array({{0, 0, 0, 0, 1}, {1, 1, 1, 1, 1}}) : Json::array({{0, 0}, {1, 1}});
        channel_ = ramp && p.y >= 77 ? std::clamp(int((p.y - 77) / 10), 0, 3) : -1;
        if (channel_ < 0) {
            selected_ = -1;
            float best = 10;
            for (size_t i = 0; i < points.size(); ++i) {
                const float x = 8 + 184 * points[i][0].get<float>(), y = ramp ? 65 : 112 - (property_ == "points" ? 104 : 88) * points[i][1].get<float>();
                const float d = ramp ? std::abs(p.x - x) : std::hypot(p.x - x, p.y - y);
                if (d < best) {
                    best = d;
                    selected_ = int(i);
                }
            }
        } else if (selected_ < 0)
            selected_ = 0;
        emit("curvefocus");
        emit("curvebegin");
        if (selected_ < 0) {
            const float x = std::clamp((p.x - 8) / 184, 0.0f, 1.0f);
            Json point;
            if (ramp) {
                const auto c = rampSample(points, x, values_.value("interpolation", std::string("linear")));
                point = {x, c[0], c[1], c[2], c[3]};
            } else
                point = {x, std::clamp((112 - p.y) / (property_ == "points" ? 104 : 88), 0.0f, 1.0f)};
            auto it = std::upper_bound(points.begin(), points.end(), x, [](float a, const Json& b) { return a < b[0].get<float>(); });
            selected_ = int(it - points.begin());
            points.insert(it, point);
        }
        dragging_ = true;
        pointer(event);
        event.StopPropagation();
    }
    void NodeCurveElement::rebuild() {
        const auto size = GetBox().GetSize(Rml::BoxArea::Content);
        const float sx = size.x / 200, sy = size.y / 120, dp = std::min(sx, sy);
        const bool editing = GetAttribute<int>("editing", 1), ramp = property_ == "stops", rgb = GetAttribute<Rml::String>("mode", "") == "rgb";
        const auto& palette = theme().palette;
        const auto dim = colour(palette.text_dim.x, palette.text_dim.y, palette.text_dim.z);
        const auto foreground = colour(palette.text.x, palette.text.y, palette.text.z);
        Rml::Mesh mesh;
        auto points = values_.value(property_, ramp ? Json::array({{0, 0, 0, 0, 1}, {1, 1, 1, 1, 1}}) : Json::array({{0, 0}, {1, 1}}));
        if (!points.is_array() || points.size() < 2)
            points = ramp ? Json::array({{0, 0, 0, 0, 1}, {1, 1, 1, 1, 1}}) : Json::array({{0, 0}, {1, 1}});
        if (ramp) {
            for (int x = 0; x < 184; ++x) {
                const auto c = rampSample(points, float(x) / 183, values_.value("interpolation", std::string("linear")));
                for (int y = 0; y < 48; y += 8) {
                    const float background = ((x / 8 + y / 8) % 2) ? 0.18f : 0.25f;
                    quad(mesh, (8 + x) * sx, (8 + y) * sy, sx, 8 * sy, colour(std::lerp(background, c[0], c[3]), std::lerp(background, c[1], c[3]), std::lerp(background, c[2], c[3])));
                }
            }
            if (editing && selected_ >= 0 && size_t(selected_) < points.size()) {
                for (int c = 0; c < 4; ++c) {
                    const float v = std::clamp(points[selected_][c + 1].get<float>(), 0.0f, 1.0f);
                    const auto tint = c == 0 ? colour(0.8f, 0.3f, 0.3f) : c == 1 ? colour(0.3f, 0.7f, 0.4f)
                                                                      : c == 2   ? colour(0.35f, 0.5f, 0.9f)
                                                                                 : foreground;
                    quad(mesh, 20 * sx, (80 + c * 10) * sy, 170 * sx, 3 * sy, colour(0.18f, 0.18f, 0.18f));
                    quad(mesh, 20 * sx, (80 + c * 10) * sy, 170 * v * sx, 3 * sy, tint);
                    disc(mesh, {(20 + 170 * v) * sx, (81.5f + c * 10) * sy}, 2.5f * dp, tint);
                }
            }
        } else {
            const float top = rgb ? 24 : 8, height = 112 - top;
            line(mesh, {8 * sx, 112 * sy}, {192 * sx, top * sy}, 0.6f * dp, colour(0.24f, 0.25f, 0.27f));
            const auto curve = lfs::nodes::curve_control_points(points);
            const auto tint = property_ == "r" ? colour(0.95f, 0.4f, 0.4f) : property_ == "g" ? colour(0.4f, 0.9f, 0.5f)
                                                                         : property_ == "b"   ? colour(0.4f, 0.6f, 1)
                                                                                              : foreground;
            for (int i = 0; i < 184; ++i)
                line(mesh, {(8 + i) * sx, (112 - height * lfs::nodes::sample_curve(curve, float(i) / 184)) * sy},
                     {(9 + i) * sx, (112 - height * lfs::nodes::sample_curve(curve, float(i + 1) / 184)) * sy}, 1.7f * dp, tint);
            if (rgb && editing) {
                const char* names[] = {"combined", "r", "g", "b"};
                for (int i = 0; i < 4; ++i)
                    quad(mesh, (8 + i * 48) * sx, 4 * sy, 40 * sx, 3 * sy, property_ == names[i] ? foreground : dim);
            }
        }
        if (editing)
            for (size_t i = 0; i < points.size(); ++i) {
                const Rml::Vector2f p{(8 + 184 * points[i][0].get<float>()) * sx, (ramp ? 65 : 112 - (rgb ? 88 : 104) * points[i][1].get<float>()) * sy};
                disc(mesh, p, (int(i) == selected_ ? 5 : 4) * dp, colour(0.08f, 0.09f, 0.10f));
                disc(mesh, p, (int(i) == selected_ ? 3.5f : 2.5f) * dp, foreground);
            }
        geometry_ = GetRenderManager()->MakeGeometry(std::move(mesh));
        dirty_ = false;
        theme_ = rml_theme::currentThemeSignature();
    }
    void NodeCurveElement::OnRender() {
        if (dirty_ || theme_ != rml_theme::currentThemeSignature())
            rebuild();
        geometry_.Render(GetAbsoluteOffset(Rml::BoxArea::Content));
    }
} // namespace lfs::vis::gui
