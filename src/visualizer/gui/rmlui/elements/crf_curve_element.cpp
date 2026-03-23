/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/elements/crf_curve_element.hpp"

#include <RmlUi/Core/RenderManager.h>
#include <algorithm>
#include <cassert>
#include <cmath>

namespace lfs::vis::gui {

    CRFCurveElement::CRFCurveElement(const Rml::String& tag) : Rml::Element(tag) {}

    bool CRFCurveElement::GetIntrinsicDimensions(Rml::Vector2f& dimensions, float& ratio) {
        dimensions = {200.f, 120.f};
        ratio = 200.f / 120.f;
        return true;
    }

    void CRFCurveElement::OnResize() { dirty_ = true; }

    void CRFCurveElement::OnAttributeChange(const Rml::ElementAttributes& changed) {
        Element::OnAttributeChange(changed);
        auto get = [this](const Rml::String& name, float fallback) {
            auto* attr = GetAttribute(name);
            return attr ? attr->Get(fallback) : fallback;
        };
        gamma_ = get("gamma", 1.0f);
        toe_ = get("toe", 0.0f);
        shoulder_ = get("shoulder", 0.0f);
        gamma_r_ = get("gamma-r", 0.0f);
        gamma_g_ = get("gamma-g", 0.0f);
        gamma_b_ = get("gamma-b", 0.0f);
        dirty_ = true;
    }

    float CRFCurveElement::applyCrf(float x, float gamma, float toe, float shoulder) {
        float y = std::pow(x, 1.0f / gamma);
        if (toe != 0.0f && x < MIDPOINT) {
            const float t_factor = 1.0f + toe * TOE_FACTOR;
            y = y * t_factor - (t_factor - 1.0f) * x * 2.0f * (MIDPOINT - x);
        }
        if (shoulder != 0.0f && x > MIDPOINT) {
            const float s_factor = 1.0f - shoulder * SHOULDER_FACTOR;
            const float blend = (x - MIDPOINT) * 2.0f;
            y = y * (1.0f - blend * (1.0f - s_factor));
        }
        return std::clamp(y, 0.0f, 1.0f);
    }

    void CRFCurveElement::rebuild() {
        auto* rm = GetRenderManager();
        if (!rm)
            return;

        const float w = GetBox().GetSize(Rml::BoxArea::Content).x;
        const float h = GetBox().GetSize(Rml::BoxArea::Content).y;
        if (w <= 0.f || h <= 0.f)
            return;

        // Background
        {
            Rml::Mesh mesh;
            Rml::ColourbPremultiplied bg_col(25, 25, 25, 255);
            mesh.vertices.push_back({{0, 0}, bg_col, {0, 0}});
            mesh.vertices.push_back({{w, 0}, bg_col, {0, 0}});
            mesh.vertices.push_back({{w, h}, bg_col, {0, 0}});
            mesh.vertices.push_back({{0, h}, bg_col, {0, 0}});
            mesh.indices = {0, 1, 2, 0, 2, 3};
            bg_geom_ = rm->MakeGeometry(std::move(mesh));
        }

        // Diagonal reference line
        {
            Rml::Mesh mesh;
            Rml::ColourbPremultiplied diag_col(76, 76, 76, 255);
            const float lw = 1.0f;
            mesh.vertices.push_back({{0, h}, diag_col, {0, 0}});
            mesh.vertices.push_back({{lw, h}, diag_col, {0, 0}});
            mesh.vertices.push_back({{w, 0}, diag_col, {0, 0}});
            mesh.vertices.push_back({{w - lw, 0}, diag_col, {0, 0}});
            mesh.indices = {0, 1, 2, 0, 2, 3};
            diag_geom_ = rm->MakeGeometry(std::move(mesh));
        }

        // Curves as quad strips with perpendicular offset
        {
            Rml::Mesh mesh;
            const bool has_per_channel = (gamma_r_ != 0.0f || gamma_g_ != 0.0f || gamma_b_ != 0.0f);
            const float half_lw = LINE_WIDTH * 0.5f;

            auto buildCurve = [&](auto evalFn, Rml::ColourbPremultiplied col) {
                struct Pt {
                    float x, y;
                };
                Pt pts[NUM_POINTS];
                for (int i = 0; i < NUM_POINTS; ++i) {
                    const float t = static_cast<float>(i) / (NUM_POINTS - 1);
                    pts[i] = {t * w, (1.0f - evalFn(t)) * h};
                }

                const int vert_base = static_cast<int>(mesh.vertices.size());
                for (int i = 0; i < NUM_POINTS; ++i) {
                    float dx, dy;
                    if (i == 0) {
                        dx = pts[1].x - pts[0].x;
                        dy = pts[1].y - pts[0].y;
                    } else if (i == NUM_POINTS - 1) {
                        dx = pts[i].x - pts[i - 1].x;
                        dy = pts[i].y - pts[i - 1].y;
                    } else {
                        dx = pts[i + 1].x - pts[i - 1].x;
                        dy = pts[i + 1].y - pts[i - 1].y;
                    }
                    const float len = std::sqrt(dx * dx + dy * dy);
                    const float nx = (len > 0.f) ? -dy / len : 0.f;
                    const float ny = (len > 0.f) ? dx / len : 1.f;

                    mesh.vertices.push_back({{pts[i].x + nx * half_lw, pts[i].y + ny * half_lw}, col, {0, 0}});
                    mesh.vertices.push_back({{pts[i].x - nx * half_lw, pts[i].y - ny * half_lw}, col, {0, 0}});

                    if (i > 0) {
                        const int b = vert_base + i * 2;
                        mesh.indices.push_back(b - 2);
                        mesh.indices.push_back(b - 1);
                        mesh.indices.push_back(b + 1);
                        mesh.indices.push_back(b - 2);
                        mesh.indices.push_back(b + 1);
                        mesh.indices.push_back(b);
                    }
                }
            };

            if (has_per_channel) {
                buildCurve([this](float x) { return applyCrf(x, gamma_ * (1.0f + gamma_r_), toe_, shoulder_); },
                           Rml::ColourbPremultiplied(230, 76, 76, 204));
                buildCurve([this](float x) { return applyCrf(x, gamma_ * (1.0f + gamma_g_), toe_, shoulder_); },
                           Rml::ColourbPremultiplied(76, 204, 76, 204));
                buildCurve([this](float x) { return applyCrf(x, gamma_ * (1.0f + gamma_b_), toe_, shoulder_); },
                           Rml::ColourbPremultiplied(76, 127, 230, 204));
            } else {
                buildCurve([this](float x) { return applyCrf(x, gamma_, toe_, shoulder_); },
                           Rml::ColourbPremultiplied(137, 180, 250, 255));
            }

            curve_geom_ = rm->MakeGeometry(std::move(mesh));
        }

        dirty_ = false;
    }

    void CRFCurveElement::OnRender() {
        if (dirty_)
            rebuild();

        auto offset = GetAbsoluteOffset(Rml::BoxArea::Content);
        bg_geom_.Render(offset);
        diag_geom_.Render(offset);
        curve_geom_.Render(offset);
    }

} // namespace lfs::vis::gui
