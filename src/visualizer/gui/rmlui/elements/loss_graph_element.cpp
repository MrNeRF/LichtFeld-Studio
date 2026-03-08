/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/elements/loss_graph_element.hpp"

#include <RmlUi/Core/RenderManager.h>
#include <algorithm>
#include <cassert>
#include <cmath>

namespace lfs::vis::gui {

    LossGraphElement::LossGraphElement(const Rml::String& tag) : Rml::Element(tag) {}

    bool LossGraphElement::GetIntrinsicDimensions(Rml::Vector2f& dimensions, float& ratio) {
        dimensions = {1.f, 60.f};
        ratio = 0.f;
        return true;
    }

    void LossGraphElement::OnResize() { dirty_ = true; }

    void LossGraphElement::setData(const std::deque<float>& data) {
        data_ = data;
        if (data_.size() > MAX_POINTS)
            data_.erase(data_.begin(), data_.begin() + static_cast<long>(data_.size() - MAX_POINTS));

        if (data_.empty()) {
            data_min_ = 0.0f;
            data_max_ = 1.0f;
        } else {
            data_min_ = *std::min_element(data_.begin(), data_.end());
            data_max_ = *std::max_element(data_.begin(), data_.end());
            if (data_min_ == data_max_) {
                data_min_ -= 1.0f;
                data_max_ += 1.0f;
            } else {
                const float margin = (data_max_ - data_min_) * MARGIN_FACTOR;
                data_min_ -= margin;
                data_max_ += margin;
            }
        }
        dirty_ = true;
    }

    void LossGraphElement::rebuild() {
        auto* rm = GetRenderManager();
        if (!rm)
            return;

        const float w = GetBox().GetSize(Rml::BoxArea::Content).x;
        const float h = GetBox().GetSize(Rml::BoxArea::Content).y;
        if (w <= 0.f || h <= 0.f)
            return;

        const float graph_top = VERT_PAD;
        const float graph_bottom = h - VERT_PAD;
        const float graph_h = graph_bottom - graph_top;
        if (graph_h <= 0.f)
            return;

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

        {
            Rml::Mesh mesh;
            Rml::ColourbPremultiplied grid_col(60, 60, 60, 255);
            const float line_h = 1.0f;

            for (int i = 0; i < TICK_COUNT; ++i) {
                const float t = static_cast<float>(i) / static_cast<float>(TICK_COUNT - 1);
                const float y = graph_top + graph_h * (1.0f - t);
                const int base = static_cast<int>(mesh.vertices.size());
                mesh.vertices.push_back({{0, y}, grid_col, {0, 0}});
                mesh.vertices.push_back({{w, y}, grid_col, {0, 0}});
                mesh.vertices.push_back({{w, y + line_h}, grid_col, {0, 0}});
                mesh.vertices.push_back({{0, y + line_h}, grid_col, {0, 0}});
                mesh.indices.push_back(base);
                mesh.indices.push_back(base + 1);
                mesh.indices.push_back(base + 2);
                mesh.indices.push_back(base);
                mesh.indices.push_back(base + 2);
                mesh.indices.push_back(base + 3);
            }
            grid_geom_ = rm->MakeGeometry(std::move(mesh));
        }

        {
            Rml::Mesh mesh;
            const int count = static_cast<int>(data_.size());
            if (count < 2) {
                line_geom_ = rm->MakeGeometry(std::move(mesh));
                dirty_ = false;
                return;
            }

            const Rml::ColourbPremultiplied line_col(137, 180, 250, 255);
            const float half_lw = LINE_WIDTH * 0.5f;
            const float range = data_max_ - data_min_;
            assert(range > 0.0f);

            const int n = std::min(count, MAX_POINTS);
            std::vector<float> px(n), py(n);
            for (int i = 0; i < n; ++i) {
                const float t = static_cast<float>(i) / static_cast<float>(n - 1);
                const float val = (data_[i] - data_min_) / range;
                px[i] = t * w;
                py[i] = graph_top + (1.0f - val) * graph_h;
            }

            mesh.vertices.reserve((n - 1) * 4);
            mesh.indices.reserve((n - 1) * 6);

            for (int i = 0; i < n - 1; ++i) {
                const float dx = px[i + 1] - px[i];
                const float dy = py[i + 1] - py[i];
                const float len = std::sqrt(dx * dx + dy * dy);
                const float nx = (len > 0.f) ? -dy / len * half_lw : 0.f;
                const float ny = (len > 0.f) ? dx / len * half_lw : half_lw;

                const int base = static_cast<int>(mesh.vertices.size());
                mesh.vertices.push_back({{px[i] + nx, py[i] + ny}, line_col, {0, 0}});
                mesh.vertices.push_back({{px[i] - nx, py[i] - ny}, line_col, {0, 0}});
                mesh.vertices.push_back({{px[i + 1] + nx, py[i + 1] + ny}, line_col, {0, 0}});
                mesh.vertices.push_back({{px[i + 1] - nx, py[i + 1] - ny}, line_col, {0, 0}});
                mesh.indices.push_back(base);
                mesh.indices.push_back(base + 1);
                mesh.indices.push_back(base + 3);
                mesh.indices.push_back(base);
                mesh.indices.push_back(base + 3);
                mesh.indices.push_back(base + 2);
            }

            line_geom_ = rm->MakeGeometry(std::move(mesh));
        }

        dirty_ = false;
    }

    void LossGraphElement::OnRender() {
        if (dirty_)
            rebuild();

        auto offset = GetAbsoluteOffset(Rml::BoxArea::Content);
        bg_geom_.Render(offset);
        grid_geom_.Render(offset);
        line_geom_.Render(offset);
    }

} // namespace lfs::vis::gui
