/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/export.hpp"
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/EventListener.h>
#include <RmlUi/Core/Geometry.h>
#include <nlohmann/json.hpp>
namespace lfs::vis::gui {
    class LFS_VIS_API NodeCurveElement final : public Rml::Element, private Rml::EventListener {
    public:
        explicit NodeCurveElement(const Rml::String& tag);

    protected:
        void OnRender() override;
        void OnResize() override;
        void OnAttributeChange(const Rml::ElementAttributes& changed) override;
        void ProcessDefaultAction(Rml::Event& event) override;
        void ProcessEvent(Rml::Event& event) override;

    private:
        void pointer(const Rml::Event& event);
        void emit(const char* event);
        void removePoint();
        void rebuild();
        Rml::Vector2f coordinates(const Rml::Event& event);
        nlohmann::json values_;
        std::string property_ = "points";
        int selected_ = -1, channel_ = -1;
        bool dragging_ = false, dirty_ = true;
        size_t theme_ = 0;
        Rml::Geometry geometry_;
    };
} // namespace lfs::vis::gui
