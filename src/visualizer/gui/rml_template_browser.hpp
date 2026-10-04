/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/uuid.hpp"
#include "gui/rmlui/rml_tooltip.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "visualizer/nodes/modifier_manager.hpp"
#include <RmlUi/Core/EventListener.h>
#include <chrono>
#include <functional>
#include <string>
#include <vector>

namespace lfs::vis::gui {
    struct PanelInputState;

    // An app-level context, just like the existing modal overlay. It never changes
    // the screen layout or borrows the Node Editor's clipping/input bounds.
    class RmlTemplateBrowser final : private Rml::EventListener {
    public:
        RmlTemplateBrowser(RmlUIManager& rml, std::function<void(std::string)> on_added);
        ~RmlTemplateBrowser();
        void open(ModifierManager& manager, core::Uuid target, std::string save_tree = {});
        void close();
        [[nodiscard]] bool isOpen() const { return open_; }
        void processInput(const PanelInputState& input, bool blocked);
        void render(int width, int height);
        void reloadResources();
        [[nodiscard]] bool needsAnimationFrame() const;

    private:
        void ProcessEvent(Rml::Event& event) override;
        void init();
        void rebuild();
        void filter();
        void select(std::string id);
        void apply(std::string id);
        void edit(std::string id);
        void updateLayout();
        void updateFooter();
        void setError(std::string message);
        void refreshTemplates();
        [[nodiscard]] const NodeGraphTemplate* selected() const;

        RmlUIManager& rml_;
        std::function<void(std::string)> on_added_;
        ModifierManager* manager_ = nullptr;
        core::Uuid target_;
        Rml::Context* context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        CachedVulkanContextRender cache_;
        RmlTooltipController tooltip_;
        std::vector<NodeGraphTemplate> templates_;
        std::string category_ = "all";
        std::string query_, selected_id_, save_tree_, rename_id_, menu_id_;
        std::string last_click_id_;
        std::chrono::steady_clock::time_point last_click_;
        std::size_t theme_signature_ = 0;
        std::uint64_t language_generation_ = 0;
        bool open_ = false;
        bool dirty_ = true;
        bool pointer_down_[3] = {};
        int width_ = 0, height_ = 0, mouse_x_ = 0, mouse_y_ = 0;
        int columns_ = 0;
        float card_width_ = 0;
    };
} // namespace lfs::vis::gui
