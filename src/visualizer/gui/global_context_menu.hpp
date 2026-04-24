/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/rmlui/rml_fbo.hpp"
#include <RmlUi/Core/DataModelHandle.h>
#include <RmlUi/Core/EventListener.h>
#include <core/export.hpp>
#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace Rml {
    class Context;
    class Element;
    class ElementDocument;
} // namespace Rml

namespace lfs::vis {
    struct Theme;
}
namespace lfs::vis::gui {

    struct PanelInputState;
    class RmlUIManager;

    struct ContextMenuItem {
        std::string label;
        std::string action;
        bool separator_before = false;
        bool is_label = false;
        bool is_submenu_item = false;
        bool is_active = false;
    };

    class LFS_VIS_API GlobalContextMenu {
    public:
        using ActionCallback = std::function<void(std::string_view)>;

        explicit GlobalContextMenu(RmlUIManager* mgr);
        ~GlobalContextMenu();

        GlobalContextMenu(const GlobalContextMenu&) = delete;
        GlobalContextMenu& operator=(const GlobalContextMenu&) = delete;

        void request(std::vector<ContextMenuItem> items, float screen_x, float screen_y,
                     ActionCallback callback = {});
        std::string pollResult();
        [[nodiscard]] bool isOpen() const { return open_ || pending_open_; }

        void processInput(const PanelInputState& input);
        void render(int screen_w, int screen_h, float screen_x, float screen_y);
        void destroyGLResources();
        void reloadResources();

    private:
        void initContext();
        void syncTheme();
        void hide();
        void focusFirstItem();

        struct EventListener : Rml::EventListener {
            GlobalContextMenu* owner = nullptr;
            void ProcessEvent(Rml::Event& event) override;
        };

        RmlUIManager* mgr_;
        Rml::Context* ctx_ = nullptr;
        Rml::ElementDocument* doc_ = nullptr;
        Rml::DataModelHandle menu_model_;
        RmlFBO fbo_;
        EventListener listener_;

        Rml::Element* el_backdrop_ = nullptr;
        Rml::Element* el_ctx_menu_ = nullptr;

        bool open_ = false;
        bool pending_open_ = false;
        bool focus_first_item_ = false;
        std::vector<ContextMenuItem> items_;
        std::vector<ContextMenuItem> pending_items_;
        ActionCallback callback_;
        float pending_x_ = 0;
        float pending_y_ = 0;
        std::string result_;

        std::string base_rcss_;
        std::size_t last_theme_signature_ = 0;
        bool has_theme_signature_ = false;
        int width_ = 0;
        int height_ = 0;
    };

} // namespace lfs::vis::gui
