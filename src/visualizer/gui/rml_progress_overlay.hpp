/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/rmlui/rmlui_manager.hpp"
#include "visualizer/app_store.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

namespace Rml {
    class Context;
    class Element;
    class ElementDocument;
    class Event;
    class EventListener;
} // namespace Rml

namespace lfs::vis {
    class RmlProgressInputRoutingTest;
}

namespace lfs::vis::gui {

    struct PanelInputState;

    struct ProgressOverlayPresentation {
        enum class Kind {
            None,
            Import,
            VideoExport,
        };

        enum class Action {
            None,
            DismissImport,
            CancelVideoExport,
        };

        Kind kind = Kind::None;
        Action action = Action::None;
        std::string title;
        std::string path;
        std::string stage;
        std::string detail;
        std::string error;
        std::string action_label;
        float progress = 0.0f;
        bool show_progress = false;
        bool success = false;

        bool operator==(const ProgressOverlayPresentation&) const = default;
    };

    [[nodiscard]] LFS_VIS_API ProgressOverlayPresentation makeProgressOverlayPresentation(
        const AppStore::ImportOverlayState& import_state,
        const AppStore::VideoExportOverlayState& video_state);

    class LFS_VIS_API RmlProgressOverlay {
    public:
        RmlProgressOverlay(RmlUIManager* rml_manager,
                           std::function<void()> dismiss_import,
                           std::function<void()> cancel_video_export);
        ~RmlProgressOverlay();

        RmlProgressOverlay(const RmlProgressOverlay&) = delete;
        RmlProgressOverlay& operator=(const RmlProgressOverlay&) = delete;

        void processInput(const PanelInputState& input, bool blocked = false);
        void render(int screen_w, int screen_h,
                    float screen_x, float screen_y,
                    float vp_x, float vp_y, float vp_w, float vp_h);
        void releaseRendererResources();
        void reloadResources();
        void preload();

        [[nodiscard]] bool isVisible() const;
        [[nodiscard]] bool blocksUnderlayInput() const { return isVisible(); }
        [[nodiscard]] bool hasPendingRenderWork() const { return isVisible(); }

    private:
        friend class lfs::vis::RmlProgressInputRoutingTest;
        struct OverlayEventListener;

        void initContext();
        bool syncTheme();
        void cacheElements();
        void applyPresentation(const ProgressOverlayPresentation& presentation);
        void invokeAction();
        void cancelPointerInput();

        RmlUIManager* rml_manager_ = nullptr;
        std::function<void()> dismiss_import_;
        std::function<void()> cancel_video_export_;

        Rml::Context* rml_context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        Rml::Element* el_backdrop_ = nullptr;
        Rml::Element* el_dialog_ = nullptr;
        Rml::Element* el_title_ = nullptr;
        Rml::Element* el_path_ = nullptr;
        Rml::Element* el_progress_row_ = nullptr;
        Rml::Element* el_progress_ = nullptr;
        Rml::Element* el_progress_text_ = nullptr;
        Rml::Element* el_stage_ = nullptr;
        Rml::Element* el_detail_ = nullptr;
        Rml::Element* el_error_ = nullptr;
        Rml::Element* el_actions_ = nullptr;
        Rml::Element* el_action_ = nullptr;
        bool elements_cached_ = false;

        std::unique_ptr<OverlayEventListener> listener_;
        ProgressOverlayPresentation presentation_;
        std::string base_rcss_;
        std::size_t last_theme_signature_ = 0;
        bool has_theme_signature_ = false;
        int width_ = 0;
        int height_ = 0;
        float last_dialog_left_ = 0.0f;
        float last_dialog_top_ = 0.0f;
        float last_dialog_content_width_ = 0.0f;
        bool dialog_position_valid_ = false;
        bool pointer_down_delivered_[3] = {};
        bool last_mouse_valid_ = false;
        int last_mouse_x_ = 0;
        int last_mouse_y_ = 0;
        CachedVulkanContextRender direct_cache_;
        bool render_needed_ = true;
    };

} // namespace lfs::vis::gui
