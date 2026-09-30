/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/area_editors.hpp"
#include "gui/panel_layout.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "screen/area_gestures.hpp"
#include "screen/screen_service.hpp"

#include <RmlUi/Core/DataModelHandle.h>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Rml {
    class Context;
    class ElementDocument;
    class Element;
} // namespace Rml

namespace lfs::vis::gui {

    class GlobalContextMenu;

    // Chrome data bound to the screen documents.
    struct ChromeItem {
        Rml::String id{};
        Rml::String label{};
        Rml::String icon{};
        Rml::String tooltip{};
        int kind = 0;
        bool active = false;
        bool closeable = false;

        bool operator==(const ChromeItem&) const = default;
    };

    struct ChromeArea {
        int id = 0;
        Rml::String left, top, width, height, header_height;
        Rml::String editor_icon;
        Rml::String editor_label;
        std::vector<ChromeItem> items;
        std::vector<ChromeItem> right_items;
        bool is_view = false;
        bool hovered = false;
        bool active = false;
        bool maximized = false;
        Rml::String view_label;

        bool operator==(const ChromeArea&) const = default;
    };

    // Hosts the screen in the window's work area: lays the areas out, draws
    // their headers, frames and gesture previews, runs the area gestures and
    // lets each area's editor draw its content.
    class ScreenHost {
    public:
        struct Services {
            screen::ScreenService* screens = nullptr;
            RmlUIManager* rml = nullptr;
            GlobalContextMenu* context_menu = nullptr;
            // A 3D view's settings or camera changed outside the renderer.
            std::function<void(screen::AreaId)> view_changed;
            // The screen's structure changed (areas, editors, active view).
            std::function<void()> screen_changed;
            // Runs a view command such as "view.frame_all" on a 3D view.
            std::function<void(screen::AreaId, std::string_view)> view_command;
        };

        ScreenHost();
        explicit ScreenHost(screen::ScreenService& screens) { services_.screens = &screens; }
        ~ScreenHost();

        void init(Services services);
        void shutdown();
        void reloadResources();

        // Lays the screen out inside `work` (window pixels) for this frame.
        void layout(const screen::Rect& work, float ui_scale);
        // Re-solve after an API-driven screen mutation using the latest window work rect.
        void refreshLayout() {
            if (services_.screens)
                layout(work_, ui_scale_);
        }
        // Runs gestures and header input. `pointer_free` is false while
        // something above the screen (menu, modal, floating panel) owns the
        // pointer.
        void processInput(const PanelInputState& input, bool pointer_free);
        // Draws every non-3D editor's content, then queues the chrome.
        void draw(const UIContext& ui, const PanelDrawContext& draw, const PanelInputState& input,
                  bool force_live, PanelAnimationDemand demand);
        void queueOverlay();

        [[nodiscard]] const std::vector<AreaFrame>& areas() const { return frames_; }
        [[nodiscard]] const AreaFrame* area(screen::AreaId id) const;
        [[nodiscard]] screen::Rect currentAreaRect(screen::AreaId id);
        [[nodiscard]] screen::AreaId areaAt(float x, float y) const;
        [[nodiscard]] screen::AreaId viewAt(float x, float y) const;
        [[nodiscard]] std::optional<screen::Rect> viewContent(screen::AreaId id) const;
        [[nodiscard]] const screen::LayoutGeometry& geometry() const { return geometry_; }

        // True where the screen's own chrome takes pointer input: headers,
        // dividers, corner zones, and everywhere while a gesture runs.
        [[nodiscard]] bool blocksPointer(float x, float y) const;
        // Divider and corner zones inside a 3D view only block presses, so
        // hovering there still reaches the viewport.
        [[nodiscard]] bool blocksPress(float x, float y) const;
        [[nodiscard]] bool gestureActive() const { return gestures_.active(); }
        [[nodiscard]] bool wantsKeyboard() const { return false; }
        [[nodiscard]] bool needsAnimationFrame() const;
        [[nodiscard]] std::string animationDemandDescription() const;
        [[nodiscard]] screen::GestureCursor cursor() const { return cursor_; }

        // Panel editors whose visibility follows a flag of their own instead
        // of the panel's enabled state (the owner syncs those).
        void setExternallyManaged(std::vector<std::string> editors);

        [[nodiscard]] PropertiesEditor& properties() { return properties_; }
        [[nodiscard]] const PropertiesEditor& properties() const { return properties_; }
        [[nodiscard]] bool isEditorVisible(std::string_view editor) const;

        // Area commands shared by headers, menus, shortcuts and scripts.
        bool toggleMaximized(screen::AreaId id);
        screen::AreaId splitArea(screen::AreaId id, screen::SplitAxis axis);
        bool closeArea(screen::AreaId id);
        bool setEditor(screen::AreaId id, std::string_view editor);
        void openEditorMenu(screen::AreaId id, float x, float y);
        void openAreaMenu(screen::AreaId id, float x, float y);

    private:
        AreaEditor& editorFor(std::string_view editor);
        void syncPanelEditors();
        void rebuildChrome();
        void syncChromeGeometry();
        void updateOverlay();
        void handleAction(int area, const std::string& action, float x, float y);
        void applyGesture(const screen::GestureCommand& command);
        void mutate(const std::function<void(screen::Screen&)>& fn);
        [[nodiscard]] bool pointerOverHeader(float x, float y) const;
        [[nodiscard]] bool updateTheme();

        Services services_;
        float ui_scale_ = 1.0f;
        screen::Rect work_;
        screen::LayoutGeometry geometry_;
        std::vector<AreaFrame> frames_;
        std::uint64_t laid_out_generation_ = 0;
        bool layout_changed_ = true;

        screen::AreaGestures gestures_;
        screen::GestureCursor cursor_ = screen::GestureCursor::Default;
        screen::GesturePreview hover_preview_;
        screen::AreaId hovered_area_;
        screen::AreaId live_capture_area_;

        View3DEditor* view3d_editor_ = nullptr;
        std::unique_ptr<View3DEditor> view3d_;
        PropertiesEditor properties_;
        ScenePanelEditor scene_;
        ConsoleEditor console_;
        PanelEditor panel_;

        // Panel editors shown last frame, to tell whether the panel's enabled
        // flag or the screen changed since.
        std::unordered_set<std::string> panel_editor_ids_;
        std::unordered_set<std::string> panel_editors_shown_;
        std::unordered_map<std::string, bool> panel_enabled_seen_;
        std::vector<std::string> externally_managed_;

        // Chrome (headers and frames, drawn behind the panels).
        Rml::Context* chrome_context_ = nullptr;
        Rml::ElementDocument* chrome_document_ = nullptr;
        Rml::DataModelHandle chrome_model_;
        std::vector<ChromeArea> chrome_areas_;
        CachedVulkanContextRender chrome_cache_;
        bool chrome_dirty_ = true;
        bool chrome_pointer_inside_ = false;
        struct PendingAction {
            int area = 0;
            std::string action;
            float x = 0.0f;
            float y = 0.0f;
        };
        std::vector<PendingAction> pending_actions_;
        float last_mouse_x_ = 0.0f;
        float last_mouse_y_ = 0.0f;

        // Overlay (gesture previews and the active-view outline, drawn on top).
        Rml::Context* overlay_context_ = nullptr;
        Rml::ElementDocument* overlay_document_ = nullptr;
        CachedVulkanContextRender overlay_cache_;
        bool overlay_dirty_ = true;
        bool overlay_visible_ = false;

        std::string chrome_base_rcss_;
        std::string overlay_base_rcss_;
        std::size_t theme_signature_ = 0;
        bool has_theme_signature_ = false;
    };

} // namespace lfs::vis::gui
