/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/global_context_menu.hpp"
#include "gui/panel_layout.hpp"
#include "gui/panel_registry.hpp"
#include "gui/ui_context.hpp"
#include "screen/screen.hpp"

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

// The GUI half of the editor types (the model half is screen/editor_type):
// how each editor draws its header controls and its content into an area.
// One instance per editor type serves every area showing it; per-area state
// lives in the area's SpaceData.
namespace lfs::vis::gui {

    struct AreaFrame {
        screen::AreaId id;
        screen::Rect rect;    // the whole area
        screen::Rect header;  // the header strip
        screen::Rect content; // below the header
        std::string editor;
    };

    // A control in an area header, drawn by the screen chrome document.
    struct HeaderItem {
        enum class Kind : std::uint8_t { Menu,
                                         Tab,
                                         Toggle,
                                         Button,
                                         Label,
                                         Spacer };
        Kind kind = Kind::Button;
        std::string id{}; // action id sent back to the editor
        std::string label{};
        std::string icon{};
        std::string tooltip{};
        bool active = false;
        bool closeable = false;
    };

    struct AreaDrawContext {
        const AreaFrame& area;
        const UIContext& ui;
        const PanelDrawContext& draw;
        const PanelInputState& input;
        bool live = true; // false: reuse cached panel textures when possible
    };

    class AreaEditor {
    public:
        virtual ~AreaEditor() = default;

        // Controls after the editor-type button, left to right. Items after
        // the first Spacer are right-aligned.
        virtual void header(const AreaFrame& /*area*/, const screen::Screen& /*screen*/,
                            std::vector<HeaderItem>& /*items*/) const {}
        // Items of the popup a Menu header item opens. Actions come back
        // through headerAction.
        [[nodiscard]] virtual std::vector<ContextMenuItem> menu(const AreaFrame& /*area*/,
                                                                const screen::Screen& /*screen*/,
                                                                std::string_view /*menu_id*/) const {
            return {};
        }
        // Handles a header control or a menu action; (x, y) is where a
        // follow-up popup would open.
        virtual void headerAction(const AreaFrame& /*area*/, screen::Screen& /*screen*/,
                                  std::string_view /*action*/, float /*x*/, float /*y*/) {}
        virtual void draw(const AreaDrawContext& /*ctx*/) {}
        // Space whose panels this editor draws, for animation demand.
        [[nodiscard]] virtual std::optional<PanelSpace> panelSpace() const { return std::nullopt; }
        [[nodiscard]] virtual bool needsAnimationFrame() const { return false; }
    };

    // The 3D viewport's header: the View menu, display mode, depth, overlays
    // and projection. Edits go straight to the area's View3DSpace; commands
    // that need the scene (framing, dataset cameras) go to `command`.
    class View3DEditor final : public AreaEditor {
    public:
        using Command = std::function<void(screen::AreaId, std::string_view)>;
        using Changed = std::function<void(screen::AreaId)>;

        View3DEditor(Command command, Changed changed)
            : command_(std::move(command)),
              changed_(std::move(changed)) {}

        void header(const AreaFrame& area, const screen::Screen& screen,
                    std::vector<HeaderItem>& items) const override;
        [[nodiscard]] std::vector<ContextMenuItem> menu(const AreaFrame& area, const screen::Screen& screen,
                                                        std::string_view menu_id) const override;
        void headerAction(const AreaFrame& area, screen::Screen& screen, std::string_view action, float x,
                          float y) override;

    private:
        Command command_;
        Changed changed_;
    };

    class PropertiesEditor final : public AreaEditor {
    public:
        void header(const AreaFrame& area, const screen::Screen& screen,
                    std::vector<HeaderItem>& items) const override;
        void headerAction(const AreaFrame& area, screen::Screen& screen, std::string_view action, float x,
                          float y) override;
        void draw(const AreaDrawContext& ctx) override;
        [[nodiscard]] std::optional<PanelSpace> panelSpace() const override { return PanelSpace::MainPanelTab; }

        [[nodiscard]] const std::string& activeTab() const { return active_tab_; }
        void setActiveTab(std::string id);
        // Focuses a tab by id or label (used by "show panel" requests).
        bool focusTab(std::string_view id_or_label);
        [[nodiscard]] float scroll() const { return scroll_; }
        void setScroll(float value) { scroll_ = value; }

    private:
        void syncTabs() const;

        mutable std::vector<PanelSummary> tabs_;
        std::string active_tab_;
        float scroll_ = 0.0f;
        float content_height_ = 0.0f;
    };

    class ScenePanelEditor final : public AreaEditor {
    public:
        void draw(const AreaDrawContext& ctx) override;
        [[nodiscard]] std::optional<PanelSpace> panelSpace() const override { return PanelSpace::SceneHeader; }
    };

    class ConsoleEditor final : public AreaEditor {
    public:
        void draw(const AreaDrawContext& ctx) override;
    };

    // Any registered panel shown as an editor of its own (the sequencer,
    // histogram, asset browser and plugin panels that used to live in docks).
    class PanelEditor final : public AreaEditor {
    public:
        void header(const AreaFrame& area, const screen::Screen& screen,
                    std::vector<HeaderItem>& items) const override;
        void draw(const AreaDrawContext& ctx) override;
        [[nodiscard]] std::optional<PanelSpace> panelSpace() const override { return PanelSpace::BottomDock; }
    };

    // Registers the editor types backed by registered panels: panels in the
    // BottomDock and LeftDock spaces become editors named after the panel.
    void installPanelEditorTypes(screen::EditorTypeRegistry& registry);

    [[nodiscard]] bool isPanelEditorSpace(PanelSpace space);

} // namespace lfs::vis::gui
