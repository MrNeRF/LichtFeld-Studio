/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/rmlui/rml_fbo.hpp"
#include "sequencer/rml_sequencer_panel.hpp"
#include <RmlUi/Core/EventListener.h>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>
#include <ImGuizmo.h>

namespace Rml {
    class Context;
    class Element;
    class ElementDocument;
} // namespace Rml

namespace lfs::vis {
    struct Theme;
    class SequencerController;
} // namespace lfs::vis

namespace lfs::vis::gui {

    class RmlUIManager;

    class RmlSequencerOverlay {
    public:
        enum class Action : uint8_t {
            ADD_KEYFRAME,
            UPDATE_KEYFRAME,
            GOTO_KEYFRAME,
            EDIT_FOCAL_LENGTH,
            SET_TRANSLATE,
            SET_ROTATE,
            SET_EASING,
            DELETE_KEYFRAME,
            DESELECT_KEYFRAME,
            APPLY_EDIT,
            REVERT_EDIT
        };

        struct PendingAction {
            Action action{};
            size_t keyframe_index = 0;
            int easing_value = 0;
        };

        struct EditResult {
            size_t index = 0;
            float value = 0.0f;
        };

        RmlSequencerOverlay(SequencerController& controller, RmlUIManager* rml_manager);
        ~RmlSequencerOverlay();

        RmlSequencerOverlay(const RmlSequencerOverlay&) = delete;
        RmlSequencerOverlay& operator=(const RmlSequencerOverlay&) = delete;

        void showContextMenu(float screen_x, float screen_y,
                             std::optional<size_t> keyframe_index,
                             ImGuizmo::OPERATION gizmo_op);
        void hideContextMenu();

        void showTimeEdit(size_t index, float current_time);
        void showFocalEdit(size_t index, float current_focal_mm);

        void updateEditOverlay(size_t selected, float pos_delta, float rot_delta,
                               float right_x, float top_y);
        void hideEditOverlay();

        void processInput(const lfs::vis::PanelInputState& input);
        void render(int screen_w, int screen_h);
        void destroyGLResources();

        [[nodiscard]] bool isContextMenuOpen() const { return context_menu_open_; }
        [[nodiscard]] bool isPopupOpen() const { return time_edit_active_ || focal_edit_active_; }
        [[nodiscard]] bool wantsInput() const { return wants_input_; }

        [[nodiscard]] std::optional<PendingAction> consumeAction();
        [[nodiscard]] std::optional<EditResult> consumeTimeEdit();
        [[nodiscard]] std::optional<EditResult> consumeFocalEdit();

    private:
        void initContext();
        void syncTheme();
        void syncLocalization();
        std::string generateThemeRCSS(const lfs::vis::Theme& t) const;
        void cacheElements();
        std::string buildContextMenuHTML(std::optional<size_t> keyframe,
                                         ImGuizmo::OPERATION gizmo_op) const;
        void submitTimeEdit();
        void submitFocalEdit();

        struct OverlayEventListener : Rml::EventListener {
            RmlSequencerOverlay* overlay = nullptr;
            void ProcessEvent(Rml::Event& event) override;
        };

        SequencerController& controller_;
        RmlUIManager* rml_manager_;
        OverlayEventListener listener_;

        Rml::Context* rml_context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        RmlFBO fbo_;

        Rml::Element* el_menu_backdrop_ = nullptr;
        Rml::Element* el_context_menu_ = nullptr;
        Rml::Element* el_popup_backdrop_ = nullptr;
        Rml::Element* el_time_popup_ = nullptr;
        Rml::Element* el_focal_popup_ = nullptr;
        Rml::Element* el_time_input_ = nullptr;
        Rml::Element* el_focal_input_ = nullptr;
        Rml::Element* el_edit_overlay_ = nullptr;
        Rml::Element* el_edit_label_ = nullptr;
        Rml::Element* el_edit_delta_ = nullptr;
        Rml::Element* el_edit_apply_ = nullptr;
        Rml::Element* el_edit_revert_ = nullptr;
        Rml::Element* el_time_popup_title_ = nullptr;
        Rml::Element* el_focal_popup_title_ = nullptr;
        Rml::Element* el_time_ok_ = nullptr;
        Rml::Element* el_time_cancel_ = nullptr;
        Rml::Element* el_focal_ok_ = nullptr;
        Rml::Element* el_focal_cancel_ = nullptr;

        bool context_menu_open_ = false;
        std::optional<size_t> context_menu_keyframe_;

        bool time_edit_active_ = false;
        size_t time_edit_index_ = 0;

        bool focal_edit_active_ = false;
        size_t focal_edit_index_ = 0;

        bool edit_overlay_visible_ = false;
        bool wants_input_ = false;
        bool has_text_focus_ = false;
        bool elements_cached_ = false;
        bool skip_next_click_ = false;

        std::vector<PendingAction> pending_actions_;
        std::optional<EditResult> pending_time_edit_;
        std::optional<EditResult> pending_focal_edit_;

        std::string base_rcss_;
        std::size_t last_theme_signature_ = 0;
        bool has_theme_signature_ = false;

        int width_ = 0;
        int height_ = 0;
    };

} // namespace lfs::vis::gui
