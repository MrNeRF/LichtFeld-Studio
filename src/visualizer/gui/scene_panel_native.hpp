/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/logger.hpp"
#include "gui/panel_registry.hpp"
#include "gui/rmlui/rml_input_utils.hpp"
#include "gui/rmlui/rml_panel_host.hpp"

#include <RmlUi/Core/EventListener.h>
#include <cstdint>
#include <string>

namespace Rml {
    class Element;
    class ElementDocument;
    class ElementFormControlSelect;
} // namespace Rml

namespace lfs::vis::gui {

    class RmlUIManager;
    class SceneGraphElement;

    class NativeScenePanel : public IPanel {
    public:
        explicit NativeScenePanel(RmlUIManager* manager);

        void draw(const PanelDrawContext& ctx) override;
        void preload(const PanelDrawContext& ctx) override;
        void preloadDirect(float w, float h, const PanelDrawContext& ctx,
                           float clip_y_min, float clip_y_max,
                           const PanelInputState* input) override;
        bool supportsDirectDraw() const override { return true; }
        void drawDirect(float x, float y, float w, float h, const PanelDrawContext& ctx) override;
        float getDirectDrawHeight() const override { return host_.getContentHeight(); }
        void setInputClipY(float y_min, float y_max) override { host_.setInputClipY(y_min, y_max); }
        void setInput(const PanelInputState* input) override { host_.setInput(input); }
        void setForcedHeight(float h) override { host_.setForcedHeight(h); }
        bool wantsKeyboard() const override { return host_.wantsKeyboard(); }
        bool needsAnimationFrame() const override { return host_.needsAnimationFrame(); }
        void reloadRmlResources() override;

    private:
        struct EventListener : Rml::EventListener {
            NativeScenePanel* owner = nullptr;
            void ProcessEvent(Rml::Event& event) override;
        };

        enum class Tab : uint8_t {
            Scene,
            History,
            Logging,
        };

        enum class FeedbackTone : uint8_t {
            Info,
            Success,
            Error,
        };

        bool ensureInitialized();
        void clearElementCache();
        void cacheElements();
        void syncPanel(const PanelDrawContext& ctx);
        bool syncSceneState(const PanelDrawContext& ctx);
        bool syncHistoryState();
        bool syncLoggingState();
        bool syncLocale();
        bool syncTabState();
        bool syncSummaryChips();
        bool syncSceneVisibility();
        bool handleEvent(Rml::Event& event);
        void applyFilterInputValue();
        void applyLogLevelSelection();
        void copyBufferedLogsToClipboard();
        void exportBufferedLogsToTextFile();
        void setLoggingFeedback(std::string message, FeedbackTone tone);
        void setTab(Tab tab);

        RmlUIManager* manager_ = nullptr;
        RmlPanelHost host_;
        EventListener listener_;
        rml_input::TextInputEscapeRevertController filter_input_revert_;

        Rml::ElementDocument* document_ = nullptr;
        SceneGraphElement* tree_el_ = nullptr;
        Rml::Element* scene_tab_el_ = nullptr;
        Rml::Element* history_tab_el_ = nullptr;
        Rml::Element* logging_tab_el_ = nullptr;
        Rml::Element* chip_row_el_ = nullptr;
        Rml::Element* summary_model_chip_el_ = nullptr;
        Rml::Element* summary_node_chip_el_ = nullptr;
        Rml::Element* summary_selection_chip_el_ = nullptr;
        Rml::Element* summary_filter_chip_el_ = nullptr;
        Rml::Element* scene_view_el_ = nullptr;
        Rml::Element* search_container_el_ = nullptr;
        Rml::Element* filter_input_el_ = nullptr;
        Rml::Element* filter_clear_el_ = nullptr;
        Rml::Element* empty_state_el_ = nullptr;
        Rml::Element* empty_primary_el_ = nullptr;
        Rml::Element* empty_secondary_el_ = nullptr;
        Rml::Element* history_container_el_ = nullptr;
        Rml::Element* history_summary_label_el_ = nullptr;
        Rml::Element* history_summary_value_el_ = nullptr;
        Rml::Element* history_transaction_el_ = nullptr;
        Rml::Element* history_undo_btn_el_ = nullptr;
        Rml::Element* history_redo_btn_el_ = nullptr;
        Rml::Element* history_clear_btn_el_ = nullptr;
        Rml::Element* history_note_el_ = nullptr;
        Rml::Element* history_undo_title_el_ = nullptr;
        Rml::Element* history_redo_title_el_ = nullptr;
        Rml::Element* history_undo_list_el_ = nullptr;
        Rml::Element* history_redo_list_el_ = nullptr;
        Rml::Element* history_empty_undo_el_ = nullptr;
        Rml::Element* history_empty_redo_el_ = nullptr;
        Rml::Element* logging_container_el_ = nullptr;
        Rml::Element* logging_summary_label_el_ = nullptr;
        Rml::Element* logging_summary_value_el_ = nullptr;
        Rml::Element* logging_level_label_el_ = nullptr;
        Rml::ElementFormControlSelect* logging_level_select_el_ = nullptr;
        Rml::Element* logging_export_btn_el_ = nullptr;
        Rml::Element* logging_copy_btn_el_ = nullptr;
        Rml::Element* logging_feedback_el_ = nullptr;
        Rml::Element* logging_note_el_ = nullptr;
        Rml::Element* logging_scroll_el_ = nullptr;
        Rml::Element* logging_list_el_ = nullptr;
        Rml::Element* logging_empty_el_ = nullptr;

        Tab active_tab_ = Tab::Scene;
        std::string last_language_;
        uint64_t last_history_generation_ = 0;
        uint64_t last_log_generation_ = 0;
        lfs::core::LogLevel last_log_level_ = lfs::core::LogLevel::Off;
        uint64_t last_prepare_frame_ = 0;
        std::string logging_feedback_text_;
        FeedbackTone logging_feedback_tone_ = FeedbackTone::Info;
        bool logging_feedback_dirty_ = false;
    };

} // namespace lfs::vis::gui
