/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/logger.hpp"
#include "gui/panel_registry.hpp"
#include "gui/rmlui/rml_input_utils.hpp"
#include "gui/rmlui/rml_panel_host.hpp"
#include "gui/scene_tree_session.hpp"

#include <RmlUi/Core/EventListener.h>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

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
        PanelRenderCapabilities renderCapabilities() const override {
            return {.direct = true};
        }
        PanelDirectRenderResult renderDirect(const PanelDirectRenderRequest& request,
                                             const PanelDrawContext& ctx) override;
        bool needsAnimationFrame() const override { return host_.needsAnimationFrame(); }
        std::optional<double> nextScheduledAnimationDelay() const override {
            return host_.nextScheduledUpdateDelay();
        }
        void reloadRmlResources() override;
        void releaseRendererResources() override { host_.releaseRendererResources(); }
        [[nodiscard]] std::string projectActiveTab() const;
        void setProjectActiveTab(std::string_view tab);
        [[nodiscard]] SceneTreeSessionChrome captureTreeChrome(
            const core::Scene& scene) const;
        void applyTreeChrome(const SceneTreeSessionChrome& chrome);
        void resetTreeChrome();
        [[nodiscard]] bool selectAllIfFocused();
        [[nodiscard]] bool toggleSelectionVisibilityIfFocused();
        [[nodiscard]] bool toggleSelectionTrainingIfFocused();
        [[nodiscard]] bool requestDeleteSelectionIfAvailable();

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

        struct SyncStamp {
            Tab active_tab = Tab::Scene;
            uint64_t scene_generation = 0;
            uint64_t selection_generation = 0;
            int64_t num_gaussians = 0;
            bool training_running = false;
            std::string training_state;
            int eval_psnr_milli = 0;
            int eval_ssim_milli = 0;
            uint64_t history_generation = 0;
            uint64_t log_generation = 0;
            lfs::core::LogLevel log_level = lfs::core::LogLevel::Off;
            uint64_t language_generation = 0;
            uint64_t render_settings_generation = 0;
            int dp_ratio_milli = 1000;
            bool invert_masks = false;
            bool scene_graph_selection_markers = false;

            bool operator==(const SyncStamp&) const = default;
        };

        bool ensureInitialized();
        void preloadDirect(float w, float h, const PanelDrawContext& ctx,
                           float clip_y_min, float clip_y_max,
                           const PanelInputState* input);
        void drawDirect(float x, float y, float w, float h, const PanelDrawContext& ctx);
        bool drawDirectCached(float x, float y, float w, float h,
                              const PanelDrawContext& ctx);
        void clearElementCache();
        void cacheElements();
        void syncPanel(const PanelDrawContext& ctx);
        bool shouldSyncPanel(const PanelInputState* input) const;
        SyncStamp makeSyncStamp() const;
        bool syncSceneState(const PanelDrawContext& ctx);
        bool syncHistoryState();
        bool syncLoggingState();
        bool syncLocale();
        bool syncTabState();
        bool syncSummaryChips();
        bool syncSelectionActions();
        bool syncSceneVisibility();
        bool handleEvent(Rml::Event& event);
        void applyFilterInputValue();
        void applyPendingTreeChrome();
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
        Rml::Element* asset_manager_button_el_ = nullptr;
        Rml::Element* chip_row_el_ = nullptr;
        Rml::Element* summary_model_chip_el_ = nullptr;
        Rml::Element* summary_node_chip_el_ = nullptr;
        Rml::Element* summary_filter_chip_el_ = nullptr;
        Rml::Element* scene_view_el_ = nullptr;
        Rml::Element* search_container_el_ = nullptr;
        Rml::Element* filter_input_el_ = nullptr;
        Rml::Element* filter_clear_el_ = nullptr;
        Rml::Element* selection_action_bar_el_ = nullptr;
        Rml::Element* selection_action_count_el_ = nullptr;
        Rml::Element* selection_clear_el_ = nullptr;
        Rml::Element* selection_visibility_el_ = nullptr;
        Rml::Element* selection_visibility_icon_el_ = nullptr;
        Rml::Element* selection_training_el_ = nullptr;
        Rml::Element* selection_training_icon_el_ = nullptr;
        Rml::Element* selection_delete_el_ = nullptr;
        std::string visible_icon_source_;
        std::string hidden_icon_source_;
        std::string locked_icon_source_;
        std::string unlocked_icon_source_;
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
        std::optional<SceneTreeSessionChrome> pending_tree_chrome_;
        std::string last_language_;
        uint64_t last_history_generation_ = 0;
        uint64_t last_log_generation_ = 0;
        lfs::core::LogLevel last_log_level_ = lfs::core::LogLevel::Off;
        uint64_t last_prepare_frame_ = 0;
        SyncStamp last_sync_stamp_{};
        bool has_last_sync_stamp_ = false;
        std::string logging_feedback_text_;
        FeedbackTone logging_feedback_tone_ = FeedbackTone::Info;
        bool logging_feedback_dirty_ = false;
        bool syncing_logging_selection_ = false;
    };

} // namespace lfs::vis::gui
