/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "gui/rmlui/rml_tooltip.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/vram_hud_overlay.hpp"
#include "visualizer/app_store.hpp"

#include <RmlUi/Core/DataModelHandle.h>
#include <RmlUi/Core/EventListener.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Rml {
    class Context;
    class ElementDocument;
    class Element;
} // namespace Rml

namespace lfs::vis {
    struct Theme;
    class RmlViewportInputRoutingTest;
} // namespace lfs::vis
namespace lfs::vis::gui {

    struct PanelInputState;

    // Frame-level blockers captured by GuiManager before underlay routing.
    // The original event stream remains available for still-owned releases.
    struct ViewportOverlayInputBlockers {
        bool startup = false;
        bool progress = false;
        bool modal = false;
        bool pending_modal = false;
        bool context_menu = false;
        bool menu_pointer = false;
        bool floating_panel = false;

        [[nodiscard]] bool blocksInput() const {
            return startup || progress || modal || pending_modal || context_menu || menu_pointer || floating_panel;
        }
    };

    // Test window-coordinate `point` against the rectangle resolveViewerPanel splits.
    // The overlay also covers the left dock for toolbars (gui_manager.cpp,
    // setViewportBounds), so an overlay press need not be a viewport press.
    [[nodiscard]] inline bool pointInsideViewport(const glm::vec2 point,
                                                  const glm::vec2 viewport_pos,
                                                  const glm::vec2 viewport_size) {
        if (viewport_size.x <= 0.0f || viewport_size.y <= 0.0f)
            return false;
        return point.x >= viewport_pos.x && point.x < viewport_pos.x + viewport_size.x &&
               point.y >= viewport_pos.y && point.y < viewport_pos.y + viewport_size.y;
    }

    // Overlay-only facts: whether a left DOWN hit a control or dismissed focused text.
    // One entry per left DOWN in the last processed frame, in SDL order.
    // Pair entry i with that frame's i-th canonical left DOWN for its coordinates
    // and GUI ownership.
    struct OverlayLeftPressClassification {
        bool on_interactive_control = false;
        bool blurred_text_input = false;
    };

    // Classify all inputs from the same DOWN's coordinates, not the latest cursor.
    struct OverlayPressFocusInputs {
        bool left_pressed = false;
        bool overlay_wants_input = false;
        bool pressed_interactive_control = false;
        bool press_blurred_text_input = false;
        bool press_inside_viewport = false;
        // Event-time GUI ownership of this DOWN.
        bool press_gui_owned = false;
    };

    // Only a left DOWN inside the viewport, outside interactive controls and not
    // GUI-owned at press time may change panel focus. The overlay also covers the
    // dock, and its resize strip overlaps the viewport; dismissing text must not
    // turn either into a focus target. Ownership captured at BUTTON_DOWN remains
    // valid after layout changes.
    // Toolbar actions must use existing focus, regardless of where controls appear.
    // Otherwise, allow focus when the overlay consumes the press or it blurred text.
    // Blur and its commit run first, so the edit reaches its original panel.
    // Process each DOWN in SDL order; each allowed press moves focus, while refusal
    // leaves any earlier focus change intact.
    [[nodiscard]] inline bool overlayPressMayFocusPanel(const OverlayPressFocusInputs& press) {
        if (!press.left_pressed || !press.press_inside_viewport ||
            press.press_gui_owned || press.pressed_interactive_control)
            return false;
        return press.overlay_wants_input || press.press_blurred_text_input;
    }

    class RmlViewportOverlay {
    public:
        struct GTMetricsOverlayState {
            bool visible = false;
            float x = 16.0f;
            float y = 16.0f;
            std::string psnr_text;
            bool show_ssim = false;
            std::string ssim_text;
        };

        struct SplitDividerOverlayState {
            bool visible = false;
            float x = 0.0f;
            float y = 0.0f;
            float width = 0.0f;
            float height = 0.0f;
        };

        struct LodStatsOverlayState {
            bool visible = false;
            float x = 52.0f;
            float y = 54.0f;
            std::string status_text;
            std::string selected_text;
            std::string budget_text;
            std::string model_text;
            std::string tree_text;
            std::string traversal_text;
            std::string stop_text;
            std::string chunks_text;
            std::string cache_text;
            std::string selector_text;
            std::string pixel_text;
            std::string render_text;
            std::string foveation_text;
            std::string hash_text;
        };

        struct ProjectDragOverlayState {
            bool visible = false;
            std::string label;
        };

        using VramHudOverlayState = VramHudOverlay::State;

        LFS_VIS_API RmlViewportOverlay();
        LFS_VIS_API ~RmlViewportOverlay();
        RmlViewportOverlay(const RmlViewportOverlay&) = delete;
        RmlViewportOverlay& operator=(const RmlViewportOverlay&) = delete;

        void init(RmlUIManager* mgr);
        LFS_VIS_API void shutdown();
        LFS_VIS_API void setViewportBounds(glm::vec2 pos, glm::vec2 size, glm::vec2 screen_origin);
        void setViewportContentOffset(float x);
        void setToolbarPanels(float primary_x, float primary_width,
                              bool show_secondary = false,
                              float secondary_x = 0.0f,
                              float secondary_width = 0.0f);
        void setLeftDockResizeIndicator(bool visible, bool active, float thickness);
        void setSplitDividerOverlay(SplitDividerOverlayState state);
        void setGTMetricsOverlay(GTMetricsOverlayState state);
        void setLodStatsOverlay(LodStatsOverlayState state);
        void setProjectDragOverlay(ProjectDragOverlayState state);
        void setVramHudOverlay(VramHudOverlayState state);
        void reloadResources();
        void render();
        void renderCached();
        void renderFrostedGlass();
        // Shared production routing boundary: input is the original frame,
        // never a masked copy with sentinel coordinates or removed events.
        LFS_VIS_API void processInput(const PanelInputState& input,
                                      const ViewportOverlayInputBlockers& blockers = {});
        bool wantsInput() const { return wants_input_; }
        // Left DOWNs from the last processInput(), classified at their own points
        // in arrival order; empty with no left press or an early blocked return.
        [[nodiscard]] const std::vector<OverlayLeftPressClassification>&
        leftPressClassifications() const {
            return left_press_classifications_;
        }
        [[nodiscard]] bool needsAnimationFrame() const {
            return render_needed_ || document_sync_dirty_ || animation_active_ || tooltip_.revealDue() ||
                   toolbar_drag_active_ ||
                   (vram_hud_ && vram_hud_->needsAnimationFrame());
        }
        // Finite RmlUi scheduled update delay (seconds) when > 0; nullopt for
        // continuous demand (0) or idle (infinity).
        [[nodiscard]] std::optional<double> nextScheduledUpdateDelay() const;
        [[nodiscard]] bool blocksPointer(double screen_x, double screen_y) const;

    private:
        struct ToolbarDragListener final : Rml::EventListener {
            RmlViewportOverlay* owner = nullptr;
            void ProcessEvent(Rml::Event& event) override;
        };

        bool updateTheme();
        void cacheBodyTemplate();
        void ensureBodyDataModelBound(Rml::Element* body);
        bool shouldRunDocumentHooks(bool force, bool prepend) const;
        bool shouldRunAnyDocumentHooks(bool force) const;
        void markDocumentSyncDirty();
        bool syncBuiltinDocument(bool force);
        bool updateToolbarRoots();
        bool updateToolbarRailLayout();
        bool applyToolbarPosition();
        void attachToolbarDragListeners();
        void resetToolbarDragListeners();
        void onToolbarDrag(Rml::Event& event);
        [[nodiscard]] float toolbarFreeGap(float toolbar_height) const;
        [[nodiscard]] float toolbarFreeTop(float toolbar_height) const;
        [[nodiscard]] float toolbarFreeTravel(float toolbar_height) const;
        void updateViewportContentOffset();
        void bindReactiveStore();
        void refreshGTMetricsOverlayFromStore();
        void applySplitDividerOverlay();
        void applyLeftDockResizeIndicator();
        void applyGTMetricsOverlay();
        void applyLodStatsOverlay();
        void applyProjectDragOverlay();
        bool applyFrameTooltip();
        void queueCachedVulkanContext(bool refresh_cache);
        enum class RenderReason : std::uint32_t {
            Initial = 1u << 0,
            Reload = 1u << 1,
            DocumentSync = 1u << 2,
            DocumentHook = 1u << 3,
            ViewportResize = 1u << 4,
            ToolbarLayout = 1u << 5,
            SplitDivider = 1u << 6,
            GTMetrics = 1u << 7,
            VramHud = 1u << 8,
            DataModelBinding = 1u << 9,
            PointerHover = 1u << 10,
            PointerButton = 1u << 11,
            PointerWheel = 1u << 12,
            PointerDrag = 1u << 13,
            Keyboard = 1u << 14,
            LodStats = 1u << 15,
            LeftDockResize = 1u << 16,
            PerfHud = 1u << 17,
            ProjectDrag = 1u << 18,
            ThemePresentation = 1u << 19,
            Tooltip = 1u << 20,
        };
        void markRenderNeeded(RenderReason reason);
        [[nodiscard]] std::string renderReasonSources() const;

        friend class lfs::vis::RmlViewportInputRoutingTest;

        RmlUIManager* rml_manager_ = nullptr;
        Rml::Context* rml_context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        Rml::Element* body_el_ = nullptr;

        glm::vec2 vp_pos_{0, 0};
        glm::vec2 vp_size_{0, 0};
        glm::vec2 screen_origin_{0, 0};
        // A collapsed layout leaves the live Rml context at its previous size.
        // Its outstanding releases still use that last valid window origin.
        std::optional<glm::vec2> last_valid_input_origin_;
        float primary_toolbar_x_ = 0.0f;
        float primary_toolbar_width_ = 0.0f;
        bool show_secondary_toolbar_ = false;
        float secondary_toolbar_x_ = 0.0f;
        float secondary_toolbar_width_ = 0.0f;
        float applied_primary_toolbar_x_ = 0.0f;
        float applied_primary_toolbar_width_ = -1.0f;
        bool applied_show_secondary_toolbar_ = false;
        float applied_secondary_toolbar_x_ = 0.0f;
        float applied_secondary_toolbar_width_ = -1.0f;
        bool toolbar_roots_dirty_ = true;
        bool toolbar_rail_layout_dirty_ = true;
        float last_toolbar_dpi_ = 0.0f;
        bool toolbar_position_preference_dirty_ = true;
        std::string viewport_toolbar_position_ = "centered";
        std::string applied_viewport_toolbar_position_;
        float viewport_toolbar_free_y_ = 0.5f;
        float applied_primary_toolbar_top_ = std::numeric_limits<float>::quiet_NaN();
        float applied_secondary_toolbar_top_ = std::numeric_limits<float>::quiet_NaN();
        Rml::Element* primary_toolbar_drag_handle_ = nullptr;
        Rml::Element* secondary_toolbar_drag_handle_ = nullptr;
        ToolbarDragListener toolbar_drag_listener_;
        bool toolbar_drag_active_ = false;
        bool applied_toolbar_drag_active_ = false;
        bool toolbar_drag_moved_ = false;
        float toolbar_drag_start_top_ = 0.0f;
        float toolbar_drag_start_mouse_y_ = 0.0f;
        float viewport_content_offset_ = 0.0f;
        bool viewport_content_offset_dirty_ = true;
        std::size_t last_theme_signature_ = 0;
        bool has_theme_signature_ = false;
        std::string viewport_chrome_style_ = "translucent";
        std::string base_rcss_;
        std::string body_template_rml_;
        bool wants_input_ = false;
        std::vector<OverlayLeftPressClassification> left_press_classifications_;
        // Per-button DOWN delivery to this RmlUi context; the matching UP is owed
        // wherever it lands. Persist across frames; clear the whole array only on
        // context destruction (rml_pointer_dispatch.hpp).
        bool pointer_down_delivered_[3] = {};
        bool doc_registered_ = false;
        bool render_needed_ = true;
        std::uint32_t render_reason_bits_ = static_cast<std::uint32_t>(RenderReason::Initial);
        bool document_sync_dirty_ = true;
        bool data_model_binding_dirty_ = true;
        bool animation_active_ = false;
        double next_update_delay_ = std::numeric_limits<double>::infinity();
        bool hovered_interactive_ = false;
        Rml::Element* last_hover_element_ = nullptr;
        bool mouse_pos_valid_ = false;
        int last_mouse_x_ = 0;
        int last_mouse_y_ = 0;
        int last_render_w_ = 0;
        int last_render_h_ = 0;
        CachedVulkanContextRender direct_cache_;
        bool left_dock_resize_visible_ = false;
        bool left_dock_resize_active_ = false;
        float left_dock_resize_thickness_ = 0.0f;
        SplitDividerOverlayState split_divider_overlay_;
        GTMetricsOverlayState gt_metrics_overlay_;
        LodStatsOverlayState lod_stats_overlay_;
        ProjectDragOverlayState project_drag_overlay_;
        lfs::vis::AppStore::GTMetricsOverlayConfig gt_metrics_config_;
        std::optional<lfs::vis::AppStore::CameraMetrics> camera_metrics_;
        lfs::core::reactive::SubscriptionToken gt_metrics_config_subscription_;
        lfs::core::reactive::SubscriptionToken camera_metrics_subscription_;
        lfs::core::reactive::SubscriptionToken vram_hud_subscription_;
        lfs::core::reactive::SubscriptionToken perf_hud_subscription_;
        std::vector<lfs::core::reactive::SubscriptionToken> document_sync_subscriptions_;
        std::unique_ptr<VramHudOverlay> vram_hud_;
        RmlTooltipController tooltip_;
        std::chrono::steady_clock::time_point last_document_hook_run_{};
        static constexpr auto kDocumentHookPollInterval = std::chrono::milliseconds(100);
    };

} // namespace lfs::vis::gui
