/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/rmlui/rml_tooltip.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/vram_hud_overlay.hpp"
#include "visualizer/app_store.hpp"

#include <RmlUi/Core/DataModelHandle.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <optional>
#include <string>

namespace Rml {
    class Context;
    class ElementDocument;
    class Element;
} // namespace Rml

namespace lfs::vis {
    struct Theme;
}
namespace lfs::vis::gui {

    struct PanelInputState;

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

        using VramHudOverlayState = VramHudOverlay::State;

        RmlViewportOverlay();
        ~RmlViewportOverlay();
        RmlViewportOverlay(const RmlViewportOverlay&) = delete;
        RmlViewportOverlay& operator=(const RmlViewportOverlay&) = delete;

        void init(RmlUIManager* mgr);
        void shutdown();
        void setViewportBounds(glm::vec2 pos, glm::vec2 size, glm::vec2 screen_origin);
        void setToolbarPanels(float primary_x, float primary_width,
                              bool show_secondary = false,
                              float secondary_x = 0.0f,
                              float secondary_width = 0.0f);
        void setGTMetricsOverlay(GTMetricsOverlayState state);
        void setVramHudOverlay(VramHudOverlayState state);
        bool isDueForVramProcessSample(std::chrono::milliseconds interval);
        void reloadResources();
        void render();
        void renderCached();
        void processInput(const PanelInputState& input);
        bool wantsInput() const { return wants_input_; }
        [[nodiscard]] bool needsAnimationFrame() const {
            return render_needed_ || document_sync_dirty_ || animation_active_ || tooltip_.needsFrame() ||
                   (vram_hud_ && vram_hud_->needsAnimationFrame());
        }
        [[nodiscard]] bool blocksPointer(double screen_x, double screen_y) const;

    private:
        bool updateTheme();
        void cacheBodyTemplate();
        void ensureBodyDataModelBound(Rml::Element* body);
        bool shouldRunDocumentHooks(bool force, bool prepend) const;
        bool shouldRunAnyDocumentHooks(bool force) const;
        void markDocumentSyncDirty();
        bool syncBuiltinDocument(bool force);
        bool updateToolbarRoots();
        void bindReactiveStore();
        void refreshGTMetricsOverlayFromStore();
        void applyGTMetricsOverlay();
        bool applyFrameTooltip();
        void queueCachedVulkanContext(bool refresh_cache);
        enum class RenderReason : std::uint32_t {
            Initial = 1u << 0,
            Reload = 1u << 1,
            DocumentSync = 1u << 2,
            DocumentHook = 1u << 3,
            ViewportResize = 1u << 4,
            ToolbarLayout = 1u << 5,
            GTMetrics = 1u << 6,
            VramHud = 1u << 7,
            DataModelBinding = 1u << 8,
            PointerHover = 1u << 9,
            PointerButton = 1u << 10,
            PointerWheel = 1u << 11,
            PointerDrag = 1u << 12,
            Keyboard = 1u << 13,
        };
        void markRenderNeeded(RenderReason reason);
        [[nodiscard]] std::string renderReasonSources() const;

        RmlUIManager* rml_manager_ = nullptr;
        Rml::Context* rml_context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        Rml::Element* body_el_ = nullptr;

        glm::vec2 vp_pos_{0, 0};
        glm::vec2 vp_size_{0, 0};
        glm::vec2 screen_origin_{0, 0};
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
        std::size_t last_theme_signature_ = 0;
        bool has_theme_signature_ = false;
        std::string base_rcss_;
        std::string body_template_rml_;
        bool wants_input_ = false;
        bool doc_registered_ = false;
        bool render_needed_ = true;
        std::uint32_t render_reason_bits_ = static_cast<std::uint32_t>(RenderReason::Initial);
        bool document_sync_dirty_ = true;
        bool data_model_binding_dirty_ = true;
        bool animation_active_ = false;
        bool hovered_interactive_ = false;
        Rml::Element* last_hover_element_ = nullptr;
        bool mouse_pos_valid_ = false;
        int last_mouse_x_ = 0;
        int last_mouse_y_ = 0;
        int last_render_w_ = 0;
        int last_render_h_ = 0;
        CachedVulkanContextRender direct_cache_;
        GTMetricsOverlayState gt_metrics_overlay_;
        lfs::vis::AppStore::GTMetricsOverlayConfig gt_metrics_config_;
        std::optional<lfs::vis::AppStore::CameraMetrics> camera_metrics_;
        lfs::core::reactive::SubscriptionToken gt_metrics_config_subscription_;
        lfs::core::reactive::SubscriptionToken camera_metrics_subscription_;
        lfs::core::reactive::SubscriptionToken vram_hud_subscription_;
        std::vector<lfs::core::reactive::SubscriptionToken> document_sync_subscriptions_;
        std::unique_ptr<VramHudOverlay> vram_hud_;
        RmlTooltipController tooltip_;
        std::chrono::steady_clock::time_point last_document_hook_run_{};
        static constexpr auto kDocumentHookPollInterval = std::chrono::milliseconds(100);
    };

} // namespace lfs::vis::gui
