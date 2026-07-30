/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "operator/poll_dependency.hpp"
#include "panel_space.hpp"

#include <core/export.hpp>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lfs::core {
    class Scene;
}

namespace lfs::vis::gui {

    struct UIContext;
    struct ViewportLayout;
    struct PanelInputState;

    struct PanelDrawBounds {
        float x = 0.0f;
        float y = 0.0f;
        float width = 0.0f;
        float height = 0.0f;

        [[nodiscard]] bool valid() const { return width > 0.0f && height > 0.0f; }
    };

    enum class PanelOption : uint32_t {
        DEFAULT_CLOSED = 1 << 0,
        HIDE_HEADER = 1 << 1,
        SELF_MANAGED = 1 << 2,
    };

    using PollDependency = lfs::vis::op::PollDependency;

    struct PanelDrawContext {
        const UIContext* ui = nullptr;
        const ViewportLayout* viewport = nullptr;
        core::Scene* scene = nullptr;
        bool ui_hidden = false;
        uint64_t frame_serial = 0;
        uint64_t scene_generation = 0;
        bool has_selection = false;
        bool is_training = false;
        bool suppress_non_native_panels = false;
        std::optional<PanelDrawBounds> bounds;
    };

    struct FloatingPanelAnchor {
        float x = 0.0f;
        float y = 0.0f;
        float width = 0.0f;
        float height = 0.0f;
    };

    struct FloatingPanelPlacement {
        float x = 0.0f;
        float y = 0.0f;
    };

    [[nodiscard]] LFS_VIS_API FloatingPanelPlacement computeFloatingPanelPlacement(
        const FloatingPanelAnchor& anchor,
        float panel_width,
        float panel_height,
        float stored_x,
        float stored_y,
        bool auto_center,
        float title_height,
        float visible_fraction = 0.1f);

    class IPanel {
    public:
        virtual ~IPanel() = default;
        virtual void draw(const PanelDrawContext& ctx) = 0;
        virtual bool poll(const PanelDrawContext& ctx) {
            (void)ctx;
            return true;
        }
        virtual bool supportsDirectDraw() const { return false; }
        virtual void preload(const PanelDrawContext& ctx) { (void)ctx; }
        virtual void preloadDirect(float w, float h, const PanelDrawContext& ctx,
                                   float clip_y_min = -1.0f, float clip_y_max = -1.0f,
                                   const PanelInputState* input = nullptr) {
            (void)w;
            (void)h;
            (void)clip_y_min;
            (void)clip_y_max;
            (void)input;
            preload(ctx);
        }
        virtual void drawDirect(float x, float y, float w, float h, const PanelDrawContext& ctx) {
            (void)x;
            (void)y;
            (void)w;
            (void)h;
            draw(ctx);
        }
        virtual bool drawDirectCached(float x, float y, float w, float h,
                                      const PanelDrawContext& ctx) {
            (void)x;
            (void)y;
            (void)w;
            (void)h;
            (void)ctx;
            return false;
        }
        virtual float getDirectDrawHeight() const { return 0.0f; }
        virtual void setInputClipY(float y_min, float y_max) {
            (void)y_min;
            (void)y_max;
        }
        virtual void setInput(const PanelInputState* input) { (void)input; }
        virtual void setForcedHeight(float h) { (void)h; }
        virtual bool wantsKeyboard() const { return false; }
        virtual bool needsAnimationFrame() const { return false; }
        virtual bool wantsExternalFloatingShadow() const { return true; }
        virtual void setPanelSpace(PanelSpace space) { (void)space; }
        virtual void reloadRmlResources() {}
        virtual void releaseRendererResources() {}
    };

    struct PanelInfo {
        std::shared_ptr<IPanel> panel;
        std::string label;
        std::string id;
        std::string parent_id;
        PanelSpace space = PanelSpace::Floating;
        int order = 100;
        bool enabled = true;
        uint32_t options = 0;
        PollDependency poll_dependencies = PollDependency::ALL;
        bool is_native = true;
        bool tab_closeable = false;
        int consecutive_errors = 0;
        bool error_disabled = false;
        float initial_width = 0;
        float initial_height = 0;
        float original_width = 0;
        float original_height = 0;
        float float_x = NAN;
        float float_y = NAN;
        bool float_auto_center = true;
        uint64_t float_stack_order = 0;
        bool float_dragging = false;
        float float_drag_ox = 0;
        float float_drag_oy = 0;
        bool float_resizing = false;
        float float_resize_start_w = 0;
        float float_resize_start_h = 0;
        float float_resize_start_mx = 0;
        float float_resize_start_my = 0;
        float float_resize_start_px = 0;
        float float_resize_start_py = 0;
        int8_t float_resize_dir_x = 0;
        int8_t float_resize_dir_y = 0;
        float float_user_height = 0;
        bool float_last_bounds_valid = false;
        float float_last_x = 0;
        float float_last_y = 0;
        float float_last_w = 0;
        float float_last_h = 0;
        static constexpr int MAX_CONSECUTIVE_ERRORS = 3;

        bool has_option(PanelOption opt) const {
            return (options & static_cast<uint32_t>(opt)) != 0;
        }
    };

    struct PanelAnimationVisibility {
        std::string_view active_main_tab;
        bool ui_visible = true;
        bool right_panel_visible = true;
        bool bottom_dock_visible = true;
        bool left_dock_visible = true;
    };

    struct PanelAnimationDemand {
        bool side_panel = false;
        bool floating = false;
        bool viewport_overlay = false;
        bool main_panel_tab = false;
        bool scene_header = false;
        bool bottom_dock = false;
        bool left_dock = false;
        bool status_bar = false;

        [[nodiscard]] bool rightPanel() const {
            return main_panel_tab || scene_header;
        }

        [[nodiscard]] bool any() const {
            return side_panel || floating || viewport_overlay || main_panel_tab ||
                   scene_header || bottom_dock || left_dock || status_bar;
        }
    };

    struct PanelSummary {
        std::string label;
        std::string id;
        PanelSpace space;
        int order;
        bool enabled;
        bool tab_closeable;
    };

    struct PanelDetails {
        std::string label;
        std::string id;
        std::string parent_id;
        PanelSpace space;
        int order;
        bool enabled;
        uint32_t options;
        PollDependency poll_dependencies;
        bool is_native;
        float initial_width;
        float initial_height;
        uint64_t float_stack_order;
    };

    struct PanelSnapshot {
        size_t index;
        IPanel* panel;
        std::string label;
        std::string id;
        std::string parent_id;
        uint32_t options;
        bool is_native;
        PollDependency poll_dependencies;
        float initial_width;
        float initial_height;
        float float_x;
        float float_y;
        int order = 100;
        uint64_t float_stack_order = 0;

        bool has_option(PanelOption opt) const {
            return (options & static_cast<uint32_t>(opt)) != 0;
        }
    };

    struct PollCacheEntry {
        bool result;
        uint64_t scene_generation;
        bool has_selection;
        bool is_training;
        PollDependency poll_dependencies;
    };

    class LFS_VIS_API PanelRegistry {
    public:
        static PanelRegistry& instance();

        bool register_panel(PanelInfo info);
        void unregister_panel(const std::string& id);
        void unregister_all_non_native();

        void draw_panels(PanelSpace space, const PanelDrawContext& ctx,
                         const PanelInputState* input = nullptr);
        void preload_panels(PanelSpace space, const PanelDrawContext& ctx);
        void draw_single_panel(const std::string& id, const PanelDrawContext& ctx);
        bool has_panels(PanelSpace space) const;

        float draw_panels_direct(PanelSpace space, float x, float y, float w, float max_h,
                                 const PanelDrawContext& ctx,
                                 const PanelInputState* input = nullptr);
        float draw_panels_direct_cached(PanelSpace space, float x, float y, float w, float max_h,
                                        const PanelDrawContext& ctx,
                                        const PanelInputState* input = nullptr);
        float preload_panels_direct(PanelSpace space, float w, float max_h,
                                    const PanelDrawContext& ctx,
                                    float clip_y_min = -1.0f, float clip_y_max = -1.0f,
                                    const PanelInputState* input = nullptr);
        float draw_single_panel_direct(const std::string& id, float x, float y, float w, float h,
                                       const PanelDrawContext& ctx,
                                       float clip_y_min = -1.0f, float clip_y_max = -1.0f,
                                       const PanelInputState* input = nullptr);
        float draw_single_panel_direct_cached(const std::string& id, float x, float y,
                                              float w, float h,
                                              const PanelDrawContext& ctx,
                                              float clip_y_min = -1.0f,
                                              float clip_y_max = -1.0f,
                                              const PanelInputState* input = nullptr);
        float preload_single_panel_direct(const std::string& id, float w, float h,
                                          const PanelDrawContext& ctx,
                                          float clip_y_min = -1.0f, float clip_y_max = -1.0f,
                                          const PanelInputState* input = nullptr);
        float preload_child_panels_direct(const std::string& parent_id, float w, float h,
                                          const PanelDrawContext& ctx,
                                          float clip_y_min = -1.0f, float clip_y_max = -1.0f,
                                          const PanelInputState* input = nullptr);
        float draw_child_panels_direct(const std::string& parent_id, float x, float y, float w, float h,
                                       const PanelDrawContext& ctx,
                                       float clip_y_min = -1.0f, float clip_y_max = -1.0f,
                                       const PanelInputState* input = nullptr);
        float draw_child_panels_direct_cached(const std::string& parent_id, float x,
                                              float y, float w, float h,
                                              const PanelDrawContext& ctx,
                                              float clip_y_min = -1.0f,
                                              float clip_y_max = -1.0f,
                                              const PanelInputState* input = nullptr);

        std::vector<PanelSummary> get_panels_for_space(PanelSpace space);
        std::vector<std::string> get_panel_names(PanelSpace space) const;
        std::optional<PanelDetails> get_panel(const std::string& id);
        bool isPositionOverFloatingPanel(double x, double y) const;
        void set_panel_enabled(const std::string& id, bool enabled);
        bool bring_panel_to_front(const std::string& id);
        bool is_panel_enabled(const std::string& id) const;
        bool apply_floating_resize_cursor() const;
        void rescale_floating_panels(float previous_scale, float new_scale);
        bool needsAnimationFrame() const;
        PanelAnimationDemand animationDemandForVisiblePanels(
            PanelAnimationVisibility visibility) const;
        bool needsAnimationFrameForVisiblePanels(PanelAnimationVisibility visibility) const;
        bool set_panel_label(const std::string& id, const std::string& new_label);
        bool set_panel_order(const std::string& id, int new_order);
        bool set_panel_space(const std::string& id, PanelSpace new_space);
        bool set_panel_parent(const std::string& id, const std::string& parent_id);
        void invalidate_poll_cache(PollDependency changed = PollDependency::ALL);
        void reload_rml_resources();

    private:
        PanelRegistry() = default;
        ~PanelRegistry() = default;
        PanelRegistry(const PanelRegistry&) = delete;
        PanelRegistry& operator=(const PanelRegistry&) = delete;

        bool check_poll(const PanelSnapshot& snap, const PanelDrawContext& ctx);
        void track_draw_result(const PanelSnapshot& snap, bool draw_succeeded);
        uint64_t alloc_float_stack_order_locked();
        void ensure_float_stack_order_locked(PanelInfo& panel);
        void bring_floating_panel_to_front_locked(PanelInfo& panel);

        mutable std::mutex mutex_;
        mutable std::mutex poll_mutex_;
        std::vector<PanelInfo> panels_;
        std::unordered_set<std::string> disabled_overrides_;
        mutable std::unordered_map<std::string, PollCacheEntry> poll_cache_;
        uint64_t next_float_stack_order_ = 1;
        int8_t floating_cursor_dir_x_ = 0;
        int8_t floating_cursor_dir_y_ = 0;
    };

} // namespace lfs::vis::gui
