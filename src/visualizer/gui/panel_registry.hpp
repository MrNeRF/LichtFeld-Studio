/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "operator/poll_dependency.hpp"
#include "panel_space.hpp"

#include <core/export.hpp>

#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace lfs::core {
    class Scene;
}

namespace lfs::vis::gui {

    struct UIContext;
    struct ViewportLayout;
    struct PanelInputState;

    enum class PanelRenderTargetKind : uint8_t {
        Space,
        Panel,
        Children,
    };

    struct PanelRenderTarget {
        PanelRenderTargetKind kind = PanelRenderTargetKind::Space;
        PanelSpace space = PanelSpace::Floating;
        std::string id;

        [[nodiscard]] static PanelRenderTarget for_space(PanelSpace panel_space) {
            return {.kind = PanelRenderTargetKind::Space, .space = panel_space, .id = {}};
        }

        [[nodiscard]] static PanelRenderTarget for_panel(std::string panel_id) {
            return {.kind = PanelRenderTargetKind::Panel, .id = std::move(panel_id)};
        }

        [[nodiscard]] static PanelRenderTarget for_children(std::string parent_id) {
            return {.kind = PanelRenderTargetKind::Children, .id = std::move(parent_id)};
        }
    };

    enum class PanelRenderMode : uint8_t {
        Standard,
        StandardPreload,
        Direct,
        DirectCached,
        DirectPreload,
    };

    struct PanelRenderOptions {
        PanelRenderTarget target;
        PanelRenderMode mode = PanelRenderMode::Standard;
        // Geometry and clip fields apply to Direct* modes; x/y are unused by DirectPreload.
        float x = 0.0f;
        float y = 0.0f;
        float width = 0.0f;
        float height = 0.0f;
        float clip_y_min = -1.0f;
        float clip_y_max = -1.0f;
        const PanelInputState* input = nullptr;
    };

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
        std::optional<PanelDrawBounds> screen_bounds;
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

    enum class PanelDirectRenderMode : uint8_t {
        Measure,
        Draw,
        Cached,
        Preload,
    };

    struct PanelDirectRenderRequest {
        PanelDirectRenderMode mode = PanelDirectRenderMode::Draw;
        PanelSpace space = PanelSpace::Floating;
        float x = 0.0f;
        float y = 0.0f;
        float width = 0.0f;
        float height = 0.0f;
        float clip_y_min = -1.0f;
        float clip_y_max = -1.0f;
        float forced_height = 0.0f;
        const PanelInputState* input = nullptr;
    };

    struct PanelDirectRenderResult {
        bool handled = false;
        float height = 0.0f;
    };

    struct PanelRenderCapabilities {
        bool direct = false;
    };

    class IPanel {
    public:
        virtual ~IPanel() = default;
        virtual void draw(const PanelDrawContext& ctx) = 0;
        virtual bool poll(const PanelDrawContext& ctx) {
            (void)ctx;
            return true;
        }
        virtual void preload(const PanelDrawContext& ctx) { (void)ctx; }
        virtual PanelRenderCapabilities renderCapabilities() const { return {}; }
        virtual PanelDirectRenderResult renderDirect(
            const PanelDirectRenderRequest& request,
            const PanelDrawContext& ctx) {
            switch (request.mode) {
            case PanelDirectRenderMode::Measure:
                return {.handled = true};
            case PanelDirectRenderMode::Draw:
                draw(ctx);
                return {.handled = true};
            case PanelDirectRenderMode::Preload:
                preload(ctx);
                return {.handled = true};
            case PanelDirectRenderMode::Cached:
                return {};
            }
            return {};
        }
        virtual bool needsAnimationFrame() const { return false; }
        // Finite scheduled animation/update delay in seconds (> 0). nullopt means
        // no scheduled wake (either continuous demand via needsAnimationFrame or idle).
        virtual std::optional<double> nextScheduledAnimationDelay() const { return std::nullopt; }
        virtual void reloadRmlResources() {}
        virtual void releaseRendererResources() {}
        // Opaque per-panel chrome JSON. Empty capture means "no payload".
        // apply of "{}" or missing keys must restore this panel's defaults.
        [[nodiscard]] virtual std::string captureChromeJson() const { return {}; }
        virtual void applyChromeJson(std::string_view json) { (void)json; }
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
        std::string default_parent_id;
        PanelSpace default_space = PanelSpace::Floating;
        int default_order = 100;
        bool default_enabled = true;
        static constexpr int MAX_CONSECUTIVE_ERRORS = 3;

        bool has_option(PanelOption opt) const {
            return (options & static_cast<uint32_t>(opt)) != 0;
        }
    };

    struct FloatingPanelInteraction {
        float x = NAN;
        float y = NAN;
        bool auto_center = true;
        uint64_t stack_order = 0;
        bool dragging = false;
        float drag_offset_x = 0.0f;
        float drag_offset_y = 0.0f;
        bool resizing = false;
        float resize_start_width = 0.0f;
        float resize_start_height = 0.0f;
        float resize_start_mouse_x = 0.0f;
        float resize_start_mouse_y = 0.0f;
        float resize_start_panel_x = 0.0f;
        float resize_start_panel_y = 0.0f;
        int8_t resize_direction_x = 0;
        int8_t resize_direction_y = 0;
        float user_height = 0.0f;
        bool last_bounds_valid = false;
        float last_x = 0.0f;
        float last_y = 0.0f;
        float last_width = 0.0f;
        float last_height = 0.0f;
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

    struct PanelProjectState {
        std::string id;
        std::string parent_id;
        PanelSpace space = PanelSpace::Floating;
        int order = 100;
        bool enabled = true;
        float float_x = NAN;
        float float_y = NAN;
        float float_user_height = 0.0f;
        bool float_last_bounds_valid = false;
        float float_last_x = 0.0f;
        float float_last_y = 0.0f;
        float float_last_w = 0.0f;
        float float_last_h = 0.0f;
        bool float_auto_center = true;
        uint64_t float_stack_order = 0;
    };

    struct PanelSnapshot {
        size_t index;
        IPanel* panel;
        std::string label;
        std::string id;
        PanelSpace space;
        uint32_t options;
        bool is_native;
        PollDependency poll_dependencies;
        float initial_width;
        float initial_height;
        float float_x;
        float float_y;
        int order = 100;
        uint64_t float_stack_order = 0;
        std::shared_ptr<IPanel> panel_holder;

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

        float render_panels(const PanelRenderOptions& options, const PanelDrawContext& ctx);
        bool has_panels(PanelSpace space) const;

        std::vector<PanelSummary> get_panels_for_space(PanelSpace space);
        std::vector<std::string> get_panel_names(PanelSpace space) const;
        std::optional<PanelDetails> get_panel(const std::string& id);
        [[nodiscard]] std::vector<PanelProjectState>
        capture_project_state() const;
        void apply_project_state(
            const std::vector<PanelProjectState>& state);
        // Drops only retained, unclamped project rectangles. Live panel
        // placement and registration are unchanged.
        void clear_project_state_retention();
        void reset_project_state();
        [[nodiscard]] std::unordered_map<std::string, std::string>
        capture_panel_payloads() const;
        // Replaces the pending payload map. Registered panels receive their
        // payload, or "{}" when absent so leftover chrome cannot leak across
        // projects. Late register_panel delivers a matching pending payload.
        void apply_panel_payloads(
            const std::unordered_map<std::string, std::string>& payloads);
        [[nodiscard]] uint64_t registration_revision() const;
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
        // Min finite scheduled delay across visible panels (same visibility rules as
        // needsAnimationFrameForVisiblePanels). nullopt if none are scheduled.
        std::optional<double> nextScheduledAnimationDelayForVisiblePanels(
            PanelAnimationVisibility visibility) const;
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
        PanelSnapshot make_snapshot_locked(size_t index, const PanelInfo& panel) const;
        std::vector<PanelSnapshot> collect_snapshots_locked(
            const PanelRenderTarget& target,
            const PanelDrawContext& ctx) const;
        float render_standard_panels(const PanelRenderOptions& options,
                                     const PanelDrawContext& ctx);
        float render_direct_panels(const PanelRenderOptions& options,
                                   const PanelDrawContext& ctx);
        void render_space_panels(PanelSpace space, const PanelDrawContext& ctx,
                                 const PanelInputState* input);
        uint64_t alloc_float_stack_order_locked();
        FloatingPanelInteraction& ensure_floating_interaction_locked(const PanelInfo& panel);
        void bring_floating_panel_to_front_locked(const PanelInfo& panel);
        void reset_project_state_locked();

        mutable std::mutex mutex_;
        mutable std::mutex poll_mutex_;
        std::vector<PanelInfo> panels_;
        std::unordered_map<std::string, FloatingPanelInteraction> floating_interactions_;
        std::unordered_set<std::string> disabled_overrides_;
        mutable std::unordered_map<std::string, PollCacheEntry> poll_cache_;
        // A monitor or user-global UI scale may clamp the live rectangle.
        // Preserve the requested project rectangle until the user explicitly
        // repositions or resizes the panel.
        std::unordered_map<std::string, PanelProjectState>
            requested_project_floating_state_;
        std::unordered_map<std::string, std::string>
            requested_panel_payloads_;
        uint64_t next_float_stack_order_ = 1;
        uint64_t registration_revision_ = 0;
        int8_t floating_cursor_dir_x_ = 0;
        int8_t floating_cursor_dir_y_ = 0;
    };

} // namespace lfs::vis::gui
