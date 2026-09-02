/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "operator/ops/depth_window_ops.hpp"
#include "core/logger.hpp"
#include "core/services.hpp"
#include "gui/gui_manager.hpp"
#include "input/input_types.hpp"
#include "input/key_codes.hpp"
#include "operation/undo_entry.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator.hpp"
#include "operator/operator_registry.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "tools/selection_tool.hpp"
#include "visualizer/app_store.hpp"
#include "visualizer_impl.hpp"
#include <algorithm>
#include <cmath>
#include <memory>

namespace lfs::vis::op {

    namespace {
        constexpr int kRequiredModifiers = input::KEYMOD_SHIFT | input::KEYMOD_ALT;
        constexpr float kSingularityEpsilon = 1.0e-6f;
        // Drags never pin the rect flush to the viewport boundary: a small
        // inset keeps the box outline visible even where the boundary has no
        // visual wall (e.g. the window's left edge).
        constexpr float kEdgeInsetPx = 6.0f;

        [[nodiscard]] glm::vec2 dragBoundsMin() { return glm::vec2(kEdgeInsetPx); }
        [[nodiscard]] glm::vec2 dragBoundsMax(const DepthWindowPanelMapping& panel) {
            return glm::max(
                glm::vec2(panel.render_width, panel.render_height) - kEdgeInsetPx,
                dragBoundsMin());
        }

        struct GrowDirection {
            bool positive_x;
            bool positive_y;
        };

        enum class MinimumWindowPolicy {
            LegacyEdge,
            PreserveAnchor,
        };

        [[nodiscard]] DepthWindowRect growToMinimumWindow(
            DepthWindowRect rect,
            const glm::vec2& anchor,
            const GrowDirection dir,
            const DepthWindowPanelMapping& panel,
            const std::optional<float> aspect_px,
            const MinimumWindowPolicy policy) {
            // Match the sanitizer's 5% floor in the drag geometry whenever the
            // bounds permit, so scale and offset are derived from the same
            // rectangle. The neither-side-fits fallback may still rely on the
            // sanitizer.
            glm::vec2 minimum{
                0.05f * static_cast<float>(panel.render_width),
                0.05f * static_cast<float>(panel.render_height),
            };
            if (aspect_px) {
                minimum.x = std::max(minimum.x, minimum.y * *aspect_px);
                minimum.y = minimum.x / *aspect_px;
            }

            const glm::vec2 bounds_lo = dragBoundsMin();
            const glm::vec2 bounds_hi = dragBoundsMax(panel);
            if (policy == MinimumWindowPolicy::LegacyEdge) {
                for (int axis = 0; axis < 2; ++axis) {
                    const float deficit = minimum[axis] - (rect.max[axis] - rect.min[axis]);
                    if (deficit <= 0.0f) {
                        continue;
                    }
                    const bool positive = axis == 0 ? dir.positive_x : dir.positive_y;
                    if (positive) {
                        rect.max[axis] = std::min(rect.max[axis] + deficit, bounds_hi[axis]);
                        rect.min[axis] = rect.max[axis] - minimum[axis];
                    } else {
                        rect.min[axis] = std::max(rect.min[axis] - deficit, bounds_lo[axis]);
                        rect.max[axis] = rect.min[axis] + minimum[axis];
                    }
                }
                return rect;
            }

            const auto available = [&](const int axis, const bool positive) {
                return std::max(
                    positive ? bounds_hi[axis] - anchor[axis]
                             : anchor[axis] - bounds_lo[axis],
                    0.0f);
            };
            const auto set_axis = [&](DepthWindowRect& out,
                                      const int axis,
                                      const bool positive,
                                      const float extent) {
                if (positive) {
                    out.min[axis] = anchor[axis];
                    out.max[axis] = anchor[axis] + extent;
                } else {
                    out.min[axis] = anchor[axis] - extent;
                    out.max[axis] = anchor[axis];
                }
            };

            if (!aspect_px) {
                for (int axis = 0; axis < 2; ++axis) {
                    if (rect.max[axis] - rect.min[axis] >= minimum[axis]) {
                        continue;
                    }
                    const bool requested = axis == 0 ? dir.positive_x : dir.positive_y;
                    const float requested_available = available(axis, requested);
                    const float flipped_available = available(axis, !requested);
                    if (requested_available >= minimum[axis]) {
                        set_axis(rect, axis, requested, minimum[axis]);
                    } else if (flipped_available >= minimum[axis]) {
                        set_axis(rect, axis, !requested, minimum[axis]);
                    } else {
                        const bool positive = requested_available >= flipped_available
                                                  ? requested
                                                  : !requested;
                        set_axis(rect, axis, positive,
                                 std::min(minimum[axis], available(axis, positive)));
                    }
                }
                return rect;
            }

            if (rect.size().x >= minimum.x && rect.size().y >= minimum.y) {
                return rect;
            }

            GrowDirection selected = dir;
            const bool x_has_side = available(0, dir.positive_x) >= minimum.x ||
                                    available(0, !dir.positive_x) >= minimum.x;
            const bool y_has_side = available(1, dir.positive_y) >= minimum.y ||
                                    available(1, !dir.positive_y) >= minimum.y;
            float scale = 1.0f;
            if (x_has_side && y_has_side) {
                if (available(0, selected.positive_x) < minimum.x) {
                    selected.positive_x = !selected.positive_x;
                }
                if (available(1, selected.positive_y) < minimum.y) {
                    selected.positive_y = !selected.positive_y;
                }
            } else {
                const GrowDirection candidates[] = {
                    dir,
                    {!dir.positive_x, dir.positive_y},
                    {dir.positive_x, !dir.positive_y},
                    {!dir.positive_x, !dir.positive_y},
                };
                scale = -1.0f;
                for (const auto candidate : candidates) {
                    const float candidate_scale = std::min({
                        1.0f,
                        available(0, candidate.positive_x) / minimum.x,
                        available(1, candidate.positive_y) / minimum.y,
                    });
                    if (candidate_scale > scale) {
                        scale = candidate_scale;
                        selected = candidate;
                    }
                }
            }
            set_axis(rect, 0, selected.positive_x, minimum.x * scale);
            set_axis(rect, 1, selected.positive_y, minimum.y * scale);
            return rect;
        }

        DepthWindowOverlayState g_overlay_state;
        std::uint64_t g_overlay_revision = 0;
        std::uint64_t g_depth_drag_revision = 0;

        [[nodiscard]] tools::SelectionTool* activeDepthWindowTool() {
            auto* const gui = services().guiOrNull();
            auto* const viewer = gui ? gui->getViewer() : nullptr;
            auto* const tool = viewer ? viewer->getSelectionTool() : nullptr;
            return tool && tool->isEnabled() ? tool : nullptr;
        }

        template <typename PanelInfo>
        [[nodiscard]] DepthWindowPanelMapping toPanelMapping(const PanelInfo& panel) {
            return {
                .panel = panel.panel,
                .x = panel.x,
                .y = panel.y,
                .width = panel.width,
                .height = panel.height,
                .render_width = panel.render_width,
                .render_height = panel.render_height,
            };
        }

        [[nodiscard]] std::optional<DepthWindowPanelMapping> resolveDepthWindowPanel(
            const glm::vec2 screen,
            const glm::vec4 viewport_bounds) {
            auto* const gui = services().guiOrNull();
            auto* const viewer = gui ? gui->getViewer() : nullptr;
            auto* const rendering = services().renderingOrNull();
            if (!viewer || !rendering || viewport_bounds.z <= 0.0f || viewport_bounds.w <= 0.0f) {
                return std::nullopt;
            }

            const auto panel = rendering->resolveViewerPanel(
                viewer->getViewport(),
                {viewport_bounds.x, viewport_bounds.y},
                {viewport_bounds.z, viewport_bounds.w},
                screen);
            return panel && panel->valid()
                       ? std::optional(toPanelMapping(*panel))
                       : std::nullopt;
        }

        void setOverlayState(const DepthWindowOverlayState& state) {
            const bool changed = g_overlay_state.visible != state.visible ||
                                 g_overlay_state.has_hovered_panel != state.has_hovered_panel ||
                                 g_overlay_state.hide_handles != state.hide_handles ||
                                 g_overlay_state.hovered_panel != state.hovered_panel ||
                                 g_overlay_state.hovered_handle != state.hovered_handle;
            g_overlay_state = state;
            ++g_overlay_revision;
            if (changed) {
                if (auto* const rendering = services().renderingOrNull()) {
                    rendering->markDirty(DirtyFlag::OVERLAY);
                }
            }
        }

        [[nodiscard]] DepthWindowSettingsState captureDepthWindowSettings(
            const RenderSettings& settings) {
            return {
                .scale_x = settings.depth_filter_scale_x,
                .scale_y = settings.depth_filter_scale_y,
                .offset_x = settings.depth_filter_offset_x,
                .offset_y = settings.depth_filter_offset_y,
            };
        }

        [[nodiscard]] bool isShiftOrAltKey(const int key) {
            return key == input::KEY_LEFT_SHIFT || key == input::KEY_RIGHT_SHIFT ||
                   key == input::KEY_LEFT_ALT || key == input::KEY_RIGHT_ALT;
        }

        [[nodiscard]] bool isControlKey(const int key) {
            return key == input::KEY_LEFT_CONTROL || key == input::KEY_RIGHT_CONTROL;
        }

        class DepthWindowDragOperator final : public Operator {
        public:
            static const OperatorDescriptor DESCRIPTOR;

            // The registry may destroy a modal operator without calling cancel().
            // Restore must precede latch release because the latch re-derives the
            // filter volume, and only this drag's still-owned state is restored.
            ~DepthWindowDragOperator() override {
                const bool terminated_mid_drag = latch_active_;
                if (terminated_mid_drag) {
                    try {
                        restoreBeforeStateIfStillOurs();
                    } catch (...) {
                        // A failed restore must not skip latch cleanup. This does
                        // not make finishLatch()'s pre-existing update path safe.
                        LOG_WARN("Depth window restore failed during operator teardown");
                    }
                }
                finishLatch();
                if ((terminated_mid_drag || modal_active_) &&
                    g_overlay_revision == overlay_revision_) {
                    clearDepthWindowHover();
                }
            }

            [[nodiscard]] const OperatorDescriptor& descriptor() const override { return DESCRIPTOR; }
            [[nodiscard]] bool poll(const OperatorContext& ctx,
                                    const OperatorProperties* props = nullptr) const override;
            OperatorResult invoke(OperatorContext& ctx, OperatorProperties& props) override;
            OperatorResult modal(OperatorContext& ctx, OperatorProperties& props) override;
            void cancel(OperatorContext& ctx) override;

        private:
            enum class DragKind : std::uint8_t {
                Draw,
                Resize,
                Edge,
                Move,
            };

            [[nodiscard]] DepthWindowRect deriveCornerRect(const glm::vec2& pointer) const;
            [[nodiscard]] DepthWindowRect deriveMoveRect(const glm::vec2& pointer) const;
            [[nodiscard]] DepthWindowRect deriveEdgeRect(const glm::vec2& pointer) const;
            void updateFromScreen(const glm::vec2& screen);
            void applyRect(const DepthWindowRect& rect);
            void restoreBeforeState();
            void restoreBeforeStateIfStillOurs();
            void finishLatch();
            void refreshOverlay();

            RenderingManager* rendering_manager_ = nullptr;
            tools::SelectionTool* selection_tool_ = nullptr;
            DepthWindowPanelMapping panel_{};
            glm::vec4 viewport_bounds_{0.0f};
            DepthWindowSettingsState before_{};
            DepthWindowSettingsState applied_{};
            DepthWindowRect initial_rect_{};
            DepthWindowRect current_rect_{};
            glm::vec2 press_render_{0.0f};
            glm::vec2 anchor_render_{0.0f};
            glm::vec2 last_screen_{0.0f};
            float aspect_px_ = 1.0f;
            int drag_button_ = static_cast<int>(input::AppMouseButton::LEFT);
            int current_modifiers_ = kRequiredModifiers;
            DragKind drag_kind_ = DragKind::Draw;
            DepthWindowHandle active_handle_ = DepthWindowHandle::None;
            bool constrained_ = false;
            bool latch_active_ = false;
            bool modal_active_ = false;
            std::uint64_t overlay_revision_ = 0;
            std::uint64_t drag_revision_ = 0;
        };

        const OperatorDescriptor DepthWindowDragOperator::DESCRIPTOR = {
            .builtin_id = BuiltinOp::DepthWindowDrag,
            .python_class_id = {},
            .label = "Drag Depth Window",
            .description = "Draw, resize, or move the selection depth window",
            .icon = "selection",
            .shortcut = "",
            .flags = OperatorFlags::REGISTER | OperatorFlags::UNDO,
            .source = OperatorSource::CPP,
            .poll_deps = PollDependency::ALL,
        };

        bool DepthWindowDragOperator::poll(const OperatorContext& /*ctx*/,
                                           const OperatorProperties* /*props*/) const {
            const auto* const tool = activeDepthWindowTool();
            auto* const rendering = services().renderingOrNull();
            return tool && tool->isDepthFilterEnabled() && rendering &&
                   !rendering->isGTComparisonActive();
        }

        OperatorResult DepthWindowDragOperator::invoke(OperatorContext& /*ctx*/,
                                                       OperatorProperties& props) {
            selection_tool_ = activeDepthWindowTool();
            rendering_manager_ = services().renderingOrNull();
            if (!selection_tool_ || !selection_tool_->isDepthFilterEnabled() || !rendering_manager_) {
                return OperatorResult::CANCELLED;
            }

            last_screen_ = {
                static_cast<float>(props.get_or<double>("x", 0.0)),
                static_cast<float>(props.get_or<double>("y", 0.0)),
            };
            viewport_bounds_ = {
                props.get_or<float>("viewport_x", 0.0f),
                props.get_or<float>("viewport_y", 0.0f),
                props.get_or<float>("viewport_width", 0.0f),
                props.get_or<float>("viewport_height", 0.0f),
            };
            const auto panel = resolveDepthWindowPanel(last_screen_, viewport_bounds_);
            if (!panel) {
                return OperatorResult::CANCELLED;
            }
            panel_ = *panel;

            drag_button_ = props.get_or<int>(
                "button", static_cast<int>(input::AppMouseButton::LEFT));
            current_modifiers_ = props.get_or<int>("modifiers", kRequiredModifiers);
            if ((current_modifiers_ & kRequiredModifiers) != kRequiredModifiers) {
                return OperatorResult::CANCELLED;
            }
            constrained_ = (current_modifiers_ & input::KEYMOD_CTRL) != 0;

            const auto settings = rendering_manager_->getSettings();
            before_ = captureDepthWindowSettings(settings);
            initial_rect_ = depthWindowRenderRect(
                panel_.render_width, panel_.render_height,
                before_.scale_x, before_.scale_y,
                before_.offset_x, before_.offset_y);
            current_rect_ = initial_rect_;
            press_render_ = glm::clamp(
                screenToRender(last_screen_, panel_),
                dragBoundsMin(),
                dragBoundsMax(panel_));

            const auto screen_rect = depthWindowScreenRect(
                panel_, before_.scale_x, before_.scale_y,
                before_.offset_x, before_.offset_y);
            active_handle_ = hitTestDepthWindowHandles(
                last_screen_, depthWindowHandleGeometry(screen_rect));
            if (active_handle_ == DepthWindowHandle::Center) {
                drag_kind_ = DragKind::Move;
            } else if (active_handle_ != DepthWindowHandle::None) {
                drag_kind_ = DragKind::Resize;
                switch (active_handle_) {
                case DepthWindowHandle::TopLeft: anchor_render_ = initial_rect_.max; break;
                case DepthWindowHandle::TopRight:
                    anchor_render_ = {initial_rect_.min.x, initial_rect_.max.y};
                    break;
                case DepthWindowHandle::BottomRight: anchor_render_ = initial_rect_.min; break;
                case DepthWindowHandle::BottomLeft:
                    anchor_render_ = {initial_rect_.max.x, initial_rect_.min.y};
                    break;
                case DepthWindowHandle::EdgeTop:
                case DepthWindowHandle::EdgeRight:
                case DepthWindowHandle::EdgeBottom:
                case DepthWindowHandle::EdgeLeft:
                    drag_kind_ = DragKind::Edge;
                    break;
                case DepthWindowHandle::Center:
                case DepthWindowHandle::None: break;
                }
            } else {
                drag_kind_ = DragKind::Draw;
                anchor_render_ = press_render_;
            }

            const float initial_width = std::abs(before_.scale_x) * panel_.render_width;
            const float initial_height = std::abs(before_.scale_y) * panel_.render_height;
            const float viewport_aspect = static_cast<float>(panel_.render_width) /
                                          static_cast<float>(panel_.render_height);
            aspect_px_ = initial_height > kSingularityEpsilon
                             ? initial_width / initial_height
                             : viewport_aspect;
            if (!std::isfinite(aspect_px_) || aspect_px_ <= kSingularityEpsilon) {
                aspect_px_ = viewport_aspect;
            }
            // Ctrl held from the onset of a fresh draw means a perfect
            // screen-pixel square. (Ctrl pressed mid-drag instead captures the
            // live rect's ratio; resize always uses the shape being resized.)
            if (constrained_ && drag_kind_ == DragKind::Draw) {
                aspect_px_ = 1.0f;
            }

            selection_tool_->setDepthWindowDragInProgress(true);
            latch_active_ = true;
            if (rendering_manager_) {
                rendering_manager_->setDepthWindowDragPreview(true);
            }
            if (drag_kind_ == DragKind::Draw) {
                updateFromScreen(last_screen_);
            } else {
                refreshOverlay();
            }
            drag_revision_ = ++g_depth_drag_revision;
            modal_active_ = true;
            return OperatorResult::RUNNING_MODAL;
        }

        DepthWindowRect DepthWindowDragOperator::deriveCornerRect(
            const glm::vec2& pointer) const {
            const GrowDirection direction{
                .positive_x = pointer.x - anchor_render_.x >= 0.0f,
                .positive_y = pointer.y - anchor_render_.y >= 0.0f,
            };
            if (!constrained_) {
                return growToMinimumWindow(
                    {
                        .min = glm::min(anchor_render_, pointer),
                        .max = glm::max(anchor_render_, pointer),
                    },
                    anchor_render_, direction, panel_, std::nullopt,
                    MinimumWindowPolicy::PreserveAnchor);
            }

            const glm::vec2 delta = pointer - anchor_render_;
            const glm::vec2 signs{
                delta.x < 0.0f ? -1.0f : 1.0f,
                delta.y < 0.0f ? -1.0f : 1.0f,
            };
            float constrained_width = std::max(std::abs(delta.x),
                                               std::abs(delta.y) * aspect_px_);
            float constrained_height = constrained_width / aspect_px_;

            // The constrained corner grows from the anchor in the pointer's
            // quadrant. If it reaches a viewport edge, both dimensions shrink by
            // the same factor so the drag-start pixel aspect remains exact.
            const glm::vec2 bounds_lo = dragBoundsMin();
            const glm::vec2 bounds_hi = dragBoundsMax(panel_);
            const float available_width = signs.x > 0.0f
                                              ? bounds_hi.x - anchor_render_.x
                                              : anchor_render_.x - bounds_lo.x;
            const float available_height = signs.y > 0.0f
                                               ? bounds_hi.y - anchor_render_.y
                                               : anchor_render_.y - bounds_lo.y;
            float shrink = 1.0f;
            if (constrained_width > kSingularityEpsilon) {
                shrink = std::min(shrink, available_width / constrained_width);
            }
            if (constrained_height > kSingularityEpsilon) {
                shrink = std::min(shrink, available_height / constrained_height);
            }
            shrink = std::clamp(shrink, 0.0f, 1.0f);
            constrained_width *= shrink;
            constrained_height *= shrink;

            const glm::vec2 corner = anchor_render_ +
                                     signs * glm::vec2(constrained_width, constrained_height);
            return growToMinimumWindow(
                {
                    .min = glm::min(anchor_render_, corner),
                    .max = glm::max(anchor_render_, corner),
                },
                anchor_render_, direction, panel_, aspect_px_,
                MinimumWindowPolicy::PreserveAnchor);
        }

        DepthWindowRect DepthWindowDragOperator::deriveMoveRect(
            const glm::vec2& pointer) const {
            glm::vec2 delta = pointer - press_render_;
            const glm::vec2 lo = dragBoundsMin();
            const glm::vec2 hi = dragBoundsMax(panel_);
            delta.x = std::clamp(delta.x,
                                 lo.x - initial_rect_.min.x,
                                 hi.x - initial_rect_.max.x);
            delta.y = std::clamp(delta.y,
                                 lo.y - initial_rect_.min.y,
                                 hi.y - initial_rect_.max.y);
            return {
                .min = initial_rect_.min + delta,
                .max = initial_rect_.max + delta,
            };
        }

        // Single-face adjust: the dragged edge follows the pointer, the other
        // three stay put; crossing the opposite edge inverts cleanly via the
        // min/max normalization. The Ctrl ratio constraint deliberately does
        // not apply to edge drags (one-dimensional by definition).
        DepthWindowRect DepthWindowDragOperator::deriveEdgeRect(
            const glm::vec2& pointer) const {
            DepthWindowRect rect = initial_rect_;
            switch (active_handle_) {
            case DepthWindowHandle::EdgeTop: rect.min.y = pointer.y; break;
            case DepthWindowHandle::EdgeRight: rect.max.x = pointer.x; break;
            case DepthWindowHandle::EdgeBottom: rect.max.y = pointer.y; break;
            case DepthWindowHandle::EdgeLeft: rect.min.x = pointer.x; break;
            default: break;
            }
            DepthWindowRect out{
                .min = glm::min(rect.min, rect.max),
                .max = glm::max(rect.min, rect.max),
            };
            const glm::vec2 initial_center = initial_rect_.center();
            return growToMinimumWindow(
                out, {}, {
                             .positive_x = pointer.x >= initial_center.x,
                             .positive_y = pointer.y >= initial_center.y,
                         },
                panel_, std::nullopt, MinimumWindowPolicy::LegacyEdge);
        }

        void DepthWindowDragOperator::applyRect(const DepthWindowRect& rect) {
            if (!rendering_manager_) {
                return;
            }

            auto settings = rendering_manager_->getSettings();
            const glm::vec2 size = glm::clamp(
                rect.size(), glm::vec2(0.0f),
                glm::vec2(panel_.render_width, panel_.render_height));
            const glm::vec2 center = rect.center();
            const float scale_x = size.x / static_cast<float>(panel_.render_width);
            const float scale_y = size.y / static_cast<float>(panel_.render_height);
            const auto offset_for_axis = [](const float center_px,
                                            const float dimension,
                                            const float scale) {
                if (std::abs(1.0f - scale) <= kSingularityEpsilon) {
                    return 0.0f;
                }
                return std::clamp(
                    (2.0f * center_px / dimension - 1.0f) / (1.0f - scale),
                    -1.0f, 1.0f);
            };

            settings.depth_filter_scale_x = scale_x;
            settings.depth_filter_scale_y = scale_y;
            settings.depth_filter_offset_x = offset_for_axis(
                center.x, static_cast<float>(panel_.render_width), scale_x);
            settings.depth_filter_offset_y = offset_for_axis(
                center.y, static_cast<float>(panel_.render_height), scale_y);
            rendering_manager_->updateSettings(settings, DirtyFlag::SELECTION);
            applied_ = captureDepthWindowSettings(rendering_manager_->getSettings());
        }

        void DepthWindowDragOperator::updateFromScreen(const glm::vec2& screen) {
            last_screen_ = screen;
            const glm::vec2 pointer = glm::clamp(
                screenToRender(screen, panel_),
                dragBoundsMin(),
                dragBoundsMax(panel_));
            current_rect_ = drag_kind_ == DragKind::Move   ? deriveMoveRect(pointer)
                            : drag_kind_ == DragKind::Edge ? deriveEdgeRect(pointer)
                                                           : deriveCornerRect(pointer);
            applyRect(current_rect_);
            refreshOverlay();
        }

        void DepthWindowDragOperator::finishLatch() {
            if (!latch_active_) {
                return;
            }
            latch_active_ = false;
            if (selection_tool_) {
                selection_tool_->setDepthWindowDragInProgress(false);
            }
            if (rendering_manager_) {
                rendering_manager_->setDepthWindowDragPreview(false);
            }
        }

        void DepthWindowDragOperator::restoreBeforeState() {
            if (rendering_manager_) {
                auto settings = rendering_manager_->getSettings();
                settings.depth_filter_scale_x = before_.scale_x;
                settings.depth_filter_scale_y = before_.scale_y;
                settings.depth_filter_offset_x = before_.offset_x;
                settings.depth_filter_offset_y = before_.offset_y;
                rendering_manager_->updateSettings(settings, DirtyFlag::SELECTION);
            }
        }

        void DepthWindowDragOperator::restoreBeforeStateIfStillOurs() {
            if (!rendering_manager_ || g_depth_drag_revision != drag_revision_) {
                return;
            }
            const auto settings = rendering_manager_->getSettings();
            if (captureDepthWindowSettings(settings) != applied_) {
                return;
            }
            restoreBeforeState();
        }

        void DepthWindowDragOperator::refreshOverlay() {
            setOverlayState({
                .visible = (current_modifiers_ & kRequiredModifiers) == kRequiredModifiers,
                .has_hovered_panel = true,
                // Handles only apply to a committed window: hide them while a
                // new box is being rubber-banded.
                .hide_handles = drag_kind_ == DragKind::Draw,
                .hovered_panel = panel_.panel,
                .hovered_handle = active_handle_,
            });
            overlay_revision_ = g_overlay_revision;
        }

        OperatorResult DepthWindowDragOperator::modal(OperatorContext& ctx,
                                                      OperatorProperties& /*props*/) {
            if (!rendering_manager_ || !selection_tool_) {
                return OperatorResult::CANCELLED;
            }

            const auto* const event = ctx.event();
            if (!event) {
                return OperatorResult::RUNNING_MODAL;
            }

            if (rendering_manager_->isGTComparisonActive()) {
                cancel(ctx);
                return OperatorResult::CANCELLED;
            }

            if (event->type == ModalEvent::Type::MOUSE_MOVE) {
                const auto* const move = event->as<MouseMoveEvent>();
                if (move) {
                    updateFromScreen(glm::vec2(move->position));
                }
                return OperatorResult::RUNNING_MODAL;
            }

            if (event->type == ModalEvent::Type::MOUSE_BUTTON) {
                const auto* const mouse_button = event->as<MouseButtonEvent>();
                if (!mouse_button || mouse_button->button != drag_button_) {
                    return OperatorResult::PASS_THROUGH;
                }
                current_modifiers_ = mouse_button->mods;
                if (mouse_button->action != input::ACTION_RELEASE) {
                    return OperatorResult::RUNNING_MODAL;
                }

                const glm::vec2 release_screen(mouse_button->position);
                if (pointInDepthWindowPanel(release_screen, panel_)) {
                    updateFromScreen(release_screen);
                }
                finishLatch();

                const auto after = captureDepthWindowSettings(
                    rendering_manager_->getSettings());
                if (after != before_) {
                    undoHistory().push(std::make_unique<DepthWindowSettingsUndoEntry>(
                        *rendering_manager_, before_, after,
                        drag_kind_ == DragKind::Draw));
                }
                if (drag_kind_ == DragKind::Draw) {
                    publish_depth_window_draw_commit();
                }
                if (auto* const selection = ctx.scene().getSelectionService()) {
                    selection->invalidateInteractiveBrushFilterCache();
                }
                modal_active_ = false;
                (void)updateDepthWindowHover(
                    release_screen, viewport_bounds_,
                    (current_modifiers_ & kRequiredModifiers) == kRequiredModifiers);
                return OperatorResult::FINISHED;
            }

            if (event->type == ModalEvent::Type::KEY) {
                const auto* const key = event->as<KeyEvent>();
                if (!key) {
                    return OperatorResult::RUNNING_MODAL;
                }
                current_modifiers_ = key->mods;
                if (key->action == input::ACTION_PRESS && key->key == input::KEY_ESCAPE) {
                    return OperatorResult::CANCELLED;
                }
                if (key->action == input::ACTION_RELEASE && isShiftOrAltKey(key->key) &&
                    (current_modifiers_ & kRequiredModifiers) != kRequiredModifiers) {
                    return OperatorResult::CANCELLED;
                }
                if (isControlKey(key->key) &&
                    (key->action == input::ACTION_PRESS ||
                     key->action == input::ACTION_RELEASE)) {
                    const bool was_constrained = constrained_;
                    constrained_ = (current_modifiers_ & input::KEYMOD_CTRL) != 0;
                    // On a fresh draw, Ctrl locks the shape being dragged right
                    // now, not the previous window — re-capture the live rect's
                    // ratio on every Ctrl press. Resize keeps the drag-start
                    // ratio (the shape being resized), where the two
                    // definitions coincide.
                    if (constrained_ && !was_constrained &&
                        drag_kind_ == DragKind::Draw) {
                        const float live_w = current_rect_.max.x - current_rect_.min.x;
                        const float live_h = current_rect_.max.y - current_rect_.min.y;
                        if (live_h > kSingularityEpsilon) {
                            const float live_aspect = live_w / live_h;
                            if (std::isfinite(live_aspect) &&
                                live_aspect > kSingularityEpsilon) {
                                aspect_px_ = live_aspect;
                            }
                        }
                    }
                    updateFromScreen(last_screen_);
                    return OperatorResult::RUNNING_MODAL;
                }
                return OperatorResult::PASS_THROUGH;
            }

            // All scroll is reserved and ignored for this modal: scroll-to-size
            // would write the window behind the drag's ownership tracking.
            if (event->type == ModalEvent::Type::MOUSE_SCROLL) {
                return OperatorResult::RUNNING_MODAL;
            }

            return OperatorResult::PASS_THROUGH;
        }

        void DepthWindowDragOperator::cancel(OperatorContext& /*ctx*/) {
            modal_active_ = false;
            restoreBeforeState();
            finishLatch();
            if ((current_modifiers_ & kRequiredModifiers) == kRequiredModifiers) {
                (void)updateDepthWindowHover(last_screen_, viewport_bounds_, true);
            } else {
                clearDepthWindowHover();
            }
        }

    } // namespace

    DepthWindowCursor updateDepthWindowHover(const glm::vec2& screen,
                                             const glm::vec4& viewport_bounds,
                                             const bool modifiers_held) {
        const auto* const tool = activeDepthWindowTool();
        auto* const rendering = services().renderingOrNull();
        if (!modifiers_held || !tool || !tool->isDepthFilterEnabled() || !rendering) {
            clearDepthWindowHover();
            return DepthWindowCursor::Default;
        }
        if (rendering->isGTComparisonActive()) {
            clearDepthWindowHover();
            return DepthWindowCursor::Default;
        }

        const auto panel = resolveDepthWindowPanel(screen, viewport_bounds);
        if (!panel) {
            setOverlayState({.visible = true});
            return DepthWindowCursor::Default;
        }

        const auto settings = rendering->getSettings();
        const auto screen_rect = depthWindowScreenRect(
            *panel,
            settings.depth_filter_scale_x,
            settings.depth_filter_scale_y,
            settings.depth_filter_offset_x,
            settings.depth_filter_offset_y);
        const auto handle = hitTestDepthWindowHandles(
            screen, depthWindowHandleGeometry(screen_rect));
        setOverlayState({
            .visible = true,
            .has_hovered_panel = true,
            .hovered_panel = panel->panel,
            .hovered_handle = handle,
        });
        const auto cursor = depthWindowCursorForHandle(handle);
        return cursor == DepthWindowCursor::Default ? DepthWindowCursor::Crosshair : cursor;
    }

    void clearDepthWindowHover() {
        setOverlayState({});
    }

    const DepthWindowOverlayState& depthWindowOverlayState() {
        return g_overlay_state;
    }

    void registerDepthWindowOperators() {
        operators().registerOperator(
            BuiltinOp::DepthWindowDrag,
            DepthWindowDragOperator::DESCRIPTOR,
            [] { return std::make_unique<DepthWindowDragOperator>(); });
    }

    void unregisterDepthWindowOperators() {
        operators().unregisterOperator(BuiltinOp::DepthWindowDrag);
    }

} // namespace lfs::vis::op
