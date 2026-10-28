/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "crop_polygon_ops.hpp"
#include "core/logger.hpp"
#include "core/property_registry.hpp"
#include "core/services.hpp"
#include "core/transform_utils.hpp"
#include "gui/gui_manager.hpp"
#include "input/key_codes.hpp"
#include "internal/viewport.hpp"
#include "operator/operator_registry.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "selection/selection_service.hpp"
#include "tools/unified_tool_registry.hpp"
#include "visualizer/gui_capabilities.hpp"
#include "visualizer_impl.hpp"
#include <array>

namespace lfs::vis::op {

    namespace crop_polygon = core::crop_polygon;

    namespace {
        // Two unprojections of the same pixel give the view ray for both projections.
        constexpr float kRayNearDepth = 1e-3f;
        constexpr const char* kCropToolId = "builtin.cropbox";

        struct PanelView {
            Viewport viewport;
            SelectionService::ViewportInfo info;
            SplitViewPanelId panel = SplitViewPanelId::Left;
            glm::vec2 render_point{0.0f};
            float focal_length_mm = 0.0f;
            bool orthographic = false;
            float ortho_scale = 1.0f;
        };

        [[nodiscard]] std::optional<PanelView> resolvePanel(const glm::vec2 screen_point) {
            auto* const rm = services().renderingOrNull();
            auto* const gm = services().guiOrNull();
            if (!rm || !gm || !gm->getViewer())
                return std::nullopt;
            const auto panel = rm->resolveViewerPanel(gm->getViewer()->getViewport(), gm->getViewportPos(),
                                                      gm->getViewportSize(), screen_point);
            if (!panel || !panel->valid())
                return std::nullopt;

            const auto settings = rm->getSettings();
            PanelView view{
                .viewport = *panel->viewport,
                .info = {.x = panel->x,
                         .y = panel->y,
                         .width = panel->width,
                         .height = panel->height,
                         .render_width = panel->render_width,
                         .render_height = panel->render_height},
                .panel = panel->panel,
                .render_point = (screen_point - glm::vec2(panel->x, panel->y)) *
                                glm::vec2(static_cast<float>(panel->render_width) / panel->width,
                                          static_cast<float>(panel->render_height) / panel->height),
                .focal_length_mm = settings.focal_length_mm,
                .orthographic = settings.orthographic,
                .ortho_scale = 1.0f,
            };
            view.viewport.windowSize = {panel->render_width, panel->render_height};
            view.ortho_scale = view.viewport.ortho_scale_override.value_or(settings.ortho_scale);
            return view;
        }

        [[nodiscard]] std::optional<crop_polygon::Ray> pixelRay(const PanelView& view) {
            const auto unproject = [&](const float depth) {
                return view.viewport.unprojectPixel(view.render_point.x, view.render_point.y, depth,
                                                    view.focal_length_mm, view.orthographic, view.ortho_scale);
            };
            const glm::vec3 near_point = unproject(kRayNearDepth);
            const glm::vec3 far_point = unproject(kRayNearDepth + 1.0f);
            if (!Viewport::isValidWorldPosition(near_point) || !Viewport::isValidWorldPosition(far_point))
                return std::nullopt;
            return crop_polygon::rayThroughPoints(near_point, far_point);
        }

        [[nodiscard]] std::optional<glm::vec2> projectToScreen(const PanelView& view, const glm::vec3& world) {
            const auto projected = rendering::projectWorldPoint(view.viewport.camera.R, view.viewport.camera.t,
                                                                view.viewport.windowSize, world, view.focal_length_mm,
                                                                view.orthographic, view.ortho_scale);
            if (!projected)
                return std::nullopt;
            return glm::vec2(view.info.x + projected->x * view.info.width / static_cast<float>(view.info.render_width),
                             view.info.y + projected->y * view.info.height / static_cast<float>(view.info.render_height));
        }

        [[nodiscard]] bool polygonCropToolActive() {
            const auto* const gui = services().guiOrNull();
            return UnifiedToolRegistry::instance().getActiveTool() == kCropToolId && gui &&
                   gui->gizmo().cropToolShape() == "polygon";
        }

    } // namespace

    const OperatorDescriptor CropPolygonDrawOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::CropPolygonDraw,
        .python_class_id = {},
        .label = "Draw Crop Polygon",
        .description = "Click points on a camera-facing plane to outline a crop volume",
        .icon = "polygon",
        .shortcut = "",
        .flags = OperatorFlags::REGISTER,
        .source = OperatorSource::CPP,
        .poll_deps = PollDependency::SCENE,
    };

    bool CropPolygonDrawOperator::poll(const OperatorContext& ctx, const OperatorProperties* props) const {
        const auto target = props ? props->get_or<int>("target_node_id", -1) : -1;
        const auto* const node = ctx.scene().getScene().getNodeById(static_cast<core::NodeId>(target));
        return node && isCropPolygonTarget(node->type);
    }

    OperatorResult CropPolygonDrawOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        target_node_id_ = static_cast<core::NodeId>(props.get_or<int>("target_node_id", -1));
        plane_.reset();
        points_.clear();
        cursor_.reset();
        clearDraft();

        // Remember what entering the drawing changes, so a cancel restores it.
        auto& scene_manager = ctx.scene();
        const auto& scene = scene_manager.getScene();
        auto* const editor = services().editorOrNull();
        start_ = StartState{
            .selection = scene_manager.getSelectedNodeIds(),
            .volume_visibility = {},
            .active_tool = UnifiedToolRegistry::instance().getActiveTool(),
            .editor_operator = editor ? editor->getActiveOperator() : std::string{},
            .editor_gizmo = editor ? editor->getGizmoType() : std::string{},
        };
        if (const auto* const target = scene.getNodeById(target_node_id_)) {
            for (const core::NodeId child_id : target->children) {
                const auto* const child = scene.getNodeById(child_id);
                if (child && (child->type == core::NodeType::CROPBOX || child->type == core::NodeType::ELLIPSOID ||
                              child->type == core::NodeType::CROP_POLYGON))
                    start_.volume_visibility.emplace_back(child_id, child->visible.get());
            }
        }

        // Selecting the splat hides its other crop volume, so no gizmo competes for the clicks.
        if (start_.selection != std::vector<core::NodeId>{target_node_id_})
            scene_manager.selectNode(target_node_id_);
        if (editor)
            editor->setActiveOperator(kCropToolId, "translate");
        UnifiedToolRegistry::instance().setActiveTool(kCropToolId);
        if (auto* const gui = services().guiOrNull())
            gui->gizmo().setCropToolShape("polygon");
        return OperatorResult::RUNNING_MODAL;
    }

    OperatorResult CropPolygonDrawOperator::cancelByUser(OperatorContext& ctx) {
        clearDraft();
        auto& scene_manager = ctx.scene();
        const auto& scene = scene_manager.getScene();
        for (const auto& [id, visible] : start_.volume_visibility) {
            const auto* const node = scene.getNodeById(id);
            if (node && node->visible.get() != visible)
                scene_manager.setNodeVisibilityTransient(id, visible);
        }
        // Reselecting a crop volume reenters the crop tool on its shape.
        if (scene_manager.getSelectedNodeIds() != start_.selection) {
            if (start_.selection.empty())
                scene_manager.clearSelection();
            else if (start_.selection.size() == 1)
                scene_manager.selectNode(start_.selection.front());
            else
                scene_manager.selectNodesById(start_.selection);
        }
        auto& registry = UnifiedToolRegistry::instance();
        if (start_.active_tool.empty())
            registry.clearActiveTool();
        else
            registry.setActiveTool(start_.active_tool);
        if (auto* const editor = services().editorOrNull()) {
            if (start_.editor_operator.empty())
                editor->clearActiveOperator();
            else
                editor->setActiveOperator(start_.editor_operator, start_.editor_gizmo);
        }
        return OperatorResult::CANCELLED;
    }

    OperatorResult CropPolygonDrawOperator::modal(OperatorContext& ctx, OperatorProperties& /*props*/) {
        // Leaving the polygon crop tool or losing the target ends the drawing.
        const auto* const target = ctx.scene().getScene().getNodeById(target_node_id_);
        if (!polygonCropToolActive() || !target || !isCropPolygonTarget(target->type))
            return OperatorResult::CANCELLED;

        const auto* event = ctx.event();
        if (!event)
            return OperatorResult::RUNNING_MODAL;

        switch (event->type) {
        case ModalEvent::Type::MOUSE_MOVE:
            if (const auto* move = event->as<MouseMoveEvent>())
                updateCursor(glm::vec2(move->position));
            // Camera navigation keeps working while drawing.
            return OperatorResult::PASS_THROUGH;

        case ModalEvent::Type::MOUSE_BUTTON: {
            const auto* button = event->as<MouseButtonEvent>();
            if (!button || event->over_gui || button->button != static_cast<int>(input::AppMouseButton::LEFT))
                return OperatorResult::PASS_THROUGH;
            if (button->action != input::ACTION_PRESS)
                return OperatorResult::RUNNING_MODAL;
            if (addVertex(glm::vec2(button->position)))
                return finish(ctx) ? OperatorResult::FINISHED : OperatorResult::RUNNING_MODAL;
            return OperatorResult::RUNNING_MODAL;
        }

        case ModalEvent::Type::ACTION: {
            const auto* action = event->as<ActionEvent>();
            if (!action)
                return OperatorResult::RUNNING_MODAL;
            switch (action->action) {
            case input::Action::CONFIRM_POLYGON:
                // Consumed even when incomplete, so the crop tool's own apply key never fires mid-draw.
                if (points_.size() >= crop_polygon::kMinVertices && finish(ctx))
                    return OperatorResult::FINISHED;
                return OperatorResult::RUNNING_MODAL;
            case input::Action::UNDO_POLYGON_VERTEX:
                if (points_.empty())
                    return cancelByUser(ctx);
                points_.pop_back();
                if (points_.empty()) {
                    plane_.reset(); // The next click fixes a new plane.
                    cursor_.reset();
                }
                publishDraft();
                return OperatorResult::RUNNING_MODAL;
            case input::Action::CANCEL_POLYGON:
                return cancelByUser(ctx);
            default:
                return OperatorResult::PASS_THROUGH;
            }
        }

        case ModalEvent::Type::NONE:
            return OperatorResult::RUNNING_MODAL;

        default:
            return OperatorResult::PASS_THROUGH;
        }
    }

    void CropPolygonDrawOperator::cancel(OperatorContext& /*ctx*/) {
        points_.clear();
        plane_.reset();
        cursor_.reset();
        clearDraft();
    }

    bool CropPolygonDrawOperator::nearFirstVertex(const glm::vec2 screen_point) const {
        if (points_.size() < crop_polygon::kMinVertices)
            return false;
        const auto view = resolvePanel(screen_point);
        const auto first = view ? projectToScreen(*view, points_.front()) : std::nullopt;
        return first && glm::distance(*first, screen_point) < POLYGON_CLOSE_DISTANCE_PX;
    }

    bool CropPolygonDrawOperator::addVertex(const glm::vec2 screen_point) {
        if (nearFirstVertex(screen_point))
            return true;
        if (points_.size() >= crop_polygon::kMaxVertices)
            return false;

        const auto view = resolvePanel(screen_point);
        if (!view)
            return false;

        if (!plane_) {
            const auto& camera = view->viewport.camera;
            const glm::vec3 forward = rendering::cameraForward(camera.R);
            std::optional<glm::vec3> first_click;
            if (!crop_polygon::pivotIsInFront(camera.t, forward, camera.pivot)) {
                if (const auto* const rm = services().renderingOrNull())
                    first_click = SelectionService::resolveViewerWorldPoint(*rm, view->viewport, view->panel,
                                                                            view->info, screen_point);
            }
            plane_ = crop_polygon::makeDrawingPlane(camera.t, forward, camera.pivot, first_click);
            camera_right_ = rendering::cameraRight(camera.R);
        }
        const auto ray = plane_ ? pixelRay(*view) : std::nullopt;
        const auto hit = ray ? crop_polygon::intersect(*ray, *plane_) : std::nullopt;
        if (!hit) {
            if (points_.empty())
                plane_.reset();
            return false;
        }
        points_.push_back(*hit);
        cursor_ = points_.back();
        publishDraft();
        return false;
    }

    void CropPolygonDrawOperator::updateCursor(const glm::vec2 screen_point) {
        if (!plane_ || points_.empty())
            return;
        if (nearFirstVertex(screen_point)) {
            cursor_ = points_.front();
        } else {
            const auto view = resolvePanel(screen_point);
            const auto ray = view ? pixelRay(*view) : std::nullopt;
            cursor_ = ray ? crop_polygon::intersect(*ray, *plane_) : std::nullopt;
        }
        publishDraft();
    }

    bool CropPolygonDrawOperator::finish(OperatorContext& ctx) {
        auto& scene_manager = ctx.scene();
        auto& scene = scene_manager.getScene();
        if (!plane_ || points_.size() < crop_polygon::kMinVertices)
            return false;

        const auto frame = crop_polygon::buildFrame(points_, *plane_, camera_right_);
        if (!frame) {
            LOG_WARN("Crop polygon outline is degenerate; add or remove vertices");
            return false;
        }

        const auto local_transform = core::finiteLocalTransform(
            scene.getWorldTransform(target_node_id_),
            rendering::visualizerWorldTransformToDataWorld(frame->world_from_local));
        if (!local_transform) {
            LOG_WARN("Cannot place the crop polygon under a singular parent transform");
            return false;
        }

        core::CropPolygonData data;
        data.vertices = frame->vertices;
        data.half_depth = frame->half_depth;
        const auto polygon_id = cap::createCropPolygon(scene_manager, services().renderingOrNull(), target_node_id_,
                                                       std::move(data), *local_transform);
        if (!polygon_id) {
            LOG_WARN("Failed to create crop polygon: {}", polygon_id.error().user_message());
            return false;
        }
        clearDraft();

        // Selecting the volume enters the crop tool; the bounds handles on the caps extrude it.
        scene_manager.selectNode(*polygon_id);
        if (auto* const editor = services().editorOrNull())
            editor->setActiveOperator(kCropToolId, "scale");
        UnifiedToolRegistry::instance().setActiveTool(kCropToolId);
        if (auto* const gui = services().guiOrNull()) {
            gui->gizmo().setCropToolShape("polygon");
            gui->gizmo().setCropToolOperation("scale");
        }
        return true;
    }

    void CropPolygonDrawOperator::publishDraft() const {
        if (auto* const rm = services().renderingOrNull()) {
            rm->setCropPolygonDraft(points_, cursor_);
            rm->markDirty(DirtyFlag::OVERLAY, FrameReason::Overlay);
        }
    }

    void CropPolygonDrawOperator::clearDraft() const {
        if (auto* const rm = services().renderingOrNull()) {
            rm->setCropPolygonDraft({}, std::nullopt);
            rm->markDirty(DirtyFlag::OVERLAY, FrameReason::Overlay);
        }
    }

    namespace {
        [[nodiscard]] std::optional<std::string> optionalNodeProp(const OperatorProperties* props) {
            if (props) {
                if (const auto value = props->get<std::string>("node"); value && !value->empty())
                    return *value;
            }
            return std::nullopt;
        }

        [[nodiscard]] bool hasCropPolygonUpdateFields(const OperatorProperties* props) {
            return props && (props->has("half_depth") || props->has("inverse") || props->has("enabled"));
        }
    } // namespace

    const OperatorDescriptor CropPolygonSetOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::CropPolygonSet,
        .python_class_id = {},
        .label = "Set Crop Polygon",
        .description = "Update a crop polygon's depth, inversion or enabled state",
        .icon = "",
        .shortcut = "",
        .flags = OperatorFlags::REGISTER | OperatorFlags::UNDO,
        .source = OperatorSource::CPP,
        .poll_deps = PollDependency::SCENE | PollDependency::SELECTION,
    };

    bool CropPolygonSetOperator::poll(const OperatorContext& ctx, const OperatorProperties* props) const {
        return hasCropPolygonUpdateFields(props) && cap::resolveCropPolygonId(ctx.scene(), optionalNodeProp(props));
    }

    OperatorResult CropPolygonSetOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        const auto polygon_id = cap::resolveCropPolygonId(ctx.scene(), optionalNodeProp(&props));
        if (!polygon_id || !hasCropPolygonUpdateFields(&props))
            return OperatorResult::CANCELLED;

        cap::CropPolygonUpdate update;
        if (props.has("half_depth"))
            update.half_depth = props.get_or<float>("half_depth", 0.0f);
        if (props.has("inverse")) {
            update.has_inverse = true;
            update.inverse = props.get_or<bool>("inverse", false);
        }
        if (props.has("enabled")) {
            update.has_enabled = true;
            update.enabled = props.get_or<bool>("enabled", false);
        }
        if (!cap::updateCropPolygon(ctx.scene(), services().renderingOrNull(), *polygon_id, update))
            return OperatorResult::CANCELLED;
        return OperatorResult::FINISHED;
    }

    void registerCropPolygonOperators() {
        core::prop::PropertyRegistry::instance().register_operator_args(
            CropPolygonSetOperator::DESCRIPTOR.id(),
            {
                core::prop::arg_string("node", "Optional crop polygon node or parent node name; defaults to the current selected crop polygon"),
                core::prop::arg_float("half_depth", "Optional half extrusion depth along the polygon's local z axis"),
                core::prop::arg_bool("inverse", "Invert the crop volume"),
                core::prop::arg_bool("enabled", "Enable crop filtering for this crop polygon"),
            });
        operators().registerOperator(BuiltinOp::CropPolygonDraw, CropPolygonDrawOperator::DESCRIPTOR,
                                     [] { return std::make_unique<CropPolygonDrawOperator>(); });
        operators().registerOperator(BuiltinOp::CropPolygonSet, CropPolygonSetOperator::DESCRIPTOR,
                                     [] { return std::make_unique<CropPolygonSetOperator>(); });
    }

    void unregisterCropPolygonOperators() {
        operators().unregisterOperator(BuiltinOp::CropPolygonSet);
        operators().unregisterOperator(BuiltinOp::CropPolygonDraw);
        core::prop::PropertyRegistry::instance().unregister_operator_args(CropPolygonSetOperator::DESCRIPTOR.id());
    }

} // namespace lfs::vis::op
