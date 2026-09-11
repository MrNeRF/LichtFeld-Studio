/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "scene_ops.hpp"

#include "core/property_registry.hpp"
#include "core/services.hpp"
#include "operator/operator_registry.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/gui_capabilities.hpp"

namespace lfs::vis::op {

    namespace {

        std::optional<std::string> optional_node_prop(const OperatorProperties* props) {
            if (!props) {
                return std::nullopt;
            }
            if (const auto value = props->get<std::string>("node"); value && !value->empty()) {
                return *value;
            }
            return std::nullopt;
        }

        std::optional<std::string> required_name_prop(const OperatorProperties& props) {
            if (const auto value = props.get<std::string>("name"); value && !value->empty()) {
                return *value;
            }
            return std::nullopt;
        }

        std::expected<core::NodeId, std::string> resolve_cropbox_id_from_props(const OperatorContext& ctx,
                                                                               const OperatorProperties* props) {
            return cap::resolveCropBoxId(ctx.scene(), optional_node_prop(props));
        }

        std::expected<core::NodeId, std::string> resolve_cropbox_parent_id_from_props(
            const OperatorContext& ctx,
            const OperatorProperties* props) {
            return cap::resolveCropBoxParentId(ctx.scene(), optional_node_prop(props));
        }

        std::expected<core::NodeId, std::string> resolve_ellipsoid_id_from_props(const OperatorContext& ctx,
                                                                                 const OperatorProperties* props) {
            return cap::resolveEllipsoidId(ctx.scene(), optional_node_prop(props));
        }

        std::expected<core::NodeId, std::string> resolve_ellipsoid_parent_id_from_props(
            const OperatorContext& ctx,
            const OperatorProperties* props) {
            return cap::resolveEllipsoidParentId(ctx.scene(), optional_node_prop(props));
        }

        bool has_cropbox_update_fields(const OperatorProperties* props) {
            return props && (props->has("min") || props->has("max") || props->has("translation") ||
                             props->has("rotation") || props->has("scale") || props->has("inverse") ||
                             props->has("enabled") || props->has("show") || props->has("use"));
        }

        bool has_ellipsoid_update_fields(const OperatorProperties* props) {
            return props && (props->has("radii") || props->has("translation") || props->has("rotation") ||
                             props->has("scale") || props->has("inverse") || props->has("enabled") ||
                             props->has("show") || props->has("use"));
        }

        void register_scene_schemas() {
            auto& properties = core::prop::PropertyRegistry::instance();
            properties.register_operator_args(SelectionClearOperator::DESCRIPTOR.id(), {});
            properties.register_operator_args(
                SceneSelectNodeOperator::DESCRIPTOR.id(),
                {
                    core::prop::arg_string("name", "Node name to select"),
                    core::prop::arg_enum(
                        "mode", "Selection update mode",
                        {
                            {.name = "Replace", .identifier = "replace", .value = 0},
                            {.name = "Add", .identifier = "add", .value = 1},
                        }),
                });
            properties.register_operator_args(
                CropBoxAddOperator::DESCRIPTOR.id(),
                {
                    core::prop::arg_string("node", "Optional splat or pointcloud node name; defaults to the current selected node"),
                });
            properties.register_operator_args(
                CropBoxSetOperator::DESCRIPTOR.id(),
                {
                    core::prop::arg_string("node", "Optional crop box node or parent node name; defaults to the current selected crop box"),
                    core::prop::arg_float_vector("min", "Optional local minimum bounds", 3),
                    core::prop::arg_float_vector("max", "Optional local maximum bounds", 3),
                    core::prop::arg_float_vector("translation", "Optional local XYZ translation", 3),
                    core::prop::arg_float_vector("rotation", "Optional local XYZ Euler rotation in radians", 3),
                    core::prop::arg_float_vector("scale", "Optional local XYZ scale", 3),
                    core::prop::arg_bool("inverse", "Invert the crop volume"),
                    core::prop::arg_bool("enabled", "Enable crop filtering for this crop box"),
                    core::prop::arg_bool("show", "Show crop boxes in the viewport"),
                    core::prop::arg_bool("use", "Use crop box filtering in rendering"),
                });
            properties.register_operator_args(
                CropBoxFitOperator::DESCRIPTOR.id(),
                {
                    core::prop::arg_string("node", "Optional crop box node or parent node name; defaults to the current selected crop box"),
                    core::prop::arg_bool("use_percentile", "Use percentile bounds instead of strict min/max"),
                });
            properties.register_operator_args(
                CropBoxResetOperator::DESCRIPTOR.id(),
                {
                    core::prop::arg_string("node", "Optional crop box node or parent node name; defaults to the current selected crop box"),
                });
            properties.register_operator_args(
                EllipsoidAddOperator::DESCRIPTOR.id(),
                {
                    core::prop::arg_string("node", "Optional splat or pointcloud node name; defaults to the current selected node"),
                });
            properties.register_operator_args(
                EllipsoidSetOperator::DESCRIPTOR.id(),
                {
                    core::prop::arg_string("node", "Optional ellipsoid node or parent node name; defaults to the current selected ellipsoid"),
                    core::prop::arg_float_vector("radii", "Optional ellipsoid radii", 3),
                    core::prop::arg_float_vector("translation", "Optional local XYZ translation", 3),
                    core::prop::arg_float_vector("rotation", "Optional local XYZ Euler rotation in radians", 3),
                    core::prop::arg_float_vector("scale", "Optional local XYZ scale", 3),
                    core::prop::arg_bool("inverse", "Invert the ellipsoid selection volume"),
                    core::prop::arg_bool("enabled", "Enable ellipsoid filtering for this helper"),
                    core::prop::arg_bool("show", "Show ellipsoids in the viewport"),
                    core::prop::arg_bool("use", "Use ellipsoid filtering in rendering"),
                });
            properties.register_operator_args(
                EllipsoidFitOperator::DESCRIPTOR.id(),
                {
                    core::prop::arg_string("node", "Optional ellipsoid node or parent node name; defaults to the current selected ellipsoid"),
                    core::prop::arg_bool("use_percentile", "Use percentile bounds instead of strict min/max"),
                });
            properties.register_operator_args(
                EllipsoidResetOperator::DESCRIPTOR.id(),
                {
                    core::prop::arg_string("node", "Optional ellipsoid node or parent node name; defaults to the current selected ellipsoid"),
                });
        }

        void unregister_scene_schemas() {
            auto& properties = core::prop::PropertyRegistry::instance();
            properties.unregister_operator_args(SelectionClearOperator::DESCRIPTOR.id());
            properties.unregister_operator_args(SceneSelectNodeOperator::DESCRIPTOR.id());
            properties.unregister_operator_args(CropBoxAddOperator::DESCRIPTOR.id());
            properties.unregister_operator_args(CropBoxSetOperator::DESCRIPTOR.id());
            properties.unregister_operator_args(CropBoxFitOperator::DESCRIPTOR.id());
            properties.unregister_operator_args(CropBoxResetOperator::DESCRIPTOR.id());
            properties.unregister_operator_args(EllipsoidAddOperator::DESCRIPTOR.id());
            properties.unregister_operator_args(EllipsoidSetOperator::DESCRIPTOR.id());
            properties.unregister_operator_args(EllipsoidFitOperator::DESCRIPTOR.id());
            properties.unregister_operator_args(EllipsoidResetOperator::DESCRIPTOR.id());
        }

    } // namespace

    const OperatorDescriptor SelectionClearOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::SelectionClear,
        .python_class_id = {},
        .label = "Clear Selection",
        .description = "Clear the current gaussian selection",
        .icon = "",
        .shortcut = "",
        .flags = OperatorFlags::REGISTER | OperatorFlags::UNDO,
        .source = OperatorSource::CPP,
        .poll_deps = PollDependency::SCENE,
    };

    bool SelectionClearOperator::poll(const OperatorContext& ctx, const OperatorProperties* /*props*/) const {
        return ctx.scene().getScene().getSelectionMask() != nullptr;
    }

    OperatorResult SelectionClearOperator::invoke(OperatorContext& ctx, OperatorProperties& /*props*/) {
        return cap::clearGaussianSelection(ctx.scene()) ? OperatorResult::FINISHED : OperatorResult::CANCELLED;
    }

    const OperatorDescriptor SceneSelectNodeOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::SceneSelectNode,
        .python_class_id = {},
        .label = "Select Node",
        .description = "Change the shared node selection",
        .icon = "",
        .shortcut = "",
        .flags = OperatorFlags::REGISTER,
        .source = OperatorSource::CPP,
        .poll_deps = PollDependency::SCENE | PollDependency::SELECTION,
    };

    bool SceneSelectNodeOperator::poll(const OperatorContext& ctx, const OperatorProperties* props) const {
        if (!props) {
            return false;
        }
        const auto name = required_name_prop(*props);
        if (!name) {
            return false;
        }
        const auto mode = props->get_or<std::string>("mode", "replace");
        if (mode != "replace" && mode != "add") {
            return false;
        }
        return ctx.scene().getScene().getNode(*name) != nullptr;
    }

    OperatorResult SceneSelectNodeOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        const auto name = required_name_prop(props);
        if (!name) {
            return OperatorResult::CANCELLED;
        }
        const auto mode = props.get_or<std::string>("mode", "replace");
        if (auto result = cap::selectNode(ctx.scene(), *name, mode); !result) {
            return OperatorResult::CANCELLED;
        }
        props.set("selected_node_names", ctx.scene().getSelectedNodeNames());
        return OperatorResult::FINISHED;
    }

    const OperatorDescriptor CropBoxAddOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::CropBoxAdd,
        .python_class_id = {},
        .label = "Add Crop Box",
        .description = "Add or reuse a crop box helper",
        .icon = "",
        .shortcut = "",
        .flags = OperatorFlags::REGISTER | OperatorFlags::UNDO,
        .source = OperatorSource::CPP,
        .poll_deps = PollDependency::SCENE | PollDependency::SELECTION,
    };

    bool CropBoxAddOperator::poll(const OperatorContext& ctx, const OperatorProperties* props) const {
        return resolve_cropbox_parent_id_from_props(ctx, props).has_value();
    }

    OperatorResult CropBoxAddOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        auto parent_id = resolve_cropbox_parent_id_from_props(ctx, &props);
        if (!parent_id) {
            return OperatorResult::CANCELLED;
        }
        auto cropbox_id = cap::ensureCropBox(ctx.scene(), services().renderingOrNull(), *parent_id);
        if (!cropbox_id) {
            return OperatorResult::CANCELLED;
        }
        props.set("resolved_cropbox_id", *cropbox_id);
        return OperatorResult::FINISHED;
    }

    const OperatorDescriptor CropBoxSetOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::CropBoxSet,
        .python_class_id = {},
        .label = "Set Crop Box",
        .description = "Update crop box bounds, transform, or render toggles",
        .icon = "",
        .shortcut = "",
        .flags = OperatorFlags::REGISTER | OperatorFlags::UNDO,
        .source = OperatorSource::CPP,
        .poll_deps = PollDependency::SCENE | PollDependency::SELECTION,
    };

    bool CropBoxSetOperator::poll(const OperatorContext& ctx, const OperatorProperties* props) const {
        return has_cropbox_update_fields(props) && resolve_cropbox_id_from_props(ctx, props).has_value();
    }

    OperatorResult CropBoxSetOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        auto cropbox_id = resolve_cropbox_id_from_props(ctx, &props);
        if (!cropbox_id || !has_cropbox_update_fields(&props)) {
            return OperatorResult::CANCELLED;
        }

        cap::CropBoxUpdate update;
        if (props.has("min")) {
            update.min_bounds = props.get_or<glm::vec3>("min", glm::vec3(0.0f));
        }
        if (props.has("max")) {
            update.max_bounds = props.get_or<glm::vec3>("max", glm::vec3(0.0f));
        }
        if (props.has("translation")) {
            update.translation = props.get_or<glm::vec3>("translation", glm::vec3(0.0f));
        }
        if (props.has("rotation")) {
            update.rotation = props.get_or<glm::vec3>("rotation", glm::vec3(0.0f));
        }
        if (props.has("scale")) {
            update.scale = props.get_or<glm::vec3>("scale", glm::vec3(1.0f));
        }
        if (props.has("inverse")) {
            update.has_inverse = true;
            update.inverse = props.get_or<bool>("inverse", false);
        }
        if (props.has("enabled")) {
            update.has_enabled = true;
            update.enabled = props.get_or<bool>("enabled", false);
        }
        if (props.has("show")) {
            update.has_show = true;
            update.show = props.get_or<bool>("show", false);
        }
        if (props.has("use")) {
            update.has_use = true;
            update.use = props.get_or<bool>("use", false);
        }

        if (auto result = cap::updateCropBox(ctx.scene(), services().renderingOrNull(), *cropbox_id, update);
            !result) {
            return OperatorResult::CANCELLED;
        }
        props.set("resolved_cropbox_id", *cropbox_id);
        return OperatorResult::FINISHED;
    }

    const OperatorDescriptor CropBoxFitOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::CropBoxFit,
        .python_class_id = {},
        .label = "Fit Crop Box",
        .description = "Fit a crop box to its parent node bounds",
        .icon = "",
        .shortcut = "",
        .flags = OperatorFlags::REGISTER | OperatorFlags::UNDO,
        .source = OperatorSource::CPP,
        .poll_deps = PollDependency::SCENE | PollDependency::SELECTION,
    };

    bool CropBoxFitOperator::poll(const OperatorContext& ctx, const OperatorProperties* props) const {
        return resolve_cropbox_id_from_props(ctx, props).has_value();
    }

    OperatorResult CropBoxFitOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        auto cropbox_id = resolve_cropbox_id_from_props(ctx, &props);
        if (!cropbox_id) {
            return OperatorResult::CANCELLED;
        }
        const bool use_percentile = props.get_or<bool>("use_percentile", true);
        if (auto result = cap::fitCropBoxToParent(ctx.scene(), services().renderingOrNull(), *cropbox_id,
                                                  use_percentile);
            !result) {
            return OperatorResult::CANCELLED;
        }
        props.set("resolved_cropbox_id", *cropbox_id);
        props.set("use_percentile", use_percentile);
        return OperatorResult::FINISHED;
    }

    const OperatorDescriptor CropBoxResetOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::CropBoxReset,
        .python_class_id = {},
        .label = "Reset Crop Box",
        .description = "Reset a crop box to default bounds and identity transform",
        .icon = "",
        .shortcut = "",
        .flags = OperatorFlags::REGISTER | OperatorFlags::UNDO,
        .source = OperatorSource::CPP,
        .poll_deps = PollDependency::SCENE | PollDependency::SELECTION,
    };

    bool CropBoxResetOperator::poll(const OperatorContext& ctx, const OperatorProperties* props) const {
        return resolve_cropbox_id_from_props(ctx, props).has_value();
    }

    OperatorResult CropBoxResetOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        auto cropbox_id = resolve_cropbox_id_from_props(ctx, &props);
        if (!cropbox_id) {
            return OperatorResult::CANCELLED;
        }
        if (auto result = cap::resetCropBox(ctx.scene(), services().renderingOrNull(), *cropbox_id); !result) {
            return OperatorResult::CANCELLED;
        }
        props.set("resolved_cropbox_id", *cropbox_id);
        return OperatorResult::FINISHED;
    }

    const OperatorDescriptor EllipsoidAddOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::EllipsoidAdd,
        .python_class_id = {},
        .label = "Add Ellipsoid",
        .description = "Add or reuse an ellipsoid helper",
        .icon = "",
        .shortcut = "",
        .flags = OperatorFlags::REGISTER | OperatorFlags::UNDO,
        .source = OperatorSource::CPP,
        .poll_deps = PollDependency::SCENE | PollDependency::SELECTION,
    };

    bool EllipsoidAddOperator::poll(const OperatorContext& ctx, const OperatorProperties* props) const {
        return resolve_ellipsoid_parent_id_from_props(ctx, props).has_value();
    }

    OperatorResult EllipsoidAddOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        auto parent_id = resolve_ellipsoid_parent_id_from_props(ctx, &props);
        if (!parent_id) {
            return OperatorResult::CANCELLED;
        }
        auto ellipsoid_id = cap::ensureEllipsoid(ctx.scene(), services().renderingOrNull(), *parent_id);
        if (!ellipsoid_id) {
            return OperatorResult::CANCELLED;
        }
        props.set("resolved_ellipsoid_id", *ellipsoid_id);
        return OperatorResult::FINISHED;
    }

    const OperatorDescriptor EllipsoidSetOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::EllipsoidSet,
        .python_class_id = {},
        .label = "Set Ellipsoid",
        .description = "Update ellipsoid radii, transform, or render toggles",
        .icon = "",
        .shortcut = "",
        .flags = OperatorFlags::REGISTER | OperatorFlags::UNDO,
        .source = OperatorSource::CPP,
        .poll_deps = PollDependency::SCENE | PollDependency::SELECTION,
    };

    bool EllipsoidSetOperator::poll(const OperatorContext& ctx, const OperatorProperties* props) const {
        return has_ellipsoid_update_fields(props) && resolve_ellipsoid_id_from_props(ctx, props).has_value();
    }

    OperatorResult EllipsoidSetOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        auto ellipsoid_id = resolve_ellipsoid_id_from_props(ctx, &props);
        if (!ellipsoid_id || !has_ellipsoid_update_fields(&props)) {
            return OperatorResult::CANCELLED;
        }

        cap::EllipsoidUpdate update;
        if (props.has("radii")) {
            update.radii = props.get_or<glm::vec3>("radii", glm::vec3(1.0f));
        }
        if (props.has("translation")) {
            update.translation = props.get_or<glm::vec3>("translation", glm::vec3(0.0f));
        }
        if (props.has("rotation")) {
            update.rotation = props.get_or<glm::vec3>("rotation", glm::vec3(0.0f));
        }
        if (props.has("scale")) {
            update.scale = props.get_or<glm::vec3>("scale", glm::vec3(1.0f));
        }
        if (props.has("inverse")) {
            update.has_inverse = true;
            update.inverse = props.get_or<bool>("inverse", false);
        }
        if (props.has("enabled")) {
            update.has_enabled = true;
            update.enabled = props.get_or<bool>("enabled", false);
        }
        if (props.has("show")) {
            update.has_show = true;
            update.show = props.get_or<bool>("show", false);
        }
        if (props.has("use")) {
            update.has_use = true;
            update.use = props.get_or<bool>("use", false);
        }

        if (auto result = cap::updateEllipsoid(ctx.scene(), services().renderingOrNull(), *ellipsoid_id, update);
            !result) {
            return OperatorResult::CANCELLED;
        }
        props.set("resolved_ellipsoid_id", *ellipsoid_id);
        return OperatorResult::FINISHED;
    }

    const OperatorDescriptor EllipsoidFitOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::EllipsoidFit,
        .python_class_id = {},
        .label = "Fit Ellipsoid",
        .description = "Fit an ellipsoid helper to its parent node bounds",
        .icon = "",
        .shortcut = "",
        .flags = OperatorFlags::REGISTER | OperatorFlags::UNDO,
        .source = OperatorSource::CPP,
        .poll_deps = PollDependency::SCENE | PollDependency::SELECTION,
    };

    bool EllipsoidFitOperator::poll(const OperatorContext& ctx, const OperatorProperties* props) const {
        return resolve_ellipsoid_id_from_props(ctx, props).has_value();
    }

    OperatorResult EllipsoidFitOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        auto ellipsoid_id = resolve_ellipsoid_id_from_props(ctx, &props);
        if (!ellipsoid_id) {
            return OperatorResult::CANCELLED;
        }
        const bool use_percentile = props.get_or<bool>("use_percentile", true);
        if (auto result = cap::fitEllipsoidToParent(ctx.scene(), services().renderingOrNull(), *ellipsoid_id,
                                                    use_percentile);
            !result) {
            return OperatorResult::CANCELLED;
        }
        props.set("resolved_ellipsoid_id", *ellipsoid_id);
        props.set("use_percentile", use_percentile);
        return OperatorResult::FINISHED;
    }

    const OperatorDescriptor EllipsoidResetOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::EllipsoidReset,
        .python_class_id = {},
        .label = "Reset Ellipsoid",
        .description = "Reset an ellipsoid helper to default radii and identity transform",
        .icon = "",
        .shortcut = "",
        .flags = OperatorFlags::REGISTER | OperatorFlags::UNDO,
        .source = OperatorSource::CPP,
        .poll_deps = PollDependency::SCENE | PollDependency::SELECTION,
    };

    bool EllipsoidResetOperator::poll(const OperatorContext& ctx, const OperatorProperties* props) const {
        return resolve_ellipsoid_id_from_props(ctx, props).has_value();
    }

    OperatorResult EllipsoidResetOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        auto ellipsoid_id = resolve_ellipsoid_id_from_props(ctx, &props);
        if (!ellipsoid_id) {
            return OperatorResult::CANCELLED;
        }
        if (auto result = cap::resetEllipsoid(ctx.scene(), services().renderingOrNull(), *ellipsoid_id);
            !result) {
            return OperatorResult::CANCELLED;
        }
        props.set("resolved_ellipsoid_id", *ellipsoid_id);
        return OperatorResult::FINISHED;
    }

    void registerSceneOperators() {
        register_scene_schemas();
        operators().registerOperator(BuiltinOp::SelectionClear, SelectionClearOperator::DESCRIPTOR,
                                     [] { return std::make_unique<SelectionClearOperator>(); });
        operators().registerOperator(BuiltinOp::SceneSelectNode, SceneSelectNodeOperator::DESCRIPTOR,
                                     [] { return std::make_unique<SceneSelectNodeOperator>(); });
        operators().registerOperator(BuiltinOp::CropBoxAdd, CropBoxAddOperator::DESCRIPTOR,
                                     [] { return std::make_unique<CropBoxAddOperator>(); });
        operators().registerOperator(BuiltinOp::CropBoxSet, CropBoxSetOperator::DESCRIPTOR,
                                     [] { return std::make_unique<CropBoxSetOperator>(); });
        operators().registerOperator(BuiltinOp::CropBoxFit, CropBoxFitOperator::DESCRIPTOR,
                                     [] { return std::make_unique<CropBoxFitOperator>(); });
        operators().registerOperator(BuiltinOp::CropBoxReset, CropBoxResetOperator::DESCRIPTOR,
                                     [] { return std::make_unique<CropBoxResetOperator>(); });
        operators().registerOperator(BuiltinOp::EllipsoidAdd, EllipsoidAddOperator::DESCRIPTOR,
                                     [] { return std::make_unique<EllipsoidAddOperator>(); });
        operators().registerOperator(BuiltinOp::EllipsoidSet, EllipsoidSetOperator::DESCRIPTOR,
                                     [] { return std::make_unique<EllipsoidSetOperator>(); });
        operators().registerOperator(BuiltinOp::EllipsoidFit, EllipsoidFitOperator::DESCRIPTOR,
                                     [] { return std::make_unique<EllipsoidFitOperator>(); });
        operators().registerOperator(BuiltinOp::EllipsoidReset, EllipsoidResetOperator::DESCRIPTOR,
                                     [] { return std::make_unique<EllipsoidResetOperator>(); });
    }

    void unregisterSceneOperators() {
        operators().unregisterOperator(BuiltinOp::SelectionClear);
        operators().unregisterOperator(BuiltinOp::SceneSelectNode);
        operators().unregisterOperator(BuiltinOp::CropBoxAdd);
        operators().unregisterOperator(BuiltinOp::CropBoxSet);
        operators().unregisterOperator(BuiltinOp::CropBoxFit);
        operators().unregisterOperator(BuiltinOp::CropBoxReset);
        operators().unregisterOperator(BuiltinOp::EllipsoidAdd);
        operators().unregisterOperator(BuiltinOp::EllipsoidSet);
        operators().unregisterOperator(BuiltinOp::EllipsoidFit);
        operators().unregisterOperator(BuiltinOp::EllipsoidReset);
        unregister_scene_schemas();
    }

} // namespace lfs::vis::op
