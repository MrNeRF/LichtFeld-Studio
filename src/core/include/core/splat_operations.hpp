/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/collision_volume.hpp"
#include "core/scene.hpp"

#include <array>
#include <expected>
#include <optional>
#include <span>
#include <string_view>
#include <variant>

namespace lfs::core {

    enum class SplatOperationKind {
        CropBox,
        Ellipsoid,
        CropPolygon,
        Collision,
        Simplify,
        Translate,
        Rotate,
        Scale,
        SetTransform,
        BakeTransform,
        Align,
        ScaleDistance,
        Select,
        SelectAll,
        SelectNone,
        SelectInvert,
        SelectGrow,
        SelectShrink,
        SelectByOpacity,
        SelectByScale,
        SelectionGroups,
        SelectionDepthFilter,
        SelectionCropFilter,
        Mirror,
        DeleteSelection,
        CopySelection,
        CutSelection,
        PasteSelection,
        Duplicate,
        Reparent,
        GroupSelection,
        WriteGaussianField
    };
    enum class SplatOperationCategory {
        CropVolume,
        DerivedData,
        GeometryReduction,
        Transform,
        Selection,
        SelectionFilter,
        DirectEdit,
        SceneOrganization
    };
    enum class SplatOperationStorage { SceneVolume,
                                       GeneratedResult,
                                       CommandOnly };

    // Scope of the authoritative state, independent of a tool's current target.
    enum class SplatOperationScope {
        Object,
        SceneSelection,
        ViewportTool
    };

    // Describes existing paths, not permission to execute a command. Callers
    // must still use the application's validation, undo and job services.
    struct SplatOperationCapabilities {
        bool edit_parameters = false;
        bool set_enabled = false;
        bool show_helper = false;
        bool fit = false;
        bool reset = false;
        bool invert = false;
        bool apply_to_geometry = false;
        bool remove_element = false;
        bool generate = false;
        bool export_result = false;
        bool reorder = false;
        bool supports_group_owner = false;
    };

    struct SplatOperationDescriptor {
        SplatOperationKind kind;
        std::string_view identifier;
        SplatOperationCategory category;
        SplatOperationStorage storage;
        SplatOperationScope scope;
        SplatOperationCapabilities capabilities;
        // Existing toolbar entry when applicable; empty for secondary commands.
        std::string_view tool_identifier;

        // A command can target splats without being owned by a splat.
        [[nodiscard]] constexpr bool supportsOwner(NodeType type) const {
            return type == NodeType::SPLAT && scope == SplatOperationScope::Object &&
                   storage != SplatOperationStorage::CommandOnly;
        }
    };

    // Classifications of command paths do not create persistent operation instances.
    [[nodiscard]] constexpr SplatOperationDescriptor splatCommandDescriptor(
        SplatOperationKind kind, std::string_view identifier, SplatOperationCategory category,
        SplatOperationScope scope, std::string_view tool_identifier = {}) {
        return {kind, identifier, category, SplatOperationStorage::CommandOnly, scope, {}, tool_identifier};
    }

    inline constexpr std::array SPLAT_OPERATION_DESCRIPTORS{
        SplatOperationDescriptor{SplatOperationKind::CropBox, "splat.crop_box", SplatOperationCategory::CropVolume, SplatOperationStorage::SceneVolume, SplatOperationScope::Object, {.edit_parameters = true, .set_enabled = true, .show_helper = true, .fit = true, .reset = true, .invert = true, .apply_to_geometry = true, .remove_element = true}, "builtin.cropbox"},
        SplatOperationDescriptor{SplatOperationKind::Ellipsoid, "splat.ellipsoid", SplatOperationCategory::CropVolume, SplatOperationStorage::SceneVolume, SplatOperationScope::Object, {.edit_parameters = true, .set_enabled = true, .show_helper = true, .fit = true, .reset = true, .invert = true, .apply_to_geometry = true, .remove_element = true}, {}},
        SplatOperationDescriptor{SplatOperationKind::CropPolygon, "splat.crop_polygon", SplatOperationCategory::CropVolume, SplatOperationStorage::SceneVolume, SplatOperationScope::Object, {.edit_parameters = true, .set_enabled = true, .show_helper = true, .fit = true, .reset = true, .invert = true, .apply_to_geometry = true, .remove_element = true}, {}},
        // Settings on a generated volume describe its completed generation;
        // they are not an editable recipe or an independent enable switch.
        SplatOperationDescriptor{SplatOperationKind::Collision, "splat.collision", SplatOperationCategory::DerivedData, SplatOperationStorage::GeneratedResult, SplatOperationScope::Object, {.remove_element = true, .generate = true, .export_result = true}, {}},
        // No persistent operation instance can be recovered from a simplified
        // splat: the current path is a command producing another splat.
        SplatOperationDescriptor{SplatOperationKind::Simplify, "splat.simplify", SplatOperationCategory::GeometryReduction, SplatOperationStorage::CommandOnly, SplatOperationScope::Object, {.generate = true}, {}},
        splatCommandDescriptor(SplatOperationKind::Translate, "splat.translate", SplatOperationCategory::Transform, SplatOperationScope::Object, "builtin.translate"),
        splatCommandDescriptor(SplatOperationKind::Rotate, "splat.rotate", SplatOperationCategory::Transform, SplatOperationScope::Object, "builtin.rotate"),
        splatCommandDescriptor(SplatOperationKind::Scale, "splat.scale", SplatOperationCategory::Transform, SplatOperationScope::Object, "builtin.scale"),
        splatCommandDescriptor(SplatOperationKind::SetTransform, "splat.set_transform", SplatOperationCategory::Transform, SplatOperationScope::Object),
        splatCommandDescriptor(SplatOperationKind::BakeTransform, "splat.bake_transform", SplatOperationCategory::Transform, SplatOperationScope::Object),
        splatCommandDescriptor(SplatOperationKind::Align, "splat.align", SplatOperationCategory::Transform, SplatOperationScope::Object, "builtin.align"),
        splatCommandDescriptor(SplatOperationKind::ScaleDistance, "splat.scale_distance", SplatOperationCategory::Transform, SplatOperationScope::Object, "builtin.scale_distance"),
        splatCommandDescriptor(SplatOperationKind::Select, "splat.select", SplatOperationCategory::Selection, SplatOperationScope::SceneSelection, "builtin.select"),
        splatCommandDescriptor(SplatOperationKind::SelectAll, "splat.select_all", SplatOperationCategory::Selection, SplatOperationScope::SceneSelection),
        splatCommandDescriptor(SplatOperationKind::SelectNone, "splat.select_none", SplatOperationCategory::Selection, SplatOperationScope::SceneSelection),
        splatCommandDescriptor(SplatOperationKind::SelectInvert, "splat.select_invert", SplatOperationCategory::Selection, SplatOperationScope::SceneSelection),
        splatCommandDescriptor(SplatOperationKind::SelectGrow, "splat.select_grow", SplatOperationCategory::Selection, SplatOperationScope::SceneSelection),
        splatCommandDescriptor(SplatOperationKind::SelectShrink, "splat.select_shrink", SplatOperationCategory::Selection, SplatOperationScope::SceneSelection),
        splatCommandDescriptor(SplatOperationKind::SelectByOpacity, "splat.select_by_opacity", SplatOperationCategory::Selection, SplatOperationScope::SceneSelection),
        splatCommandDescriptor(SplatOperationKind::SelectByScale, "splat.select_by_scale", SplatOperationCategory::Selection, SplatOperationScope::SceneSelection),
        splatCommandDescriptor(SplatOperationKind::SelectionGroups, "splat.selection_groups", SplatOperationCategory::Selection, SplatOperationScope::SceneSelection),
        splatCommandDescriptor(SplatOperationKind::SelectionDepthFilter, "splat.selection_depth_filter", SplatOperationCategory::SelectionFilter, SplatOperationScope::ViewportTool),
        splatCommandDescriptor(SplatOperationKind::SelectionCropFilter, "splat.selection_crop_filter", SplatOperationCategory::SelectionFilter, SplatOperationScope::ViewportTool),
        splatCommandDescriptor(SplatOperationKind::Mirror, "splat.mirror", SplatOperationCategory::DirectEdit, SplatOperationScope::SceneSelection, "builtin.mirror"),
        splatCommandDescriptor(SplatOperationKind::DeleteSelection, "splat.delete_selection", SplatOperationCategory::DirectEdit, SplatOperationScope::SceneSelection),
        splatCommandDescriptor(SplatOperationKind::CopySelection, "splat.copy_selection", SplatOperationCategory::DirectEdit, SplatOperationScope::SceneSelection),
        splatCommandDescriptor(SplatOperationKind::CutSelection, "splat.cut_selection", SplatOperationCategory::DirectEdit, SplatOperationScope::SceneSelection),
        splatCommandDescriptor(SplatOperationKind::PasteSelection, "splat.paste_selection", SplatOperationCategory::DirectEdit, SplatOperationScope::SceneSelection),
        splatCommandDescriptor(SplatOperationKind::Duplicate, "splat.duplicate", SplatOperationCategory::SceneOrganization, SplatOperationScope::Object),
        splatCommandDescriptor(SplatOperationKind::Reparent, "splat.reparent", SplatOperationCategory::SceneOrganization, SplatOperationScope::Object),
        splatCommandDescriptor(SplatOperationKind::GroupSelection, "splat.group_selection", SplatOperationCategory::SceneOrganization, SplatOperationScope::SceneSelection),
        splatCommandDescriptor(SplatOperationKind::WriteGaussianField, "splat.write_gaussian_field", SplatOperationCategory::DirectEdit, SplatOperationScope::Object),
    };

    [[nodiscard]] constexpr std::span<const SplatOperationDescriptor> splatOperationDescriptors() {
        return SPLAT_OPERATION_DESCRIPTORS;
    }

    [[nodiscard]] constexpr const SplatOperationDescriptor* splatOperationDescriptor(SplatOperationKind kind) {
        for (const auto& descriptor : SPLAT_OPERATION_DESCRIPTORS)
            if (descriptor.kind == kind)
                return &descriptor;
        return nullptr;
    }

    using SplatOperationParameters = std::variant<CropBoxData, EllipsoidData, CropPolygonData, CollisionSettings>;

    // Detached metadata only: no geometry tensors, volume words or borrowed
    // pointers. Read on the scene-owning thread (or under its read protection).
    // Parameters remain authoritative in the existing node payloads.
    struct SplatOperationSnapshot {
        SplatOperationKind kind;
        Uuid owner_uuid;
        Uuid element_uuid;
        glm::mat4 local_transform{1.0f};
        bool node_visible = true;
        bool node_locked = false;
        // Local payload flag only. Global renderer gates and ancestor state
        // are deliberately not represented as effective operation state.
        std::optional<bool> enabled;
        SplatOperationParameters parameters;
    };

    enum class SplatOperationReadError { UnsupportedOwner,
                                         InvalidIdentity,
                                         NotDirectChild,
                                         UnsupportedElement,
                                         MissingPayload };

    // No selection fallback, name lookup, ancestor traversal, hydration,
    // persistence mutation or synthetic operation UUID generation.
    [[nodiscard]] LFS_CORE_API std::expected<SplatOperationSnapshot, SplatOperationReadError>
    readSplatOperation(const SceneNode& owner, const SceneNode& element);

} // namespace lfs::core
