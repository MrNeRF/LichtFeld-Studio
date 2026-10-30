/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/collision_volume.hpp"
#include "core/error.hpp"
#include "core/scene.hpp"

#include <array>
#include <optional>
#include <span>
#include <string_view>
#include <variant>

namespace lfs::core {

    enum class SplatOperationKind { CropBox,
                                    Ellipsoid,
                                    CropPolygon,
                                    Collision };
    enum class SplatOperationCategory { CropVolume,
                                        DerivedData };

    // Only elements consumed by the scene tree belong here. Actions refer
    // directly to the existing operator registry; an empty ID means unavailable.
    struct SplatOperationDescriptor {
        SplatOperationKind kind;
        NodeType node_type;
        SplatOperationCategory category;
        std::string_view set_operator;
        std::string_view duplicate_operator;
    };

    inline constexpr std::array SPLAT_OPERATION_DESCRIPTORS{
        SplatOperationDescriptor{SplatOperationKind::CropBox, NodeType::CROPBOX, SplatOperationCategory::CropVolume, "crop_box.set", "crop_volume.duplicate"},
        SplatOperationDescriptor{SplatOperationKind::Ellipsoid, NodeType::ELLIPSOID, SplatOperationCategory::CropVolume, "ellipsoid.set", "crop_volume.duplicate"},
        SplatOperationDescriptor{SplatOperationKind::CropPolygon, NodeType::CROP_POLYGON, SplatOperationCategory::CropVolume, "crop_polygon.set", "crop_volume.duplicate"},
        SplatOperationDescriptor{SplatOperationKind::Collision, NodeType::COLLISION, SplatOperationCategory::DerivedData, {}, {}},
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

    [[nodiscard]] constexpr const SplatOperationDescriptor* splatOperationDescriptor(NodeType type) {
        for (const auto& descriptor : SPLAT_OPERATION_DESCRIPTORS)
            if (descriptor.node_type == type)
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

    // No selection fallback, name lookup, ancestor traversal, hydration,
    // persistence mutation or synthetic operation UUID generation.
    [[nodiscard]] LFS_CORE_API lfs::Result<SplatOperationSnapshot>
    readSplatOperation(const SceneNode& owner, const SceneNode& element);

} // namespace lfs::core
