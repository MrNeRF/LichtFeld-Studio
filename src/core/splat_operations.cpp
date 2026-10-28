/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_operations.hpp"

#include <algorithm>

namespace lfs::core {

    std::expected<SplatOperationSnapshot, SplatOperationReadError>
    readSplatOperation(const SceneNode& owner, const SceneNode& element) {
        if (owner.type != NodeType::SPLAT)
            return std::unexpected(SplatOperationReadError::UnsupportedOwner);
        if (owner.id < 0 || element.id < 0 || owner.id == element.id ||
            owner.uuid.bytes == Uuid{}.bytes || element.uuid.bytes == Uuid{}.bytes || owner.uuid == element.uuid)
            return std::unexpected(SplatOperationReadError::InvalidIdentity);
        if (element.parent_id != owner.id || std::ranges::find(owner.children, element.id) == owner.children.end())
            return std::unexpected(SplatOperationReadError::NotDirectChild);

        SplatOperationKind kind;
        SplatOperationParameters parameters;
        std::optional<bool> enabled;
        switch (element.type) {
        case NodeType::CROPBOX:
            if (!element.cropbox)
                return std::unexpected(SplatOperationReadError::MissingPayload);
            kind = SplatOperationKind::CropBox;
            parameters = *element.cropbox;
            enabled = element.cropbox->enabled;
            break;
        case NodeType::ELLIPSOID:
            if (!element.ellipsoid)
                return std::unexpected(SplatOperationReadError::MissingPayload);
            kind = SplatOperationKind::Ellipsoid;
            parameters = *element.ellipsoid;
            enabled = element.ellipsoid->enabled;
            break;
        case NodeType::CROP_POLYGON:
            if (!element.crop_polygon)
                return std::unexpected(SplatOperationReadError::MissingPayload);
            kind = SplatOperationKind::CropPolygon;
            parameters = *element.crop_polygon;
            enabled = element.crop_polygon->enabled;
            break;
        case NodeType::COLLISION:
            if (!element.collision)
                return std::unexpected(SplatOperationReadError::MissingPayload);
            kind = SplatOperationKind::Collision;
            parameters = element.collision->settings;
            break;
        default:
            return std::unexpected(SplatOperationReadError::UnsupportedElement);
        }

        return SplatOperationSnapshot{
            .kind = kind,
            .owner_uuid = owner.uuid,
            .element_uuid = element.uuid,
            .local_transform = element.transform(),
            .node_visible = element.visible.get(),
            .node_locked = element.locked.get(),
            .enabled = enabled,
            .parameters = std::move(parameters),
        };
    }

} // namespace lfs::core
