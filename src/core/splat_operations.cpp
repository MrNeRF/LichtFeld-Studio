/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_operations.hpp"

#include <algorithm>

namespace lfs::core {
    namespace {
        lfs::Error readError(lfs::ErrorCode code, std::string detail) {
            return lfs::make_error({.code = code, .domain = lfs::ErrorDomain::Core, .user_message = "Cannot inspect this splat element.", .detail = std::move(detail), .detection = LFS_SOURCE_SITE_CURRENT()});
        }
    } // namespace

    lfs::Result<SplatOperationSnapshot>
    readSplatOperation(const SceneNode& owner, const SceneNode& element) {
        if (owner.type != NodeType::SPLAT)
            return readError(lfs::ErrorCode::Unsupported, "The owner must be a SPLAT.");
        if (owner.id < 0 || element.id < 0 || owner.id == element.id ||
            owner.uuid.bytes == Uuid{}.bytes || element.uuid.bytes == Uuid{}.bytes || owner.uuid == element.uuid)
            return readError(lfs::ErrorCode::InvalidArgument, "Owner and element need distinct valid node IDs and UUIDs.");
        if (element.parent_id != owner.id || std::ranges::find(owner.children, element.id) == owner.children.end())
            return readError(lfs::ErrorCode::FailedPrecondition, "The owner and element must have reciprocal direct parent-child links.");

        SplatOperationKind kind;
        SplatOperationParameters parameters;
        std::optional<bool> enabled;
        switch (element.type) {
        case NodeType::CROPBOX:
            if (!element.cropbox)
                return readError(lfs::ErrorCode::FailedPrecondition, "The scene element is missing its required payload.");
            kind = SplatOperationKind::CropBox;
            parameters = *element.cropbox;
            enabled = element.cropbox->enabled;
            break;
        case NodeType::ELLIPSOID:
            if (!element.ellipsoid)
                return readError(lfs::ErrorCode::FailedPrecondition, "The scene element is missing its required payload.");
            kind = SplatOperationKind::Ellipsoid;
            parameters = *element.ellipsoid;
            enabled = element.ellipsoid->enabled;
            break;
        case NodeType::CROP_POLYGON:
            if (!element.crop_polygon)
                return readError(lfs::ErrorCode::FailedPrecondition, "The scene element is missing its required payload.");
            kind = SplatOperationKind::CropPolygon;
            parameters = *element.crop_polygon;
            enabled = element.crop_polygon->enabled;
            break;
        case NodeType::COLLISION:
            if (!element.collision)
                return readError(lfs::ErrorCode::FailedPrecondition, "The scene element is missing its required payload.");
            kind = SplatOperationKind::Collision;
            parameters = element.collision->settings;
            break;
        default:
            return readError(lfs::ErrorCode::Unsupported, "This node type is not an inspectable splat element.");
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
