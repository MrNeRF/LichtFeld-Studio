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

    lfs::Result<void>
    validateSplatOperation(const SceneNode& owner, const SceneNode& element) {
        if (owner.type != NodeType::SPLAT)
            return lfs::Result<void>::failure(readError(lfs::ErrorCode::Unsupported, "The owner must be a SPLAT."));
        if (owner.id < 0 || element.id < 0 || owner.id == element.id ||
            owner.uuid.bytes == Uuid{}.bytes || element.uuid.bytes == Uuid{}.bytes || owner.uuid == element.uuid)
            return lfs::Result<void>::failure(readError(lfs::ErrorCode::InvalidArgument, "Owner and element need distinct valid node IDs and UUIDs."));
        if (element.parent_id != owner.id || std::ranges::find(owner.children, element.id) == owner.children.end())
            return lfs::Result<void>::failure(readError(lfs::ErrorCode::FailedPrecondition, "The owner and element must have reciprocal direct parent-child links."));

        const bool has_payload =
            (element.type == NodeType::CROPBOX && element.cropbox) ||
            (element.type == NodeType::ELLIPSOID && element.ellipsoid) ||
            (element.type == NodeType::CROP_POLYGON && element.crop_polygon) ||
            (element.type == NodeType::COLLISION && element.collision);
        if (!splatOperationDescriptor(element.type))
            return lfs::Result<void>::failure(readError(lfs::ErrorCode::Unsupported, "This node type is not an inspectable splat element."));
        if (!has_payload)
            return lfs::Result<void>::failure(readError(lfs::ErrorCode::FailedPrecondition, "The scene element is missing its required payload."));
        return {};
    }

    lfs::Result<SplatOperationSnapshot>
    readSplatOperation(const SceneNode& owner, const SceneNode& element) {
        if (auto valid = validateSplatOperation(owner, element); !valid)
            return valid.error();
        SplatOperationKind kind;
        SplatOperationParameters parameters;
        std::optional<bool> enabled;
        switch (element.type) {
        case NodeType::CROPBOX:
            kind = SplatOperationKind::CropBox;
            parameters = *element.cropbox;
            enabled = element.cropbox->enabled;
            break;
        case NodeType::ELLIPSOID:
            kind = SplatOperationKind::Ellipsoid;
            parameters = *element.ellipsoid;
            enabled = element.ellipsoid->enabled;
            break;
        case NodeType::CROP_POLYGON:
            kind = SplatOperationKind::CropPolygon;
            parameters = *element.crop_polygon;
            enabled = element.crop_polygon->enabled;
            break;
        case NodeType::COLLISION:
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
