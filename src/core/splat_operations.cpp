/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_operations.hpp"
#include "core/splat_data_transform.hpp"
#include "geometry/bounding_box.hpp"

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace lfs::core {
    namespace {
        lfs::Error readError(lfs::ErrorCode code, std::string detail) {
            return lfs::make_error({.code = code, .domain = lfs::ErrorDomain::Core, .user_message = "Cannot inspect this splat element.", .detail = std::move(detail), .detection = LFS_SOURCE_SITE_CURRENT()});
        }
    } // namespace

    std::expected<void, CropApplyError>
    applySplatCrop(SplatData& model, const glm::mat4& crop_to_owner,
                   const SplatOperationParameters& parameters) {
        return std::visit([&](const auto& volume) -> std::expected<void, CropApplyError> {
            using Volume = std::decay_t<decltype(volume)>;
            if constexpr (std::is_same_v<Volume, CollisionSettings>) {
                return std::unexpected(CropApplyError::Unsupported);
            } else {
                if (!volume.enabled)
                    return {};
                const float determinant = glm::determinant(crop_to_owner);
                if (!std::isfinite(determinant) || determinant == 0.f)
                    return std::unexpected(CropApplyError::InvalidTransform);
                const auto to_crop = glm::inverse(crop_to_owner);
                for (int col = 0; col < 4; ++col)
                    for (int row = 0; row < 4; ++row)
                        if (!std::isfinite(to_crop[col][row]))
                            return std::unexpected(CropApplyError::InvalidTransform);
                const auto finite = [](const glm::vec3& value) {
                    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
                };
                if constexpr (std::is_same_v<Volume, CropBoxData>) {
                    if (!finite(volume.min) || !finite(volume.max) || glm::any(glm::lessThanEqual(volume.max, volume.min)))
                        return std::unexpected(CropApplyError::InvalidVolume);
                    geometry::BoundingBox bounds;
                    bounds.setBounds(volume.min, volume.max);
                    bounds.setworld2BBox(to_crop);
                    (void)soft_crop_by_cropbox(model, bounds, volume.inverse);
                } else if constexpr (std::is_same_v<Volume, EllipsoidData>) {
                    if (!finite(volume.radii) || glm::any(glm::lessThanEqual(volume.radii, glm::vec3(0))))
                        return std::unexpected(CropApplyError::InvalidVolume);
                    (void)soft_crop_by_ellipsoid(model, to_crop, volume.radii, volume.inverse);
                } else {
                    if (!crop_polygon::isValidOutline(volume.vertices) || !crop_polygon::isValidHalfDepth(volume.half_depth))
                        return std::unexpected(CropApplyError::InvalidVolume);
                    (void)soft_crop_by_polygon(model, to_crop, volume);
                }
                return {};
            }
        },
                          parameters);
    }

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
