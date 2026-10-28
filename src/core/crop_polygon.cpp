/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/crop_polygon.hpp"

namespace lfs::core::crop_polygon {

    namespace {
        [[nodiscard]] bool isFinite(const glm::vec3& v) {
            return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
        }
    } // namespace

    std::optional<Plane> makePlane(const glm::vec3& camera_forward, const glm::vec3& point) {
        const float length = glm::length(camera_forward);
        if (!isFinite(camera_forward) || !isFinite(point) || !(length > 1e-8f))
            return std::nullopt;
        return Plane{.point = point, .normal = camera_forward / length};
    }

    bool pivotIsInFront(const glm::vec3& camera_position, const glm::vec3& camera_forward,
                        const glm::vec3& pivot) {
        const float length = glm::length(camera_forward);
        if (!isFinite(camera_position) || !isFinite(camera_forward) || !isFinite(pivot) || !(length > 1e-8f))
            return false;
        return glm::dot(pivot - camera_position, camera_forward / length) > kMinHalfDepth;
    }

    std::optional<Plane> makeDrawingPlane(const glm::vec3& camera_position, const glm::vec3& camera_forward,
                                          const glm::vec3& pivot,
                                          const std::optional<glm::vec3>& first_click_point) {
        if (pivotIsInFront(camera_position, camera_forward, pivot))
            return makePlane(camera_forward, pivot);
        return first_click_point ? makePlane(camera_forward, *first_click_point) : std::nullopt;
    }

    std::optional<Ray> rayThroughPoints(const glm::vec3& near_point, const glm::vec3& far_point) {
        const glm::vec3 direction = far_point - near_point;
        const float length = glm::length(direction);
        if (!isFinite(near_point) || !isFinite(far_point) || !std::isfinite(length) || !(length > 1e-12f))
            return std::nullopt;
        return Ray{.origin = near_point, .direction = direction / length};
    }

    std::optional<glm::vec3> intersect(const Ray& ray, const Plane& plane) {
        const float denominator = glm::dot(ray.direction, plane.normal);
        if (!std::isfinite(denominator) || std::abs(denominator) < 1e-6f)
            return std::nullopt;
        const float t = glm::dot(plane.point - ray.origin, plane.normal) / denominator;
        if (!std::isfinite(t) || t < 0.0f)
            return std::nullopt;
        const glm::vec3 hit = ray.origin + ray.direction * t;
        return isFinite(hit) ? std::optional<glm::vec3>(hit) : std::nullopt;
    }

    std::optional<Frame> buildFrame(const std::span<const glm::vec3> world_points,
                                    const Plane& plane,
                                    const glm::vec3& camera_right) {
        const float normal_length = glm::length(plane.normal);
        if (world_points.size() < kMinVertices || world_points.size() > kMaxVertices ||
            !isFinite(plane.point) || !isFinite(plane.normal) || !(normal_length > 1e-8f))
            return std::nullopt;

        const glm::vec3 z = plane.normal / normal_length;
        glm::vec3 x = camera_right - glm::dot(camera_right, z) * z;
        if (!isFinite(x) || !(glm::length(x) > 1e-6f))
            return std::nullopt;
        x = glm::normalize(x);
        // x right, y = z cross x points down the image: the dataset camera's right-handed axes.
        const glm::vec3 y = glm::cross(z, x);

        std::vector<glm::vec2> in_plane;
        in_plane.reserve(world_points.size());
        glm::vec2 lo(std::numeric_limits<float>::max());
        glm::vec2 hi(std::numeric_limits<float>::lowest());
        for (const glm::vec3& point : world_points) {
            const glm::vec3 offset = point - plane.point;
            const glm::vec2 uv(glm::dot(offset, x), glm::dot(offset, y));
            lo = glm::min(lo, uv);
            hi = glm::max(hi, uv);
            in_plane.push_back(uv);
        }
        const glm::vec2 centre = (lo + hi) * 0.5f;
        for (glm::vec2& uv : in_plane)
            uv -= centre;
        if (!isValidOutline(in_plane))
            return std::nullopt;

        const glm::vec3 origin = plane.point + x * centre.x + y * centre.y;
        const glm::vec2 half_extents = (hi - lo) * 0.5f;
        const float half_depth = std::max(std::max(half_extents.x, half_extents.y), kMinHalfDepth);
        if (!std::isfinite(half_depth) || !isFinite(origin))
            return std::nullopt;

        return Frame{
            .world_from_local = glm::mat4(glm::vec4(x, 0.0f), glm::vec4(y, 0.0f), glm::vec4(z, 0.0f),
                                          glm::vec4(origin, 1.0f)),
            .vertices = std::move(in_plane),
            .half_depth = half_depth,
        };
    }

} // namespace lfs::core::crop_polygon
